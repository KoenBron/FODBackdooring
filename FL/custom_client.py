import os

import torch
from torch.utils.data import DataLoader

from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.dataset import CustomCocoDataset
from utils.global_settings import ProjectPaths, BackdoorProperties, FLSettings, TrainingSettings
from utils.noniid import (
    allocate_label_skew,
    composition_rows,
    global_marginal,
    print_composition_table,
    write_composition_csv,
)
from utils.partitioning import build_iid_partitions, partition_dataset

class ClientManager:
    def __init__(self, num_clients: int, logger):
        # Logger holds the run's output paths plus per-protocol sub-loggers
        # (e.g. logger.pilot_a). AttackProtocols read self.manager.logger
        # to dispatch their writes; keeps I/O concerns out of the protocols.
        self.logger = logger
        self.is_attacker = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create the benign and malicious subset for each client once
        ds_mal, ds_benign = self._setup_datasets(num_clients=num_clients)
        client_indices = self._build_client_partitions(ds_benign, num_clients)
        self.subsets_mal = partition_dataset(ds_mal, client_indices)
        self.subsets_benign = partition_dataset(ds_benign, client_indices)

        # Clients work on the same model instance but load in their parameters
        num_classes = ds_benign.get_total_classes_count() + 1
        self.wrapper = FasterRCNNWrapper(num_classes=num_classes)

        # Round-to-round cache of global weights. Rolled forward by update_round_cache().
        # Used by Neurotoxin to estimate the benign-gradient direction as theta_t - theta_{t-1}.
        self.prev_global_weights = None
        self.cached_global_weights = None
        self._neurotoxin_mask_logged = False

    def _build_client_partitions(self, ds_benign, num_clients):
        """One index list per client, then the composition report.

        Two modes. 'iid' is the historical seeded shuffle sliced into equal chunks
        (optionally target-quota rebalanced first) — unchanged. 'dirichlet' gives every
        client the same number of images but skews each client's CLASS composition; see
        utils/noniid.py for why sizes are held fixed.

        Both modes are fully derived from PARTITIONING_SEED (plus the partitioning flags
        and the annotations file), so a resume reproduces the split exactly with nothing
        persisted — the same property the IID path has always had. The only randomness is
        numpy Generators seeded from that value; nothing here reads module-level RNG state
        or iteration order of a dict/set. smoketest_noniid [7] holds that line.
        """
        labels, names = ds_benign.get_partition_labels(
            FLSettings.DIRICHLET_LABEL_RULE, FLSettings.PARTITIONING_SEED,
            FLSettings.DIRICHLET_LABEL_P)
        target_bucket = ds_benign.get_target_class_bucket(names)

        if FLSettings.PARTITION_MODE == "dirichlet":
            client_indices, requested = allocate_label_skew(
                labels=labels,
                n_buckets=len(names),
                n_clients=num_clients,
                attacker_ids=FLSettings.ATTACKER_IDS,
                alpha=FLSettings.DIRICHLET_ALPHA,
                seed=FLSettings.PARTITIONING_SEED,
                target_idx=target_bucket,
                target_min_count=FLSettings.ATTACKER_TARGET_MIN_COUNT,
                target_min_fraction=FLSettings.ATTACKER_TARGET_MIN_FRACTION,
            )
        else:
            client_indices = build_iid_partitions(
                len(ds_benign), num_clients, ds_benign.get_target_image_indices())
            # An IID partition is drawn from the global marginal, so that is the reference
            # its drift is measured against — giving the baseline row for the same CSV.
            requested = [global_marginal(labels, len(names))] * num_clients

        self._report_partitions(client_indices, labels, requested, names,
                                ds_benign.get_target_image_indices())
        return client_indices

    def _report_partitions(self, client_indices, labels, requested, names,
                           target_image_indices):
        """Write partition_composition.csv and print the table.

        Written for BOTH modes so an IID run produces a directly comparable baseline.
        The partitioning flags themselves are already recorded in config.txt's
        'Partitioning' section, and the partition is fully derived from them plus
        PARTITIONING_SEED, so the CSV is a view of the split rather than the record of it.
        """
        rows = composition_rows(
            client_indices, labels, requested, names, FLSettings.ATTACKER_IDS,
            target_name=BackdoorProperties.TARGET_CLASS,
            contains_target=set(target_image_indices),
        )
        out = getattr(self.logger, "output_path", None)
        if out is not None:
            write_composition_csv(os.path.join(out, "partition_composition.csv"), rows)
        print_composition_table(rows, names)

    def _setup_datasets(self, num_clients=None):
        """
        Returns the benign and malicious datasets in the dataset as a set
        """
        if num_clients is None:
            raise Exception("Client manager improperly instantiated, no total number of clients given when initializing")
        mal_ds = CustomCocoDataset(
            image_folder=ProjectPaths.TRAIN_IMAGES_DIR,
            annotations_file=ProjectPaths.TRAIN_ANNOTATIONS_FILE,
            width=BackdoorProperties.RESIZE_WIDTH,
            height=BackdoorProperties.RESIZE_HEIGHT,
            mode=FLSettings.ATTACKER_DATASET_MODE,
        )

        benign_ds = CustomCocoDataset(
            image_folder=ProjectPaths.TRAIN_IMAGES_DIR,
            annotations_file=ProjectPaths.TRAIN_ANNOTATIONS_FILE,
            width=BackdoorProperties.RESIZE_WIDTH,
            height=BackdoorProperties.RESIZE_HEIGHT,
            mode="always_clean",
        )

        return mal_ds, benign_ds
        
    def _retrieve_client_dataloader(self, client_index, is_attacker):
        """
        Retrieves the dataloader for the part of the dataset assigned to the given client based on client_index and selects from the correct update.
        client_index is 0-indexed.
        """
        if is_attacker:
            subsets = self.subsets_mal
        else:
            subsets = self.subsets_benign

        return DataLoader(
                subsets[client_index],
                batch_size=TrainingSettings.TRAINING_BATCH_SIZE,
                num_workers=TrainingSettings.NUM_WORKERS,
                collate_fn=FasterRCNNWrapper.get_collate_fn(),
                pin_memory=True,
                shuffle=True,
            )

    def update_round_cache(self, server_weights):
        """
        Roll the round-to-round cache so that prev_global_weights = theta_{t-1} and
        cached_global_weights = theta_t. Must be called once per round, before any
        client trains. Skipped entirely when Neurotoxin is disabled since nothing
        consumes the cache.

        No clone() needed: server.get_global_weights() already returns a fresh dict,
        and no downstream caller mutates the broadcast dict in place
        (set_weights -> load_state_dict copies; _do_model_replacement uses * which
        allocates new tensors).
        """
        if not FLSettings.NEUROTOXIN_ENABLED:
            return
        self.prev_global_weights = self.cached_global_weights
        self.cached_global_weights = server_weights

    def _compute_neurotoxin_mask(self, ratio):
        """
        Per-layer Neurotoxin mask. For each named parameter, build a binary tensor
        where M=0 on the top-`ratio` largest |theta_t - theta_{t-1}| coords (the
        directions benign clients are pushing hardest) and M=1 elsewhere.

        Returns dict[name -> mask Tensor on CPU] keyed by named-parameter name so the
        wrapper can apply it per-grad inside train_step.
        """
        mask = {}
        param_names = [n for n, _ in self.wrapper.model.named_parameters()]
        for name in param_names:
            if name not in self.cached_global_weights or name not in self.prev_global_weights:
                continue
            delta = (self.cached_global_weights[name] - self.prev_global_weights[name]).abs()
            flat = delta.flatten()
            if flat.numel() == 0:
                continue
            k = max(1, int(ratio * flat.numel()))
            # kthvalue with j = numel - k + 1 returns the k-th largest magnitude (the threshold).
            threshold = torch.kthvalue(flat, flat.numel() - k + 1).values
            # Bottom (1 - ratio) coords kept (mask=1), top `ratio` coords masked (mask=0).
            layer_mask = (delta < threshold).to(self.cached_global_weights[name].dtype)
            mask[name] = layer_mask

        # One-shot sanity print: confirms per-layer sparsity matches the requested ratio
        if not self._neurotoxin_mask_logged and mask:
            sparsities = [(name, (m == 0).float().mean().item()) for name, m in mask.items()]
            vals = [s for _, s in sparsities]
            print(f"\n[NEUROTOXIN] One-shot mask sanity check (ratio={ratio}):")
            print(f"  layers masked: {len(sparsities)}")
            print(f"  fraction-zeros per layer  min={min(vals):.4f}  mean={sum(vals)/len(vals):.4f}  max={max(vals):.4f}")
            print("  first/last 3 layers:")
            for name, s in sparsities[:3] + sparsities[-3:]:
                print(f"    {name}: zeros={s:.4f}")
            self._neurotoxin_mask_logged = True
        return mask

    def _replacement_gamma(self, m):
        """Bagdasaryan model-replacement scale  γ = k / (η · m),  clamped at
        FLSettings.GAMMA_CAP (None = uncapped).

        SINGLE SOURCE OF TRUTH for the replacement factor, shared by full model
        replacement (_do_model_replacement, whole state dict) and partial
        amplification (Partial protocol, carrying-set only). Both then apply the
        SAME  γ·local − (γ−1)·global  formula; they differ only in WHICH tensors
        it touches, not in how γ is derived.

        m: number of malicious clients that round.
        """
        k = FLSettings.CLIENTS_PER_ROUND
        eta = FLSettings.SERVER_LR

        gamma_uncapped = k / (eta * m)
        # GAMMA_CAP=None means "no cap" — honor it instead of crashing inside min().
        gamma = gamma_uncapped if FLSettings.GAMMA_CAP is None else min(gamma_uncapped, FLSettings.GAMMA_CAP)
        if FLSettings.GAMMA_CAP and gamma_uncapped > FLSettings.GAMMA_CAP:
            print(f"[MODEL REPLACEMENT] gamma capped: {gamma_uncapped:.3f} -> {gamma:.3f} (k={k}, eta={eta}, m={m})")
        return gamma

    def _do_model_replacement(self, local_weights, global_weights, m):
        """
        Scales the local weights so that — after server-side FedOpt aggregation with
        learning rate η — the global model lands on the attacker's backdoored model:

            w_submit = γ · local_weights − (γ − 1) · global_weights,   γ = k / (η · m)

        Derived from solving (1−η)·w_global + η·[(m/k)·w_submit + (1 − m/k)·w_global] = w_X
        under the standard Bagdasaryan assumption that the average benign client
        update is close to the prior global. γ is clamped at FLSettings.GAMMA_CAP to
        prevent norm explosion / trivial detection when η is small. With η = 1 and an
        uncapped γ this reduces to the original (k/m)·w_local − ((k−m)/m)·w_global.

        local_weights: tensor with the malicious update (unscaled)
        global_weights: tensor with the original server provided weights
        m: number of malicious clients that round
        """
        gamma = self._replacement_gamma(m)

        if gamma == 1.0:
            # No-op: η = 1 and m = k means submitting the local weights as-is reaches w_X
            return local_weights

        scale_global = gamma - 1.0

        replaced_weights = {}
        for key in global_weights:
            if local_weights[key].is_floating_point():
                replaced_weights[key] = (
                    gamma * local_weights[key] - scale_global * global_weights[key]
                )
            else:
                # Integer tensors don't contain learnable params that need to be amplified
                replaced_weights[key] = local_weights[key]

        return replaced_weights

    def get_parameters(self):
        # Delegate to wrapper to make client implementation model agnsostic
        return self.wrapper.get_weights()

    def set_weights(self, weights):
        # Delegate to wrapper to make client implementation model agnsostic
        self.wrapper.set_weights(weights=weights)

    def client_side_train(self, server_weights=None, client_config=None):
        """Dispatch entry point.

        Loads server_weights onto the wrapper and routes to the benign path or
        to the AttackProtocol selected by FLSettings.ATTACK_PROTOCOL. Per the
        Phase 0a/0b layout, the outer FL loop knows nothing about which
        protocol runs — every protocol must return the same
        (updated_params, num_samples, metrics) shape.
        """
        try:
            client_index = client_config["index"]
            client_is_attacker = client_config["is_attacker"]
        except KeyError:
            raise Exception("Client manager error: Failed to retrieve needed information from supplied client_config. Expected client_config keys: 'index', 'is_attacker'")

        print(f"  [client {client_index}] start", flush=True)
        if server_weights is None:
            raise Exception("Client manager error: No server_weights were supplied for training")

        # Load the broadcast model weights
        self.set_weights(weights=server_weights)

        # Benign clients
        if not client_is_attacker:
            return self._benign_path(client_index)

        # Attacker routing. Protocol imports are lazy so adding a new
        # AttackProtocol means touching only this dispatcher and the new file.
        protocol = FLSettings.ATTACK_PROTOCOL
    
        # Legacy malicious approach
        if protocol == "composable":
            return self._composable_attacker_path(server_weights, client_config)
        if protocol == "pilot_a":
            from FL.attack_protocols.pilot_a import PilotAProtocol
            return PilotAProtocol(self).run(server_weights, client_config)
        if protocol == "pilot_b":
            from FL.attack_protocols.pilot_b import PilotBProtocol
            return PilotBProtocol(self).run(server_weights, client_config)
        if protocol == "partial":
            from FL.attack_protocols.partial import PartialAmplificationProtocol
            return PartialAmplificationProtocol(self).run(server_weights, client_config)
        raise NotImplementedError(
            f"--attack-protocol {protocol!r} selected but not yet wired. "
            f"LaySelFL lands in a later phase."
        )

    # ---- Benign and composable-attacker paths ----

    def _benign_path(self, client_index):
        """Honest client. One pass on the clean partition; no grad mask, no
        augmenter, no submission scaling. Byte-identical to the pre-Phase-0b
        benign branch."""
        train_dl = self._retrieve_client_dataloader(client_index=client_index, is_attacker=False)
        _, num_samples, final_epoch_losses, self.optimizer = self.wrapper.train_step(
            dataloader=train_dl,
            epochs=TrainingSettings.BENIGN_CLIENT_LOCAL_EPOCHS,
            device=self.device,
            optimizer_config={"lr": TrainingSettings.BENIGN_CLIENT_LEARNING_RATE},
            partition_id=client_index,
            optimizer=None,
            grad_mask=None,
        )
        updated_params = self.get_parameters()
        return updated_params, num_samples, self._build_metrics(client_index, final_epoch_losses)

    def _composable_attacker_path(self, server_weights, client_config):
        """Today's customizable malicious-client procedure.

        Composes independently-toggled knobs:
          --enable-neurotoxin       : gradient mask built from theta_t - theta_{t-1}
          --enable-model-replacement: Bagdasaryan submission-time gamma-scaling
          --attack-type TARGETED_*  : dataset-level poisoning (handled inside
                                      CustomCocoDataset, not here)

        Lives on ClientManager rather than as an AttackProtocol subclass because
        its surface IS the knob set; the preset Protocols (PilotA / LaySelFL /
        Ours) have their own per-round procedures and don't expose these knobs.
        """
        client_index = client_config["index"]
        train_dl = self._retrieve_client_dataloader(client_index=client_index, is_attacker=True)

        # Neurotoxin mask: requires a cached previous global to estimate the
        # benign-gradient direction. First attack round after start/resume
        # falls back to no mask (prev_global_weights is None then).
        grad_mask = None
        if FLSettings.NEUROTOXIN_ENABLED and self.prev_global_weights is not None:
            grad_mask = self._compute_neurotoxin_mask(ratio=FLSettings.NEUROTOXIN_MASK_RATIO)

        _, num_samples, final_epoch_losses, self.optimizer = self.wrapper.train_step(
            dataloader=train_dl,
            epochs=TrainingSettings.MALICIOUS_CLIENT_LOCAL_EPOCHS,
            device=self.device,
            optimizer_config={"lr": TrainingSettings.MALICIOUS_CLIENT_LEARNING_RATE},
            partition_id=client_index,
            optimizer=None,
            grad_mask=grad_mask,
        )
        updated_params = self.get_parameters()

        if FLSettings.DO_MODEL_REPLACEMENT:
            updated_params = self._do_model_replacement(
                local_weights=updated_params,
                global_weights=server_weights,
                m=client_config["n_mal_clients"],
            )

        return updated_params, num_samples, self._build_metrics(client_index, final_epoch_losses)

    def _build_metrics(self, client_index, final_epoch_losses):
        """Shared metrics packaging — partition id + the averaged final-epoch
        loss components produced by wrapper.train_step."""
        metrics = {"partition_id": client_index}
        if final_epoch_losses is not None:
            for k, v in final_epoch_losses.items():
                if hasattr(v, 'item'):
                    metrics[k] = v.item()
                else:
                    metrics[k] = float(v)
        return metrics

