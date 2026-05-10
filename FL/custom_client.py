import torch
from torch.utils.data import DataLoader

from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.dataset import CustomCocoDataset
from utils.global_settings import ProjectPaths, BackdoorProperties, FLSettings, TrainingSettings
from utils.partitioning import get_shuffled_indices, partition_dataset

class ClientManager:
    def __init__(self, num_clients: int):
        self.is_attacker = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create the benign and malicious subset for each client once
        ds_mal, ds_benign = self._setup_datasets(num_clients=num_clients)
        shuffled_indices = get_shuffled_indices(len(ds_benign))
        self.subsets_mal, self.subsets_benign = partition_dataset(dataset=ds_mal, n_clients=num_clients, indices=shuffled_indices), partition_dataset(dataset=ds_benign, n_clients=num_clients, indices=shuffled_indices)

        # Clients work on the same model instance but load in their parameters
        num_classes = ds_benign.get_total_classes_count() + 1
        self.wrapper = FasterRCNNWrapper(num_classes=num_classes)

        # Round-to-round cache of global weights. Rolled forward by update_round_cache().
        # Used by Neurotoxin to estimate the benign-gradient direction as theta_t - theta_{t-1}.
        self.prev_global_weights = None
        self.cached_global_weights = None
        self._neurotoxin_mask_logged = False

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
        client trains. The clone() ensures later in-place ops on server_weights don't
        retroactively mutate the cache.
        """
        self.prev_global_weights = self.cached_global_weights
        self.cached_global_weights = {k: v.clone() for k, v in server_weights.items()}

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

    def _do_model_replacement(self, local_weights, global_weights, m):
        """
        Scales the local weights according to model replacement:
            w_mal = (k / m) * local_weights - (k - m) / m * global_weights

        local_weights: tensor with the malicious update (unscaled)
        global_weights: tensor with the original server provided weights
        m: number of malicious clients that round
        """
        k = FLSettings.CLIENTS_PER_ROUND

        if m==k:
            return local_weights

        scale_mal = k / m
        scale_global = (k - m) / m

        replaced_weights = {}
        for key in global_weights:
            if local_weights[key].is_floating_point():
                replaced_weights[key] = (
                    scale_mal * local_weights[key] - scale_global * global_weights[key]
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
        """
        First loads the server_weights onto the client manager model and then
        performs the client side training and returns the training results as pytorch tensors
        """
        try:
            client_index = client_config["index"]
            client_is_attacker = client_config["is_attacker"]
        except KeyError:
            raise Exception("Client manager error: Failed to retrieve needed information from supplied client_config. Expected client_config keys: 'index', 'is_attacker'")

        if server_weights is None:
            raise Exception("Client manager error: No server_weights were supplied for training")

        # Load starting point for training
        self.set_weights(weights=server_weights)
        train_dl = self._retrieve_client_dataloader(client_index=client_index, is_attacker=client_is_attacker)

        num_epochs = TrainingSettings.MALICIOUS_CLIENT_LOCAL_EPOCHS if client_is_attacker else TrainingSettings.BENIGN_CLIENT_LOCAL_EPOCHS

        # Build Neurotoxin mask for malicious clients when enabled and a previous-round
        # global is cached. First attacker round after start/resume falls back to no mask.
        grad_mask = None
        if client_is_attacker and FLSettings.NEUROTOXIN_ENABLED and self.prev_global_weights is not None:
            grad_mask = self._compute_neurotoxin_mask(ratio=FLSettings.NEUROTOXIN_MASK_RATIO)

        _, num_samples, final_epoch_losses, self.optimizer = self.wrapper.train_step(
            dataloader=train_dl,
            epochs= num_epochs,
            device=self.device,
            optimizer_config={},
            partition_id=client_index,
            optimizer=None,
            grad_mask=grad_mask,
        )

        # Retrieve the updated parameters
        updated_params = self.get_parameters()

        # Bundle metrics and info needed by the logger
        metrics = {
            "partition_id": client_index
        }
        if final_epoch_losses is not None:
            for k, v in final_epoch_losses.items():
                if hasattr(v, 'item'):
                    # Tensor case
                    metrics[k] = v.item()
                else:
                    # Just in case
                    metrics[k] = float(v)

        # Amplify backdoor using model replacement if enabled
        if client_is_attacker and FLSettings.DO_MODEL_REPLACEMENT:
            updated_params = self._do_model_replacement(local_weights=updated_params, global_weights=server_weights, m=client_config["n_mal_clients"])

        return updated_params, num_samples, metrics

