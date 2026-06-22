"""PilotA observational protocol.

Purpose
-------
First malicious round in the per-group constrain-and-scale pipeline. PilotA is
purely OBSERVATIONAL: it runs three trajectories from the same theta_t in
lockstep — one malicious and two benign — ALL ON THE ATTACKER'S OWN PARTITION,
differentiated only by data poisoning (mal vs benign) and RNG seed (ben1 vs
ben2). Computes per-group L2 divergence with a real benign-drift floor b_g,
runs the candidate selectors at a sweep of (k, c), and calibrates beta_max via
measure-without-apply probe steps AT THE BROADCAST THETA_T (where L_det's
gradients are sharpest — see "Round flow" below). It then SUBMITS the
benign-anchor (T_ben1) endpoint so global aggregation receives an honest-
looking update.

Round flow (mirrors what the real OursProtocol will do)
-------------------------------------------------------
    1. ben1 trajectory      : produces T_ben1 (the benign anchor)
    2. beta_max calibration : probes at theta_t against anchor T_ben1
    3. mal trajectory       : (pilot does not apply beta_max — observational)
    4. ben2 trajectory       : (pilot-only — produces the b_g noise floor)

beta_max sits between ben1 and mal because in production the malicious round
needs beta_max ready BEFORE it starts training. Probing at theta_t (rather
than the post-training mal endpoint) sizes the penalty against the largest
L_det gradients the round will see — the same regime in which the penalty
will be applied during malicious training.

Trajectory triple (load-bearing design choice)
----------------------------------------------
    mal   : subsets_mal[client_id],    seed = S       (poisoned, fixed RNG)
    ben1  : subsets_benign[client_id], seed = S       (clean counterfactual)
    ben2  : subsets_benign[client_id], seed = S + 1   (RNG noise probe)

This isolates two things cleanly:
  - div_g = ||mal - ben1|| / sqrt(d_g) measures the poison's effect on group g,
    holding partition and seed fixed. Anything moving is the trigger insertion
    and label flip.
  - b_g  = ||ben1 - ben2|| / sqrt(d_g) measures the RNG noise floor on the SAME
    partition under the SAME (clean) data. The only remaining stochastic source
    is the PRNG, so b_g is what RNG alone produces for this group.

Subsets_mal and subsets_benign are partitioned with the same shuffled indices
(see ClientManager.__init__), so subsets_benign[client_id] is exactly the
clean version of subsets_mal[client_id] — no data-heterogeneity confound.

PilotA does not write to AttackState — it does not drive subsequent rounds.

Outputs
-------
One JSON file per (round, client_id) under <output_path>/pilot_a/:

    {
      "round": r, "client_id": i, "n_epochs": N,
      "groups": [...], "group_dim": {g: d_g},
      "epochs": [
        {"n": 1, "div": {g: ...}, "b": {g: ...}, "rho": {g: ...},
         "selections": {"topk": {k: [...]}, "cutoff": {c: [...]}}},
        ...
      ],
      "beta_max": {
        "value_ratios": [...], "grad_ratios": [...],
        "value_median": ..., "grad_median": ...,
        "agreement_ratio": ...
      }
    }

Per-group scalars only (~5 KB per round). Per-tensor data is intentionally
not persisted — pilot 2 (LaySelFL cross-granularity comparison) will add
per-tensor dumps when it lands.
"""
from __future__ import annotations

import json
import math
import os
import random as pyrandom
import statistics

import torch
from torch.utils.data import DataLoader

try:
    import numpy as _np
except ImportError:
    _np = None

from FL.attack_protocols.base import AttackProtocol
from FL.groups import build_group_map, group_l2, per_tensor_sq_l2
from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.global_settings import FLSettings, TrainingSettings


# Sweep grids. Final-epoch values are the headline; per-epoch series feed the
# stability plot. Picked to span an interesting range: topk from "is just
# cls_score enough?" to "almost all heads"; cutoff from "barely above benign
# drift" to "clearly above benign drift".
_TOPK_VALUES = (1, 3, 5, 7)
_CUTOFF_VALUES = (1.5, 2.0, 3.0)

# Median of 10 probe ratios is robust to outliers; more would burn cluster
# time without sharpening the conclusion.
_NUM_PROBE_STEPS = 10


def _worker_init_fn(worker_id):
    """Per-worker RNG seeding for num_workers > 0.

    PyTorch auto-seeds each worker's torch RNG from the DataLoader's
    generator + worker_id; torch.initial_seed() inside the worker returns
    that derived seed. We use it to seed numpy and python random so any
    transform/augmentation that consults them is also deterministic per-
    worker. Without this, those two RNGs drift between runs even with a
    seeded DataLoader generator.
    """
    base = torch.initial_seed() % (2**32)
    pyrandom.seed(base)
    if _np is not None:
        _np.random.seed(base)


class PilotALogger:
    """Sub-logger owned by ModelLogger; writes one JSON per (round, client) to
    <output>/attack_logs/pilot_a/. The schema is owned here next to
    PilotAProtocol so a payload change updates writer and reader together
    without crossing the utils/FL package boundary.
    """

    def __init__(self, attack_logs_dir):
        self.dir = os.path.join(attack_logs_dir, "pilot_a")
        os.makedirs(self.dir, exist_ok=True)

    def write_round(self, current_round, client_id, n_epochs, groups, group_dim, epochs, beta_max):
        target = os.path.join(self.dir, f"round_{current_round}_client_{client_id}.json")
        payload = {
            "round": current_round,
            "client_id": client_id,
            "n_epochs": n_epochs,
            "groups": groups,
            "group_dim": group_dim,
            "epochs": epochs,
            "beta_max": beta_max,
        }
        with open(target, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(
            f"  [pilot_a client={client_id} round={current_round}] log written: {target}",
            flush=True,
        )
        # Return the path relative to the run dir so callers can stash it
        # in their metrics without re-deriving it.
        return os.path.join("attack_logs", "pilot_a", os.path.basename(target))


class PilotAProtocol(AttackProtocol):
    """Three-trajectory observational localizer. See module docstring."""

    name: str = "pilot_a"

    def run(self, server_weights, client_config):
        client_id = client_config["index"]
        current_round = client_config.get("round")
        if current_round is None:
            raise KeyError(
                "PilotAProtocol requires 'round' in client_config — added in "
                "custom_fl_train.py's per-client client_config build."
            )

        n_epochs = TrainingSettings.MALICIOUS_CLIENT_LOCAL_EPOCHS

        # Seed recipe: mal and ben1 share seed S so the only difference between
        # their trajectories is the poison. ben2 uses S+1 so b_g measures pure
        # RNG noise on the same (clean) partition. The 1000* spacing keeps
        # different rounds' seeds well apart.
        seed_mal = FLSettings.SEED + 1000 * int(current_round) + int(client_id)
        seed_ben1 = seed_mal
        seed_ben2 = seed_mal + 1
        print(
            f"  [pilot_a client={client_id} round={current_round}] "
            f"trajectories on partition {client_id}: "
            f"mal(seed={seed_mal}) / ben1(seed={seed_ben1}) / ben2(seed={seed_ben2})",
            flush=True,
        )

        # Trajectory order mirrors production: ben1 (anchor) -> beta_max -> mal -> ben2.
        # ben2 is pilot-only (noise floor for b_g); production would stop after mal.
        # Each trajectory restores wrapper state from server_weights, seeds the
        # RNGs, builds a fresh optimizer, and collects per-epoch CPU snapshots.
        ben1_snaps = self._run_trajectory(
            server_weights, partition_id=client_id, is_attacker=False,
            n_epochs=n_epochs, label="ben1", seed=seed_ben1,
        )

        # Beta_max calibration AT THETA_T against the just-computed anchor.
        # The probe dataloader is built with seed=seed_mal+2 so the probe-step
        # order is reproducible run-to-run (matches the seed passed below).
        # This matches the production flow: in the real OursProtocol the
        # calibration runs at the top of the malicious round (after ben1, before
        # mal training) so beta_max is ready as a coefficient on the constrain
        # penalty during the upcoming mal trajectory. Probing at theta_t sizes
        # the penalty against the largest L_det gradients the round will see —
        # probing at the mal endpoint (where L_det has decayed and ||theta - anchor||
        # has grown) would under-calibrate beta_max for the deployed regime.
        # Probe gradients discarded; only the two scalar ratios per step survive.
        mal_dl = self._dataloader_for(client_id, is_attacker=True, seed=seed_mal + 2)
        beta_max = self._calibrate_beta_max(
            probe_state=server_weights,
            anchor_state=ben1_snaps[-1],
            dataloader=mal_dl,
            seed=seed_mal + 2,
        )

        mal_snaps = self._run_trajectory(
            server_weights, partition_id=client_id, is_attacker=True,
            n_epochs=n_epochs, label="mal", seed=seed_mal,
        )
        ben2_snaps = self._run_trajectory(
            server_weights, partition_id=client_id, is_attacker=False,
            n_epochs=n_epochs, label="ben2", seed=seed_ben2,
        )

        # Per-epoch metrics. Pure CPU pass over the snapshots — no GPU needed.
        epochs_data = self._compute_epoch_metrics(mal_snaps, ben1_snaps, ben2_snaps)

        log_path = self.manager.logger.pilot_a.write_round(
            current_round=current_round,
            client_id=client_id,
            n_epochs=n_epochs,
            groups=epochs_data["groups"],
            group_dim=epochs_data["group_dim"],
            epochs=epochs_data["epochs"],
            beta_max=beta_max,
        )
        self._print_headline(epochs_data, beta_max, client_id, current_round)

        # Submit T_ben1 endpoint so aggregation receives a benign-looking update.
        # PilotA is observational; no scaling, no model replacement.
        num_samples = len(self.manager.subsets_mal[client_id])
        metrics = {
            "partition_id": client_id,
            "protocol": "pilot_a",
            "pilot_log": log_path,
        }
        return ben1_snaps[-1], num_samples, metrics

    # ---- Trajectory execution ----

    def _dataloader_for(self, partition_id, is_attacker, seed=None):
        """Mirror of ClientManager._retrieve_client_dataloader, exposed locally
        so we can route between the poisoned and clean view of the SAME
        partition without going through the manager's routing layer.

        `seed` (optional): when provided, scopes data-side RNG to this DL
        instance. The torch.Generator drives the RandomSampler's shuffle
        order; worker_init_fn re-seeds each worker's numpy + python random
        from torch.initial_seed() so transforms running in worker processes
        are also deterministic. (Without worker_init_fn, torch auto-seeds
        only the worker's torch RNG; numpy/random would still drift between
        runs.)

        Partition assignment is unaffected — that happens once in
        ClientManager.__init__ from a separately seeded shuffle, well before
        any DataLoader exists.
        """
        subsets = self.manager.subsets_mal if is_attacker else self.manager.subsets_benign
        generator = None
        init_fn = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
            init_fn = _worker_init_fn
        return DataLoader(
            subsets[partition_id],
            batch_size=TrainingSettings.TRAINING_BATCH_SIZE,
            num_workers=TrainingSettings.NUM_WORKERS,
            collate_fn=FasterRCNNWrapper.get_collate_fn(),
            pin_memory=True,
            shuffle=True,
            generator=generator,
            worker_init_fn=init_fn,
        )

    def _seed_model_rngs(self, seed):
        """Seed RNGs used by the MODEL forward path.

        Faster R-CNN's RPN positive/negative subsampling calls torch.randperm
        on torch.default_generator (or its CUDA counterpart), which can't be
        scoped via a DataLoader's generator — so we have to set it globally.
        Same for any other model-internal torch.randX call.

        Data-side RNG (DataLoader shuffle + per-worker transforms) is scoped
        per-DL via _dataloader_for's generator + worker_init_fn — those don't
        need this.

        Residual non-determinism we accept: non-deterministic cuDNN kernels
        (turning these off via torch.use_deterministic_algorithms breaks some
        torchvision ops and isn't worth it for a noise-floor measurement —
        the residual shows up as a constant in b_g, where it belongs).
        """
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def _run_trajectory(self, server_weights, partition_id, is_attacker, n_epochs, label, seed):
        """One N-epoch SGD trajectory from theta_t. Returns a list of N CPU
        state_dict snapshots (one per epoch end). Fresh optimizer each call —
        no momentum carryover from prior trajectories.

        Seeding is split: the DataLoader carries its own generator +
        worker_init_fn (scoped to this DL only), and torch.default_generator
        is reseeded right before train_step iterates so the model's internal
        RNG (RPN sampler, etc.) is in scope too.
        """
        self.manager.wrapper.set_weights(server_weights)
        dl = self._dataloader_for(partition_id, is_attacker=is_attacker, seed=seed)
        lr = (
            TrainingSettings.MALICIOUS_CLIENT_LEARNING_RATE
            if is_attacker else TrainingSettings.BENIGN_CLIENT_LEARNING_RATE
        )

        self._seed_model_rngs(seed)

        snapshots = []

        def callback(epoch_idx, state_cpu):
            snapshots.append(state_cpu)

        self.manager.wrapper.train_step(
            dataloader=dl,
            epochs=n_epochs,
            device=self.manager.device,
            optimizer_config={"lr": lr},
            partition_id=partition_id,
            optimizer=None,
            grad_mask=None,
            loss_augment_fn=None,
            freeze_groups=None,
            epoch_callback=callback,
        )
        if len(snapshots) != n_epochs:
            raise RuntimeError(
                f"PilotA trajectory '{label}' collected {len(snapshots)} snapshots, "
                f"expected {n_epochs}. epoch_callback should fire once per epoch."
            )
        return snapshots

    # ---- Per-epoch metrics ----

    def _compute_epoch_metrics(self, mal_snaps, ben1_snaps, ben2_snaps):
        """Per-epoch div_g, b_g, rho_g + selector outputs.

        All three trajectories start from theta_t, so
            Delta_X,g - Delta_Y,g = (theta_X,n - theta_t) - (theta_Y,n - theta_t)
                                  = theta_X,n - theta_Y,n
        Pairwise differences between same-epoch states are equivalent to
        differences-of-deltas — saves one redundant subtraction per epoch.
        """
        model = self.manager.wrapper.model
        tensor_list, tensor_to_group, group_to_tensors = build_group_map(model)
        param_numel = {n: p.numel() for n, p in model.named_parameters()}
        group_dim = {
            g: sum(param_numel[t] for t in group_to_tensors[g])
            for g in group_to_tensors
        }
        sqrt_d = {g: math.sqrt(group_dim[g]) for g in group_dim}

        epochs_data = []
        for n, (m, b1, b2) in enumerate(zip(mal_snaps, ben1_snaps, ben2_snaps), start=1):
            # L2 norm of the malicious-benign divergence
            l2_mal_ben1 = group_l2(per_tensor_sq_l2(m, b1, tensor_list), tensor_to_group)
            # L2 norm of the benign-benign divergence
            l2_ben1_ben2 = group_l2(per_tensor_sq_l2(b1, b2, tensor_list), tensor_to_group)

            div_g = {g: l2_mal_ben1[g] / sqrt_d[g] for g in l2_mal_ben1}
            b_g = {g: l2_ben1_ben2[g] / sqrt_d[g] for g in l2_ben1_ben2}
            rho_g = {g: max(0.0, div_g[g] - b_g[g]) for g in div_g}

            selections = {
                "topk": {str(k): _select_topk(rho_g, k) for k in _TOPK_VALUES},
                "cutoff": {str(c): _select_cutoff(div_g, b_g, c) for c in _CUTOFF_VALUES},
            }

            epochs_data.append({
                "n": n,
                "div": div_g,
                "b": b_g,
                "rho": rho_g,
                "selections": selections,
            })
        return {
            "group_dim": group_dim,
            "groups": list(group_dim.keys()),
            "epochs": epochs_data,
        }

    # ---- Beta_max calibration ----

    def _calibrate_beta_max(self, probe_state, anchor_state, dataloader, seed):
        """Probe-step calibration: how big a penalty matches the task loss?

        value_ratio = L_det / penalty             (scalar magnitude match)
        grad_ratio  = ||grad L_det|| / ||grad penalty||

        `probe_state` is the model state at which the ratios are measured.
        In production OursProtocol (and now in this pilot) it's the broadcast
        theta_t — the point where L_det's gradients are sharpest and where
        the penalty will first apply during the upcoming malicious round.

        Penalty is sum_p ||theta_p - anchor_p||^2 over ALL params (the worst-
        case non-carrying set; the real attack will exclude carrying-group
        params from the sum). For pilot 1 the simplification is intentional:
        we just want to confirm value_ratio and grad_ratio agree, which is
        roughly invariant to the param subset.

        Probe gradients are computed via torch.autograd.grad and discarded —
        no .grad accumulation, no optimizer.step(). Model state is restored
        to probe_state before probing and not used afterwards (the protocol
        returns the pre-saved ben1 snapshot, not the probed wrapper state).
        BN running stats may shift slightly under the 10 train()-mode
        forwards; subsequent trajectories' set_weights(server_weights) wipes
        that, so no contamination of the mal/ben2 trajectories.

        `seed` makes the probe order (DataLoader shuffle, RPN sampling)
        reproducible across pilot runs with the same FLSettings.SEED.
        """
        wrapper = self.manager.wrapper
        wrapper.set_weights(probe_state)
        self._seed_model_rngs(seed)
        model = wrapper.model
        device = self.manager.device
        model.to(device)
        model.train()

        # Anchor as fp32 constants on device. Same fp32 reasoning as the
        # loss_augment_fn cast in wrapper.train_step — tiny anchor distances
        # round to zero in fp16 and a penalty would silently vanish.
        anchor_dev = {
            n: t.to(device).float() for n, t in anchor_state.items() if t.is_floating_point()
        }
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        param_list = [p for _, p in named]

        value_ratios = []
        grad_ratios = []

        it = iter(dataloader)
        steps_done = 0
        while steps_done < _NUM_PROBE_STEPS:
            # Iterate throught the batches
            try:
                images, targets = next(it)
            except StopIteration:
                it = iter(dataloader)
                images, targets = next(it)

            # Batch the images and their targets
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [
                {k: v.to(device, non_blocking=True) for k, v in t.items()}
                for t in targets
            ]

            # Build detection loss
            loss_dict = model(images, targets)
            L_det = sum(loss_dict.values()).float()

            # Build penalty loss (from Tensors that have their gradietns checked)
            penalty = torch.zeros((), device=device)
            for name, p in named:
                if name in anchor_dev:
                    # Build squared l2 loss
                    penalty = penalty + (p.float() - anchor_dev[name]).pow(2).sum()# .float() for fp32 precision

            # Get loss values to CPU
            v_L = L_det.detach().item()
            v_pen = penalty.detach().item()
            if v_pen > 0:
                value_ratios.append(v_L / v_pen)

            # Separate-graph autograd.grad calls. Each call consumes its own
            # graph; the two graphs share the param leaves but are otherwise
            # independent (model forward vs scalar sum over params), so no
            # retain_graph is needed.
            grads_L = torch.autograd.grad(
                L_det, param_list, retain_graph=False, allow_unused=True,
            )
            grads_P = torch.autograd.grad(
                penalty, param_list, retain_graph=False, allow_unused=True,
            )
            
            # Compute l2 norm of the gradients
            norm_L = math.sqrt(sum(g.detach().pow(2).sum().item() for g in grads_L if g is not None))
            norm_P = math.sqrt(sum(g.detach().pow(2).sum().item() for g in grads_P if g is not None))
            if norm_P > 0:
                grad_ratios.append(norm_L / norm_P)

            steps_done += 1

        model.cpu()
        torch.cuda.empty_cache()

        v_med = statistics.median(value_ratios) if value_ratios else None
        g_med = statistics.median(grad_ratios) if grad_ratios else None
        agreement = None
        if v_med is not None and g_med is not None and min(v_med, g_med) > 0:
            agreement = max(v_med, g_med) / min(v_med, g_med)

        return {
            "value_ratios": value_ratios,
            "grad_ratios": grad_ratios,
            "value_median": v_med,
            "grad_median": g_med,
            "agreement_ratio": agreement,
        }

    # ---- Headline print ----

    def _print_headline(self, epochs_data, beta_max, client_id, current_round):
        """Eye-level summary of the round so the user doesn't have to open
        the JSON to check whether the pilot is producing sensible numbers."""
        final = epochs_data["epochs"][-1]
        ranked = sorted(final["rho"].items(), key=lambda kv: -kv[1])
        top3 = ", ".join(f"{g}={v:.3g}" for g, v in ranked[:3])
        v_med = beta_max.get("value_median")
        g_med = beta_max.get("grad_median")
        agree = beta_max.get("agreement_ratio")
        print(f"[pilot_a client={client_id} round={current_round}] HEADLINE:")
        print(f"    final-epoch rho_g top-3: {top3}")
        print(
            f"    beta_max: value_med={_fmt(v_med)}  grad_med={_fmt(g_med)}  "
            f"agreement={_fmt(agree)}"
        )


# ---- Selectors (module-level so the analysis script can reuse them) ----

def _select_topk(score_map, k):
    """Top-k groups by score (descending). Returns sorted list for stable diffs."""
    ranked = sorted(score_map.items(), key=lambda kv: -kv[1])
    return sorted(g for g, _ in ranked[:k])


def _select_cutoff(div_map, b_map, c):
    """Groups whose div exceeds c * benign-drift floor."""
    return sorted(g for g in div_map if div_map[g] > c * b_map[g])


def _fmt(x):
    return f"{x:.3g}" if isinstance(x, (int, float)) else str(x)
