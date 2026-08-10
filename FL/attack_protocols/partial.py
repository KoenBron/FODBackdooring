"""Partial-amplification attack protocol (--attack-protocol partial).

Standalone AttackProtocol. It deliberately does NOT inherit from the localization
pilots (PilotA / PilotB) — those are diagnostic scaffolding, not thesis machinery.
The trajectory engine (_run_trajectory / _dataloader_for / _seed_model_rngs) is
COPIED from PilotA and simplified to return only the trajectory ENDPOINT; this
protocol needs the final weights, not the per-epoch snapshot series.

The layer-selection math lives in FL.scoring.l2_based / FL.groups, which are
shared libraries (not pilot code), so it is IMPORTED rather than copied.

INCREMENT 2 SCOPE — dynamic per-tensor carrying-layer selection + amplification.
Each round, on the attacker's own partition:
  1. train a benign anchor (ben1) and a PRNG-noise reference (ben2) on the clean view,
  2. train a malicious trajectory theta_mal on the poisoned view,
  3. select the carrying set C = eligible tensors whose DENOISED per-tensor
     divergence of (theta_mal vs ben1) clears the ben2 floor — PilotB's math,
  4. persist C to AttackState, then AMPLIFY only Delta_C = theta_mal - theta_t by
     PARTIAL_AMP_GAMMA and submit (non-carrying params are submitted as theta_mal).

The norm budget / relocation step (pulling the total ||Delta|| back down to a
benign norm) is a LATER increment: with gamma > 1 this submission's total norm is
elevated, which is expected here — the point of this increment is to concentrate
the update energy in C, not yet to hide it.
"""
import json
import math
import os
import random

import torch
from torch.utils.data import DataLoader

from FL.attack_protocols.base import AttackProtocol
from FL.attack_protocols.amplify import partial_amplify
from FL.attack_state import AttackState
from FL.groups import build_group_map, build_eligibility
from FL.scoring.l2_based import (
    numel_per_tensor,
    squared_divergence_per_tensor,
    identity_partition,
    normalized_l2_divergence,
    select_topk_signed,
    select_cutoff,
)
from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.global_settings import FLSettings, TrainingSettings

try:
    import numpy as _np
except ImportError:
    _np = None


def _worker_init_fn(worker_id):
    """Per-worker RNG seeding for num_workers > 0 (copied from PilotA).

    torch auto-seeds each worker's torch RNG from the DataLoader generator +
    worker_id; torch.initial_seed() inside the worker returns that derived seed.
    We reuse it to seed numpy and python random so any transform that consults
    them is deterministic per-worker too.
    """
    base = torch.initial_seed() % (2**32)
    random.seed(base)
    if _np is not None:
        _np.random.seed(base)


class PartialLogger:
    """Sub-logger owned by ModelLogger; one JSON per (round, client) under
    <output>/attack_logs/partial/. Schema owned here next to the protocol, mirroring
    PilotBLogger. Torch-free (plain dict -> JSON), so it never touches the GPU.
    """

    def __init__(self, attack_logs_dir):
        self.dir = os.path.join(attack_logs_dir, "partial")
        os.makedirs(self.dir, exist_ok=True)

    def write_round(self, current_round, client_id, select_mode, select_param, gamma,
                    carrying, carrying_by_group, top_carriers, norms):
        target = os.path.join(self.dir, f"round_{current_round}_client_{client_id}.json")
        payload = {
            "round": current_round,
            "client_id": client_id,
            "select_mode": select_mode,
            "select_param": select_param,
            "gamma": gamma,
            "carrying_size": len(carrying),
            "carrying": sorted(carrying),
            "carrying_by_group": carrying_by_group,   # {group_name: #carrying tensors}
            "top_carriers": top_carriers,             # [[name, score], ...] (div or rho)
            "norms": norms,                           # see _norm_diagnostics
        }
        with open(target, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"  [partial client={client_id} round={current_round}] log written: {target}", flush=True)
        return os.path.join("attack_logs", "partial", os.path.basename(target))


class PartialAmplificationProtocol(AttackProtocol):
    """Dynamic per-tensor carrying-layer selection (+ amplification, Increment 2)."""

    name: str = "partial"

    # ---- trajectory engine (copied from PilotA, endpoint-only) ----

    def _dataloader_for(self, partition_id, is_attacker, seed=None):
        """Route between the poisoned and clean view of the SAME partition. When
        `seed` is given, a torch.Generator scopes the sampler shuffle and
        worker_init_fn re-seeds each worker's numpy/python RNG for determinism.
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
        """Seed the MODEL forward-path RNG (Faster R-CNN RPN subsampling calls
        torch.randperm on the default generator, which a DataLoader generator
        cannot scope), so trajectories are reproducible across runs.
        """
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def _run_trajectory(self, server_weights, partition_id, is_attacker, n_epochs, seed):
        """One N-epoch SGD trajectory from theta_t; returns the FINAL CPU state_dict.

        Only the endpoint is needed — this protocol does no per-epoch analysis — so the
        final weights are read with get_weights() after training. No per-epoch snapshot
        series is collected (that would hold N full state_dicts in CPU RAM for nothing).

        Fresh optimizer each call (no momentum carryover). The mal and benign
        trajectories MUST train at the same LR: a mal/ben LR gap would confound the
        per-tensor divergence (theta_mal vs T_ben) with LR-induced movement rather
        than the poison — the same invariant the pilots rely on.
        """
        assert (
            TrainingSettings.MALICIOUS_CLIENT_LEARNING_RATE
            == TrainingSettings.BENIGN_CLIENT_LEARNING_RATE
        ), (
            "partial protocol trajectories require MALICIOUS_CLIENT_LEARNING_RATE "
            "== BENIGN_CLIENT_LEARNING_RATE; a differing LR confounds the "
            "divergence signal with an LR artifact."
        )
        self.manager.wrapper.set_weights(server_weights)
        dl = self._dataloader_for(partition_id, is_attacker=is_attacker, seed=seed)
        lr = (
            TrainingSettings.MALICIOUS_CLIENT_LEARNING_RATE
            if is_attacker else TrainingSettings.BENIGN_CLIENT_LEARNING_RATE
        )
        self._seed_model_rngs(seed)

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
        )
        # train_step leaves the trained model on CPU; get_weights() returns a fresh
        # CPU-cloned full state_dict — the trajectory endpoint.
        return self.manager.wrapper.get_weights()

    # ---- layer selection (PilotB math; imported atoms) ----

    def _select_carrying(self, theta_mal, theta_ben1, theta_ben2):
        """Per-tensor carrying-layer selection (PilotB math), restricted to ELIGIBLE
        tensors (trainable, non-batch-norm) via FL.groups.build_eligibility.

            div_t = ||theta_mal - theta_ben1||_t / sqrt(numel_t)   (per-tensor normalized divergence)
            b_t   = ||theta_ben1 - theta_ben2||_t / sqrt(numel_t)  (PRNG-noise floor from a 2nd benign run)
            select:
              'cutoff' -> { t : div_t > PARTIAL_CUTOFF * b_t }        (variable-size; PilotB's selector;
                                                                       cutoff=1.0 is exactly the rho>0 gate)
              'topk'   -> { top-PARTIAL_TOPK t by SIGNED rho_t = div_t - b_t (unclamped,
                                                       ungated) — floor still subtracted,
                                                       but always returns exactly k }

        The floor is the per-tensor PRNG-noise floor from a second benign trajectory
        (ben1 vs ben2). The cheap cross-tensor sigma floor was dropped: the pilots
        showed it does not separate carriers (strong carriers self-inflate a single
        global threshold), so only the per-tensor ben2 floor remains.

        BN exclusion is EXPLICIT here: build_eligibility detects batch-norm by module
        TYPE (not name), so learnable BN scale/shift never enter div_t — unlike PilotB,
        which excludes only requires_grad=False tensors and relies on this model's BN
        being FrozenBatchNorm (buffers, not parameters) to keep BN out incidentally.
        The eligible set also matches the stealth harness's compute_delta coordinate
        set, so the attack and the measurement agree on what an update is.

        Returns a dict with keys: carrying (set), ranked ([(name, score)] descending —
        score is div in cutoff mode, rho in topk mode), div_t, elig_list (eligible
        tensor names, for the norm diagnostics), tensor_to_group (for the group rollup).
        """
        model = self.manager.wrapper.model
        tensor_list, tensor_to_group, _ = build_group_map(model)
        numel = numel_per_tensor(model)
        id_part = identity_partition(tensor_list)
        eligible_names, _eligible_buckets = build_eligibility(model, id_part)
        elig_list = [t for t in tensor_list if t in eligible_names]

        # div_t over ELIGIBLE tensors only (BN + frozen never enter the selection).
        sq_div = squared_divergence_per_tensor(theta_mal, theta_ben1, elig_list)
        div_t = normalized_l2_divergence(sq_div, numel, id_part)   # keys = eligible tensors

        # Per-tensor PRNG-noise floor from the second benign trajectory.
        sq_b = squared_divergence_per_tensor(theta_ben1, theta_ben2, elig_list)
        b_t = normalized_l2_divergence(sq_b, numel, id_part)

        mode = FLSettings.PARTIAL_SELECT_MODE
        if mode == "cutoff":
            # div_t > cutoff * b_t; variable-size set. cutoff=1.0 == the rho>0 gate.
            ranked = select_cutoff(div_t, b_t, FLSettings.PARTIAL_CUTOFF)   # [(name, div)] desc
        elif mode == "topk":
            # SIGNED rho = div_t - b_t, UNCLAMPED and UNGATED: still subtracts the
            # benign-to-benign floor b_t (denoising preserved), but keeps a full total
            # order so select_topk_signed always returns exactly k carriers instead of
            # collapsing to the (possibly < k) rho>0 survivors. Frozen/BN tensors are
            # already absent (div_t/b_t are keyed on elig_list), so the "pre-exclude
            # frozen" contract of select_topk_signed is satisfied here.
            signed_rho = {t: div_t[t] - b_t[t] for t in div_t}
            ranked = select_topk_signed(signed_rho, FLSettings.PARTIAL_TOPK)  # [(name, signed rho)] desc
        else:
            raise ValueError(f"Unknown PARTIAL_SELECT_MODE {mode!r}; expected 'cutoff' or 'topk'.")

        carrying = {name for name, _ in ranked}
        return {
            "carrying": carrying,
            "ranked": ranked,
            "div_t": div_t,
            "elig_list": elig_list,
            "tensor_to_group": tensor_to_group,
        }

    # ---- orchestration ----

    def run(self, server_weights, client_config):
        cid = client_config["index"]
        current_round = client_config.get("round")
        if current_round is None:
            raise KeyError(
                "PartialAmplificationProtocol requires 'round' in client_config "
                "(set in custom_fl_train.py's per-client client_config build)."
            )

        n_epochs = TrainingSettings.MALICIOUS_CLIENT_LOCAL_EPOCHS

        # mal and ben1 share a seed (they differ ONLY by the poison); ben2 = +1 so
        # the floor measures pure PRNG noise between two benign runs.
        seed_mal = FLSettings.SEED + 1000 * int(current_round) + int(cid)
        seed_ben1 = seed_mal
        seed_ben2 = seed_mal + 1

        print(
            f"  [partial client={cid} round={current_round}] trajectories on "
            f"partition {cid}: ben1(seed={seed_ben1}) / mal(seed={seed_mal}) / ben2(seed={seed_ben2})",
            flush=True,
        )

        theta_ben1 = self._run_trajectory(
            server_weights, cid, is_attacker=False, n_epochs=n_epochs, seed=seed_ben1)
        theta_mal = self._run_trajectory(
            server_weights, cid, is_attacker=True, n_epochs=n_epochs, seed=seed_mal)
        theta_ben2 = self._run_trajectory(
            server_weights, cid, is_attacker=False, n_epochs=n_epochs, seed=seed_ben2)

        sel = self._select_carrying(theta_mal, theta_ben1, theta_ben2)
        carrying, ranked = sel["carrying"], sel["ranked"]
        AttackState.set_carrying_set(cid, carrying)
        self._print_selection(cid, current_round, carrying, ranked)

        # Amplify ONLY the carrying tensors' update Delta_C = theta_mal - theta_t by
        # PARTIAL_AMP_GAMMA; non-carrying params submit as theta_mal. gamma=1.0 is an
        # exact no-op (submission == theta_mal).
        gamma = FLSettings.PARTIAL_AMP_GAMMA
        submission = partial_amplify(server_weights, theta_mal, carrying, gamma)

        # Per-round diagnostics JSON (attack_logs/partial/): selection, group rollup,
        # and the norm inflation the amplification produced.
        mode = FLSettings.PARTIAL_SELECT_MODE
        select_param = FLSettings.PARTIAL_CUTOFF if mode == "cutoff" else FLSettings.PARTIAL_TOPK
        norms = self._norm_diagnostics(
            server_weights, theta_mal, submission, carrying, sel["elig_list"], gamma)
        log_path = self.manager.logger.partial.write_round(
            current_round=current_round,
            client_id=cid,
            select_mode=mode,
            select_param=select_param,
            gamma=gamma,
            carrying=carrying,
            carrying_by_group=self._carrying_by_group(carrying, sel["tensor_to_group"]),
            top_carriers=[[n, float(s)] for n, s in ranked[:10]],
            norms=norms,
        )

        num_samples = len(self.manager.subsets_mal[cid])
        metrics = {
            "partition_id": cid,
            "protocol": "partial",
            "select_mode": mode,
            "select_param": select_param,
            "carrying_size": len(carrying),
            "amp_gamma": gamma,
            "partial_log": log_path,
        }
        return submission, num_samples, metrics

    def _print_selection(self, cid, current_round, carrying, ranked):
        mode = FLSettings.PARTIAL_SELECT_MODE
        param = FLSettings.PARTIAL_CUTOFF if mode == "cutoff" else FLSettings.PARTIAL_TOPK
        top = ", ".join(f"{name}={score:.3g}" for name, score in ranked[:5]) or "(none)"
        print(
            f"[partial client={cid} round={current_round}] carrying set |C|={len(carrying)} "
            f"(mode={mode}, param={param}, gamma={FLSettings.PARTIAL_AMP_GAMMA})",
            flush=True,
        )
        print(f"    top carriers ({'div' if mode == 'cutoff' else 'rho'}): {top}", flush=True)

    @staticmethod
    def _carrying_by_group(carrying, tensor_to_group):
        """Roll the per-tensor carrying set up to the 13 groups: {group: #tensors}."""
        counts = {}
        for name in carrying:
            g = tensor_to_group.get(name, "other")
            counts[g] = counts.get(g, 0) + 1
        return counts

    def _norm_diagnostics(self, theta_t, theta_mal, submission, carrying, elig_list, gamma):
        """Norm/energy diagnostics for the submission, all over the ELIGIBLE tensor set
        (the stealth harness's coordinate set). norm_inflation_ratio =
        ||submission - theta_t|| / ||theta_mal - theta_t|| is the exact quantity a
        future norm budget will drive back toward 1.0.
        """
        def l2(a, b, names):
            s = 0.0
            for n in names:
                s += float((a[n].detach().float() - b[n].detach().float()).pow(2).sum())
            return math.sqrt(s)

        carrying_list = [n for n in elig_list if n in carrying]
        dC_pre = l2(theta_mal, theta_t, carrying_list)       # ||Delta_C|| pre-amplification
        d_pre = l2(theta_mal, theta_t, elig_list)            # unamplified total (== theta_mal update)
        d_post = l2(submission, theta_t, elig_list)          # submitted total (post-amplification)
        return {
            "delta_carrying_l2_pre": dC_pre,
            "delta_carrying_l2_post": gamma * dC_pre,        # amplification scales Delta_C by gamma
            "delta_total_l2_unamplified": d_pre,
            "delta_total_l2_submitted": d_post,
            "norm_inflation_ratio": (d_post / d_pre) if d_pre > 0 else float("nan"),
        }
