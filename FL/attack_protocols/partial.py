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
    select_by_mode,
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

    def write_round(self, current_round, client_id, phase, select_mode, select_param,
                    gamma, gamma_source, carrying, carrying_by_group, top_carriers,
                    norms, freeze=None):
        target = os.path.join(self.dir, f"round_{current_round}_client_{client_id}.json")
        payload = {
            "round": current_round,
            "client_id": client_id,
            "phase": phase,                           # 'disabled' | 'plastic' | 'exploit'
            "select_mode": select_mode,
            "select_param": select_param,
            "gamma": gamma,
            "gamma_source": gamma_source,             # 'plastic_no_amp' | 'replacement' | 'fixed'
            "carrying_size": len(carrying),
            "carrying": sorted(carrying),
            "carrying_by_group": carrying_by_group,   # {group_name: #carrying tensors}
            "top_carriers": top_carriers,             # this round's per-round ranking (drift check in exploit)
            "norms": norms,                           # see _norm_diagnostics
            # Present ONLY on the round the plastic phase freezes; records how the
            # single shared carrying set was chosen from the pooled means. None otherwise.
            "freeze": freeze,
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

    def _eligible_tensors(self):
        """Eligible (trainable, non-batch-norm) tensor names + the tensor->group map
        and identity partition, derived from the model alone (no trajectories). Shared
        by the per-round localizer (_select_carrying) and the exploit path (_exploit_sel),
        which needs the same coordinate set for logging/diagnostics but skips the anchors.
        """
        model = self.manager.wrapper.model
        tensor_list, tensor_to_group, _ = build_group_map(model)
        id_part = identity_partition(tensor_list)
        eligible_names, _eligible_buckets = build_eligibility(model, id_part)
        elig_list = [t for t in tensor_list if t in eligible_names]
        return elig_list, tensor_to_group, id_part

    def _exploit_sel(self):
        """The `sel` dict for an exploit round, built WITHOUT the benign anchors.

        No new trajectories run, but the frozen set was itself selected from the POOLED
        per-bucket divergence scores accumulated during the plastic phase — and those
        means are persisted in AttackState. So we can still rank the frozen layers by
        the exact scores that selected them: re-run the SAME select_by_mode(mean_div,
        mean_b, ...) call _freeze used. `ranked` is therefore the frozen carrying set
        with its normalized divergence score per layer (div in cutoff mode, denoised
        rho in topk mode), descending. It is STATIC across exploit rounds (it's the
        frozen decision, not a per-round remeasurement). div_t/b_t carry the pooled
        means for reference; they are unused downstream in exploit (accumulation is
        closed, _norm_diagnostics doesn't read them).
        """
        elig_list, tensor_to_group, _ = self._eligible_tensors()
        mean_div, mean_b = AttackState.plastic_means()
        ranked = select_by_mode(
            mean_div, mean_b,
            FLSettings.PARTIAL_SELECT_MODE,
            FLSettings.PARTIAL_CUTOFF, FLSettings.PARTIAL_TOPK,
        )
        return {
            "carrying": AttackState.plastic_frozen_set,
            "ranked": ranked,
            "div_t": mean_div,
            "b_t": mean_b,
            "elig_list": elig_list,
            "tensor_to_group": tensor_to_group,
        }

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
        elig_list, tensor_to_group, id_part = self._eligible_tensors()
        numel = numel_per_tensor(self.manager.wrapper.model)

        # div_t over ELIGIBLE tensors only (BN + frozen never enter the selection).
        sq_div = squared_divergence_per_tensor(theta_mal, theta_ben1, elig_list)
        div_t = normalized_l2_divergence(sq_div, numel, id_part)   # keys = eligible tensors

        # Per-tensor PRNG-noise floor from the second benign trajectory.
        sq_b = squared_divergence_per_tensor(theta_ben1, theta_ben2, elig_list)
        b_t = normalized_l2_divergence(sq_b, numel, id_part)

        # Selector dispatch lives in l2_based.select_by_mode so the per-round path and
        # the plastic freeze pick carriers IDENTICALLY (only the div/b inputs differ).
        ranked = select_by_mode(
            div_t, b_t,
            FLSettings.PARTIAL_SELECT_MODE,
            FLSettings.PARTIAL_CUTOFF,
            FLSettings.PARTIAL_TOPK,
        )

        carrying = {name for name, _ in ranked}
        return {
            "carrying": carrying,
            "ranked": ranked,
            "div_t": div_t,
            "b_t": b_t,   # per-bucket benign floor — needed by the plastic accumulator
            "elig_list": elig_list,
            "tensor_to_group": tensor_to_group,
        }

    # ---- plastic phase (which carrying set) ----

    def _resolve_carrying(self, sel, in_plastic, plastic_on):
        """Pick THIS round's carrying set and drive the pooled plastic accumulator.

            plastic disabled -> per-round selection (unchanged legacy behavior)
            plastic, frozen  -> the single shared frozen set (exploit)
            plastic, active  -> accumulate this round's div/b into the pooled means,
                                use the per-round set, and freeze once the pooled
                                injection count reaches PARTIAL_PLASTIC_ROUNDS.
        """
        if not plastic_on:
            return sel["carrying"]
        if not in_plastic:
            return AttackState.plastic_frozen_set
        AttackState.accumulate_plastic(sel["div_t"], sel["b_t"])
        if AttackState.plastic_count >= FLSettings.PARTIAL_PLASTIC_ROUNDS:
            self._freeze()
        return sel["carrying"]

    def _freeze(self):
        """Reduce the pooled accumulator to ONE shared carrying set: mean(div) -
        mean(b) -> select_by_mode (the SAME cutoff/topk selector as the per-round
        path) -> stored in AttackState for every later round."""
        mean_div, mean_b = AttackState.plastic_means()
        ranked = select_by_mode(
            mean_div, mean_b,
            FLSettings.PARTIAL_SELECT_MODE,
            FLSettings.PARTIAL_CUTOFF,
            FLSettings.PARTIAL_TOPK,
        )
        AttackState.set_plastic_frozen({name for name, _ in ranked})
        print(
            f"  [partial] PLASTIC PHASE FROZEN after {AttackState.plastic_count} pooled "
            f"injection(s): |C|={len(AttackState.plastic_frozen_set)} via "
            f"{FLSettings.PARTIAL_SELECT_MODE}",
            flush=True,
        )

    # ---- gamma source (how much amplification) — orthogonal to carrying set ----

    def _resolve_gamma(self, client_config, in_plastic):
        """Amplification factor for this round, plus a short source label for logging.

        During the plastic phase with PARTIAL_PLASTIC_NO_AMP, submit UNAMPLIFIED
        (gamma=1) so the localizer observes on an un-deformed global. Otherwise draw
        gamma from PARTIAL_GAMMA_MODE: 'replacement' = k/(eta*m) capped (identical to
        full model replacement, carriers land on theta_mal post-aggregation); 'fixed'
        = the hand-set PARTIAL_AMP_GAMMA sweep knob.

        Returns (gamma, source) where source is one of 'plastic_no_amp' | 'replacement'
        | 'fixed'.
        """
        if in_plastic and FLSettings.PARTIAL_PLASTIC_NO_AMP:
            return 1.0, "plastic_no_amp"
        if FLSettings.PARTIAL_GAMMA_MODE == "replacement":
            return self.manager._replacement_gamma(client_config["n_mal_clients"]), "replacement"
        return FLSettings.PARTIAL_AMP_GAMMA, "fixed"

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

        # Once the carrying set is frozen (exploit), the benign anchors are unused:
        # the frozen set drives amplification and the per-round localizer is only a
        # (now forfeited) drift check. Skip BOTH benign trajectories then and train
        # only the poisoned run. Each trajectory re-seeds all its RNGs and starts from
        # theta_t with a fresh optimizer, so theta_mal is identical whether or not the
        # anchors ran before it.
        in_exploit = FLSettings.PARTIAL_PLASTIC_ROUNDS > 0 and AttackState.is_plastic_frozen()

        if in_exploit:
            print(
                f"  [partial client={cid} round={current_round}] exploit: mal-only "
                f"trajectory on partition {cid} (seed={seed_mal}); benign anchors skipped",
                flush=True,
            )
            theta_mal = self._run_trajectory(
                server_weights, cid, is_attacker=True, n_epochs=n_epochs, seed=seed_mal)
            sel = self._exploit_sel()
        else:
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
        ranked = sel["ranked"]

        # Phase captured BEFORE any accumulate/freeze mutation, so the round that
        # TRIGGERS the freeze is still treated as plastic (its sample counts, and it
        # stays unamplified under --partial-plastic-no-amp); the freeze affects the
        # NEXT round onward. Which carrying set (chunk: plastic phase) and how much
        # amplification (chunk: gamma source) are resolved independently.
        plastic_on = FLSettings.PARTIAL_PLASTIC_ROUNDS > 0
        in_plastic = plastic_on and not AttackState.is_plastic_frozen()

        carrying = self._resolve_carrying(sel, in_plastic, plastic_on)
        # in_plastic was captured pre-mutation; if we're now frozen, the freeze fired
        # THIS round (its sample was the Nth). phase names the round for the logs.
        froze_this_round = in_plastic and AttackState.is_plastic_frozen()
        phase = "disabled" if not plastic_on else ("plastic" if in_plastic else "exploit")
        gamma, gamma_source = self._resolve_gamma(client_config, in_plastic)
        AttackState.set_carrying_set(cid, carrying)
        self._print_selection(cid, current_round, carrying, ranked, gamma, phase)

        # Amplify ONLY the carrying tensors' update Delta_C = theta_mal - theta_t.
        # partial_amplify applies theta_t + gamma*(theta_mal - theta_t), i.e. exactly
        # the model-replacement form  gamma*theta_mal - (gamma-1)*theta_t  restricted to C.
        submission = partial_amplify(server_weights, theta_mal, carrying, gamma)

        # When the freeze fires THIS round, capture the pooled decision so the log
        # records exactly how the frozen set was chosen (set + top denoised means).
        freeze = None
        if froze_this_round:
            mdiv, mb = AttackState.plastic_means()
            signed = sorted(((n, mdiv[n] - mb[n]) for n in mdiv), key=lambda x: (-x[1], x[0]))
            freeze = {
                "pooled_injections": AttackState.plastic_count,
                "frozen_set": sorted(AttackState.plastic_frozen_set),
                "top_mean_carriers": [[n, float(s)] for n, s in signed[:10]],  # mean(div)-mean(b)
            }

        # Per-round diagnostics JSON (attack_logs/partial/): selection, group rollup,
        # and the norm inflation the amplification produced.
        mode = FLSettings.PARTIAL_SELECT_MODE
        select_param = FLSettings.PARTIAL_CUTOFF if mode == "cutoff" else FLSettings.PARTIAL_TOPK
        norms = self._norm_diagnostics(
            server_weights, theta_mal, submission, carrying, sel["elig_list"], gamma)
        log_path = self.manager.logger.partial.write_round(
            current_round=current_round,
            client_id=cid,
            phase=phase,
            select_mode=mode,
            select_param=select_param,
            gamma=gamma,
            gamma_source=gamma_source,
            carrying=carrying,
            carrying_by_group=self._carrying_by_group(carrying, sel["tensor_to_group"]),
            top_carriers=[[n, float(s)] for n, s in ranked[:10]],
            norms=norms,
            freeze=freeze,
        )

        num_samples = len(self.manager.subsets_mal[cid])
        metrics = {
            "partition_id": cid,
            "protocol": "partial",
            "phase": phase,
            "select_mode": mode,
            "select_param": select_param,
            "carrying_size": len(carrying),
            "amp_gamma": gamma,
            "gamma_source": gamma_source,
            "partial_log": log_path,
        }
        return submission, num_samples, metrics

    def _print_selection(self, cid, current_round, carrying, ranked, gamma, phase):
        mode = FLSettings.PARTIAL_SELECT_MODE
        param = FLSettings.PARTIAL_CUTOFF if mode == "cutoff" else FLSettings.PARTIAL_TOPK
        top = ", ".join(f"{name}={score:.3g}" for name, score in ranked[:5]) or "(none)"
        print(
            f"[partial client={cid} round={current_round}] phase={phase} carrying set "
            f"|C|={len(carrying)} (mode={mode}, param={param}, gamma={gamma:.3g})",
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
