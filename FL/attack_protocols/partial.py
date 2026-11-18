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
     PARTIAL_AMP_GAMMA and submit (non-carrying params are submitted as theta_mal,
     or at theta_t under --partial-mask-noncarrying).

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
from FL.attack_protocols.amplify import partial_amplify, allocate_norm_budget, mask_noncarrying
from FL.attack_state import AttackState
from FL.normclip_report import update_norm
from FL.groups import build_group_map, build_eligibility
from FL.scoring.l2_based import (
    numel_per_tensor,
    squared_divergence_per_tensor,
    identity_partition,
    normalized_l2_divergence,
    select_by_mode,
)
from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.global_settings import ANSIColor, FLSettings, TrainingSettings

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
                    norms, freeze=None, constrain=None, freeze_noncarrying=None,
                    mask_noncarrying=None, norm_budget=None, submitted=None):
        target = os.path.join(self.dir, f"round_{current_round}_client_{client_id}.json")
        payload = {
            "round": current_round,
            "client_id": client_id,
            "phase": phase,                           # 'disabled' | 'plastic' | 'exploit'
            # What was actually handed to the server: 'malicious' normally, 'benign_anchor'
            # on plastic rounds under PARTIAL_PLASTIC_SUBMIT_BENIGN. The norms block still
            # describes the poisoned submission that WOULD have been sent, so the two modes
            # stay comparable; this field says whether it was.
            "submitted": submitted,
            "select_mode": select_mode,
            "select_param": select_param,
            # The amplification factor applied to every carrier — EXCEPT under
            # gamma_source 'budgeted', where this is the per-carrier CEILING and the
            # factor actually applied to each carrier lives in norm_budget below.
            "gamma": gamma,
            # 'plastic_no_amp' | 'replacement' | 'fixed' | 'budgeted' | 'budgeted_pre_exploit'
            "gamma_source": gamma_source,
            "carrying_size": len(carrying),
            "carrying": sorted(carrying),
            "carrying_by_group": carrying_by_group,   # {group_name: #carrying tensors}
            "top_carriers": top_carriers,             # this round's per-round ranking (drift check in exploit)
            "norms": norms,                           # see _norm_diagnostics
            # Present ONLY on the round the plastic phase freezes; records how the
            # single shared carrying set was chosen from the pooled means. None otherwise.
            "freeze": freeze,
            # Present ONLY on exploit rounds where the constrain penalty was applied
            # (lambda + #non-carrying tensors pulled toward theta_t). None otherwise.
            "constrain": constrain,
            # Present ONLY on exploit rounds where the non-carrying gradient freeze was
            # applied (#tensors whose gradient was zeroed). None otherwise.
            "freeze_noncarrying": freeze_noncarrying,
            # Present ONLY on exploit rounds where the submission-side non-carrying mask was
            # applied (#tensors reset to theta_t in the submission). None otherwise. Mutually
            # exclusive with freeze_noncarrying and constrain.
            "mask_noncarrying": mask_noncarrying,
            # Present ONLY on exploit rounds under --partial-gamma-mode budgeted: this
            # client's budget, how the excess was rationed across carriers (which one it
            # ran out on, each carrier's applied gamma), and realized vs budget norm.
            "norm_budget": norm_budget,
        }
        with open(target, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"  [partial client={client_id} round={current_round}] log written: {target}", flush=True)
        return os.path.join("attack_logs", "partial", os.path.basename(target))

    def write_freeze_summary(self, current_round, select_mode, select_param,
                             pooled_injections, frozen_set, rows,
                             carrying_params=None, total_params=None):
        """Write the one-look summary of the frozen carrying set: frozen_set.md.

        Called ONCE, on the round the plastic phase freezes. The same facts are
        recoverable from round_<freeze>_client_<id>.json's 'freeze' block and from
        attack_state.json, but only interleaved with the full 72-tensor div/b dumps;
        this file exists so the decision can be read at a glance.

        rows: one dict per ELIGIBLE tensor, with keys name, group, mean_div, mean_b,
        ratio (mean_div/mean_b -- the quantity compared against a cutoff), signed
        (mean_div - mean_b -- the quantity ranked by topk), selected (bool).

        carrying_params / total_params: element counts (numel) of the frozen set and of
        the whole model, so the header reports the carrying set's size as a fraction of
        the model's parameters (not just its tensor count).

        Tensors are listed by ratio descending and split at the selection boundary,
        so the rejected near-misses are visible too: that is what says whether the
        chosen threshold sits on a cliff or on a plateau of the ratio distribution.
        """
        target = os.path.join(self.dir, "frozen_set.md")
        ranked = sorted(rows, key=lambda r: (-r["ratio"], r["name"]))
        sel = [r for r in ranked if r["selected"]]
        rej = [r for r in ranked if not r["selected"]]

        by_group = {}
        for r in sel:
            by_group[r["group"]] = by_group.get(r["group"], 0) + 1

        def table(entries):
            out = ["| ratio = div/b | signed = div-b | group | tensor |",
                   "|---:|---:|---|---|"]
            for r in entries:
                out.append(f"| {r['ratio']:.3f} | {r['signed']:.3e} | "
                           f"{r['group']} | `{r['name']}` |")
            return out

        lines = [
            "# Frozen carrying set",
            "",
            f"- froze on round: **{current_round}** (after {pooled_injections} pooled injection(s))",
            f"- selector: **{select_mode}**, param **{select_param}**",
            f"- |C| = **{len(frozen_set)}** of {len(rows)} eligible tensors",
            f"- carrying params: **{carrying_params:,}** of {total_params:,} model params "
            f"(**{100.0 * carrying_params / total_params:.3f}%**)"
            if carrying_params is not None and total_params else "- carrying params: (n/a)",
            f"- by group: {', '.join(f'{g}={n}' for g, n in sorted(by_group.items())) or '(none)'}",
            "",
            "## Selected (the frozen set)",
            "",
        ]
        lines += table(sel) if sel else ["_(empty)_"]
        # Only the near-misses matter for threshold sensitivity; the long negative
        # tail is already in attack_state.json for anyone who wants all of it.
        lines += ["", "## Rejected — 15 nearest the boundary", ""]
        lines += table(rej[:15]) if rej else ["_(none)_"]
        lines += [""]

        with open(target, "w") as f:
            f.write("\n".join(lines))
        print(f"  [partial] frozen-set summary written: {target}", flush=True)
        return os.path.join("attack_logs", "partial", "frozen_set.md")


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

    def _run_trajectory(self, server_weights, partition_id, is_attacker, n_epochs, seed,
                        loss_augment_fn=None, freeze_names=None):
        """One N-epoch SGD trajectory from theta_t; returns the FINAL CPU state_dict.

        Only the endpoint is needed — this protocol does no per-epoch analysis — so the
        final weights are read with get_weights() after training. No per-epoch snapshot
        series is collected (that would hold N full state_dicts in CPU RAM for nothing).

        Fresh optimizer each call (no momentum carryover). The mal and benign
        trajectories MUST train at the same LR: a mal/ben LR gap would confound the
        per-tensor divergence (theta_mal vs T_ben) with LR-induced movement rather
        than the poison — the same invariant the pilots rely on.

        loss_augment_fn: optional (model -> scalar Tensor) added to the task loss each
        step (wrapper.train_step's fp32-outside-autocast hook). Used by the exploit-phase
        constrain penalty; None for the benign anchors and for unconstrained mal runs.

        freeze_names: optional iterable of parameter names to hard-freeze for this
        trajectory by setting requires_grad=False (the exploit-phase non-carrying freeze).
        A frozen leaf gets no gradient at all, so SGD skips it entirely — including weight
        decay — and it stays bit-exact at theta_t while the carriers co-adapt. requires_grad
        is PERSISTENT state on the shared wrapper.model, so it is set here and ALWAYS
        restored in a finally, or it would leak into later trajectories/rounds.
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

        # Hard-freeze the requested params for the duration of this trajectory only.
        # requires_grad lives on the shared wrapper.model and survives set_weights(), so
        # the toggle MUST be undone in the finally even if train_step raises.
        frozen_params = []
        if freeze_names:
            wanted = set(freeze_names)
            frozen_params = [p for n, p in self.manager.wrapper.model.named_parameters()
                             if n in wanted]
            for p in frozen_params:
                p.requires_grad_(False)
        try:
            self.manager.wrapper.train_step(
                dataloader=dl,
                epochs=n_epochs,
                device=self.manager.device,
                optimizer_config={"lr": lr},
                partition_id=partition_id,
                optimizer=None,
                grad_mask=None,
                loss_augment_fn=loss_augment_fn,
                freeze_groups=None,
            )
        finally:
            for p in frozen_params:
                p.requires_grad_(True)
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
        ranked by its selection score per layer (the ratio mean_div/mean_b in cutoff
        mode, signed rho mean_div-mean_b in topk mode), descending. It is STATIC across
        exploit rounds (it's the frozen decision, not a per-round remeasurement).
        div_t/b_t carry the pooled means for reference; they are unused downstream in
        exploit (accumulation is closed, _norm_diagnostics doesn't read them).
        """
        elig_list, tensor_to_group, _ = self._eligible_tensors()
        mean_div, mean_b = AttackState.plastic_means()
        ranked = select_by_mode(
            mean_div, mean_b,
            FLSettings.PARTIAL_SELECT_MODE,
            FLSettings.PARTIAL_CUTOFF, FLSettings.PARTIAL_TOPK,
            FLSettings.PARTIAL_CARRYING_NAMES,
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
        score is the ratio div/b in cutoff mode, signed rho div-b in topk mode), div_t,
        elig_list (eligible tensor names, for the norm diagnostics), tensor_to_group
        (for the group rollup).
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
            FLSettings.PARTIAL_CARRYING_NAMES,
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

    def _resolve_carrying(self, sel, in_plastic, plastic_on, cid=None, benign_norms=None,
                          current_round=None):
        """Pick THIS round's carrying set and drive the pooled plastic accumulator.

            plastic disabled -> per-round selection (unchanged legacy behavior)
            plastic, frozen  -> the single shared frozen set (exploit)
            plastic, active  -> accumulate this round's div/b into the pooled means,
                                use the per-round set, and freeze once the pooled
                                injection count reaches PARTIAL_PLASTIC_ROUNDS.

        benign_norms: this round's benign-anchor update norms (ben1 and ben2) for THIS
        attacker, recorded under its own cid — never pooled across clients, unlike div/b.
        None when nothing consumes them (see _needs_benign_refs).

        current_round: identifies this injection together with cid. accumulate_plastic is
        idempotent in that pair, so a round replayed after a crash-and-resume replaces its
        earlier sample instead of pooling both and reweighting the freeze.
        """
        if not plastic_on:
            return sel["carrying"]
        if not in_plastic:
            return AttackState.plastic_frozen_set
        AttackState.accumulate_plastic(sel["div_t"], sel["b_t"], current_round, cid)
        if benign_norms:
            AttackState.accumulate_benign_norms(cid, benign_norms)
        if AttackState.plastic_count >= FLSettings.PARTIAL_PLASTIC_ROUNDS:
            self._freeze()
        return sel["carrying"]

    @staticmethod
    def _needs_benign_refs():
        """Does anything this run consume the per-client benign-anchor references?

        Two independent consumers: the observational norm-clip report (its
        ratio_to_benign / exceeds_benign columns) and budgeted gamma (the norm budget is
        B x this client's reference). Collection is gated on the union, NOT on the
        norm-clip flag alone — otherwise running budgeted mode with the report disabled
        would silently leave every client without a reference.
        """
        return FLSettings.DO_NORMCLIP_REPORT or FLSettings.PARTIAL_GAMMA_MODE == "budgeted"

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
            FLSettings.PARTIAL_CARRYING_NAMES,
        )
        AttackState.set_plastic_frozen({name for name, _ in ranked})
        # Freeze each attacker's OWN mean benign-anchor norm (empty when nothing collects
        # them, i.e. neither the norm-clip report nor budgeted gamma is on).
        AttackState.freeze_benign_refs()
        print(
            f"  [partial] PLASTIC PHASE FROZEN after {AttackState.plastic_count} pooled "
            f"injection(s): |C|={len(AttackState.plastic_frozen_set)} via "
            f"{FLSettings.PARTIAL_SELECT_MODE}",
            flush=True,
        )
        if self._needs_benign_refs():
            self._report_benign_ref_coverage()

    @staticmethod
    def _report_benign_ref_coverage():
        """Runtime backstop for the startup coverage check (parse_args).

        parse_args halts before training when a client SCHEDULE proves an attacker never
        acts during the plastic phase. Under random client selection that is unknowable in
        advance, so this reports the realised coverage the moment it is fixed — at the
        freeze — and names any attacker left without a reference.

        Reporting, not enforcement: under budgeted gamma the run WILL fail if such an
        attacker is later selected (_amplify raises), but it may simply never be selected
        again, so this is an early heads-up rather than a fatal check.
        """
        counts = AttackState.benign_ref_sample_counts()
        covered = sorted(AttackState.plastic_benign_ref)
        missing = sorted(set(FLSettings.ATTACKER_IDS) - set(covered))
        detail = ", ".join(
            f"cid={c}: ref={AttackState.plastic_benign_ref[c]:.4g} from {counts.get(c, 0)} anchor(s)"
            for c in covered) or "(none)"
        print(f"  [partial] benign references frozen (per client, not pooled) -> {detail}", flush=True)
        if missing:
            extra = ("; under --partial-gamma-mode budgeted the run will FAIL if any of them "
                     "is selected for an exploit round (no budget can be sized)"
                     if FLSettings.PARTIAL_GAMMA_MODE == "budgeted" else
                     "; their norm-clip ratio_to_benign will be NaN")
            print(
                f"  [partial] {ANSIColor.YELLOW}[WARNING]{ANSIColor.RESET} attacker(s) {missing} "
                f"never acted during the plastic phase and have NO benign reference{extra}.",
                flush=True,
            )

    # ---- gamma source (how much amplification) — orthogonal to carrying set ----

    def _resolve_gamma(self, client_config, in_plastic, in_exploit=False):
        """Amplification factor for this round, plus a short source label for logging.

        During the plastic phase with PARTIAL_PLASTIC_NO_AMP, submit UNAMPLIFIED
        (gamma=1) so the localizer observes on an un-deformed global. Otherwise draw
        gamma from PARTIAL_GAMMA_MODE: 'replacement' = k/(eta*m) capped (identical to
        full model replacement, carriers land on theta_mal post-aggregation); 'fixed'
        = the hand-set PARTIAL_AMP_GAMMA sweep knob.

        In 'budgeted' mode this returns the per-carrier CEILING — the replacement gamma —
        which _resolve_budgeted_gammas then rations across carriers under the norm budget.
        Budgeted is exploit-only: before the freeze the carrying set and its ranking still
        oscillate round to round, and there is no frozen benign reference to size a budget
        against, so it degrades to gamma=1 rather than guessing.

        Returns (gamma, source) where source is one of 'plastic_no_amp' | 'replacement'
        | 'fixed' | 'budgeted' | 'budgeted_pre_exploit'.
        """
        if in_plastic and FLSettings.PARTIAL_PLASTIC_NO_AMP:
            return 1.0, "plastic_no_amp"
        if FLSettings.PARTIAL_GAMMA_MODE == "budgeted":
            if not in_exploit:
                return 1.0, "budgeted_pre_exploit"
            return self.manager._replacement_gamma(client_config["n_mal_clients"]), "budgeted"
        if FLSettings.PARTIAL_GAMMA_MODE == "replacement":
            return self.manager._replacement_gamma(client_config["n_mal_clients"]), "replacement"
        return FLSettings.PARTIAL_AMP_GAMMA, "fixed"

    @staticmethod
    def _carrier_energies(theta_t, theta_mal, names):
        """{name: ||theta_mal_p - theta_t_p||^2} — per-tensor update ENERGY.

        Energy (not norm) because it is additive across disjoint tensors, so the allocator
        can ration it with plain subtraction and take one square root at the end. Named for
        its main caller, but it is plain per-tensor energy over whatever `names` it is
        given — the budget path also uses it to total the masked tensors' discarded energy.
        """
        out = {}
        for n in names:
            out[n] = float((theta_mal[n].detach().float() - theta_t[n].detach().float()).pow(2).sum())
        return out

    def _resolve_budgeted_gammas(self, cid, theta_t, theta_mal, sel, carrying, gamma_full,
                                 mask_names=None):
        """Per-carrier gammas under this client's norm budget. Returns (gammas, report).

        The budget is B x THIS client's own frozen mean benign-anchor norm — never a
        cross-client statistic, so it stays meaningful when partitions go non-IID.

        Coordinate set: the budget is enforced on the WHOLE-update L2 norm (update_norm,
        every float tensor), because that is what a norm-clip defense measures, what the
        norm-clip report compares against, and what the server scores. That is consistent
        even though carriers are a subset of the eligible tensors: partial_amplify
        perturbs ONLY carrier coordinates, so every non-carrier contributes the same
        constant to the unamplified norm and to the submitted norm. The non-carrying MASK
        is the one thing that breaks that constancy — it sends its coordinates to theta_t —
        so its energy is subtracted below, leaving the allocator's `unamplified_energy` the
        energy of the update ACTUALLY submitted at gamma=1.

        Carrier ORDER is sel["ranked"] (descending selection score). `ranked` is recomputed
        from the pooled means while `carrying` is the persisted frozen set; they normally
        agree exactly, but a selector-setting change across a resume can desync them, so
        ranked is filtered to `carrying` and any carrier missing from ranked is appended at
        the end (it still gets a gamma, never silently dropped).

        Returns (None, report) with no gammas when this client has no benign reference —
        the caller (_amplify) treats that as fatal rather than falling back to gamma=1,
        which would make the attacker's exploit rounds silent no-ops.
        """
        ref = AttackState.get_benign_ref(cid)
        if ref is None or ref <= 0:
            return None, {"no_benign_ref": True, "cid": cid,
                          "budget_l2": None, "gamma_full": gamma_full}

        ordered = [n for n, _ in sel["ranked"] if n in carrying]
        appended = sorted(set(carrying) - set(ordered))
        if appended:
            ordered += appended

        energy = self._carrier_energies(theta_t, theta_mal, ordered)
        unamp_l2 = update_norm(theta_mal, theta_t)      # whole-update, matches `ref`
        unamp_energy = unamp_l2 * unamp_l2
        masked_energy = sum(self._carrier_energies(theta_t, theta_mal, mask_names).values()) \
            if mask_names else 0.0
        budget = FLSettings.PARTIAL_NORM_BUDGET * ref

        gammas, report = allocate_norm_budget(
            ranked=[(n, 0.0) for n in ordered],
            energy=energy,
            unamplified_energy=max(0.0, unamp_energy - masked_energy),
            budget=budget,
            gamma_full=gamma_full,
        )
        report.update({
            "no_benign_ref": False,
            "cid": cid,
            "B": FLSettings.PARTIAL_NORM_BUDGET,
            "benign_ref": ref,
            "benign_ref_samples": len(AttackState.plastic_benign_norms.get(cid, [])),
            "carriers_not_in_ranked": appended,
            # Energy the mask threw away, i.e. how much budget the mask freed up. 0.0 when
            # the mask is off; report["unamplified_l2"] is already net of it.
            "masked_energy": masked_energy,
        })
        return gammas, report

    # ---- non-carrying handling (constrain penalty / gradient freeze) ----

    @staticmethod
    def _noncarrying_names(carrying, elig_list):
        """The eligible tensors NOT in the carrying set — the shared coordinate set for
        both non-carrying interventions (the constrain penalty and the gradient freeze),
        so they always agree on which params count as 'non-carrying'. BN/frozen params
        are outside elig_list and so are never included."""
        return [n for n in elig_list if n not in carrying]

    def _constrain_penalty_fn(self, theta_t, carrying, elig_list):
        """Build the exploit-phase constrain penalty, or None when it's inactive.

        Returns a closure (model -> scalar Tensor) computing

            LAMBDA * sum_{p in (eligible \\ C)} || theta_p - theta_t,p ||^2

        i.e. the UNNORMALIZED sum of squared errors between the NON-carrying eligible
        params and the broadcast model theta_t (server_weights), scaled by
        PARTIAL_CONSTRAIN_LAMBDA. Carrying params C are excluded so the backdoor moves
        freely; the penalty pulls the rest of the update back onto theta_t (the benign
        cluster). BN/frozen params are already outside `elig_list`, so they never enter.

        SSE, not a mean: dividing by the non-carrying element count (~1e7) shrinks the
        term ~7 orders of magnitude below the task loss, so at any sane LAMBDA it would
        be negligible. The SSE keeps the penalty in the SAME magnitude regime as the task
        loss (LAMBDA ~ O(1) actually bites), matching PilotA's _calibrate_beta_max
        (penalty = sum_p ||theta_p - anchor_p||^2) and the deferred beta_max auto-sizing
        that is defined on that SSE.

        Returns None (no penalty) when the switch is off, LAMBDA <= 0, or there are no
        non-carrying eligible params — so run() can pass the result straight through to
        _run_trajectory without a guard.

        The reference tensors are materialized as fp32 constants on the training device
        ONCE here; the closure re-reads them every step. Same fp32 reasoning as
        wrapper.train_step's loss_augment_fn cast — a small penalty rounds to zero in
        fp16 and would silently vanish.
        """
        lam = FLSettings.PARTIAL_CONSTRAIN_LAMBDA
        if not FLSettings.PARTIAL_CONSTRAIN or lam <= 0:
            return None
        constrained = self._noncarrying_names(carrying, elig_list)
        if not constrained:
            return None

        device = self.manager.device
        ref = {n: theta_t[n].to(device).float() for n in constrained}

        def penalty(model):
            acc = torch.zeros((), device=device)
            for name, p in model.named_parameters():
                if name in ref:
                    acc = acc + (p.float() - ref[name]).pow(2).sum()
            return lam * acc

        return penalty

    def _noncarrying_freeze_names(self, carrying, elig_list):
        """The names to hard-freeze for the exploit-phase non-carrying freeze, or None.

        Returns the non-carrying eligible tensor names (`eligible \\ C`); _run_trajectory
        sets requires_grad=False on exactly these for the mal trajectory. A frozen leaf
        gets NO gradient, so SGD skips it entirely — weight decay included — and it stays
        bit-exact at theta_t while the carriers co-adapt. Carrying tensors are absent (they
        train normally); BN/frozen are outside elig_list.

        Returns None (unfrozen training) when the switch is off or there are no non-carrying
        eligible params, so run() can pass the result straight through.
        """
        if not FLSettings.PARTIAL_FREEZE_NONCARRYING:
            return None
        frozen = self._noncarrying_names(carrying, elig_list)
        return frozen or None

    def _noncarrying_mask_names(self, carrying, elig_list):
        """The names to reset to theta_t at SUBMISSION time, or None.

        Same coordinate set as the freeze (`eligible \\ C`), applied to the finished
        submission instead of to training: theta_mal is trained completely unrestricted and
        mask_noncarrying then zeroes these tensors' pseudo-gradients. Freeze and mask put
        the same SUPPORT on the wire while differing in ||Delta_C|| — under the freeze the
        carriers absorb the pinned tensors' share of the task loss, under the mask they do
        not — which is the ablation the pair exists for.

        Returns None (nothing masked) when the switch is off or there are no non-carrying
        eligible params, so callers can pass the result straight through.
        """
        if not FLSettings.PARTIAL_MASK_NONCARRYING:
            return None
        masked = self._noncarrying_names(carrying, elig_list)
        return masked or None

    # ---- orchestration ----

    @staticmethod
    def _seeds(current_round, cid):
        """(mal, ben1, ben2). mal and ben1 SHARE a seed so the pair differs ONLY by the
        poison; ben2 is +1 so ben1-vs-ben2 measures pure PRNG noise."""
        seed_mal = FLSettings.SEED + 1000 * int(current_round) + int(cid)
        return seed_mal, seed_mal, seed_mal + 1

    def _exploit_trajectory(self, server_weights, cid, current_round, n_epochs, seed_mal):
        """Exploit round: train the poisoned run alone against the frozen carrying set.

        Both benign anchors are skipped once the set is frozen (they only fed the
        per-round localizer) -- roughly a third of the round's training cost. theta_mal is
        unchanged by the skip: every trajectory re-seeds its RNGs and restarts from
        theta_t with a fresh optimizer. `sel` is trajectory-free, so it resolves BEFORE
        the run -- constrain and freeze both need the frozen C to know what to exclude.

        Returns (theta_mal, sel, constrain_fn, freeze_names).
        """
        sel = self._exploit_sel()
        constrain_fn = self._constrain_penalty_fn(server_weights, sel["carrying"], sel["elig_list"])
        freeze_names = self._noncarrying_freeze_names(sel["carrying"], sel["elig_list"])

        n_nonc = len(sel["elig_list"]) - len(sel["carrying"])
        if constrain_fn:
            extra = f"; constrain lambda={FLSettings.PARTIAL_CONSTRAIN_LAMBDA:g} on {n_nonc} non-carrying tensors"
        elif freeze_names:
            extra = f"; freeze (requires_grad=False) {n_nonc} non-carrying tensors"
        else:
            extra = ""
        print(f"  [partial client={cid} round={current_round}] exploit: mal-only trajectory "
              f"on partition {cid} (seed={seed_mal}); benign anchors skipped{extra}", flush=True)

        theta_mal = self._run_trajectory(
            server_weights, cid, is_attacker=True, n_epochs=n_epochs, seed=seed_mal,
            loss_augment_fn=constrain_fn, freeze_names=freeze_names)
        return theta_mal, sel, constrain_fn, freeze_names

    def _plastic_trajectories(self, server_weights, cid, current_round, n_epochs, seeds):
        """Plastic/disabled round: all three trajectories, then per-round selection.

        Constrain and freeze do not apply here -- both exclude C, which is only known
        AFTER selection. Both benign anchors are measured for this client's own benign
        reference; ben2 is already trained for the noise floor, so the second sample is
        free (2 samples per plastic round this client acts in).

        Returns (theta_mal, sel, benign_norms, theta_ben1). theta_ben1 is returned so the
        caller can submit it instead of theta_mal under PARTIAL_PLASTIC_SUBMIT_BENIGN --
        the measurement is unaffected either way, since div/b come from the trained
        trajectories and not from whatever was submitted.
        """
        seed_mal, seed_ben1, seed_ben2 = seeds
        print(f"  [partial client={cid} round={current_round}] trajectories on partition "
              f"{cid}: ben1(seed={seed_ben1}) / mal(seed={seed_mal}) / ben2(seed={seed_ben2})",
              flush=True)
        theta_ben1 = self._run_trajectory(server_weights, cid, is_attacker=False,
                                          n_epochs=n_epochs, seed=seed_ben1)
        theta_mal = self._run_trajectory(server_weights, cid, is_attacker=True,
                                         n_epochs=n_epochs, seed=seed_mal)
        theta_ben2 = self._run_trajectory(server_weights, cid, is_attacker=False,
                                          n_epochs=n_epochs, seed=seed_ben2)

        sel = self._select_carrying(theta_mal, theta_ben1, theta_ben2)
        benign_norms = None
        if self._needs_benign_refs():
            benign_norms = [update_norm(theta_ben1, server_weights),
                            update_norm(theta_ben2, server_weights)]
        return theta_mal, sel, benign_norms, theta_ben1

    def _stash_round_state(self, cid, carrying, benign_norms, phase, in_exploit):
        """Publish this round's cross-round state: the carrying set, plus (for the
        server-side norm-clip reporter) this client's benign reference -- the live mean of
        this round's anchors before the freeze, its own frozen mean after. The reporter
        context is ephemeral: the server reads it this round and nothing persists it."""
        if FLSettings.DO_NORMCLIP_REPORT:
            live = (sum(benign_norms) / len(benign_norms)) if benign_norms else None
            ref = AttackState.get_benign_ref(cid) if in_exploit else live
            AttackState.set_normclip_ctx(cid, ref, phase)
        AttackState.set_carrying_set(cid, carrying)

    def _amplify(self, cid, current_round, server_weights, theta_mal, sel, carrying,
                 gamma, gamma_source, mask_names=None):
        """Build the submission: theta_t + gamma*(theta_mal - theta_t) on C only, i.e. the
        model-replacement form restricted to the carriers. Budgeted mode passes a
        PER-CARRIER gamma map instead of one flat factor.

        mask_names (the exploit-phase non-carrying mask, None when off) is applied AFTER
        amplification — it only ever touches tensors outside C, so the two never collide —
        and is folded into the budget: masked coordinates are submitted at theta_t and so
        contribute nothing to the submitted norm, which is why their energy is deducted
        from the unamplified energy the allocator rations against. Budgeting against drift
        that never leaves the client would leave most of the budget unspent, i.e. silently
        cancel the point of pairing the mask with a budget.

        RAISES if budgeted mode reaches a client with no benign reference. There is no
        budget to size, so the only thing this could do is submit unamplified — an attack
        that runs and does nothing, which is precisely the silent failure the startup
        check exists to prevent. parse_args catches this ahead of training whenever a
        client schedule makes coverage predictable; under random selection it is not
        predictable, so this is where it has to be caught.

        Returns (submission, norm_budget); norm_budget is None outside budgeted mode.
        """
        if mask_names:
            print(f"  [partial client={cid} round={current_round}] mask: {len(mask_names)} "
                  f"non-carrying tensor(s) reset to theta_t at submission (zero pseudo-gradient)",
                  flush=True)

        if gamma_source != "budgeted":
            submission = partial_amplify(server_weights, theta_mal, carrying, gamma)
            return mask_noncarrying(submission, server_weights, mask_names), None

        gamma_map, report = self._resolve_budgeted_gammas(
            cid, server_weights, theta_mal, sel, carrying, gamma, mask_names)
        if gamma_map is None:
            raise RuntimeError(
                f"[partial client={cid} round={current_round}] --partial-gamma-mode budgeted "
                f"needs a benign reference to size the norm budget, but client {cid} has none: "
                f"it never acted during the plastic phase "
                f"({AttackState.plastic_count} pooled injection(s) covered clients "
                f"{sorted(AttackState.plastic_benign_ref)}). References are per client and are "
                f"never pooled, so there is no fallback. Submitting unamplified would make this "
                f"attacker's exploit rounds silently no-ops, so the run stops here.\n"
                f"Fix by raising --partial-plastic-rounds so every attacker acts at least once "
                f"before the freeze, by using --use-client-schedule to guarantee coverage (which "
                f"also moves this check to startup), or by narrowing --attacker-fraction."
            )

        submission = partial_amplify(server_weights, theta_mal, carrying, gamma_map)
        submission = mask_noncarrying(submission, server_weights, mask_names)
        self._print_budget(cid, current_round, report)
        return submission, report

    def _freeze_block(self, sel, current_round):
        """Log record of a freeze that fired THIS round: the pooled decision, plus the
        standalone frozen_set.md one-look summary unless it is switched off."""
        mdiv, mb = AttackState.plastic_means()
        signed = sorted(((n, mdiv[n] - mb[n]) for n in mdiv), key=lambda x: (-x[1], x[0]))
        block = {
            "pooled_injections": AttackState.plastic_count,
            "frozen_set": sorted(AttackState.plastic_frozen_set),
            "top_mean_carriers": [[n, float(s)] for n, s in signed[:10]],   # mean(div)-mean(b)
        }
        if not FLSettings.LOG_FROZEN_SET_SUMMARY:
            return block

        numel = numel_per_tensor(self.manager.wrapper.model)
        self.manager.logger.partial.write_freeze_summary(
            current_round=current_round,
            select_mode=FLSettings.PARTIAL_SELECT_MODE,
            select_param=self._select_param(),
            pooled_injections=AttackState.plastic_count,
            frozen_set=AttackState.plastic_frozen_set,
            carrying_params=sum(numel.get(n, 0) for n in AttackState.plastic_frozen_set),
            total_params=sum(numel.values()),
            rows=[{
                "name": n,
                "group": sel["tensor_to_group"].get(n, "other"),
                "mean_div": float(mdiv[n]),
                "mean_b": float(mb[n]),
                # b is a PRNG floor, 0 only for frozen tensors (already ineligible) -- guard anyway.
                "ratio": (float(mdiv[n] / mb[n]) if mb[n] > 0 else float("inf")),
                "signed": float(mdiv[n] - mb[n]),
                "selected": n in AttackState.plastic_frozen_set,
            } for n in mdiv],
        )
        return block

    def run(self, server_weights, client_config):
        cid = client_config["index"]
        current_round = client_config.get("round")
        if current_round is None:
            raise KeyError(
                "PartialAmplificationProtocol requires 'round' in client_config "
                "(set in custom_fl_train.py's per-client client_config build)."
            )
        n_epochs = TrainingSettings.MALICIOUS_CLIENT_LOCAL_EPOCHS
        seeds = self._seeds(current_round, cid)
        in_exploit = FLSettings.PARTIAL_PLASTIC_ROUNDS > 0 and AttackState.is_plastic_frozen()

        # ---- 1. train the trajectories and pick this round's candidate carriers ----
        if in_exploit:
            theta_mal, sel, constrain_fn, freeze_names = self._exploit_trajectory(
                server_weights, cid, current_round, n_epochs, seeds[0])
            benign_norms = None
            theta_ben1 = None            # no anchors after the freeze
        else:
            theta_mal, sel, benign_norms, theta_ben1 = self._plastic_trajectories(
                server_weights, cid, current_round, n_epochs, seeds)
            constrain_fn = freeze_names = None

        # ---- 2. resolve the round: phase, carrying set, gamma ----
        # Phase is captured BEFORE _resolve_carrying, which may fire the freeze: the
        # triggering round still counts as plastic and the freeze takes effect next round.
        plastic_on = FLSettings.PARTIAL_PLASTIC_ROUNDS > 0
        in_plastic = plastic_on and not AttackState.is_plastic_frozen()
        carrying = self._resolve_carrying(sel, in_plastic, plastic_on, cid, benign_norms,
                                          current_round)
        froze_this_round = in_plastic and AttackState.is_plastic_frozen()
        phase = "disabled" if not plastic_on else ("plastic" if in_plastic else "exploit")
        gamma, gamma_source = self._resolve_gamma(client_config, in_plastic, in_exploit)

        # ---- 3. publish cross-round state, then build the submission ----
        # The mask edits the SUBMISSION only, so unlike freeze_names it is resolved here
        # (after training) and never reaches _run_trajectory.
        mask_names = self._noncarrying_mask_names(carrying, sel["elig_list"]) if in_exploit else None
        self._stash_round_state(cid, carrying, benign_norms, phase, in_exploit)
        self._print_selection(cid, current_round, carrying, sel["ranked"], gamma, phase)
        submission, norm_budget = self._amplify(
            cid, current_round, server_weights, theta_mal, sel, carrying, gamma, gamma_source,
            mask_names)

        # Plastic phase only: hand in the benign anchor rather than the poisoned model.
        # Deliberately AFTER _amplify so the logged norms still describe what the poisoned
        # submission WOULD have been, which is what makes the two modes comparable; only
        # the returned weights change. The localizer is untouched -- sel, div/b and the
        # pooled accumulator all come from the trained trajectories above.
        plastic_submit_benign = (in_plastic and FLSettings.PARTIAL_PLASTIC_SUBMIT_BENIGN
                                 and theta_ben1 is not None)
        if plastic_submit_benign:
            submission = theta_ben1
            print(f"  [partial client={cid} round={current_round}] plastic: submitting the "
                  f"BENIGN anchor (ben1); the poisoned model is measured but not sent",
                  flush=True)

        # ---- 4. per-round log blocks; each stays None unless its feature ran ----
        freeze = self._freeze_block(sel, current_round) if froze_this_round else None
        constrain = None if constrain_fn is None else {
            "lambda": FLSettings.PARTIAL_CONSTRAIN_LAMBDA,
            "constrained_tensors": len(sel["elig_list"]) - len(carrying),
        }
        freeze_noncarrying = None if freeze_names is None else {"frozen_tensors": len(freeze_names)}
        mask_noncarrying_block = None if mask_names is None else {"masked_tensors": len(mask_names)}

        norms = self._norm_diagnostics(server_weights, theta_mal, submission, carrying,
                                       sel["elig_list"])
        # realized_l2 is measured off the submission while the allocator's predicted_l2 is
        # its own arithmetic, so the two agreeing is an independent check on the allocator.
        if norm_budget is not None and not norm_budget.get("no_benign_ref"):
            norm_budget["realized_l2"] = update_norm(submission, server_weights)
            norm_budget["within_budget"] = (
                norm_budget["realized_l2"] <= norm_budget["budget_l2"] * (1.0 + 1e-6))

        # ---- 5. write the round log and return the submission ----
        mode = FLSettings.PARTIAL_SELECT_MODE
        select_param = self._select_param()
        log_path = self.manager.logger.partial.write_round(
            current_round=current_round, client_id=cid, phase=phase,
            select_mode=mode, select_param=select_param,
            gamma=gamma, gamma_source=gamma_source,
            carrying=carrying,
            carrying_by_group=self._carrying_by_group(carrying, sel["tensor_to_group"]),
            top_carriers=[[n, float(s)] for n, s in sel["ranked"][:10]],
            norms=norms, freeze=freeze, constrain=constrain,
            freeze_noncarrying=freeze_noncarrying, mask_noncarrying=mask_noncarrying_block,
            norm_budget=norm_budget,
            submitted=("benign_anchor" if plastic_submit_benign else "malicious"),
        )

        metrics = {
            "partition_id": cid,
            "protocol": "partial",
            "phase": phase,
            "select_mode": mode,
            "select_param": select_param,
            "carrying_size": len(carrying),
            "amp_gamma": gamma,
            "gamma_source": gamma_source,
            "constrain_lambda": FLSettings.PARTIAL_CONSTRAIN_LAMBDA if constrain_fn is not None else 0.0,
            "freeze_noncarrying": freeze_names is not None,
            "mask_noncarrying": mask_names is not None,
            "norm_budget_l2": (norm_budget or {}).get("budget_l2"),
            "norm_budget_realized_l2": (norm_budget or {}).get("realized_l2"),
            "partial_log": log_path,
        }
        return submission, len(self.manager.subsets_mal[cid]), metrics

    @staticmethod
    def _select_param():
        """The selector's own parameter, for the logs: cutoff value, k, or the names."""
        mode = FLSettings.PARTIAL_SELECT_MODE
        if mode == "cutoff":
            return FLSettings.PARTIAL_CUTOFF
        if mode == "names":
            return ", ".join(FLSettings.PARTIAL_CARRYING_NAMES)
        return FLSettings.PARTIAL_TOPK

    def _print_selection(self, cid, current_round, carrying, ranked, gamma, phase):
        mode = FLSettings.PARTIAL_SELECT_MODE
        param = self._select_param()
        top = ", ".join(f"{name}={score:.3g}" for name, score in ranked[:5]) or "(none)"
        # In budgeted mode gamma is the per-carrier CEILING, not a factor applied to all
        # of C — say so, or the line reads as if every carrier got it (the norm-budget
        # line printed just after shows what was actually spent).
        g_label = ("gamma<=" if FLSettings.PARTIAL_GAMMA_MODE == "budgeted" else "gamma=")
        print(
            f"[partial client={cid} round={current_round}] phase={phase} carrying set "
            f"|C|={len(carrying)} (mode={mode}, param={param}, {g_label}{gamma:.3g})",
            flush=True,
        )
        print(f"    top carriers ({'ratio=div/b' if mode == 'cutoff' else 'rho=div-b'}): {top}", flush=True)

    @staticmethod
    def _print_budget(cid, current_round, rep):
        """One-look line on how the norm budget was spent this round."""
        if rep.get("deficit"):
            over = rep["unamplified_l2"] / rep["budget_l2"] if rep["budget_l2"] else float("nan")
            print(
                f"  [partial client={cid} round={current_round}] norm budget DEFICIT: "
                f"unamplified ||Delta||={rep['unamplified_l2']:.4g} already "
                f"{over:.2f}x the budget {rep['budget_l2']:.4g} "
                f"(B={rep['B']:g} x benign_ref={rep['benign_ref']:.4g}) -> no amplification. "
                f"Lower the update with --partial-constrain / --partial-freeze-noncarrying "
                f"/ --partial-mask-noncarrying, or raise --partial-norm-budget.",
                flush=True,
            )
            return
        where = rep["throttled_at"] or "(budget not exhausted)"
        print(
            f"  [partial client={cid} round={current_round}] norm budget: "
            f"{rep['budget_l2']:.4g} (B={rep['B']:g} x benign_ref={rep['benign_ref']:.4g} from "
            f"{rep['benign_ref_samples']} anchor(s)); gamma_full={rep['gamma_full']:.3g} to "
            f"{rep['n_full']} carrier(s), throttled at {where}, {rep['n_unamplified']} left "
            f"unamplified; predicted ||Delta||={rep['predicted_l2']:.4g}",
            flush=True,
        )

    @staticmethod
    def _carrying_by_group(carrying, tensor_to_group):
        """Roll the per-tensor carrying set up to the 13 groups: {group: #tensors}."""
        counts = {}
        for name in carrying:
            g = tensor_to_group.get(name, "other")
            counts[g] = counts.get(g, 0) + 1
        return counts

    def _norm_diagnostics(self, theta_t, theta_mal, submission, carrying, elig_list):
        """Norm/energy diagnostics for the submission, all over the ELIGIBLE tensor set
        (the stealth harness's coordinate set). norm_inflation_ratio =
        ||submission - theta_t|| / ||theta_mal - theta_t|| is the exact quantity a
        future norm budget will drive back toward 1.0.

        Also reports how the update's L2 ENERGY (||.||^2) splits between the carrying and
        non-carrying sets BEFORE vs AFTER amplification, as percentages side by side —
        the direct read on how much amplification concentrated (blew up) the update in C.
        And the carrying set's size as a fraction of the model's parameters.
        """
        def l2(a, b, names):
            s = 0.0
            for n in names:
                s += float((a[n].detach().float() - b[n].detach().float()).pow(2).sum())
            return math.sqrt(s)

        carrying_list = [n for n in elig_list if n in carrying]
        noncarrying_list = [n for n in elig_list if n not in carrying]
        dC_pre = l2(theta_mal, theta_t, carrying_list)       # ||Delta_C|| pre-amplification
        d_pre = l2(theta_mal, theta_t, elig_list)            # unamplified total (== theta_mal update)
        d_post = l2(submission, theta_t, elig_list)          # submitted total (post-amplification)
        # Non-carrying drift from theta_t, measured on BOTH sides. partial_amplify leaves it
        # alone, but the submission-side mask sends it to exactly 0 — so pre and post differ
        # whenever --partial-mask-noncarrying is on, and taking one for the other would put
        # the energy percentages on inconsistent totals. It is also what the constrain
        # penalty drives toward 0 (smaller => the update hides better under L2 / Krum).
        d_nonC_pre = l2(theta_mal, theta_t, noncarrying_list)
        d_nonC = l2(submission, theta_t, noncarrying_list)

        # Energy split (%). Energy = L2^2, so the carrying/non-carrying energies add up to
        # the total on each side (carrying_list and noncarrying_list partition elig_list),
        # giving tot_pre == d_pre^2 and tot_post == d_post^2. Pre vs post is then the direct
        # read on how much amplification (and, under the mask, discarding) shifted the
        # update's mass into C.
        #
        # The post-amplification carrying norm is MEASURED off the submission rather than
        # derived as gamma*dC_pre. Those agree under a single scalar gamma, but budgeted
        # mode gives every carrier its own gamma, for which no scalar factor exists — so
        # measuring keeps this correct for all three gamma modes.
        dC_post = l2(submission, theta_t, carrying_list)
        e_carry_pre = dC_pre ** 2
        e_nonc_pre = d_nonC_pre ** 2
        e_nonc = d_nonC ** 2                                  # == e_nonc_pre unless masked
        e_carry_post = dC_post ** 2
        tot_pre = e_carry_pre + e_nonc_pre                    # == d_pre^2
        tot_post = e_carry_post + e_nonc                      # == d_post^2
        def pct(part, tot):
            return (100.0 * part / tot) if tot > 0 else float("nan")

        # Carrying-set size as a fraction of ALL model parameters (not just tensor count).
        numel = numel_per_tensor(self.manager.wrapper.model)
        total_params = sum(numel.values())
        carrying_params = sum(numel.get(n, 0) for n in carrying_list)

        return {
            "delta_carrying_l2_pre": dC_pre,
            "delta_carrying_l2_post": dC_post,               # measured, not gamma*dC_pre
            "delta_total_l2_unamplified": d_pre,
            "delta_total_l2_submitted": d_post,
            "delta_noncarrying_l2_pre": d_nonC_pre,          # theta_mal's own drift off C
            "delta_noncarrying_l2_submitted": d_nonC,        # constrain target (-> 0; exactly 0 when masked)
            "norm_inflation_ratio": (d_post / d_pre) if d_pre > 0 else float("nan"),
            # Carrying set size vs the whole model.
            "carrying_params": carrying_params,
            "total_params": total_params,
            "carrying_params_pct": pct(carrying_params, total_params),
            # Energy split before vs after amplification, side by side.
            "energy_pct": {
                "carrying_pre": pct(e_carry_pre, tot_pre),
                "noncarrying_pre": pct(e_nonc_pre, tot_pre),
                "carrying_post": pct(e_carry_post, tot_post),
                "noncarrying_post": pct(e_nonc, tot_post),
            },
        }
