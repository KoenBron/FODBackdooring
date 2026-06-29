"""PilotB observational protocol — per-tensor granularity.

Same three-trajectory observational design as PilotA (mal / ben1 / ben2 on the
attacker's own partition, differing only by poison and RNG seed), but the
localization signal is computed at PER-TENSOR granularity instead of the 13
groups: every named parameter is its own bucket. This is the fine-grained half
of the cross-granularity comparison — does the backdoor concentrate in a few
individual tensors, and do those tensors fall inside the groups PilotA's coarse
localizer already picks?

PilotB reuses PilotA's trajectory engine wholesale by subclassing it
(_run_trajectory, _dataloader_for, _seed_model_rngs, _calibrate_beta_max are
inherited unchanged). Only the per-epoch metric computation and the headline are
per-tensor specific, so only those are overridden.

Scope: OUR L2 localization at per-tensor granularity, plus a per-round LaySelFL
sensitivity selection compared against it (same top-k selector, so the contrast
is purely score-vs-score). The per-group roll-up consistency check lands later;
the JSON already carries `tensor_to_group` so that roll-up needs no re-run.

Outputs one JSON per (round, client_id) under <output>/attack_logs/pilot_b/.
The schema mirrors PilotA's but keyed by tensor name, plus `tensor_to_group`
for offline coarsening.
"""
from __future__ import annotations

import json
import os
import statistics

from FL.attack_protocols.pilot_a import PilotAProtocol, _fmt
from FL.groups import build_group_map
from FL.scoring.l2_based import (
    numel_per_tensor,
    frozen_tensors,
    squared_divergence_per_tensor,
    identity_partition,
    normalized_l2_divergence,
    rho,
    select_topk,
    select_cutoff,
)
from FL.scoring.layselfl import laysel_sensitivity, select_topk as laysel_select_topk
from utils.global_settings import FLSettings, TrainingSettings


# Per-tensor selection sweeps. k is in tens (≈265 tensors to rank), not ones as
# at group granularity. cutoff is div-based (div > c*floor) and uses the SAME
# grid as PilotA so the cross-granularity cutoff comparison is apples-to-apples.
_TOPK_VALUES = (5, 10, 20, 50)
_CUTOFF_VALUES = (1.5, 2.0, 3.0)


class PilotBLogger:
    """Sub-logger owned by ModelLogger; one JSON per (round, client) under
    <output>/attack_logs/pilot_b/. Schema owned here next to PilotBProtocol.
    """

    def __init__(self, attack_logs_dir):
        self.dir = os.path.join(attack_logs_dir, "pilot_b")
        os.makedirs(self.dir, exist_ok=True)

    def write_round(self, current_round, client_id, n_epochs, tensors, numel,
                    tensor_to_group, frozen_tensors, epochs, laysel, beta_max):
        target = os.path.join(self.dir, f"round_{current_round}_client_{client_id}.json")
        payload = {
            "round": current_round,
            "client_id": client_id,
            "n_epochs": n_epochs,
            "granularity": "per_tensor",
            "tensors": tensors,
            "numel": numel,
            "tensor_to_group": tensor_to_group,
            "frozen_tensors": frozen_tensors,
            "epochs": epochs,
            "laysel": laysel,
            "beta_max": beta_max,
        }
        with open(target, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(
            f"  [pilot_b client={client_id} round={current_round}] log written: {target}",
            flush=True,
        )
        return os.path.join("attack_logs", "pilot_b", os.path.basename(target))


class PilotBProtocol(PilotAProtocol):
    """Per-tensor observational localizer. Inherits PilotA's trajectory engine."""

    name: str = "pilot_b"

    def run(self, server_weights, client_config):
        client_id = client_config["index"]
        current_round = client_config.get("round")
        if current_round is None:
            raise KeyError(
                "PilotBProtocol requires 'round' in client_config — added in "
                "custom_fl_train.py's per-client client_config build."
            )

        n_epochs = TrainingSettings.MALICIOUS_CLIENT_LOCAL_EPOCHS

        # Same seed recipe as PilotA: mal and ben1 share seed S (only difference
        # is the poison); ben2 uses S+1 so the floor measures pure RNG noise.
        seed_mal = FLSettings.SEED + 1000 * int(current_round) + int(client_id)
        seed_ben1 = seed_mal
        seed_ben2 = seed_mal + 1
        print(
            f"  [pilot_b client={client_id} round={current_round}] "
            f"trajectories on partition {client_id}: "
            f"mal(seed={seed_mal}) / ben1(seed={seed_ben1}) / ben2(seed={seed_ben2})",
            flush=True,
        )

        # ben1 (anchor) -> beta_max -> mal -> ben2. Engine inherited from PilotA.
        ben1_snaps = self._run_trajectory(
            server_weights, partition_id=client_id, is_attacker=False,
            n_epochs=n_epochs, label="ben1", seed=seed_ben1,
        )
        # mal_dl = self._dataloader_for(client_id, is_attacker=True, seed=seed_mal + 2)
        # beta_max = self._calibrate_beta_max(
        #     probe_state=server_weights,
        #     anchor_state=ben1_snaps[-1],
        #     dataloader=mal_dl,
        #     seed=seed_mal + 2,
        # )
        # PilotB does not calibrate beta_max (localization-only). Use a schema-
        # consistent sentinel (None) rather than a free string so analysis code
        # never branches on type — PilotA writes a dict here, PilotB writes null.
        beta_max = None

        mal_snaps = self._run_trajectory(
            server_weights, partition_id=client_id, is_attacker=True,
            n_epochs=n_epochs, label="mal", seed=seed_mal,
        )
        ben2_snaps = self._run_trajectory(
            server_weights, partition_id=client_id, is_attacker=False,
            n_epochs=n_epochs, label="ben2", seed=seed_ben2,
        )

        # Per-tensor metrics. Pure CPU pass over the snapshots. server_weights IS
        # theta_t (the round's broadcast); our L2 div doesn't need it, but the
        # LaySelFL volatility uses it as the epoch-0 anchor for the first step.
        epochs_data = self._compute_epoch_metrics(
            mal_snaps, ben1_snaps, ben2_snaps, server_weights,
        )

        log_path = self.manager.logger.pilot_b.write_round(
            current_round=current_round,
            client_id=client_id,
            n_epochs=n_epochs,
            tensors=epochs_data["tensors"],
            numel=epochs_data["numel"],
            tensor_to_group=epochs_data["tensor_to_group"],
            frozen_tensors=epochs_data["frozen_tensors"],
            epochs=epochs_data["epochs"],
            laysel=epochs_data["laysel"],
            beta_max=beta_max,
        )
        self._print_headline(epochs_data, beta_max, client_id, current_round)

        # Observational: submit the benign-anchor endpoint, no scaling.
        num_samples = len(self.manager.subsets_mal[client_id])
        metrics = {
            "partition_id": client_id,
            "protocol": "pilot_b",
            "pilot_log": log_path,
        }
        return ben1_snaps[-1], num_samples, metrics

    # ---- Per-tensor metrics (overrides PilotA's per-group version) ----

    def _compute_epoch_metrics(self, mal_snaps, ben1_snaps, ben2_snaps, server_weights):
        """Per-epoch div_t, b_t, rho_t + selectors at PER-TENSOR granularity, plus
        a per-round LaySelFL sensitivity selection compared against ours.

        Identity partition (each tensor its own bucket) routed through the same
        normalized_l2_divergence as PilotA's groups, so the fine scores are exactly
        the per-group ones un-coarsened. div_t = ||mal - ben1||_t / sqrt(numel_t);
        b_t is the expensive floor from ben2; sigma is the cheap floor (stdev of
        div_t over trainable tensors — ≈265 samples, far more stable than the
        11-trainable-group sigma).

        LaySelFL sensitivity is a single per-round score (it averages volatility
        over all epochs and adds the endpoint gap), so it is computed once after
        the loop and compared against OUR final-epoch rho selection.
        """
        model = self.manager.wrapper.model
        tensor_list, tensor_to_group, _group_to_tensors = build_group_map(model)
        numel = numel_per_tensor(model)
        id_part = identity_partition(tensor_list)
        # theta_t pinned to CPU (no-op if already CPU) and restricted to the
        # tensors we compare, so LaySelFL's first volatility step ||theta_1 - theta_t||
        # matches the CPU snapshots.
        theta_t = {n: server_weights[n].cpu() for n in tensor_list}

        frozen = frozen_tensors(model)
        trainable = [t for t in tensor_list if t not in frozen]

        epochs_data = []
        for n, (m, b1, b2) in enumerate(zip(mal_snaps, ben1_snaps, ben2_snaps), start=1):
            # Canonical additive squared atoms, computed once per pair.
            sq_div = squared_divergence_per_tensor(m, b1, tensor_list)   # mal vs ben1
            sq_b = squared_divergence_per_tensor(b1, b2, tensor_list)    # ben1 vs ben2

            div_t = normalized_l2_divergence(sq_div, numel, id_part)
            b_t = normalized_l2_divergence(sq_b, numel, id_part)
            rho_t = rho(div_t, b_t)

            # Cheap floor: cross-tensor sigma of div_t over trainable tensors.
            trainable_divs = [div_t[t] for t in trainable]
            sigma = statistics.stdev(trainable_divs) if len(trainable_divs) > 1 else 0.0
            b_cheap = {t: sigma for t in div_t}
            rho_cheap = rho(div_t, b_cheap)

            # topk ranks the denoised rho; cutoff thresholds the RAW div against
            # c*floor (div > c*floor), matching PilotA's _select_cutoff exactly.
            selections = {
                "topk":   {str(k): select_topk(rho_t, k) for k in _TOPK_VALUES},
                "cutoff": {str(c): select_cutoff(div_t, b_t, c) for c in _CUTOFF_VALUES},
            }
            selections_cheap = {
                "topk":   {str(k): select_topk(rho_cheap, k) for k in _TOPK_VALUES},
                "cutoff": {str(c): select_cutoff(div_t, b_cheap, c) for c in _CUTOFF_VALUES},
            }

            epochs_data.append({
                "n": n,
                "div": div_t,
                "b": b_t,
                "rho": rho_t,
                "sigma": sigma,
                "b_cheap": b_cheap,
                "selections": selections,
                "selections_cheap": selections_cheap,
            })

        # ---- LaySelFL cross-method comparison (per-round, post-loop) ----
        # sens_d / sens_c kept separate (not just the sum) so we can see whether
        # the convergent endpoint gap dominates the dynamic volatility term.
        sensitivity, sens_d, sens_c = laysel_sensitivity(
            ben_snaps=ben1_snaps, mal_snaps=mal_snaps,
            theta_t=theta_t, tensor_list=tensor_list,
        )
        # LaySelFL's own filter-free top-k (no score>0 gate) — faithful to the
        # baseline, which ranks raw sensitivity including non-positive layers.
        laysel_selections = {str(k): laysel_select_topk(sensitivity, k) for k in _TOPK_VALUES}

        # Compare against OUR final-epoch rho top-k. Exclude frozen tensors from
        # BOTH sides first: LaySelFL's unfiltered top-k can pull in frozen layers
        # (sensitivity 0) when fewer than k positive scores exist, but a frozen
        # layer is an untrainable non-carrier — counting it would distort the
        # Jaccard. We still report which frozen layers LaySelFL selected.
        frozen_set = set(frozen)
        final_sel = epochs_data[-1]["selections"]["topk"]
        vs_ours = {}
        for k in _TOPK_VALUES:
            ours = {g for g, _ in final_sel[str(k)]} - frozen_set      # rho>0 already excludes these
            theirs_raw = {t for t, _ in laysel_selections[str(k)]}
            theirs = theirs_raw - frozen_set
            union = ours | theirs
            vs_ours[str(k)] = {
                "jaccard": (len(ours & theirs) / len(union)) if union else 1.0,
                "ours_only": sorted(ours - theirs),                  # our floor keeps, they miss
                "laysel_only": sorted(theirs - ours),                # they pick, our floor rejects
                "laysel_frozen_selected": sorted(theirs_raw & frozen_set),  # contamination report
            }

        laysel = {
            "sensitivity": sensitivity,
            "sens_dynamic": sens_d,
            "sens_convergent": sens_c,
            "selections": laysel_selections,
            "vs_ours": vs_ours,
        }

        return {
            "tensors": tensor_list,
            "numel": numel,
            "tensor_to_group": tensor_to_group,   # lets analysis roll up to groups
            "frozen_tensors": sorted(frozen),
            "epochs": epochs_data,
            "laysel": laysel,
        }

    # ---- Headline (per-tensor) ----

    def _print_headline(self, epochs_data, beta_max, client_id, current_round):
        final = epochs_data["epochs"][-1]
        t2g = epochs_data["tensor_to_group"]
        ranked = sorted(final["rho"].items(), key=lambda kv: -kv[1])
        top5 = ", ".join(f"{t}[{t2g[t]}]={v:.3g}" for t, v in ranked[:5])
        n_tensors = len(epochs_data["tensors"])
        n_frozen = len(epochs_data["frozen_tensors"])
        # Cheap-vs-expensive sanity check at k=20 (mid of the per-tensor sweep).
        sel_exp = {t for t, _ in final["selections"]["topk"]["20"]}
        sel_cheap = {t for t, _ in final["selections_cheap"]["topk"]["20"]}
        union = sel_exp | sel_cheap
        jaccard_k20 = (len(sel_exp & sel_cheap) / len(union)) if union else 1.0
        laysel_jac = epochs_data["laysel"]["vs_ours"]["20"]["jaccard"]
        print(f"[pilot_b client={client_id} round={current_round}] HEADLINE (per-tensor):")
        print(f"    final-epoch rho top-5: {top5}")
        print(f"    frozen tensors: {n_frozen}/{n_tensors}")
        print(f"    top-20 Jaccard (cheap floor vs expensive): {jaccard_k20:.3g}")
        print(f"    top-20 Jaccard (LaySelFL sensitivity vs ours): {laysel_jac:.3g}")
