import torch

from collections import OrderedDict

from FL.groups import name_to_group
from FL.strategy import BaseStrategy
from FL.strategies._robust import foolsgold_weights


# A client is reported as suppressed when it receives less than this fraction of the
# fair share 1/n it would have got under plain FedAvg. Only affects REPORTING — the
# aggregation always uses the continuous weights — but it is what makes the server's
# kept/rejected line, and its "attackers rejected k/N" count, mean what a reader assumes.
SUPPRESSED_SHARE = 0.5


class FoolsGold(BaseStrategy):
    """FoolsGold sybil-resistant aggregation (Fung et al., RAID 2020), server-side.

    Like every strategy here it sees only the submitted updates (weights +
    num_samples), NEVER an attacker/benign label. Each round it:
      1. forms each client's update Delta_i = w_i - w_global, restricted to the
         INDICATIVE FEATURES (see below),
      2. accumulates it into that client's cumulative history H_i,
      3. computes pairwise cosine similarity between the participants' histories,
      4. turns those into per-client weights via the shared torch-free
         FL.strategies._robust.foolsgold_weights, and
      5. adds the weighted sum of pseudo-gradients to the broadcast model.

    Unlike Multi-Krum this RE-WEIGHTS rather than selects: a suspicious client is
    scaled toward zero, not removed from a kept set.

    Deltas, not weights
    -------------------
    This codebase transports ABSOLUTE weights (FL.custom_client.get_parameters returns
    the full state_dict; FedAvg.aggregate starts from zeros and never re-adds the
    global). Multi-Krum can ignore that because pairwise distances are translation
    invariant, but FoolsGold cannot: the history sums across rounds whose broadcast
    globals differ, so accumulating raw weights would add sum_t theta_t into every
    client's history and swamp the direction signal the cosine is meant to read.
    Hence the explicit subtraction against `global_weights` in aggregate().

    Indicative features
    -------------------
    History over the whole model would be ~41M floats per client (~3.3 GB for 20
    clients). The paper independently recommends restricting to the output layer's
    "indicative features", which both sharpens the sybil signal and shrinks the state
    to ~21.5K floats per client (~1.7 MB for 20). For a TARGETED_MISLABELING backdoor
    the classification head is exactly where the signal lives; a box-transform attack
    would point FOOLSGOLD_FEATURE_GROUP at roi.bbox_pred instead.

    Partial participation
    ---------------------
    The paper assumes every client reports every round; here 9 of 20 do. History
    PERSISTS for absent clients, and each round's cosines are computed among the
    round's participants but over their FULL histories. That is the natural
    adaptation, and a deviation from the paper worth stating in any write-up.
    """

    def __init__(self, model_state_dict: OrderedDict, kappa: float,
                 feature_group: str, normalise: str = "none") -> None:
        self.model_state_dict = model_state_dict
        self.kappa = kappa
        self.normalise = normalise
        self.feature_group = feature_group
        self.feature_names = self._resolve_features(model_state_dict, feature_group)
        # cid -> 1-D float tensor: the cumulative sum of that client's updates over the
        # indicative features. Survives resume via state_dict()/load_state_dict().
        self.history = {}
        self._degenerate_warned = False
        self.reset_state()

    @staticmethod
    def _resolve_features(model_state_dict, feature_group):
        """The state_dict keys making up the indicative-feature group, sorted.

        Sorted so every client's history lives in the same coordinate system. Raises if
        the group matches nothing: an empty feature set would make every history the
        zero vector, every cosine 0 and every weight uniform — a defense that reports
        itself as running while doing nothing at all.
        """
        names = sorted(k for k in model_state_dict
                       if torch.is_floating_point(model_state_dict[k])
                       and name_to_group(k) == feature_group)
        if not names:
            groups = sorted({g for g in (name_to_group(k) for k in model_state_dict) if g})
            raise ValueError(
                f"FoolsGold: --foolsgold-feature-group {feature_group!r} matched no "
                f"floating-point parameter in the model. Tracking an empty feature set "
                f"would silently disable the defense. Available groups: {groups}"
            )
        return names

    def reset_state(self):
        # Round state only. `history` is deliberately NOT cleared — it is the whole
        # point of the strategy and must accumulate across rounds (and across resumes).
        self.last_selection = None

    def _delta_vector(self, weights, global_weights):
        """Flatten w - w_global over the indicative features into one vector."""
        parts = [(weights[k].detach().cpu().float()
                  - global_weights[k].detach().cpu().float()).flatten()
                 for k in self.feature_names]
        return torch.cat(parts)

    @staticmethod
    def _cosine(a, b):
        """cos(a, b), with 0 for a zero-norm history.

        A client whose cumulative update is exactly zero has no direction to compare;
        returning 0 treats it as maximally dissimilar, which leaves it un-penalised.
        """
        na = float(torch.linalg.vector_norm(a))
        nb = float(torch.linalg.vector_norm(b))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return float(torch.dot(a, b)) / (na * nb)

    def _warn_degenerate(self, n):
        if self._degenerate_warned:
            return
        self._degenerate_warned = True
        print(
            f"[WARNING] FoolsGold found every one of this round's n={n} clients maximally "
            f"similar to some peer, so all confidence scores collapsed to zero. Falling "
            f"back to UNIFORM weights for such rounds — without that fallback the convex "
            f"combination would be all-zero and would overwrite the global model with "
            f"zeros. Treat these rounds as 'defense had no opinion', not 'all clean'.",
            flush=True,
        )

    def _print_weights(self, cids, weights, max_cs, trust, weight_sum):
        """One line per round: the actual output of the defense.

        The server's generic robust-strategy log reports kept/rejected, which is the
        right shape for a SELECTING defense like Multi-Krum. FoolsGold re-weights, so
        the interesting quantity is the weight each client received and the peak history
        cosine that produced it — neither of which the kept/rejected view shows.
        """
        body = "  ".join(
            f"{c}:w={weights[c]:.3f}(x{trust[c]:.2f}),cs={max_cs[c]:+.2f}" for c in cids)
        # sum(w) is the effective step multiplier under the paper-faithful "none" mode:
        # 1.0 means the aggregate is one average client's delta, 3.0 means three times
        # that. It grows with the number of trusted participants, which is the property
        # the paper's un-normalised weights have and the reason to keep it in the log.
        print(f"  [foolsgold] {body}   sum(w)={weight_sum:.2f} "
              f"[step multiplier; norm={self.normalise}] "
              f"[xR<{SUPPRESSED_SHARE} = suppressed]", flush=True)

    def aggregate(self, client_updates: list, global_weights=None) -> OrderedDict:
        # Clear FIRST, before anything can raise, so a failed aggregation cannot leave
        # the previous round's decision in place for the server to log as this round's.
        self.last_selection = None

        if not client_updates:
            raise Exception("[ERROR] No client updates given to aggregate")
        if global_weights is None:
            raise ValueError(
                "FoolsGold needs the broadcast global to form Delta_i = w_i - w_global; "
                "clients submit absolute weights, and the history sums across rounds "
                "whose globals differ."
            )

        # Fixed cid order: float addition is not associative, so an unsorted combine
        # would make the aggregate depend on client arrival order.
        updates = sorted(client_updates, key=lambda u: u["cid"])
        cids = [u["cid"] for u in updates]
        n = len(cids)

        # FoolsGold's alpha REPLACES the sample weighting rather than multiplying it
        # (paper-faithful). That is only equivalent to FedAvg's num_samples weighting
        # when the partitions are equal, which this project's partitioner guarantees
        # (see NONIID_PARTITIONING.md: label skew at fixed size). Under UNEQUAL sizes the
        # two are genuinely different aggregations, and silently picking one would be a
        # landmine, so refuse rather than choose on the caller's behalf.
        sizes = {u["cid"]: u["num_samples"] for u in updates}
        distinct = set(sizes.values())
        if len(distinct) > 1:
            raise ValueError(
                f"FoolsGold weights clients by alpha ALONE, which replaces FedAvg's "
                f"sample weighting. That is only equivalent when partitions are equal, but "
                f"this round has {len(distinct)} distinct partition sizes "
                f"({dict(sorted(sizes.items()))}). Decide explicitly: keep equal-size "
                f"partitions (--partition-mode iid / dirichlet both produce them), or "
                f"extend _combine to weight by alpha_i * n_i."
            )

        # 1-2. accumulate this round's update into each participant's history.
        for u in updates:
            delta = self._delta_vector(u["weights"], global_weights)
            prev = self.history.get(u["cid"])
            self.history[u["cid"]] = delta if prev is None else prev + delta

        # 3. pairwise cosine over the participants' FULL histories.
        cosine = {i: {} for i in cids}
        for a in range(n):
            i = cids[a]
            for b in range(a + 1, n):
                j = cids[b]
                c = self._cosine(self.history[i], self.history[j])
                cosine[i][j] = cosine[j][i] = c

        # 4. weights.
        res = foolsgold_weights(cosine, cids, self.kappa, self.normalise)
        weights = res["weights"]
        if res["degenerate"]:
            self._warn_degenerate(n)

        # Populate the shape the server already logs for robust strategies.
        #
        # FoolsGold re-weights rather than selects, so "rejected" needs defining. Using
        # weight == 0 would UNDERSTATE the defense badly: because the weights are
        # normalised to a convex combination, a client is only exactly zero if the logit
        # clipped it there, and one left at a hundredth of its fair share still counts as
        # "kept". The server's log line — and the "attackers rejected k/N" it prints —
        # would then read as though the defense did nothing while it had in fact silenced
        # the attacker.
        #
        # So suppression is measured against what a FULLY TRUSTED client receives, which
        # differs by mode: 1.0 under the paper-faithful "none" (the weight IS the trust
        # score), and the fair share 1/n under "sum". `trust_ratio` divides by that
        # reference, so it reads the same way in both — 1.0 = full contribution, 0 =
        # silenced — and is what the CSV and the round line report.
        full = 1.0 if self.normalise == "none" else 1.0 / n
        share = {c: (weights[c] / full if full > 0 else 0.0) for c in cids}
        rejected = sorted(c for c in cids if share[c] < SUPPRESSED_SHARE)
        self.last_selection = {
            "kept": sorted(c for c in cids if share[c] >= SUPPRESSED_SHARE),
            "rejected": rejected,
            "scores": {c: res["max_cs"][c] for c in cids},   # peak history cosine
            "weights": {c: weights[c] for c in cids},
            "trust_ratio": share,          # 1.0 == a fully-trusted client's contribution
            "silenced": sorted(c for c in cids if weights[c] <= 1e-12),
            "suppressed_threshold": SUPPRESSED_SHARE,
            "selected": None,                                 # no single winner
            "m": n - len(rejected),
            "f": None,
            "n": n,
            "weight_sum": res["weight_sum"],
            "normalise": self.normalise,
            "degenerate": res["degenerate"],
        }

        self._print_weights(cids, weights, res["max_cs"], share, res["weight_sum"])

        # 5. combine, in the paper's own form: weight the PSEUDO-GRADIENTS, sum them,
        # and add the result to the broadcast model.
        return self._combine(updates, weights, global_weights)

    @staticmethod
    def _combine(updates, weights, global_weights):
        """theta_{t+1} = theta_t + sum_i w_i * Delta_i — the paper's formulation.

        FoolsGold is defined over pseudo-gradients, so the combine is written that way:
        weight each client's Delta_i = w_i - w_global, sum, add to the broadcast model.
        With sum(w) == 1 this is algebraically identical to the weighted mean of absolute
        weights that FedAvg computes,

            sum_i w_i * theta_i = theta_t * sum_i w_i + sum_i w_i * Delta_i,

        so this is not a numerical change — it is the form the algorithm is stated in, it
        keeps theta_t intact independently of what the weights sum to, and it drops the
        previous reuse of FedAvg's `num_samples` slot to smuggle alpha through.

        Non-float entries (integer buffers) have no meaningful pseudo-gradient. They are
        combined by the same weighted average FedAvg has always applied to them, which is
        identical under sum(w) == 1 — preserved deliberately so switching the combine form
        does not quietly change anything beyond the intended change.
        """
        out = OrderedDict()
        for key, ref in global_weights.items():
            if not torch.is_floating_point(ref):
                acc = torch.zeros_like(ref, dtype=torch.float32)
                for u in updates:
                    acc += u["weights"][key] * weights[u["cid"]]
                out[key] = acc
                continue
            base = ref.detach().cpu().float()
            acc = torch.zeros_like(base)
            for u in updates:
                w = weights[u["cid"]]
                if w == 0.0:
                    continue                      # fully suppressed: contributes nothing
                acc += (u["weights"][key].detach().cpu().float() - base) * w
            out[key] = base + acc
        return out

    # ---- persistence (folded into the server's existing checkpoint) ----

    def state_dict(self):
        """Cumulative histories, for the round checkpoint.

        Written into the same checkpoint file as the model and round number, so a resume
        cannot desync the history from the model it was accumulated against.
        """
        return {"history": {int(c): v.clone() for c, v in self.history.items()},
                "feature_group": self.feature_group,
                "feature_dim": sum(self.model_state_dict[k].numel()
                                   for k in self.feature_names)}

    def load_state_dict(self, state):
        """Restore histories on resume; ignore silently when there is nothing stored.

        Refuses a history built over a DIFFERENT feature set: resuming with a changed
        --foolsgold-feature-group would otherwise concatenate incompatible coordinate
        systems into one running sum and produce meaningless cosines.
        """
        if not state:
            return
        want_dim = sum(self.model_state_dict[k].numel() for k in self.feature_names)
        if state.get("feature_group") != self.feature_group or \
                state.get("feature_dim") != want_dim:
            raise ValueError(
                f"FoolsGold: checkpoint history was built over feature group "
                f"{state.get('feature_group')!r} (dim {state.get('feature_dim')}), but this "
                f"run uses {self.feature_group!r} (dim {want_dim}). Resuming would sum "
                f"incompatible coordinate systems. Restore the original "
                f"--foolsgold-feature-group, or start a fresh run."
            )
        self.history = {int(c): v.detach().cpu().float()
                        for c, v in state.get("history", {}).items()}
        print(f"[FoolsGold] resumed cumulative histories for {len(self.history)} client(s)",
              flush=True)
