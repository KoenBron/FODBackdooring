import torch

from collections import OrderedDict

from FL.normclip_report import update_norm
from FL.strategy import BaseStrategy
from FL.strategies._robust import normclip_scales


class NormClip(BaseStrategy):
    """Norm-bound clipping aggregation (FLAME-style median rule), server-side.

    Like every strategy here it sees only the submitted updates (weights + num_samples),
    NEVER an attacker/benign label. Each round it:
      1. forms each client's update Delta_i = w_i - w_global and measures ||Delta_i||,
      2. sets the bound M = c * median_j ||Delta_j||,
      3. RESCALES every over-bound update onto the sphere of radius M, and
      4. adds the sample-weighted sum of the clipped pseudo-gradients to the broadcast
         model, optionally with Gaussian noise.

    It neither selects (Multi-Krum) nor re-weights by trust (FoolsGold): it bounds
    magnitude and nothing else.

    Direction is untouched
    ----------------------
    s_i is a positive scalar, so clipping preserves each update's direction EXACTLY.
    cos_to_median is unchanged by this defense; only update_l2 moves. That is precisely
    why it belongs in the set — it isolates the magnitude axis of the stealth trade-off
    and says nothing about the angular separation Krum and cosine clustering consume.

    One source of truth for the norm
    --------------------------------
    ||Delta_i|| comes from FL.normclip_report.update_norm — every float tensor, the whole
    update a real norm-clipping server would see. That is the SAME function behind
    normclip_report.csv, so the active defense and the passive per-attacker report measure
    the same quantity and their thresholds are directly comparable.

    Note this deliberately differs from the stealth harness's `normclip` block, which
    scores ELIGIBLE-only deltas (trainable, non-BN) and uses a different multiplier
    (--normclip-c, the detector). Both are kept as-is so previously collected runs stay
    comparable; --normclip-agg-c is the aggregation's own policy knob.
    """

    def __init__(self, model_state_dict: OrderedDict, c: float,
                 noise_sigma: float = 0.0, seed: int = 0) -> None:
        self.model_state_dict = model_state_dict
        self.c = c
        self.noise_sigma = noise_sigma
        self.seed = seed
        # Rounds this strategy has aggregated. Only used to seed the noise generator, and
        # persisted so a resume continues the noise stream instead of replaying it.
        self._rounds_seen = 0
        self.reset_state()

    def reset_state(self):
        self.last_selection = None

    def _noise_generator(self):
        """A generator seeded from (run seed, rounds aggregated).

        Deterministic so a run is reproducible and a resume does not silently re-use the
        same draw; per-round so successive rounds get independent noise.
        """
        gen = torch.Generator()
        gen.manual_seed(int(self.seed) * 1000003 + int(self._rounds_seen))
        return gen

    def aggregate(self, client_updates: list, global_weights=None) -> OrderedDict:
        # Clear FIRST so a failed aggregation cannot leave the previous round's decision
        # for the server to log as this round's.
        self.last_selection = None

        if not client_updates:
            raise Exception("[ERROR] No client updates given to aggregate")
        if global_weights is None:
            raise ValueError(
                "NormClip needs the broadcast global to form Delta_i = w_i - w_global; "
                "clients submit absolute weights."
            )

        # Fixed cid order: float addition is not associative, so an unsorted combine
        # would make the aggregate depend on client arrival order.
        updates = sorted(client_updates, key=lambda u: u["cid"])
        cids = [u["cid"] for u in updates]

        norms = {u["cid"]: update_norm(u["weights"], global_weights) for u in updates}
        res = normclip_scales(norms, cids, self.c)
        scales = res["scales"]

        total_samples = sum(u["num_samples"] for u in updates)
        if total_samples <= 0:
            raise ValueError("NormClip: total num_samples is zero; cannot weight the average")

        self.last_selection = {
            # A norm-clipping server rejects nobody, so "rejected" is defined as CLIPPED —
            # the clients the defense actually acted on. Note a client at scale 0.99 is
            # barely touched yet still counts as clipped, so the per-client `scales` below
            # are the real story, not the kept/rejected split. At c = 1.0 roughly half the
            # round sits above the median by construction, so a large clipped count is the
            # expected state, not a signal.
            "kept": sorted(c for c in cids if scales[c] >= 1.0),
            "rejected": sorted(res["clipped"]),
            "scores": {c: norms[c] for c in cids},        # ||Delta_i||
            "scales": scales,
            "threshold": res["threshold"],
            "median_norm": res["median"],
            "noise_sigma": self.noise_sigma,
            "selected": None,
            "m": len(cids) - len(res["clipped"]),
            "f": None,
            "n": len(cids),
        }
        self._print_round(cids, norms, scales, res)

        out = self._combine(updates, scales, total_samples, global_weights)
        if self.noise_sigma > 0.0:
            out = self._add_noise(out, global_weights, res["threshold"], len(cids))
        self._rounds_seen += 1
        return out

    @staticmethod
    def _combine(updates, scales, total_samples, global_weights):
        """theta_{t+1} = theta_t + sum_i (n_i/N) * s_i * Delta_i.

        Delta form, matching FoolsGold: the pseudo-gradients are what get clipped, so the
        combine is written over them and theta_t is preserved by construction rather than
        by the weights happening to sum to 1. Sample weighting is RETAINED here (unlike
        FoolsGold, whose alpha replaces it), so this stays plain FedAvg with a per-client
        scalar applied to each delta — and it needs no equal-partition guard.

        Non-float entries have no meaningful pseudo-gradient and keep the plain
        sample-weighted average the other strategies give them.
        """
        out = OrderedDict()
        for key, ref in global_weights.items():
            if not torch.is_floating_point(ref):
                acc = torch.zeros_like(ref, dtype=torch.float32)
                for u in updates:
                    acc += u["weights"][key] * (u["num_samples"] / total_samples)
                out[key] = acc
                continue
            base = ref.detach().cpu().float()
            acc = torch.zeros_like(base)
            for u in updates:
                w = (u["num_samples"] / total_samples) * scales[u["cid"]]
                acc += (u["weights"][key].detach().cpu().float() - base) * w
            out[key] = base + acc
        return out

    def _add_noise(self, aggregate, global_weights, threshold, n_clients):
        """Add N(0, (sigma * M / n)^2) to the aggregated FLOAT tensors.

        Scaled by the clip bound M rather than given in absolute units, following the
        DP-FedAvg convention: the aggregate is an average over n clients whose individual
        contributions are bounded by M, so M/n is the natural noise unit and sigma is
        "noise as a fraction of one client's worth of clipped signal". That makes sigma
        interpretable without knowing the model's parameter magnitudes — O(1e-3) is a
        sensible starting scale, and sigma = 0 is an exact no-op.

        Which tensors are "float" is decided from GLOBAL_WEIGHTS, not from the aggregate.
        _combine upcasts integer buffers to float32 (inherited FedAvg behaviour), so the
        aggregate's own dtypes would report those buffers as floating point and they would
        be noised — a counter perturbed by Gaussian noise, silently.
        """
        std = self.noise_sigma * threshold / max(1, n_clients)
        if std <= 0.0:
            return aggregate
        gen = self._noise_generator()
        for key, val in aggregate.items():
            if not torch.is_floating_point(global_weights[key]):
                continue
            aggregate[key] = val + torch.normal(
                0.0, std, size=val.shape, generator=gen, dtype=val.dtype)
        return aggregate

    def _print_round(self, cids, norms, scales, res):
        clipped = res["clipped"]
        body = "  ".join(f"{c}:|d|={norms[c]:.4g}" + ("" if scales[c] >= 1.0
                                                      else f",s={scales[c]:.3f}")
                         for c in cids)
        print(f"  [normclip] median={res['median']:.4g} M={res['threshold']:.4g} "
              f"(c={self.c:g})  clipped {len(clipped)}/{len(cids)} {clipped}"
              + (f"  noise_sigma={self.noise_sigma:g}" if self.noise_sigma > 0 else ""),
              flush=True)
        print(f"  [normclip] {body}", flush=True)

    # ---- persistence (folded into the server's existing checkpoint) ----

    def state_dict(self):
        """Only the round counter, which seeds the noise stream. Nothing else here is
        stateful — the clip bound is recomputed from each round's own submissions."""
        return {"rounds_seen": self._rounds_seen, "noise_sigma": self.noise_sigma,
                "seed": self.seed}

    def load_state_dict(self, state):
        """Restore the noise round counter on resume, so the stream continues rather than
        replaying draws already used. Silently ignores an absent/foreign state."""
        if not state or "rounds_seen" not in state:
            return
        self._rounds_seen = int(state["rounds_seen"])
        if self.noise_sigma > 0.0:
            print(f"[NormClip] resumed noise stream at round {self._rounds_seen}", flush=True)
