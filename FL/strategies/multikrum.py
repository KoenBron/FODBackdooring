import torch

from collections import OrderedDict

from FL.strategy import BaseStrategy
from FL.strategies.fedavg import FedAvg
from FL.strategies._robust import krum_select, krum_min_clients


class MultiKrum(BaseStrategy):
    """Multi-Krum robust aggregation (Blanchard et al. 2017), server-side.

    Purely from the aggregation server's point of view — it sees only the submitted
    updates (weights + num_samples), NEVER any attacker/benign label, so it preserves
    the adversarial model. Each round it:
      1. flattens every submitted update into one vector,
      2. computes pairwise squared L2 distances,
      3. scores each client (sum of the n-f-2 smallest distances) and keeps the m
         lowest-scoring — the Krum score/selection is delegated to the shared
         FL.strategies._robust.krum_select (single source of truth with the stealth
         harness), and
      4. combines the kept clients with plain FedAvg (sample-weighted mean).

    Distance note: every client trains from the same broadcast global, so the absolute
    weights give the same pairwise distances as the deltas would — no need to subtract
    the global here.

    The kept/rejected decision is stashed on `last_selection` so the server can log
    which clients the defense filtered (observational; not fed back into aggregation).
    """

    def __init__(self, model_state_dict: OrderedDict, f: int, m: int) -> None:
        self.model_state_dict = model_state_dict
        self.f = f
        self.m = m
        # Reuse FedAvg for the final combine of the kept set (DRY: one averaging path).
        self._combiner = FedAvg(model_state_dict=model_state_dict)
        # (n, f) pairs already warned about, so an out-of-regime config is reported
        # once per distinct round shape instead of every round. Run-level, NOT round
        # state — deliberately not cleared by reset_state().
        self._resilience_warned = set()
        self.reset_state()

    def reset_state(self):
        self.last_selection = None
        self._combiner.reset_state()

    def _warn_if_not_resilient(self, n):
        """Warn ONCE per (n, f) when the round is outside Krum's resilience regime.

        Blanchard et al. prove Byzantine resilience only for n >= 2f + 3. Below it the
        selection still returns a set — silently, and that silence is the danger: the
        defense looks like it is working while its guarantee is void. n varies with
        partial participation, so dedup on (n, f) rather than warning once per run.
        """
        need = krum_min_clients(self.f)
        if n >= need or (n, self.f) in self._resilience_warned:
            return
        self._resilience_warned.add((n, self.f))
        print(
            f"[WARNING] Multi-Krum is outside its resilience regime this round: "
            f"n={n} participating client(s) but Krum (Blanchard et al. 2017) requires "
            f"n >= 2f+3 = {need} for f={self.f}. Selection still runs, but its "
            f"Byzantine-resilience guarantee does NOT hold — treat the kept/rejected "
            f"sets as uncalibrated. Raise --clients-per-round or lower --krum-f.",
            flush=True,
        )

    @staticmethod
    def _flatten(weights):
        """One vector per update: every FLOAT tensor, concatenated in a fixed (sorted)
        key order so all clients live in the same coordinate system. Integer buffers
        (e.g. num_batches_tracked) carry no update signal and are skipped."""
        parts = [weights[k].detach().cpu().float().flatten()
                 for k in sorted(weights) if torch.is_floating_point(weights[k])]
        return torch.cat(parts) if parts else torch.zeros(1)

    def aggregate(self, client_updates: list, global_weights=None) -> OrderedDict:
        # Clear FIRST, before anything can raise. Otherwise a failed aggregation would
        # leave last_selection holding the PREVIOUS round's decision, and the server's
        # `if selection is not None` log would report it as if it were this round's.
        # None means "no decision was made", which is exactly right on a failure.
        self.last_selection = None

        if not client_updates:
            raise Exception("[ERROR] No client updates given to aggregate")

        # Sort by cid so the FedAvg summation below runs in a fixed order — float
        # addition is not associative, so an unsorted kept_updates would make the
        # aggregated weights depend on client arrival order. (krum_select sorts its
        # own cids, so tie-breaking no longer relies on this.)
        updates = sorted(client_updates, key=lambda u: u["cid"])
        cids = [u["cid"] for u in updates]
        n = len(cids)
        self._warn_if_not_resilient(n)

        # Degenerate case: with a single update there is nothing to filter.
        if n == 1:
            self.last_selection = {"kept": list(cids), "rejected": [], "scores": {cids[0]: 0.0},
                                   "selected": cids[0], "m": 1, "f": self.f, "n": 1}
            return self._combiner.aggregate(updates, global_weights)

        vecs = {u["cid"]: self._flatten(u["weights"]) for u in updates}
        sqdist = {i: {} for i in cids}
        for a in range(n):
            i = cids[a]
            for b in range(a + 1, n):
                j = cids[b]
                d = float(torch.sum((vecs[i] - vecs[j]) ** 2))
                sqdist[i][j] = sqdist[j][i] = d

        res = krum_select(sqdist, cids, self.f, self.m)
        kept = res["kept"]

        self.last_selection = {
            "kept": sorted(kept),
            "rejected": sorted(set(cids) - kept),
            "scores": res["scores"],
            "selected": res["selected"],
            "m": res["m"],
            "f": self.f,
            "n": n,
        }

        # Combine ONLY the kept updates, via the shared FedAvg averaging.
        kept_updates = [u for u in updates if u["cid"] in kept]
        return self._combiner.aggregate(kept_updates, global_weights)
