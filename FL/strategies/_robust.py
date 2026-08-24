"""Torch-free robust-aggregation core.

SINGLE SOURCE OF TRUTH for the Krum score/selection, shared by:
  - the Multi-Krum aggregation strategy (FL.strategies.multikrum), and
  - the observational stealth harness (FL.stealth_defenses.krum),
so the two can never drift. Only the *distance* input differs between them (the
harness scores eligible-param deltas; the aggregator scores whole-update vectors);
the scoring itself — score = sum of the (n-f-2) smallest squared distances, argmin
for Krum, m lowest for Multi-Krum — lives here and nowhere else.

Torch-free on purpose: the scoring is pure arithmetic over a squared-distance map,
so it unit-tests without torch/GPU (the dev sandbox and CI). Distance computation,
which needs torch, stays in each caller.
"""


def krum_min_clients(f):
    """Smallest n for which Krum's Byzantine-resilience theorem holds: n >= 2f + 3.

    Blanchard et al. prove (alpha, f)-Byzantine resilience only under this condition.
    Below it the scoring still computes a number, but the guarantee is void — the
    (n-f-2) neighbourhood is no longer large enough to be majority-honest. Exposed as
    a predicate so callers can WARN without re-deriving the paper's condition.
    """
    return 2 * f + 3


def krum_select(sqdist, cids, f, m=None):
    """Krum / Multi-Krum selection (Blanchard et al. 2017).

    sqdist: symmetric nested map; sqdist[i][j] = squared L2 distance between clients
            i and j (j != i). Only off-diagonal entries are read.
    cids:   iterable of client ids (keys of sqdist). Sorted internally, so equal-score
            ties always break by ascending cid regardless of the order passed in —
            callers cannot accidentally make the selection non-deterministic.
    f:      assumed Byzantine client count. score_i = sum of the (n-f-2) smallest
            squared distances from i to the other clients; lower = more 'central'.
    m:      how many clients Multi-Krum keeps (the m lowest-scoring). None or <= 0
            auto-sets to n - f (at least 1) — the common default. Clamped to n.

    Returns a dict:
      scores:    {cid: krum_score}
      ranked:    cids sorted by score ascending (most central first)
      selected:  the single Krum winner (argmin score), or None if cids is empty
      kept:      set of the m lowest-scoring cids (the Multi-Krum selection)
      m:         the resolved m actually used — ALWAYS == len(kept)
      resilient: does n >= 2f + 3 hold? False = the selection ran but Krum's
                 Byzantine-resilience guarantee does not apply (callers should warn).
    """
    # Sort here rather than trusting the caller: `ranked` below is a STABLE sort by
    # score, so the incoming order is what breaks equal-score ties. Leaving that to
    # the caller made tie-breaking silently order-dependent for anyone who forgot.
    cids = sorted(cids)
    n = len(cids)
    # (n-f-2) closest neighbours; guard tiny n so k stays >= 1.
    k = max(1, n - f - 2)
    scores = {}
    for i in cids:
        dists = sorted(sqdist[i][j] for j in cids if j != i)
        scores[i] = float(sum(dists[:k]))
    # Stable sort over the sorted cids => equal scores break by ascending cid.
    ranked = sorted(cids, key=lambda c: scores[c])
    selected = ranked[0] if ranked else None
    m_req = m if (m is not None and m > 0) else max(1, n - f)
    # Clamp to n: requesting more than participated keeps everyone. Reporting the
    # UNclamped request would make the server's round log self-contradictory
    # ("m=99 ... n=4" beside four kept clients), so `m` is the count actually used.
    m_res = min(m_req, n)
    kept = set(ranked[:m_res])
    return {"scores": scores, "ranked": ranked, "selected": selected,
            "kept": kept, "m": m_res, "resilient": n >= krum_min_clients(f)}
