"""Torch-free robust-aggregation core.

SINGLE SOURCE OF TRUTH for the numeric heart of every robust aggregator:

  - Krum / Multi-Krum scoring (krum_select), shared by the Multi-Krum strategy
    (FL.strategies.multikrum) and the observational stealth harness
    (FL.stealth_defenses.krum), so the two can never drift. Only the *distance*
    input differs between them (the harness scores eligible-param deltas; the
    aggregator scores whole-update vectors); the scoring itself — score = sum of
    the (n-f-2) smallest squared distances, argmin for Krum, m lowest for
    Multi-Krum — lives here and nowhere else.

  - FoolsGold re-weighting (foolsgold_weights), used by FL.strategies.foolsgold.

  - Norm-clip scaling (normclip_scales), used by FL.strategies.normclip.

Torch-free on purpose: both are pure arithmetic over a similarity/distance map, so
they unit-test without torch/GPU (the dev sandbox and CI). The vector work that
needs torch — distances, cosines, update histories — stays in each caller.
"""
import math
import statistics


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


def _degenerate(cids, alpha, max_cs, normalise):
    """Every client equally suspicious, so the rescale alpha/max(alpha) is undefined.

    The right fallback depends on the mode, and in both cases it is the SAFE one:

      "none" — return all zeros. The combine is theta_t + sum(w_i * Delta_i), so a
          zero weight vector simply leaves the global model untouched for that round.
          "The defense had no confidence, so do not move" is both conservative and
          the honest reading; the paper does not define this case at all.
      "sum"  — return uniform 1/n. All zeros is NOT available here: the caller would
          divide by sum(w) == 0, and a convex combination of zero weights collapses
          the model rather than leaving it alone.
    """
    n = len(cids)
    weights = {i: 0.0 for i in cids} if normalise == "none" else {i: 1.0 / n for i in cids}
    return {"weights": weights, "alpha": alpha, "max_cs": max_cs,
            "weight_sum": sum(weights.values()), "degenerate": True}


def foolsgold_weights(cosine, cids, kappa=1.0, normalise="none"):
    """FoolsGold aggregation weights (Fung et al., RAID 2020).


    cosine: symmetric nested map, cosine[i][j] = cos(H_i, H_j) for i != j, where H_i
            is client i's cumulative update. Diagonal entries are never read.
    cids:   client ids to weight. Sorted internally, so the output cannot depend on
            the order the caller happened to iterate its updates in.
    kappa:  confidence parameter scaling the logit before the [0, 1] clip. NOT a
            severity dial: the logit crosses zero at alpha ~ 0.3775, and any client
            below that is clipped to 0 for EVERY kappa > 0, so kappa never changes
            WHICH clients are excluded. What it changes is how decisively the
            survivors are admitted — larger kappa pushes borderline clients up toward
            full weight, i.e. a sharper, more binary split (and slightly more
            permissive at the margin). Paper default 1.0.

    normalise:
        "none" (default, PAPER-FAITHFUL) — return the post-logit weights as they are,
            each in [0, 1]. The paper treats w_i as a per-client TRUST score, not a
            share: several clients may sit at 1.0 simultaneously, and the aggregate
            sum(w_i * Delta_i) therefore grows with how many clients are trusted.
            Safe here only because the combiner is delta-form (theta_t + sum w_i D_i),
            which leaves theta_t intact whatever the weights sum to.
        "sum" — divide by sum(w), giving a convex combination and a step size that does
            not move with the trust count. Defensible but NOT the paper's rule; kept as
            the recovery path if the paper-faithful sum destabilises training.

    Returns {"weights": {cid: w}, "alpha": {cid: rescaled confidence},
             "max_cs": {cid: v_i}, "weight_sum": float, "degenerate": bool}.

    `weight_sum` is the effective step multiplier under "none" (1.0 == one average
    client's delta), so it is worth logging: it is the quantity that grows with the
    number of trusted participants.
    """
    cids = sorted(cids)
    n = len(cids)
    if n == 0:
        return {"weights": {}, "alpha": {}, "max_cs": {}, "weight_sum": 0.0,
                "degenerate": False}
    if n == 1:
        c = cids[0]
        return {"weights": {c: 1.0}, "alpha": {c: 1.0}, "max_cs": {c: 0.0},
                "weight_sum": 1.0, "degenerate": False}

    # v_i = max_j cs_ij, with the implicit 0 from the zeroed diagonal.
    v = {i: max([0.0] + [cosine[i][j] for j in cids if j != i]) for i in cids}

    # Pardoning: an honest client that happens to resemble a sybil should not be
    # punished for the sybil's own high similarity. Scale i's view of j down by the
    # ratio of their peak similarities whenever j is the more suspicious of the two.
    # v[j] > v[i] >= 0 implies v[j] > 0, so the division is always safe.
    alpha = {}
    for i in cids:
        adjusted = [0.0]
        for j in cids:
            if j == i:
                continue
            c = cosine[i][j]
            if v[j] > v[i]:
                c *= v[i] / v[j]
            adjusted.append(c)
        alpha[i] = 1.0 - max(adjusted)

    alpha = {i: min(1.0, max(0.0, a)) for i, a in alpha.items()}

    # Rescale so the least-suspicious client sits at 1. If every client is maximally
    # similar to some peer, every alpha is 0 and this would divide by zero — which
    # would hand back all-zero weights and, because the combiner is a convex
    # combination over ABSOLUTE weights, silently replace the global model with
    # zeros. Fall back to uniform and let the caller warn.
    a_max = max(alpha.values())
    if a_max <= 0.0:
        return _degenerate(cids, alpha, v, normalise)
    alpha = {i: a / a_max for i, a in alpha.items()}

    # Logit, which sharpens the split between "clearly honest" and "clearly colluding".
    # alpha == 1 would be +inf, so the reference clamps it to 0.99 first; alpha == 0 is
    # -inf, which the [0, 1] clip below sends to 0 (fully excluded).
    out = {}
    for i in cids:
        a = alpha[i]
        if a >= 1.0:
            a = 0.99
        if a <= 0.0:
            out[i] = 0.0
            continue
        w = kappa * (math.log(a / (1.0 - a)) + 0.5)
        out[i] = min(1.0, max(0.0, w))

    total = sum(out.values())
    if total <= 0.0:
        return _degenerate(cids, alpha, v, normalise)
    if normalise == "sum":
        out = {i: w / total for i, w in out.items()}
        total = 1.0
    return {"weights": out, "alpha": alpha, "max_cs": v,
            "weight_sum": total, "degenerate": False}


def normclip_scales(norms, cids, c):
    """Norm-bound clipping (FLAME-style median rule), server-side.

    Bounds every update's L2 norm at M = c * median(||Delta_j||) by RESCALING rather than
    rejecting: an update over the bound is projected onto the sphere of radius M, one
    under it passes through untouched.

        M   = c * median_j ||Delta_j||
        s_i = min(1, M / ||Delta_i||)

    Rescaling is a pure positive scalar, so it changes MAGNITUDE ONLY — the update's
    direction is preserved exactly. That is the point of including this defense: it is
    the one that acts purely on the norm axis, and it leaves cosine/Krum-style direction
    separation untouched (asserted in smoketest_normclip_agg).

    norms: {cid: ||Delta_i||}, whole-update L2. The caller supplies these, so the
           coordinate set is the caller's choice — FL.strategies.normclip uses
           FL.normclip_report.update_norm (every float tensor), the same quantity
           normclip_report.csv reports, so the active defense and the passive report
           agree on what is being measured.
    cids:  clients to score. Sorted internally so the output cannot depend on the order
           the caller iterated its updates in.
    c:     the multiplier. 1.0 clips at the median itself (FLAME's rule).

    statistics.median (not torch.median) on purpose: torch returns the LOWER of the two
    middle values on even input, while statistics averages them. The observational
    normclip_report already uses statistics.median, so this keeps the active defense and
    that report identical for even participant counts too.

    Returns {"scales": {cid: s}, "threshold": M, "median": med, "clipped": [cids]}.
    A zero-norm update, or a round whose median is 0, yields s = 1 (nothing to clip)
    rather than a division by zero.
    """
    cids = sorted(cids)
    if not cids:
        return {"scales": {}, "threshold": 0.0, "median": 0.0, "clipped": []}

    med = float(statistics.median([norms[i] for i in cids]))
    threshold = c * med

    scales = {}
    for i in cids:
        n_i = norms[i]
        if threshold <= 0.0 or n_i <= 0.0 or n_i <= threshold:
            scales[i] = 1.0
        else:
            scales[i] = threshold / n_i
    return {"scales": scales, "threshold": threshold, "median": med,
            "clipped": [i for i in cids if scales[i] < 1.0]}
