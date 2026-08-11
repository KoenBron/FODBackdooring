"""L2-divergence localization scoring at either granularity.

The whole module rests on one fact: squared-L2 and element-count are ADDITIVE
across tensors, but L2 norm and sqrt(count) are NOT. A bucket's update is the
concatenation of its member tensors' difference vectors, so

    ||Delta_bucket|| = sqrt( sum_t ||Delta_t||^2 )      # squares under the root
    sqrt_d_bucket    = sqrt( sum_t numel_t )            # counts  under the root

You may never sum the per-tensor norms (or the per-tensor sqrt(numel)) directly.

So the pipeline keeps two additive per-tensor atoms -- squared divergence and
numel -- sums each into buckets, and only THEN takes the roots. Granularity is
just the choice of bucketing:

    per-group  : name_to_bucket = tensor_to_group   (build_group_map's 2nd return)
    per-tensor : name_to_bucket = identity_partition(tensor_list)

Both flow through the same normalized_divergence, so the coarse and fine scores
are guaranteed consistent (the per-group result equals the per-tensor squared
atoms re-summed into groups) -- which is exactly what the cross-granularity
pilot relies on.
"""
import math

from FL.groups import per_tensor_sq_l2


# ---- additive per-tensor atoms (the only things that aggregate linearly) ----

def numel_per_tensor(model):
    """{name -> numel}. Additive: bucket dimension = sum of member numels."""
    return {n: p.numel() for n, p in model.named_parameters()}


def frozen_tensors(model):
    """Names of params with requires_grad=False.

    Their divergence is identically 0 (no gradient, no update), so they are
    excluded from the cheap sigma floor — otherwise their zeros would deflate
    the cross-tensor standard deviation and shrink the floor toward 0.
    """
    return {n for n, p in model.named_parameters() if not p.requires_grad}


def squared_divergence_per_tensor(state_a, state_b, tensor_list):
    """{name -> ||a - b||^2}. Additive: bucket squared-norm = sum of members.

    Thin alias over groups.per_tensor_sq_l2 so the 'atom' vocabulary lives here.
    """
    return per_tensor_sq_l2(state_a=state_a, state_b=state_b, tensor_list=tensor_list)


# ---- generic bucket aggregation ----

def bucket_sum(per_tensor, name_to_bucket):
    """Sum an additive per-tensor dict into buckets keyed by name_to_bucket[name]."""
    agg = {}
    for name, v in per_tensor.items():
        b = name_to_bucket[name]
        agg[b] = agg.get(b, 0.0) + v
    return agg


def identity_partition(tensor_list):
    """Per-tensor 'partition': each tensor is its own bucket."""
    return {n: n for n in tensor_list}


# ---- normalized divergence at ANY granularity ----

def normalized_l2_divergence(sq, numel, name_to_bucket):
    """Per-bucket normalized L2 divergence = sqrt(sum sq) / sqrt(sum numel).

    sq, numel       : additive per-tensor dicts (squared distances, element counts).
    name_to_bucket  : tensor_to_group (per-GROUP) | identity_partition (per-TENSOR).

    Sums the SQUARES under the root (true Euclidean norm of the concatenated
    bucket update) and the raw numels under the root for the dimension. Buckets
    are taken from sq, so empty groups simply do not appear.
    """
    sq_b = bucket_sum(sq, name_to_bucket)# Squared total divergence in a bucket
    dim_b = bucket_sum(numel, name_to_bucket)# Total parameters in a bucket
    # Total divergence is treated as a vector of per vector divergencesummed under the root and the it is normalized through dividing by the the root of the number of parameters.
    return {b: math.sqrt(sq_b[b]) / math.sqrt(dim_b[b]) for b in sq_b}


# ---- convenience wrappers so call sites read at their granularity ----

def divergence_norm_per_group(state_a, state_b, tensor_list, tensor_to_group, numel):
    """Per-group normalized divergence (this is Pilot A's div_g)."""
    sq = squared_divergence_per_tensor(state_a, state_b, tensor_list)
    return normalized_l2_divergence(sq, numel, tensor_to_group)


def divergence_norm_per_tensor(state_a, state_b, tensor_list, numel):
    """Per-tensor normalized divergence (Pilot B's fine view)."""
    sq = squared_divergence_per_tensor(state_a, state_b, tensor_list)
    return normalized_l2_divergence(sq, numel, identity_partition(tensor_list))


# ---- denoising + selection (granularity-agnostic: keys are bucket names) ----

def rho(div, baseline):
    """
    Compute the rho divergence using the normalized l2 divergence per group and determine the floor by subtracting the baseline divergence (or noise) and tune the sensitivity using the cutoff parameter.
    """
    return {g: max(0.0, div[g] - baseline[g]) for g in div}


def select_cutoff(div, baseline, cutoff):
    """
    Select carrying layers whose RAW divergence exceeds cutoff * baseline
    (div > cutoff * baseline).

    Takes div (not rho) so the rule is identical to PilotA's inline
    _select_cutoff(div_g, b_g, c) — same semantics at per-group and per-tensor
    granularity, so the cross-granularity cutoff comparison is not contaminated
    by a (1 + cutoff) vs cutoff mismatch.

    Strict '>' (not '>=') so frozen layers, which have div = baseline = 0 under
    the expensive floor, are NOT selected (0 > 0 is False).

    Returns a list of (name, div) tuples ORDERED by div descending (ties broken
    by name). The order is the amplification priority: with a limited budget the
    consumer amplifies from the front of the list.
    """
    selected = [(g, div[g]) for g in div if div[g] > baseline[g] * cutoff]
    return sorted(selected, key=lambda x: (-x[1], x[0]))


def select_topk(rho, k):
    """
    Select the top-k groups by rho, among those with positive (non-floored) rho.

    Returns fewer than k groups when fewer than k have rho > 0 — a group with
    zero denoised signal is never a carrier, so it is not padded into the set.

    Returns a list of (name, rho) tuples ORDERED by rho descending (ties broken
    by name) — the amplification priority order, not an alphabetical set. Callers
    that need a set extract {n for n, _ in result}.
    """
    candidates = [(g, s) for g, s in rho.items() if s > 0.0]
    ranked = sorted(candidates, key=lambda x: (-x[1], x[0]))
    return ranked[:k]


def select_topk_signed(score, k):
    """Top-k buckets by RAW (possibly negative) score, descending — NO score>0 gate.

    Unlike select_topk, this does not filter on sign, so it always returns
    min(k, len(score)) entries and keeps a total order even where the score is
    <= 0. Built for the cross-method ranking comparison against an ungated
    baseline (LaySelFL) using SIGNED rho (div - b, unclamped): the signed score
    matches the clamped/gated rho on its positive prefix (same carriers, same
    order) but, unlike clamped rho, does not collapse sub-floor buckets into a
    0.0 tie — so a full k can be drawn for a like-for-like overlap.

    The caller MUST pre-exclude any bucket it does not want ranked. In particular
    frozen tensors have div = b = 0 -> signed rho 0, which sits MID-ranking (above
    the negative tail), so they would otherwise be padded into the set; filter to
    trainable buckets before calling.

    Returns a list of (name, score) tuples ORDERED by score descending (ties by
    name).
    """
    ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:k]


def select_by_mode(div, b, mode, cutoff, topk):
    """Route a (div, b) score pair to the configured carrying-layer selector.

    SINGLE source of the cutoff-vs-topk decision, so the per-round path and the
    plastic-phase freeze select IDENTICALLY -- they differ only in whether div/b are
    one round's raw scores or the pooled plastic means:

        'cutoff' -> select_cutoff(div, b, cutoff)         # variable size, div > cutoff*b
        'topk'   -> select_topk_signed(div - b, topk)     # fixed k by SIGNED denoised rho

    Both subtract the benign-to-benign floor b (denoising preserved). Returns ranked
    [(name, score)] descending. div/b keys MUST be pre-restricted to eligible tensors
    (frozen/BN excluded) -- select_topk_signed relies on that (its 0-signed frozen
    entries would otherwise pad the top-k).
    """
    if mode == "cutoff":
        return select_cutoff(div, b, cutoff)
    if mode == "topk":
        signed = {n: div[n] - b[n] for n in div}
        return select_topk_signed(signed, topk)
    raise ValueError(f"Unknown select mode {mode!r}; expected 'cutoff' or 'topk'.")
