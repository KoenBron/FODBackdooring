"""
Passive backdoor-detectability harness (B1).

OBSERVATIONAL ONLY. Every function here reads client updates and emits detection
scores/decisions for logging; nothing modifies an update or the aggregation.
Training is byte-identical whether the harness runs or not.

Two clearly separated layers:
  * Label-BLIND decisions  -> krum/multikrum, norm-clip, cosine-clustering,
    distance-to-median. These see only the updates, exactly like a real
    defender would.
  * Label-AWARE scoring    -> AUC / Cohen's d separability and per-defense
    TPR/FPR. These use the known attacker ids only to *grade* the blind
    decisions. Keep the layers apart: the blind decisions must never read
    attacker_ids, or the detectability numbers become circular.

Modularity:
  * Each metric is a toggleable block in the registry (see MetricBlock / _BLOCKS):
    a new defense is write two functions + append one MetricBlock. Columns are a
    function of the enabled set; the logger writes both CSV headers lazily from
    the first row's keys, so there is no static schema constant to keep in sync.
  * Bucketing (per-group vs per-tensor) is NOT defined here. FL.groups is the
    single source of truth: the server injects name_to_bucket built via
    FL.groups.build_bucket_map, and column/iteration order comes from
    FL.groups.ordered_buckets. This module never re-derives a partition.
"""

from __future__ import annotations
import math
import torch

from FL.groups import ordered_buckets

# ----------------------------------------------------------------------------
# CSV schema. Identity columns are always present; every other column belongs
# to a toggleable metric block (see the registry lower down). The full column
# order is therefore a FUNCTION of the enabled set (and, for the energy block,
# the granularity). The logger writes both headers lazily from the first row's
# keys, so it never needs the enabled set — but per_client_columns() /
# summary_columns() are provided for introspection and tests. Both are defined
# after the registry.
#
# name_to_bucket is injected by the caller (built via FL.groups.build_bucket_map,
# over named_parameters() only). Full-state-dict floats that are not in the map
# — BN running stats etc. — are routed to the "other" bucket via
# name_to_bucket.get(name, "other"); "other" is always the last bucket (see
# FL.groups.ordered_buckets).
# ----------------------------------------------------------------------------

_PER_CLIENT_BASE_COLUMNS = ["round", "cid", "type", "is_attacker", "num_samples"]
_SUMMARY_BASE_COLUMNS = ["round", "n_clients", "n_attackers"]


# ----------------------------------------------------------------------------
# Update extraction
# ----------------------------------------------------------------------------

def compute_delta(client_weights, global_weights, name_to_bucket, eligible_names):
    """
    Delta = theta_client - theta_broadcast over ELIGIBLE parameters only.
    Returns (flat_delta: 1-D float32 CPU tensor, group_l2: {bucket: L2 norm}).

    eligible_names (from FL.groups.build_eligibility) is the set of trainable,
    non-batch-norm named parameters. Everything else is excluded from BOTH the
    flat delta and the per-bucket energy:
      - frozen params (requires_grad=False) contribute ~0 update anyway;
      - BN scale/shift and BN running-stat buffers are momentum statistics, not
        gradient updates, and their drift would otherwise inflate the norm /
        cosine / distance signals (see FL/groups.py state_dict caveat);
      - integer buffers (num_batches_tracked) carry no learnable signal.
    Keeping one eligible delta means all five metrics (l2, krum, median, cosine,
    energy) read the same coordinate set.

    name_to_bucket (from FL.groups.build_bucket_map) fixes the energy-decomposition
    granularity. The flat delta is granularity-INDEPENDENT — every pairwise defense
    downstream uses it as-is. Buckets with no eligible params stay at 0.0.
    """
    parts = []
    group_sq = {b: 0.0 for b in ordered_buckets(name_to_bucket)}
    for k, w in client_weights.items():
        if k not in eligible_names:
            continue
        d = (w.detach().cpu().float() - global_weights[k].detach().cpu().float()).flatten()
        parts.append(d)
        group_sq[name_to_bucket.get(k, "other")] += float(torch.dot(d, d))
    flat = torch.cat(parts) if parts else torch.zeros(1)
    group_l2 = {g: math.sqrt(max(0.0, s)) for g, s in group_sq.items()}
    return flat, group_l2


class PairwiseStats:
    """Per-client norms plus pairwise cosine and squared-distance over deltas."""

    def __init__(self, deltas):  # deltas: {cid: flat tensor}
        self.cids = sorted(deltas.keys())
        self.norm = {c: float(torch.norm(deltas[c])) for c in self.cids}
        self.cos = {c: {} for c in self.cids}
        self.sqdist = {c: {} for c in self.cids}
        for a in range(len(self.cids)):
            i = self.cids[a]
            for b in range(a + 1, len(self.cids)):
                j = self.cids[b]
                dot = float(torch.dot(deltas[i], deltas[j]))
                ni, nj = self.norm[i], self.norm[j]
                cos = dot / (ni * nj) if ni > 0 and nj > 0 else 0.0
                sq = max(0.0, ni * ni + nj * nj - 2.0 * dot)
                self.cos[i][j] = self.cos[j][i] = cos
                self.sqdist[i][j] = self.sqdist[j][i] = sq

    def mean_cos_to(self, cid, others):
        vals = [self.cos[cid][o] for o in others if o != cid and o in self.cos[cid]]
        return sum(vals) / len(vals) if vals else float("nan")


# ----------------------------------------------------------------------------
# Label-blind defenses
# ----------------------------------------------------------------------------

def krum(stats, f, m):
    """
    Krum / Multi-Krum (Blanchard 2017). score_i = sum of the (n-f-2) smallest
    squared distances from i to the other clients. Krum selects argmin; Multi-
    Krum keeps the m lowest-scoring clients. Lower score = more 'central'.
    """
    cids = stats.cids
    n = len(cids)
    k = max(1, n - f - 2)
    score = {}
    for i in cids:
        dists = sorted(stats.sqdist[i][j] for j in cids if j != i)
        score[i] = float(sum(dists[:k]))
    ranked = sorted(cids, key=lambda c: score[c])
    selected = ranked[0]
    m = m if (m is not None and m > 0) else max(1, n - f)
    keep = set(ranked[:m])
    return {c: {"krum_score": score[c],
                "krum_selected": (c == selected),
                "multikrum_selected": (c in keep)} for c in cids}


def norm_clip(norms, cids, c):
    """Norm-bound detector (logged only, never applied). Flag ||Delta|| > c*median.

    Takes the per-client norm dict directly (not PairwiseStats) so it needs no
    O(n^2) pairwise pass — the norms are the only thing it reads.
    """
    vals = [norms[i] for i in cids]
    med = float(torch.median(torch.tensor(vals))) if vals else 0.0
    thr = c * med
    return {i: {"normclip_flagged": norms[i] > thr,
                "normclip_threshold": thr} for i in cids}


def cosine_cluster(stats):
    """
    Dependency-free clustering. Split clients by mean pairwise cosine similarity
    (co-similarity) at the largest gap. Backdoor clients pursue one objective ->
    unusually high mutual cosine -> the tight MINORITY group is flagged as
    outliers. If the high-similarity group is the majority, nothing is flagged
    (no clear outlier, avoids false positives when there is no structure).
    """
    cids = stats.cids
    n = len(cids)
    if n < 3:
        return {i: {"cluster_label": 0, "cluster_outlier": False} for i in cids}
    sim = {i: stats.mean_cos_to(i, cids) for i in cids}
    order = sorted(cids, key=lambda c: sim[c])
    gaps = [(sim[order[t + 1]] - sim[order[t]], t) for t in range(n - 1)]
    _, split = max(gaps)
    high = set(order[split + 1:])           # higher co-similarity = suspect side
    is_minority = len(high) <= n // 2
    return {c: {"cluster_label": (1 if c in high else 0),
                "cluster_outlier": (c in high) and is_minority} for c in cids}


# The coordinate-wise median update vector (the robust reference shared by the
# 'median' and 'cosine' blocks) is computed once in analyze() and carried on the
# StealthContext, so the two blocks never recompute it and always agree on it.


# ----------------------------------------------------------------------------
# Label-aware scoring (uses ground-truth attacker ids ONLY to grade decisions)
# ----------------------------------------------------------------------------

def _auc(values, labels):
    """AUC = P(value_attacker > value_benign), Mann-Whitney with tie correction."""
    pos = [v for v, l in zip(values, labels) if l and not _isnan(v)]
    neg = [v for v, l in zip(values, labels) if not l and not _isnan(v)]
    if not pos or not neg:
        return float("nan")
    paired = [(v, l) for v, l in zip(values, labels) if not _isnan(v)]
    vals = [v for v, _ in paired]
    labs = [l for _, l in paired]
    order = sorted(range(len(vals)), key=lambda t: vals[t])
    ranks = [0.0] * len(vals)
    t = 0
    while t < len(order):
        u = t
        while u + 1 < len(order) and vals[order[u + 1]] == vals[order[t]]:
            u += 1
        avg = (t + u) / 2.0 + 1.0  # average 1-based rank for ties
        for r in range(t, u + 1):
            ranks[order[r]] = avg
        t = u + 1
    rank_pos = sum(ranks[i] for i in range(len(vals)) if labs[i])
    n_pos, n_neg = len(pos), len(neg)
    return (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _cohens_d(values, labels):
    pos = [v for v, l in zip(values, labels) if l and not _isnan(v)]
    neg = [v for v, l in zip(values, labels) if not l and not _isnan(v)]
    if not pos or not neg:
        return float("nan")
    mp, mn = sum(pos) / len(pos), sum(neg) / len(neg)
    vp = sum((v - mp) ** 2 for v in pos) / max(1, len(pos) - 1)
    vn = sum((v - mn) ** 2 for v in neg) / max(1, len(neg) - 1)
    pooled = math.sqrt(((len(pos) - 1) * vp + (len(neg) - 1) * vn) /
                       max(1, len(pos) + len(neg) - 2))
    return (mp - mn) / pooled if pooled > 0 else float("nan")


def _rate(flags, labels, want_label):
    """Fraction of clients with the given true label that are flagged suspicious."""
    denom = sum(1 for l in labels if l == want_label)
    if denom == 0:
        return float("nan")
    num = sum(1 for fl, l in zip(flags, labels) if l == want_label and fl)
    return num / denom


def _isnan(x):
    return isinstance(x, float) and math.isnan(x)


# ----------------------------------------------------------------------------
# Shared per-round context
# ----------------------------------------------------------------------------

class StealthContext:
    """Everything the metric blocks share for one round, computed once.

    Cheap fields (deltas, group_l2, norms) are always present. The O(n^2 d)
    PairwiseStats matrix is built only when an enabled block needs it (krum /
    cluster; stats is None otherwise), and the coordinate-wise median_delta only
    when the median or cosine block is enabled — so disabling the pairwise
    defenses actually skips the pairwise pass.
    """

    def __init__(self, round, cids, attacker_ids, num_samples, deltas, group_l2,
                 buckets, eligible_buckets, group_numel, norms, median_delta, median_norm, stats,
                 krum_f, multikrum_m, normclip_c):
        self.round = round
        self.cids = cids
        self.n = len(cids)
        self.attacker_ids = attacker_ids
        self.attacker_cids = [c for c in cids if c in attacker_ids]
        self.benign_cids = [c for c in cids if c not in attacker_ids]
        self.num_samples = num_samples
        self.deltas = deltas
        self.group_l2 = group_l2
        self.buckets = buckets                    # ordered_buckets: dl2_* column order (all buckets)
        self.eligible_buckets = eligible_buckets  # trainable, non-BN buckets: entropy support + denom
        self.group_numel = group_numel            # eligible element count per bucket (n_g); per-param norm
        self.norms = norms
        self.median_delta = median_delta   # coord-wise median update; robust reference
        self.median_norm = median_norm     # shared by the 'median' and 'cosine' blocks
        self.stats = stats
        self.krum_f = krum_f
        self.multikrum_m = multikrum_m
        self.normclip_c = normclip_c
        # Filled after the per-client phase, for the label-aware summary phase.
        self.rows = None
        self.labels = None

    def col(self, name):
        return [r[name] for r in self.rows]


# ----------------------------------------------------------------------------
# Metric blocks: each is an independently toggleable unit contributing
# per-client columns and (optionally) label-aware summary columns. Add a
# defense = write its two functions + append one MetricBlock to _BLOCKS.
# Each block reads only its OWN per-client columns in its summary, so there are
# no cross-block dependencies: a disabled block removes both its per-client and
# its summary columns cleanly.
# ----------------------------------------------------------------------------

def _spread_density(energy, denom):
    """(top_group_frac, normalized-entropy density in [0,1]) for a per-bucket energy dict.

    density = H / log(|buckets|) with H = -sum p_g log p_g, p_g = energy_g / sum energy.
    Dividing by log(|buckets|) (a per-round constant) rescales to [0,1] but does NOT
    change the per-round ranking, so a metric's AUC is identical with or without it —
    the reported density and the underlying raw entropy H grade the same.
    """
    tot = sum(energy.values())
    if tot <= 0:
        return float("nan"), float("nan")
    top = max(energy.values()) / tot
    if denom <= 0:
        return top, 0.0   # single eligible bucket: no spread is possible
    ps = [e / tot for e in energy.values() if e > 0]
    H = -sum(p * math.log(p) for p in ps)
    return top, H / denom


def _energy_per_client(ctx):
    """Per-bucket update-energy decomposition and its normalized Shannon spread,
    on TWO energy bases (only ELIGIBLE buckets enter — frozen groups and BN
    coordinates are excluded; BN is already dropped from the delta):

      RAW           energy_g = ||delta_g||^2              -> group_entropy_raw
      PER-PARAMETER energy_g = ||delta_g||^2 / n_g        -> group_entropy_density

    n_g = eligible element count in bucket g (ctx.group_numel). The raw basis is
    biased toward larger buckets (more coordinates -> more total energy at equal
    per-coordinate magnitude), so its entropy partly reflects the parameter-count
    distribution; the per-parameter basis (mean squared coordinate magnitude)
    removes that bias, so a uniform per-coordinate update -> density 1.0 regardless
    of bucket sizes. Both densities normalize by log(|eligible_buckets|) -> [0, 1].
    dl2_* raw columns are still emitted for ALL buckets (frozen ones read 0.0).
    """
    buckets = ctx.eligible_buckets
    denom = math.log(len(buckets)) if len(buckets) > 1 else 0.0
    out = {}
    for c in ctx.cids:
        gl = ctx.group_l2[c]
        energy = {g: gl.get(g, 0.0) ** 2 for g in buckets}
        energy_pp = {}
        for g in buckets:
            n = ctx.group_numel.get(g, 0)
            energy_pp[g] = (gl.get(g, 0.0) ** 2 / n) if n > 0 else 0.0
        top_frac, entropy_raw = _spread_density(energy, denom)
        _, entropy_density = _spread_density(energy_pp, denom)
        row = {f"dl2_{g}": gl.get(g, 0.0) for g in ctx.buckets}
        row["top_group_frac"] = top_frac
        row["group_entropy_raw"] = entropy_raw          # raw energy basis: ||delta_g||^2
        row["group_entropy_density"] = entropy_density  # per-parameter basis: ||delta_g||^2 / n_g
        out[c] = row
    return out


def _energy_summary(ctx):
    # A localized backdoor concentrates update energy in few buckets -> LOWER
    # entropy density for attackers, on either basis; read separability as
    # |AUC - 0.5|. It only separates when the attack actually localizes energy —
    # a diffuse (e.g. whole-update-scaled) attack leaves density near the benign
    # spread and the AUC sits near 0.5. Both the raw and the per-parameter-normalized
    # entropy are graded (they differ because the per-param basis removes the
    # bucket-size bias, so their AUCs can genuinely diverge).
    return {"auc_group_entropy_raw": _auc(ctx.col("group_entropy_raw"), ctx.labels),
            "auc_top_group_frac": _auc(ctx.col("top_group_frac"), ctx.labels),
            "cohens_d_group_entropy_raw": _cohens_d(ctx.col("group_entropy_raw"), ctx.labels),
            "auc_group_entropy_density": _auc(ctx.col("group_entropy_density"), ctx.labels),
            "cohens_d_group_entropy_density": _cohens_d(ctx.col("group_entropy_density"), ctx.labels)}


def _energy_columns(name_to_bucket):
    return [*(f"dl2_{b}" for b in ordered_buckets(name_to_bucket)),
            "top_group_frac", "group_entropy_raw", "group_entropy_density"]


def _norm_per_client(ctx):
    return {c: {"update_l2": ctx.norms[c]} for c in ctx.cids}


def _norm_summary(ctx):
    return {"auc_update_l2": _auc(ctx.col("update_l2"), ctx.labels),
            "cohens_d_update_l2": _cohens_d(ctx.col("update_l2"), ctx.labels)}


def _cosine_per_client(ctx):
    """Direction signal against a ROBUST reference: cosine to the coordinate-wise
    median update. The median (not the mean) is used deliberately — a high-norm
    replacement attacker drags the MEAN direction toward itself, so cos-to-mean
    would read the attacker as maximally aligned with consensus exactly when it is
    most malicious. The median is not corruptible that way, matching dist_to_median.
    """
    out = {}
    for c in ctx.cids:
        cos_to_median = float("nan")
        if ctx.median_delta is not None and ctx.norms[c] > 0 and ctx.median_norm > 0:
            cos_to_median = float(torch.dot(ctx.deltas[c], ctx.median_delta)) / (ctx.norms[c] * ctx.median_norm)
        out[c] = {"cos_to_median": cos_to_median}
    return out


def _cosine_summary(ctx):
    # Attackers deviate from the benign consensus direction, so their cos_to_median
    # is LOWER -> AUC below 0.5. Read separability as |AUC - 0.5| (or 1 - AUC).
    return {"auc_cos_to_median": _auc(ctx.col("cos_to_median"), ctx.labels),
            "cohens_d_cos_to_median": _cohens_d(ctx.col("cos_to_median"), ctx.labels)}


def _krum_per_client(ctx):
    if ctx.n < 2:
        return {c: {"krum_score": float("nan"), "krum_selected": 0,
                    "multikrum_selected": 0} for c in ctx.cids}
    out = krum(ctx.stats, ctx.krum_f, ctx.multikrum_m)
    return {c: {"krum_score": out[c]["krum_score"],
                "krum_selected": int(out[c]["krum_selected"]),
                "multikrum_selected": int(out[c]["multikrum_selected"])} for c in ctx.cids}


def _krum_summary(ctx):
    suspect = [not bool(r["multikrum_selected"]) for r in ctx.rows]  # rejected by Multi-Krum
    return {"multikrum_tpr": _rate(suspect, ctx.labels, 1),
            "multikrum_fpr": _rate(suspect, ctx.labels, 0),
            "auc_krum_score": _auc(ctx.col("krum_score"), ctx.labels),
            "cohens_d_krum_score": _cohens_d(ctx.col("krum_score"), ctx.labels)}


def _median_per_client(ctx):
    if ctx.median_delta is None:
        return {c: {"dist_to_median": float("nan")} for c in ctx.cids}
    return {c: {"dist_to_median": float(torch.norm(ctx.deltas[c] - ctx.median_delta))}
            for c in ctx.cids}


def _median_summary(ctx):
    return {"auc_dist_to_median": _auc(ctx.col("dist_to_median"), ctx.labels),
            "cohens_d_dist_to_median": _cohens_d(ctx.col("dist_to_median"), ctx.labels)}


def _normclip_per_client(ctx):
    nc = norm_clip(ctx.norms, ctx.cids, ctx.normclip_c)
    return {c: {"normclip_flagged": int(nc[c]["normclip_flagged"]),
                "normclip_threshold": nc[c]["normclip_threshold"]} for c in ctx.cids}


def _normclip_summary(ctx):
    suspect = [bool(r["normclip_flagged"]) for r in ctx.rows]
    return {"normclip_tpr": _rate(suspect, ctx.labels, 1),
            "normclip_fpr": _rate(suspect, ctx.labels, 0)}


def _cluster_per_client(ctx):
    cl = cosine_cluster(ctx.stats)
    return {c: {"cluster_label": cl[c]["cluster_label"],
                "cluster_outlier": int(cl[c]["cluster_outlier"])} for c in ctx.cids}


def _cluster_summary(ctx):
    suspect = [bool(r["cluster_outlier"]) for r in ctx.rows]
    return {"cluster_tpr": _rate(suspect, ctx.labels, 1),
            "cluster_fpr": _rate(suspect, ctx.labels, 0)}


class MetricBlock:
    """One toggleable stealth metric.

    needs_pairwise flags the O(n^2 d) PairwiseStats dependency (cos / sqdist) so
    the orchestrator can skip that pass when no enabled block needs it.
    per_client_cols is a callable(name_to_bucket) -> [col names] (only the energy
    block actually varies with granularity; the rest ignore the argument).
    summary may be None for a block with no label-aware grading (all current
    blocks provide one).
    """

    def __init__(self, name, needs_pairwise, per_client, summary,
                 per_client_cols, summary_cols):
        self.name = name
        self.needs_pairwise = needs_pairwise
        self.per_client = per_client
        self.summary = summary
        self.per_client_cols = per_client_cols
        self.summary_cols = summary_cols


# Registry order fixes the CSV column order. Kept aligned with the pre-refactor
# per-client layout so existing (all-on) runs get an identical per-client schema.
_BLOCKS = [
    MetricBlock("norm", False, _norm_per_client, _norm_summary,
                lambda m: ["update_l2"],
                ["auc_update_l2", "cohens_d_update_l2"]),
    MetricBlock("cosine", False, _cosine_per_client, _cosine_summary,
                lambda m: ["cos_to_median"],
                ["auc_cos_to_median", "cohens_d_cos_to_median"]),
    MetricBlock("krum", True, _krum_per_client, _krum_summary,
                lambda m: ["krum_score", "krum_selected", "multikrum_selected"],
                ["multikrum_tpr", "multikrum_fpr", "auc_krum_score", "cohens_d_krum_score"]),
    MetricBlock("median", False, _median_per_client, _median_summary,
                lambda m: ["dist_to_median"],
                ["auc_dist_to_median", "cohens_d_dist_to_median"]),
    MetricBlock("normclip", False, _normclip_per_client, _normclip_summary,
                lambda m: ["normclip_flagged", "normclip_threshold"],
                ["normclip_tpr", "normclip_fpr"]),
    MetricBlock("cluster", True, _cluster_per_client, _cluster_summary,
                lambda m: ["cluster_label", "cluster_outlier"],
                ["cluster_tpr", "cluster_fpr"]),
    MetricBlock("energy", False, _energy_per_client, _energy_summary,
                _energy_columns,
                ["auc_group_entropy_raw", "auc_top_group_frac",
                 "cohens_d_group_entropy_raw",
                 "auc_group_entropy_density",
                 "cohens_d_group_entropy_density"]),
]
_BLOCK_BY_NAME = {b.name: b for b in _BLOCKS}
DEFENSE_NAMES = tuple(b.name for b in _BLOCKS)


def _resolve_enabled(enabled):
    """Normalize an enabled iterable (None = all) to registry order; validate names."""
    if enabled is None:
        return list(DEFENSE_NAMES)
    enabled = set(enabled)
    unknown = enabled - set(DEFENSE_NAMES)
    if unknown:
        raise ValueError(f"Unknown stealth metric(s): {sorted(unknown)}. "
                         f"Known: {list(DEFENSE_NAMES)}")
    return [n for n in DEFENSE_NAMES if n in enabled]


# ---- column helpers (schema introspection; the logger uses lazy headers) ----

def per_client_columns(name_to_bucket, enabled=None):
    """Full per-client column order for a granularity (name_to_bucket) and enabled set."""
    cols = list(_PER_CLIENT_BASE_COLUMNS)
    for name in _resolve_enabled(enabled):
        cols.extend(_BLOCK_BY_NAME[name].per_client_cols(name_to_bucket))
    return cols


def summary_columns(enabled=None):
    """Full summary column order for an enabled set (granularity-independent)."""
    cols = list(_SUMMARY_BASE_COLUMNS)
    for name in _resolve_enabled(enabled):
        cols.extend(_BLOCK_BY_NAME[name].summary_cols)
    return cols


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------

def analyze(round, client_weights, global_weights, attacker_ids,
            name_to_bucket, eligible_names, eligible_buckets,
            num_samples=None, krum_f=1, multikrum_m=None,
            normclip_c=2.0, enabled=None):
    """
    Run the enabled passive metric blocks for one round.

    name_to_bucket:   the granularity partition from FL.groups.build_bucket_map.
    eligible_names:   trainable, non-BN named parameters (FL.groups.build_eligibility).
                      Every metric's delta is restricted to these coordinates.
    eligible_buckets: buckets holding >=1 eligible param, canonical order; its
                      length is the group_entropy_raw normalization denominator.
    enabled: iterable of block names (see DEFENSE_NAMES); None = all on. A
    disabled block contributes NO columns (not NaN columns) to either output.
    Returns (per_client_rows, summary_row) as dicts; pure/observational, inputs
    not mutated. Column order follows the registry order.
    """
    enabled = _resolve_enabled(enabled)
    enabled_set = set(enabled)
    cids = sorted(client_weights.keys())
    attacker_ids = set(attacker_ids)
    num_samples = num_samples or {}
    buckets = ordered_buckets(name_to_bucket)

    # Always-cheap shared state: deltas, per-bucket energy, per-client norms.
    deltas, group_l2 = {}, {}
    for c in cids:
        flat, gl = compute_delta(client_weights[c], global_weights, name_to_bucket, eligible_names)
        deltas[c], group_l2[c] = flat, gl
    norms = {c: float(torch.norm(deltas[c])) for c in cids}

    # Per-bucket eligible element count n_g, for the per-parameter energy
    # normalization in the energy block. Structural (identical every round / every
    # client), so read once from any participating client's weights — a plain
    # lookup, no extra plumbing through build_eligibility or the server.
    group_numel = {b: 0 for b in buckets}
    if cids:
        ref = client_weights[cids[0]]
        for name in eligible_names:
            if name in ref:
                group_numel[name_to_bucket.get(name, "other")] += ref[name].numel()

    # Expensive shared state, built only when an enabled block needs it.
    need_pairwise = any(_BLOCK_BY_NAME[n].needs_pairwise for n in enabled)
    stats = PairwiseStats(deltas) if need_pairwise else None

    # Coordinate-wise median update: the robust reference shared by the 'median'
    # (euclidean) and 'cosine' (angular) blocks. Built once, only when needed.
    median_delta, median_norm = None, 0.0
    if ("median" in enabled_set or "cosine" in enabled_set) and len(cids) >= 2:
        stacked = torch.stack([deltas[c] for c in cids], dim=0)
        median_delta = torch.median(stacked, dim=0).values
        median_norm = float(torch.norm(median_delta))
        del stacked

    ctx = StealthContext(round, cids, attacker_ids, num_samples, deltas, group_l2,
                         buckets, eligible_buckets, group_numel, norms,
                         median_delta, median_norm, stats,
                         krum_f, multikrum_m, normclip_c)

    # Per-client phase: base identity columns, then each enabled block in order.
    rows_by_cid = {c: {"round": round, "cid": c,
                       "type": "Attacker" if c in attacker_ids else "Benign",
                       "is_attacker": int(c in attacker_ids),
                       "num_samples": num_samples.get(c, "")} for c in cids}
    for name in enabled:
        contribution = _BLOCK_BY_NAME[name].per_client(ctx)
        for c in cids:
            rows_by_cid[c].update(contribution[c])
    rows = [rows_by_cid[c] for c in cids]

    # Label-aware summary phase (grades the blind decisions; never feeds them).
    ctx.rows = rows
    ctx.labels = [1 if c in attacker_ids else 0 for c in cids]
    summary = {"round": round, "n_clients": ctx.n, "n_attackers": sum(ctx.labels)}
    for name in enabled:
        block = _BLOCK_BY_NAME[name]
        if block.summary is not None:
            summary.update(block.summary(ctx))

    del deltas, median_delta
    return rows, summary
