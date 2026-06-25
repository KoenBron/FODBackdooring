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
  * PER_CLIENT_COLUMNS / SUMMARY_COLUMNS are the single source of truth for the
    CSV schema; the logger imports them, so adding a metric is a one-line edit
    here plus filling it in analyze().
  * Each defense is a standalone function over a shared PairwiseStats, so a new
    defense is: write the fn -> call it in analyze() -> add its columns.
  * Per-group decomposition uses the 13-group map from FL.groups — same map the
    localization protocols (PilotA, future LaySelFL / Ours) use. The harness
    has no parallel group definition; FL.groups is the single source of truth.
"""

from __future__ import annotations
import math
import torch

from FL.groups import GROUP_NAMES, name_to_group

# ----------------------------------------------------------------------------
# Parameter grouping (delegated to FL.groups)
# ----------------------------------------------------------------------------

# State_dict entries that aren't in any group (e.g. BN running_mean / running_var
# — they're floats and propagate during aggregation, so they belong to the
# energy budget, just not to any learnable-param group). "other" catches them.
_STEALTH_GROUPS = list(GROUP_NAMES) + ["other"]


def _group_of(name):
    """Group bucket for a state_dict key.

    FL.groups.name_to_group is keyed on named_parameters() names; it returns
    None for anything else (BN buffers etc.). The harness measures the FULL
    state_dict delta, so unmatched names fall into 'other' rather than being
    dropped.
    """
    return name_to_group(name) or "other"


# ----------------------------------------------------------------------------
# CSV schema (single source of truth, imported by the logger)
# ----------------------------------------------------------------------------

_GROUP_COLS = [f"dl2_{g}" for g in _STEALTH_GROUPS]

PER_CLIENT_COLUMNS = [
    "round", "cid", "type", "is_attacker", "num_samples",
    "update_l2", "update_cos_to_mean", "mean_cos_to_others",
    "mean_cos_to_attackers", "mean_cos_to_benign",
    "krum_score", "krum_selected", "multikrum_selected",
    "dist_to_median", "normclip_flagged", "normclip_threshold",
    "cluster_label", "cluster_outlier",
    *_GROUP_COLS,
    "top_group_frac", "group_entropy",
]

SUMMARY_COLUMNS = [
    "round", "n_clients", "n_attackers",
    "multikrum_tpr", "multikrum_fpr",
    "normclip_tpr", "normclip_fpr",
    "cluster_tpr", "cluster_fpr",
    "auc_update_l2", "auc_krum_score", "auc_dist_to_median",
    "auc_update_cos_to_mean", "auc_mean_cos_to_others",
    "cohens_d_update_l2", "cohens_d_krum_score", "cohens_d_dist_to_median",
]


# ----------------------------------------------------------------------------
# Update extraction
# ----------------------------------------------------------------------------

def compute_delta(client_weights, global_weights):
    """
    Delta = theta_client - theta_broadcast over floating-point parameters only.
    Returns (flat_delta: 1-D float32 CPU tensor, group_l2: {group: L2 norm}).
    Integer buffers (e.g. num_batches_tracked) are skipped: no learnable signal,
    and they would otherwise pollute the norm.
    """
    parts = []
    group_sq = {g: 0.0 for g in _STEALTH_GROUPS}
    for k, w in client_weights.items():
        if not torch.is_floating_point(w):
            continue
        d = (w.detach().cpu().float() - global_weights[k].detach().cpu().float()).flatten()
        parts.append(d)
        group_sq[_group_of(k)] += float(torch.dot(d, d))
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


def norm_clip(stats, c):
    """Norm-bound detector (logged only, never applied). Flag ||Delta|| > c*median."""
    norms = [stats.norm[i] for i in stats.cids]
    med = float(torch.median(torch.tensor(norms))) if norms else 0.0
    thr = c * med
    return {i: {"normclip_flagged": stats.norm[i] > thr,
                "normclip_threshold": thr} for i in stats.cids}


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


def median_distance(deltas):
    """L2 distance from each Delta to the coordinate-wise median (Median/TM signal)."""
    cids = sorted(deltas.keys())
    if len(cids) < 2:
        return {c: float("nan") for c in cids}
    stacked = torch.stack([deltas[c] for c in cids], dim=0)
    med = torch.median(stacked, dim=0).values
    out = {c: float(torch.norm(deltas[c] - med)) for c in cids}
    del stacked, med
    return out


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
# Orchestrator
# ----------------------------------------------------------------------------

def analyze(round, client_weights, global_weights, attacker_ids,
            num_samples=None, krum_f=1, multikrum_m=None, normclip_c=2.0):
    """
    Run the full passive battery for one round.

    Returns (per_client_rows, summary_row) as dicts keyed by the column
    constants. Pure/observational: inputs are not mutated.
    """
    cids = sorted(client_weights.keys())
    attacker_ids = set(attacker_ids)
    num_samples = num_samples or {}
    gnames = _STEALTH_GROUPS

    # 1. Updates + per-group energy
    deltas, group_l2 = {}, {}
    for c in cids:
        flat, gl = compute_delta(client_weights[c], global_weights)
        deltas[c], group_l2[c] = flat, gl

    stats = PairwiseStats(deltas)

    # Mean update (accumulate to avoid a second giant stack)
    mean_delta = None
    for c in cids:
        mean_delta = deltas[c].clone() if mean_delta is None else mean_delta + deltas[c]
    if mean_delta is not None:
        mean_delta /= len(cids)
        mean_norm = float(torch.norm(mean_delta))
    else:
        mean_norm = 0.0

    # 2. Label-blind defenses
    n = len(cids)
    krum_out = krum(stats, krum_f, multikrum_m) if n >= 2 else {}
    nc_out = norm_clip(stats, normclip_c) if n >= 1 else {}
    cl_out = cosine_cluster(stats)
    md_out = median_distance(deltas)

    attacker_cids = [c for c in cids if c in attacker_ids]
    benign_cids = [c for c in cids if c not in attacker_ids]

    # 3. Per-client rows
    rows = []
    for c in cids:
        gl = group_l2[c]
        energy = {g: gl[g] ** 2 for g in gnames}
        tot_e = sum(energy.values())
        top_frac = (max(energy.values()) / tot_e) if tot_e > 0 else float("nan")
        if tot_e > 0:
            ps = [e / tot_e for e in energy.values() if e > 0]
            H = -sum(p * math.log(p) for p in ps)
            ent = H / math.log(len(energy)) if len(energy) > 1 else 0.0
        else:
            ent = float("nan")

        cos_to_mean = float("nan")
        if mean_delta is not None and stats.norm[c] > 0 and mean_norm > 0:
            cos_to_mean = float(torch.dot(deltas[c], mean_delta)) / (stats.norm[c] * mean_norm)

        row = {
            "round": round, "cid": c,
            "type": "Attacker" if c in attacker_ids else "Benign",
            "is_attacker": int(c in attacker_ids),
            "num_samples": num_samples.get(c, ""),
            "update_l2": stats.norm[c],
            "update_cos_to_mean": cos_to_mean,
            "mean_cos_to_others": stats.mean_cos_to(c, cids),
            "mean_cos_to_attackers": stats.mean_cos_to(c, attacker_cids),
            "mean_cos_to_benign": stats.mean_cos_to(c, benign_cids),
            "krum_score": krum_out.get(c, {}).get("krum_score", float("nan")),
            "krum_selected": int(krum_out.get(c, {}).get("krum_selected", False)),
            "multikrum_selected": int(krum_out.get(c, {}).get("multikrum_selected", False)),
            "dist_to_median": md_out.get(c, float("nan")),
            "normclip_flagged": int(nc_out.get(c, {}).get("normclip_flagged", False)),
            "normclip_threshold": nc_out.get(c, {}).get("normclip_threshold", float("nan")),
            "cluster_label": cl_out.get(c, {}).get("cluster_label", 0),
            "cluster_outlier": int(cl_out.get(c, {}).get("cluster_outlier", False)),
            "top_group_frac": top_frac,
            "group_entropy": ent,
        }
        for g in gnames:
            row[f"dl2_{g}"] = gl.get(g, 0.0)
        rows.append(row)

    # 4. Label-aware scoring of the blind decisions
    labels = [1 if c in attacker_ids else 0 for c in cids]

    def col(name):
        return [r[name] for r in rows]

    # Each defense's "suspicion" = 1 means flagged malicious.
    mk_suspect = [not bool(r["multikrum_selected"]) for r in rows]  # rejected by Multi-Krum
    nc_suspect = [bool(r["normclip_flagged"]) for r in rows]
    cl_suspect = [bool(r["cluster_outlier"]) for r in rows]

    summary = {
        "round": round, "n_clients": n, "n_attackers": sum(labels),
        "multikrum_tpr": _rate(mk_suspect, labels, 1),
        "multikrum_fpr": _rate(mk_suspect, labels, 0),
        "normclip_tpr": _rate(nc_suspect, labels, 1),
        "normclip_fpr": _rate(nc_suspect, labels, 0),
        "cluster_tpr": _rate(cl_suspect, labels, 1),
        "cluster_fpr": _rate(cl_suspect, labels, 0),
        "auc_update_l2": _auc(col("update_l2"), labels),
        "auc_krum_score": _auc(col("krum_score"), labels),
        "auc_dist_to_median": _auc(col("dist_to_median"), labels),
        "auc_update_cos_to_mean": _auc(col("update_cos_to_mean"), labels),
        "auc_mean_cos_to_others": _auc(col("mean_cos_to_others"), labels),
        "cohens_d_update_l2": _cohens_d(col("update_l2"), labels),
        "cohens_d_krum_score": _cohens_d(col("krum_score"), labels),
        "cohens_d_dist_to_median": _cohens_d(col("dist_to_median"), labels),
    }

    del deltas, mean_delta
    return rows, summary
