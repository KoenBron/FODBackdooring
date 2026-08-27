"""Observational norm-clip report (Partial protocol).

For each malicious update in a round, record whether its L2 norm exceeds
  - x * (benign-trajectory norm)   — the attacker's OWN mean anchor norm, and
  - x * (median submitted norm)    — the round's robust central norm,
across a sweep of x. OBSERVATIONAL ONLY — this never touches aggregation or
training; it simulates the norm-clip defense being considered separately.

Split in two so the threshold/flag logic is torch-free and unit-testable:
  - update_norm(...) is the whole-update L2 norm of one update (needs torch;
    imported lazily so importing this module does NOT drag torch into the dev
    sandbox / smoketests);
  - build_report_rows(...) is pure Python over those floats.

The benign reference is supplied per-cid by the caller (the server reads it from
AttackState, where the Partial protocol stashes it) and is PER CLIENT, never pooled:
the live mean of that round's anchors (ben1 and ben2) in the plastic/disabled phases,
and that client's own frozen mean in the exploit phase. A client that never acted
during the plastic phase has no reference — its ratio_to_benign is NaN and its
exceeds_benign flags are False (see build_report_rows).
"""
import statistics


def update_norm(client_weights, global_weights):
    """||theta_client - theta_broadcast||_2 over the ENTIRE update — one scalar.

    Plain L2 over every floating-point tensor in the state dict (integer buffers like
    num_batches_tracked are skipped). No per-bucket decomposition and no eligibility
    map: this is the whole-update norm a norm-clip defense would actually see. The same
    function scores the malicious/benign trajectories (Partial protocol) and every
    submitted update (server), so all norms in the report are comparable. torch is
    imported HERE so the module stays torch-free at import time (the row builder below,
    and its smoketest, need no torch).
    """
    import torch
    sq = 0.0
    for k, w in client_weights.items():
        g = global_weights.get(k)
        if g is None or not torch.is_floating_point(w):
            continue
        sq += float((w.detach().float() - g.detach().float()).pow(2).sum())
    return sq ** 0.5


def build_report_rows(current_round, norms, attacker_ids, benign_ref_by_cid,
                      phase_by_cid, x_values):
    """One row dict per malicious client. Pure Python (floats in, dicts out).

    norms:              {cid: L2 norm} for EVERY client this round (the median is
                        taken over all of them).
    attacker_ids:       which cids are malicious (the rows we emit).
    benign_ref_by_cid:  {cid: benign-trajectory norm or None}.
    phase_by_cid:       {cid: 'plastic'|'exploit'|'disabled'}.
    x_values:           iterable of thresholds; each adds exceeds_median@x /
                        exceeds_benign@x columns (flagged when norm > x*reference).

    exceedance is strict (>), matching a norm bound at exactly x*reference. When a
    reference is missing/<=0 the ratio is NaN and its flags are False.
    """
    all_norms = [norms[c] for c in norms]
    median_norm = float(statistics.median(all_norms)) if all_norms else 0.0

    rows = []
    for cid in sorted(attacker_ids):
        if cid not in norms:
            continue  # a scheduled attacker that didn't actually submit this round
        mal = norms[cid]
        benign_ref = benign_ref_by_cid.get(cid)
        has_benign = benign_ref is not None and benign_ref > 0

        row = {
            "round": current_round,
            "cid": cid,
            "phase": phase_by_cid.get(cid, ""),
            "mal_norm": mal,
            "median_norm": median_norm,
            "benign_ref": benign_ref if benign_ref is not None else float("nan"),
            "ratio_to_median": (mal / median_norm) if median_norm > 0 else float("nan"),
            "ratio_to_benign": (mal / benign_ref) if has_benign else float("nan"),
        }
        for x in x_values:
            row[f"exceeds_median@{x}"] = bool(median_norm > 0 and mal > x * median_norm)
            row[f"exceeds_benign@{x}"] = bool(has_benign and mal > x * benign_ref)
        rows.append(row)
    return rows
