"""Smoketest: benign-band statistics, the cohort block, and the round digest.

Covers what a benign-only run and a "how does the attacker differ from the benign
cluster" claim depend on:

  1. Attacker-free round: analyze() runs, the benign band is measured, every
     attacker-side field and z is NaN, and nothing raises. This is the benign-only
     baseline path (--attacker-fraction 0 --stealth-on-benign-rounds).
  2. Band arithmetic: benign mean/sd and z = (attacker mean - benign mean)/benign sd
     match a hand computation, for magnitude and for direction.
  3. Cohort columns exist and exclude self: a benign client's mean_cos_to_benign
     averages over the OTHER benign clients only.
  4. An attacker far from the cohort reads as high magnitude and low direction
     agreement on both the median-referenced and the cohort-referenced columns.
  5. Declared schema equals the emitted dict keys, per-client and summary.
  6. Disabling 'cohort' removes exactly its columns and leaves the rest intact.
  7. format_round_digest renders for both an attacker round and a benign-only round
     and reports the defence counts split by cohort.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch

from FL.stealth_defenses import (analyze, format_round_digest,
                                 per_client_columns, summary_columns)

FAILURES = []


def _expect(cond, msg):
    if not cond:
        FAILURES.append(msg)


def _isnan(x):
    return isinstance(x, float) and math.isnan(x)


GLOBAL = {"roi.weight": torch.zeros(400), "fpn.weight": torch.zeros(200)}
N2B = {"roi.weight": "roi", "fpn.weight": "fpn"}
ELIG = list(GLOBAL)
EBUCKETS = ["roi", "fpn"]


def _client(seed, scale, shift=0.0):
    """A client update: shared drift (shift) plus per-client noise (scale)."""
    torch.manual_seed(seed)
    return {k: v + shift + scale * torch.randn_like(v) for k, v in GLOBAL.items()}


def main():
    # Benign clients share a descent direction (shift) so cos-to-cohort is a real
    # signal rather than noise around zero, matching a converging FL run.
    benign = {c: _client(c, 0.05, shift=0.10) for c in range(5)}

    # ---- 1. attacker-free round ----------------------------------------------
    rows, summ = analyze(7, benign, GLOBAL, [], N2B, ELIG, EBUCKETS)
    _expect(len(rows) == 5, f"benign-only: expected 5 rows, got {len(rows)}")
    _expect(summ["n_attackers"] == 0, "benign-only: n_attackers should be 0")
    for k in ("benign_l2_mean", "benign_l2_sd", "benign_cos_to_median_mean",
              "benign_cos_to_benign_mean", "benign_dist_to_benign_mean"):
        _expect(not _isnan(summ[k]), f"benign-only: {k} should be measured, got NaN")
    for k in ("attacker_l2_mean", "l2_z", "attacker_cos_to_median_mean",
              "cos_to_median_z", "attacker_cos_to_benign_mean", "cos_to_benign_z"):
        _expect(_isnan(summ[k]), f"benign-only: {k} should be NaN with no attacker")

    # ---- 2. band arithmetic ---------------------------------------------------
    # A backdoor attacker is not just a louder benign client: it is loud AND pointed
    # somewhere else. Scaling the shared drift up would raise cohort agreement, so
    # the attacker here pushes the head tensor the opposite way while the benign
    # cohort drifts positive on both.
    torch.manual_seed(99)
    attacker = {"roi.weight": GLOBAL["roi.weight"] - 0.50
                + 0.05 * torch.randn_like(GLOBAL["roi.weight"]),
                "fpn.weight": GLOBAL["fpn.weight"] + 0.10
                + 0.05 * torch.randn_like(GLOBAL["fpn.weight"])}
    mixed = dict(benign)
    mixed[9] = attacker
    rows, summ = analyze(29, mixed, GLOBAL, [9], N2B, ELIG, EBUCKETS)

    def hand_band(col):
        ben = [r[col] for r in rows if not r["is_attacker"]]
        att = [r[col] for r in rows if r["is_attacker"]]
        m = sum(ben) / len(ben)
        sd = math.sqrt(sum((v - m) ** 2 for v in ben) / len(ben))
        return m, sd, sum(att) / len(att), (sum(att) / len(att) - m) / sd

    for col, keys in (("update_l2", ("benign_l2_mean", "benign_l2_sd",
                                     "attacker_l2_mean", "l2_z")),
                      ("cos_to_median", ("benign_cos_to_median_mean",
                                         "benign_cos_to_median_sd",
                                         "attacker_cos_to_median_mean",
                                         "cos_to_median_z")),
                      ("mean_cos_to_benign", ("benign_cos_to_benign_mean",
                                              "benign_cos_to_benign_sd",
                                              "attacker_cos_to_benign_mean",
                                              "cos_to_benign_z"))):
        want = hand_band(col)
        for w, k in zip(want, keys):
            _expect(abs(summ[k] - w) < 1e-9,
                    f"band mismatch on {col}/{k}: {summ[k]} vs hand {w}")
    _expect(abs(summ["l2_ratio"] - summ["attacker_l2_mean"] / summ["benign_l2_mean"]) < 1e-9,
            "l2_ratio must be attacker mean over benign mean")

    # ---- 3. cohort excludes self ---------------------------------------------
    by_cid = {r["cid"]: r for r in rows}
    # Benign client 0 averaged over benign 1..4 (4 peers), never itself.
    from FL.stealth_defenses import PairwiseStats, compute_delta
    deltas = {c: compute_delta(w, GLOBAL, N2B, ELIG)[0] for c, w in mixed.items()}
    st = PairwiseStats(deltas)
    peers = [1, 2, 3, 4]
    want = sum(st.cos[0][p] for p in peers) / len(peers)
    _expect(abs(by_cid[0]["mean_cos_to_benign"] - want) < 1e-9,
            "benign client's mean_cos_to_benign must exclude itself")
    # The attacker averages over ALL five benign clients.
    want_a = sum(st.cos[9][p] for p in range(5)) / 5
    _expect(abs(by_cid[9]["mean_cos_to_benign"] - want_a) < 1e-9,
            "attacker's mean_cos_to_benign must cover every benign client")

    # ---- 4. direction of the signals -----------------------------------------
    _expect(summ["l2_z"] > 3, f"a far-out attacker should read loud on magnitude, z={summ['l2_z']}")
    _expect(summ["cos_to_benign_z"] < 0,
            f"attacker should agree with the cohort LESS than benign peers do, "
            f"z={summ['cos_to_benign_z']}")
    _expect(summ["dist_to_benign_z"] > 0, "attacker should sit further from the cohort")

    # ---- 5. declared schema == emitted keys ----------------------------------
    _expect(per_client_columns(N2B) == list(rows[0].keys()),
            "per-client schema drifted from the emitted keys")
    _expect(summary_columns() == list(summ.keys()),
            "summary schema drifted from the emitted keys")

    # ---- 6. cohort is cleanly disableable ------------------------------------
    keep = [n for n in ("norm", "cosine", "krum", "median", "normclip", "cluster", "energy")]
    rows_off, summ_off = analyze(29, mixed, GLOBAL, [9], N2B, ELIG, EBUCKETS, enabled=keep)
    for c in ("mean_cos_to_benign", "mean_dist_to_benign"):
        _expect(c not in rows_off[0], f"disabled cohort still emitted per-client {c}")
    for c in ("benign_cos_to_benign_mean", "cos_to_benign_z"):
        _expect(c not in summ_off, f"disabled cohort still emitted summary {c}")
    _expect("benign_l2_mean" in summ_off, "disabling cohort must not drop the norm band")
    _expect(per_client_columns(N2B, enabled=keep) == list(rows_off[0].keys()),
            "per-client schema drifted when cohort was disabled")
    _expect(summary_columns(enabled=keep) == list(summ_off.keys()),
            "summary schema drifted when cohort was disabled")

    # ---- 7. digest ------------------------------------------------------------
    text = format_round_digest(rows, summ)
    for token in ("round 29", "magnitude", "direction", "cos->benign",
                  "defences", "multi-krum KEPT", "per client"):
        _expect(token in text, f"digest missing '{token}'")
    _expect("1 attacker (9)" in text, "digest should name the attacker cid")
    _expect("attacker 0/1" in text or "attacker 1/1" in text,
            "digest should split defence counts by cohort")

    rows_b, summ_b = analyze(7, benign, GLOBAL, [], N2B, ELIG, EBUCKETS)
    text_b = format_round_digest(rows_b, summ_b)
    _expect("0 attacker (none)" in text_b, "benign-only digest should say there is no attacker")
    _expect("n/a" in text_b, "benign-only digest should show n/a for attacker fields")
    _expect(format_round_digest([], {}) == "", "empty input should give an empty digest")

    if FAILURES:
        print("  FAIL:")
        for m in FAILURES:
            print("    -", m)
        sys.exit(1)
    print("  PASS: benign band, cohort block, schema and round digest all OK")


if __name__ == "__main__":
    main()
