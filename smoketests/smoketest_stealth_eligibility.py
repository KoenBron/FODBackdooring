"""Smoke test for the stealth harness eligibility + the five separability metrics.

Run from project root:
    python smoketests/smoketest_stealth_eligibility.py

Asserts, against the real torchvision fasterrcnn_resnet50_fpn:
  1. build_eligibility drops frozen backbone groups (stem, layer1 by default) and
     every batch-norm parameter; eligible params are a strict, non-empty subset.
  2. compute_delta restricted to eligible_names ignores BN-buffer / frozen drift.
  3. analyze() emits exactly the five requested metrics' columns (per-client AND
     summary AUCs) and NOT the old mean-based cosine columns.
  4. An update spread evenly across all eligible buckets scores density ~= 1.0,
     and an update localized to one bucket scores ~= 0.0 (the denominator fix);
     the energy summary grades that separation (auc_group_entropy_raw).

CPU-only, random weights (no checkpoint). Needs torch + torchvision.
"""
import sys
import math
import torch

from models.fasterrcnn.model import get_model
from FL.groups import build_bucket_map, build_eligibility, _batchnorm_param_names, ordered_buckets
from FL.stealth_defenses import analyze, compute_delta


def _expect(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)


def _isnan(x):
    return isinstance(x, float) and math.isnan(x)


def _perturb(sd, names, scale):
    """Return a state_dict copy with +scale added to every tensor in `names`."""
    out = {k: v.detach().clone() for k, v in sd.items()}
    for n in names:
        out[n] = out[n] + scale
    return out


def _first_param_per_bucket(name_to_bucket, eligible_names, eligible_buckets):
    """One representative eligible param name per eligible bucket."""
    return {b: next(n for n in sorted(eligible_names) if name_to_bucket[n] == b)
            for b in eligible_buckets}


def _even_energy_spread(sd, reps):
    """State_dict with EQUAL update energy in every eligible bucket -> density 1.0.

    Bucket energy = ||delta_bucket||^2. Perturbing one param of numel n by a scalar
    a adds a^2*n energy, so a = 1/sqrt(n) makes every bucket contribute exactly 1 —
    a uniform ENERGY distribution (not a uniform per-parameter bump, which would be
    weighted by each bucket's parameter count)."""
    out = {k: v.detach().clone() for k, v in sd.items()}
    for pname in reps.values():
        a = (1.0 / out[pname].numel()) ** 0.5
        out[pname] = out[pname] + a
    return out


def main():
    print("[smoketest_stealth_eligibility] building fasterrcnn_resnet50_fpn (num_classes=20)...")
    model = get_model(num_classes=20)
    name_to_bucket = build_bucket_map(model, "group")
    eligible_names, eligible_buckets = build_eligibility(model, name_to_bucket)

    all_param_names = {n for n, _ in model.named_parameters()}
    bn_names = _batchnorm_param_names(model)
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    all_buckets = ordered_buckets(name_to_bucket)

    # (1) eligibility is a strict non-empty subset; excludes BN + frozen
    _expect(eligible_names, "no eligible params")
    _expect(eligible_names <= all_param_names, "eligible names not a subset of named_parameters")
    _expect(not (eligible_names & bn_names), "BN params leaked into eligible set")
    _expect(not (eligible_names & frozen), "frozen params leaked into eligible set")
    dropped = all_param_names - eligible_names
    _expect(dropped <= (frozen | bn_names),
            f"params dropped for an unexpected reason: {dropped - (frozen | bn_names)}")

    _expect(len(eligible_buckets) < len(all_buckets),
            "eligible_buckets should exclude at least the frozen backbone groups")
    print(f"  named params : {len(all_param_names)}  eligible: {len(eligible_names)}  "
          f"(dropped {len(dropped)}: {len(frozen)} frozen + BN)")
    print(f"  buckets      : {len(all_buckets)} total -> {len(eligible_buckets)} eligible")
    print(f"  excluded buckets: {[b for b in all_buckets if b not in eligible_buckets]}")

    # (2) BN-buffer / frozen drift is invisible to compute_delta
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    noise_targets = list(frozen)[:1]
    bn_buffers = [k for k in sd.keys()
                  if k not in all_param_names and ("running_mean" in k or "running_var" in k)]
    noise_targets += bn_buffers[:1]
    sd_noise = _perturb(sd, noise_targets, 5.0)
    flat, gl = compute_delta(sd_noise, sd, name_to_bucket, eligible_names)
    _expect(float(torch.norm(flat)) == 0.0,
            f"compute_delta picked up non-eligible drift from {noise_targets}")
    print(f"  compute_delta ignored drift on {len(noise_targets)} non-eligible tensors: OK")

    # (3) + (4) end-to-end analyze on 4 synthetic clients that separate the two
    # energy bases:
    #   client 0 (attacker): energy localized to ONE eligible bucket -> both densities ~0
    #   client 1           : equal RAW energy per bucket   -> raw density ~1.0, perparam <1
    #   clients 2,3        : uniform PER-COORDINATE bump    -> perparam density ~1.0, raw <1
    reps = _first_param_per_bucket(name_to_bucket, eligible_names, eligible_buckets)
    one_bucket = eligible_buckets[-1]  # e.g. roi.bbox_pred
    local_client = _perturb(sd, [reps[one_bucket]], 0.05)         # all energy in one bucket
    raw_even_client = _even_energy_spread(sd, reps)               # equal raw energy per bucket
    pp_even_client = _perturb(sd, list(eligible_names), 0.02)     # uniform per-coordinate magnitude

    client_weights = {0: local_client, 1: raw_even_client, 2: pp_even_client, 3: pp_even_client}
    rows, summary = analyze(
        round=1, client_weights=client_weights, global_weights=sd,
        attacker_ids={0}, name_to_bucket=name_to_bucket,
        eligible_names=eligible_names, eligible_buckets=eligible_buckets,
        num_samples={c: 100 for c in client_weights},
    )

    by_cid = {r["cid"]: r for r in rows}
    row0, row1, row2 = by_cid[0], by_cid[1], by_cid[2]

    # per-client columns present (both density bases); old cosine columns gone
    for col in ("update_l2", "krum_score", "dist_to_median", "cos_to_median",
                "group_entropy_raw", "group_entropy_density", "top_group_frac"):
        _expect(col in row0, f"missing expected per-client column {col!r}")
    for gone in ("update_cos_to_mean", "mean_cos_to_others", "group_entropy"):
        _expect(gone not in row0, f"stale column {gone!r} still present")
    for col in ("auc_update_l2", "auc_krum_score", "auc_dist_to_median",
                "auc_cos_to_median", "cohens_d_cos_to_median",
                "auc_group_entropy_raw", "auc_top_group_frac",
                "cohens_d_group_entropy_raw",
                "auc_group_entropy_density",
                "cohens_d_group_entropy_density"):
        _expect(col in summary, f"missing expected summary column {col!r}")

    # density bounds on both bases
    for r in rows:
        for key in ("group_entropy_raw", "group_entropy_density"):
            d = r[key]
            _expect(_isnan(d) or (-1e-9 <= d <= 1.0 + 1e-9), f"{key} out of [0,1]: {d}")

    # localized attacker: ~0 on both bases
    _expect(row0["group_entropy_raw"] < 0.05,
            f"attacker raw density should be ~0, got {row0['group_entropy_raw']}")
    _expect(row0["group_entropy_density"] < 0.05,
            f"attacker perparam density should be ~0, got {row0['group_entropy_density']}")
    # equal RAW energy -> raw density 1.0 (but NOT perparam, which reweights by 1/n_g)
    _expect(row1["group_entropy_raw"] > 0.999,
            f"equal-raw-energy client raw density should be ~1.0, got {row1['group_entropy_raw']}")
    _expect(row1["group_entropy_density"] < 0.999,
            "equal-raw-energy client should NOT be uniform on the perparam basis")
    # uniform per-coordinate -> perparam density 1.0 (but NOT raw, which is bucket-size weighted)
    _expect(row2["group_entropy_density"] > 0.999,
            f"uniform-per-coordinate client perparam density should be ~1.0, got {row2['group_entropy_density']}")
    _expect(row2["group_entropy_raw"] < 0.999,
            "uniform-per-coordinate client should NOT be uniform on the raw basis")

    # both energy summaries separate the localized attacker (lowest density) -> AUC 0.0
    _expect(summary["auc_group_entropy_raw"] == 0.0,
            f"raw energy: expected low-side separation, got {summary['auc_group_entropy_raw']}")
    _expect(summary["auc_group_entropy_density"] == 0.0,
            f"perparam energy: expected low-side separation, got {summary['auc_group_entropy_density']}")

    print(f"  group_entropy_raw    : attacker={row0['group_entropy_raw']:.4f} "
          f"raw-even={row1['group_entropy_raw']:.4f} pp-even={row2['group_entropy_raw']:.4f}")
    print(f"  group_entropy_density: attacker={row0['group_entropy_density']:.4f} "
          f"raw-even={row1['group_entropy_density']:.4f} pp-even={row2['group_entropy_density']:.4f}")
    print(f"  energy summary       : auc_group_entropy_raw={summary['auc_group_entropy_raw']:.2f} "
          f"auc_group_entropy_density={summary['auc_group_entropy_density']:.2f}")
    print("[smoketest_stealth_eligibility] PASS")


if __name__ == "__main__":
    main()
