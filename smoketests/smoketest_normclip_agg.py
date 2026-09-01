"""Smoke test for norm-clip AGGREGATION (--aggregation-strategy normclip).

Run from project root:
    python smoketests/smoketest_normclip_agg.py

The clipping maths lives in the torch-free FL.strategies._robust, so this drives it with
plain float dicts and needs neither torch nor a GPU — the same split that keeps
krum_select and foolsgold_weights testable. Asserts:

  1. An over-threshold update is rescaled to EXACTLY the bound M, not merely below it.
  2. An under-threshold update is untouched (scale is exactly 1.0, not 0.999...).
  3. All-equal norms -> nothing clipped (median == every norm, so no one exceeds it).
  4. DIRECTION IS PRESERVED EXACTLY. Clipping is a positive scalar, so the clipped update
     is parallel to the original: cosine == 1. This is the thesis claim — norm clipping
     acts on the magnitude axis only and leaves cos_to_median/Krum-style angular
     separation untouched — so it is asserted rather than assumed.
  5. Degenerate rounds do not divide by zero: a zero-norm client, and an all-zero round.
  6. Even participant counts average the two middle norms (statistics.median), which is
     what keeps this defense consistent with normclip_report.csv. torch.median would
     return the LOWER middle value instead.
  7. c scales the bound linearly, and c = 1.0 (FLAME's rule) clips everything strictly
     above the median — by construction roughly half a round.
  8. Ordering: the result cannot depend on the order cids are passed in.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FL.strategies._robust import normclip_scales


def _expect(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)


def main():
    # ---- 1/2. exact rescale to M, exact pass-through below it -----------------------
    norms = {0: 1.0, 1: 0.9, 2: 1.1, 3: 10.0}
    cids = [0, 1, 2, 3]
    r = normclip_scales(norms, cids, 1.0)
    M = r["threshold"]
    _expect(abs(norms[3] * r["scales"][3] - M) < 1e-12,
            f"clipped update must land on EXACTLY M={M}, got {norms[3] * r['scales'][3]}")
    _expect(r["scales"][1] == 1.0,
            f"an under-threshold update must be exactly 1.0, got {r['scales'][1]!r}")
    _expect(r["scales"][0] == 1.0, "a norm equal to the bound is not clipped (strict >)")
    print(f"  [1] over-threshold 10.0 -> {norms[3] * r['scales'][3]:.6f} == M ({M:.4f})")
    print(f"  [2] under-threshold untouched: scale is exactly 1.0")

    # ---- 3. all-equal norms: nothing to clip -----------------------------------------
    flat = normclip_scales({c: 2.5 for c in cids}, cids, 1.0)
    _expect(flat["clipped"] == [], f"equal norms must clip nobody, got {flat['clipped']}")
    _expect(all(s == 1.0 for s in flat["scales"].values()), "all scales must be exactly 1.0")
    print("  [3] all-equal norms -> nothing clipped")

    # ---- 4. THE THESIS CLAIM: direction is preserved exactly --------------------------
    # Apply the scale to a real vector and check the clipped update is parallel to the
    # original. A defense that changed direction would show up here as cos < 1.
    delta = [3.0, -4.0, 12.0]                       # ||delta|| = 13
    n13 = math.sqrt(sum(x * x for x in delta))
    r4 = normclip_scales({0: n13, 1: 1.0, 2: 1.0}, [0, 1, 2], 1.0)
    s = r4["scales"][0]
    clipped = [x * s for x in delta]
    dot = sum(a * b for a, b in zip(delta, clipped))
    cos = dot / (n13 * math.sqrt(sum(x * x for x in clipped)))
    _expect(abs(cos - 1.0) < 1e-12,
            f"clipping must preserve direction exactly, got cos={cos!r}")
    _expect(abs(math.sqrt(sum(x * x for x in clipped)) - r4["threshold"]) < 1e-12,
            "and the clipped vector's norm must equal the bound")
    _expect(s < 1.0, "fixture must actually clip, else the test is vacuous")
    print(f"  [4] direction preserved: cos(delta, clipped) = {cos:.15f}; only the norm moved "
          f"({n13:.1f} -> {r4['threshold']:.1f})")

    # ---- 5. degenerate rounds ---------------------------------------------------------
    z = normclip_scales({0: 0.0, 1: 1.0, 2: 2.0}, [0, 1, 2], 1.0)
    _expect(z["scales"][0] == 1.0, "a zero-norm client must not divide by zero")
    allz = normclip_scales({c: 0.0 for c in cids}, cids, 1.0)
    _expect(allz["threshold"] == 0.0 and allz["clipped"] == []
            and all(s == 1.0 for s in allz["scales"].values()),
            f"an all-zero round must clip nobody, got {allz}")
    _expect(normclip_scales({}, [], 1.0)["scales"] == {}, "empty round must be handled")
    print("  [5] zero-norm client, all-zero round and empty round handled")

    # ---- 6. even-n median averages the two middle values -------------------------------
    ev = normclip_scales({0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}, [0, 1, 2, 3], 1.0)
    _expect(abs(ev["median"] - 2.5) < 1e-12,
            f"even n must average the middle two (2.5), got {ev['median']} — torch.median "
            f"would give 2.0 and diverge from normclip_report.csv")
    print(f"  [6] even-n median = {ev['median']} (statistics convention, matches the report)")

    # ---- 7. c scales the bound; c=1.0 clips everything above the median ----------------
    spread = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}
    scids = list(spread)
    med = normclip_scales(spread, scids, 1.0)["median"]
    for c in (0.5, 1.0, 2.0, 4.0):
        rc = normclip_scales(spread, scids, c)
        _expect(abs(rc["threshold"] - c * med) < 1e-12, f"threshold must be c*median for c={c}")
    above = [i for i in scids if spread[i] > med]
    _expect(normclip_scales(spread, scids, 1.0)["clipped"] == above,
            "c=1.0 must clip exactly those strictly above the median")
    _expect(normclip_scales(spread, scids, 4.0)["clipped"] == [],
            "a large c must clip nobody")
    print(f"  [7] threshold = c x median; at c=1.0 clipped {len(above)}/{len(scids)} "
          f"(above the median by construction)")

    # ---- 8. order independence ---------------------------------------------------------
    a = normclip_scales(norms, [0, 1, 2, 3], 1.0)
    b = normclip_scales(norms, [3, 1, 0, 2], 1.0)
    _expect(a == b, f"result must not depend on cid order:\n  {a}\n  {b}")
    print("  [8] identical result for shuffled cid order")

    print("\nAll norm-clip aggregation smoke tests passed.")


if __name__ == "__main__":
    main()
