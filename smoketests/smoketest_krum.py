"""Smoke test for the shared Krum core (FL.strategies._robust.krum_select).

Run from project root:
    python smoketests/smoketest_krum.py

The Krum score/selection is pure arithmetic over a squared-distance map, so this
test needs NEITHER torch NOR a GPU. It drives the SINGLE SOURCE OF TRUTH used by both
the Multi-Krum aggregation strategy and the observational stealth harness.

Asserts:
  1. A lone outlier gets the highest Krum score, is never the Krum winner, and is the
     first client Multi-Krum rejects.
  2. m auto-resolves to n-f when m<=0; an explicit m is honoured; |kept| == resolved m.
  3. The harness wrapper schema (krum_score / krum_selected / multikrum_selected) is
     reproduced from the same core result, so passive and active Krum agree.
  4. Logging integrity: the reported m ALWAYS equals |kept| (a request larger than n
     is clamped, so the server's round log can never read "m=99 ... n=4").
  5. Tie-breaking is deterministic REGARDLESS of the cid order passed in — krum_select
     sorts internally, so a caller cannot silently make selection order-dependent.
  6. `resilient` reports Blanchard's n >= 2f+3 condition, so the strategy can warn when
     the Byzantine-resilience guarantee does not apply.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FL.strategies._robust import krum_select, krum_min_clients


def _expect(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)


def _sqdist_from_points(points):
    """Symmetric sqdist map from {cid: 1-D coordinate} (squared 1-D distance)."""
    cids = list(points)
    sq = {i: {} for i in cids}
    for a in range(len(cids)):
        for b in range(a + 1, len(cids)):
            i, j = cids[a], cids[b]
            d = (points[i] - points[j]) ** 2
            sq[i][j] = sq[j][i] = d
    return sq


def main():
    # 5 tight, honest clients near 0 and one far outlier at 100 (the "attacker").
    points = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 100.0}
    sq = _sqdist_from_points(points)

    # 1. Outlier behaviour. f=1 -> k = n-f-2 = 3 nearest neighbours.
    # Scores: 0->14, 1->6, 2->6, 3->6, 4->14, 5->huge (clients 1,2,3 tie as most central).
    res = krum_select(sq, sorted(points), f=1, m=None)
    scores = res["scores"]
    _expect(max(scores, key=scores.get) == 5, f"outlier (5) should have the highest score, got {scores}")
    _expect(res["selected"] != 5, "Krum winner must never be the outlier")
    # Three-way tie at score 6 (clients 1,2,3); the tie breaks to the lowest cid.
    _expect(res["selected"] == 1, f"tie among {{1,2,3}} should resolve to lowest cid 1, got {res['selected']}")

    # 2. m auto = n - f = 6 - 1 = 5, and the outlier is exactly the one dropped.
    _expect(res["m"] == 5, f"auto m should be n-f=5, got {res['m']}")
    _expect(len(res["kept"]) == 5, f"|kept| must equal resolved m=5, got {len(res['kept'])}")
    _expect(5 not in res["kept"], "the outlier must be rejected by Multi-Krum")
    _expect(res["kept"] == {0, 1, 2, 3, 4}, f"kept set wrong: {res['kept']}")

    # explicit m honoured (keep only the 2 most central).
    res2 = krum_select(sq, sorted(points), f=1, m=2)
    _expect(res2["m"] == 2 and len(res2["kept"]) == 2, f"explicit m=2 not honoured: {res2['m']}, {res2['kept']}")
    _expect(res2["kept"] == {1, 2}, f"top-2 most central should be {{1,2}}, got {res2['kept']}")

    # 3. Harness wrapper schema reproduced from the same core result (mirror of
    #    stealth_defenses.krum). Passive detector and active defense agree by construction.
    selected, keep, score = res["selected"], res["kept"], res["scores"]
    harness = {c: {"krum_score": score[c],
                   "krum_selected": (c == selected),
                   "multikrum_selected": (c in keep)} for c in sorted(points)}
    _expect(harness[1]["krum_selected"] and not harness[5]["krum_selected"],
            "harness krum_selected flag wrong")
    _expect(harness[0]["multikrum_selected"] and not harness[5]["multikrum_selected"],
            "harness multikrum_selected flag wrong")
    _expect(sum(1 for c in harness if harness[c]["multikrum_selected"]) == 5,
            "exactly m=5 clients should be multikrum_selected")

    # 4. LOGGING INTEGRITY: reported m == |kept|, always. Requesting more than the
    #    number of participants keeps everyone; reporting the unclamped request would
    #    make the server's round log self-contradictory ("m=99" beside 6 kept clients).
    res3 = krum_select(sq, sorted(points), f=1, m=99)
    _expect(res3["m"] == len(res3["kept"]) == 6,
            f"m>n must clamp to n: m={res3['m']} |kept|={len(res3['kept'])}")
    for mm in (None, -1, 0, 1, 3, 6, 99):
        r = krum_select(sq, sorted(points), f=1, m=mm)
        _expect(r["m"] == len(r["kept"]),
                f"reported m ({r['m']}) != |kept| ({len(r['kept'])}) for m={mm}")

    # 5. TIE DETERMINISM independent of caller order. Four points on a unit square give
    #    four identical Krum scores; the winner must be the lowest cid no matter how the
    #    cids are ordered going in (krum_select sorts them itself).
    square = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0), 3: (0.0, 1.0)}
    sq_sym = {i: {} for i in square}
    for i in square:
        for j in square:
            if i != j:
                sq_sym[i][j] = (square[i][0] - square[j][0]) ** 2 + (square[i][1] - square[j][1]) ** 2
    tied = {round(v, 9) for v in krum_select(sq_sym, [0, 1, 2, 3], f=0, m=2)["scores"].values()}
    _expect(len(tied) == 1, f"symmetric square should tie all scores, got {tied}")
    for order in ([0, 1, 2, 3], [3, 1, 0, 2], [2, 3, 1, 0], [1, 0, 3, 2]):
        r = krum_select(sq_sym, order, f=0, m=2)
        _expect(r["selected"] == 0 and r["kept"] == {0, 1},
                f"cid order {order} changed the tie-break: selected={r['selected']} kept={r['kept']}")

    # 6. Resilience flag mirrors Blanchard's n >= 2f+3 (drives the strategy's warning).
    _expect(krum_min_clients(1) == 5 and krum_min_clients(4) == 11, "krum_min_clients wrong")
    _expect(krum_select(sq, sorted(points), f=1, m=None)["resilient"] is True,
            "n=6, f=1 (needs 5) should be flagged resilient")
    _expect(krum_select(sq, sorted(points), f=2, m=None)["resilient"] is False,
            "n=6, f=2 (needs 7) should be flagged NOT resilient")

    print("  PASS: Krum core scoring, selection, auto/explicit m, harness schema, "
          "m==|kept| logging integrity, order-independent ties, and resilience flag all OK")


if __name__ == "__main__":
    main()
