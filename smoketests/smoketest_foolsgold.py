"""Smoke test for FoolsGold aggregation (--aggregation-strategy foolsgold).

Run from project root:
    python smoketests/smoketest_foolsgold.py

The re-weighting maths lives in the torch-free FL.strategies._robust, so this drives it
with plain float dicts and needs neither torch nor a GPU — the same split that keeps
krum_select testable. Asserts:

  1. Sybils are found: near-identical cumulative directions collapse to weight ~0 while
     mutually dissimilar honest clients keep the mass.
  2. Both normalisation modes are well-formed: "none" (PAPER-FAITHFUL) returns weights
     in [0,1] whose SUM GROWS with the trusted count — the paper's per-client trust
     semantics, and the effective step multiplier; "sum" returns a convex combination.
  3. Pardoning: an honest client that happens to resemble one sybil is not punished as
     though it were the sybil. Without pardoning it loses most of its weight.
  4. The degenerate guard at cosine == 1.0, where alpha collapses to 0 everywhere and the
     rescale would divide by zero. Mode-dependent and safe in each: "none" returns all
     zeros (delta-form combine leaves the model untouched — "no confidence, do not move"),
     "sum" returns uniform (all-zero would divide by zero and collapse the model).
  5. Equal suspicion CANCELS. The rescale divides by max(alpha), so uniformly-similar
     clients all normalise back to 1 and the round is plain FedAvg. FoolsGold cannot
     down-weight everyone — relevant because IID benign clients are already similar.
  6. What kappa really does: it scales the logit before clipping, so it CANNOT change
     which clients are excluded (weight==0 iff the pre-kappa logit <=0). It only decides
     how decisively borderline survivors are admitted — larger kappa is more permissive
     at the margin, not more aggressive.
  7. Order independence: the result cannot depend on the order cids are passed in.
  8. Edge cases: n=0, n=1, and an all-negative similarity row (the implicit-zero
     diagonal convention).
  9. The strategy's feature resolution refuses an empty indicative-feature set, which
     would silently disable the defense.
 10. Partial suppression is reachable: a client can hold a non-zero weight that is still
     far below its 1/n fair share. Defining "rejected" as weight == 0 would report such a
     client as kept, understating what the defense did.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FL.strategies._robust import foolsgold_weights


def _expect(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)


def sim(cids, pairs, default=0.0):
    """Symmetric similarity map; `pairs` overrides {frozenset({i,j}): value}."""
    cs = {i: {} for i in cids}
    for a_i, i in enumerate(cids):
        for j in cids[a_i + 1:]:
            v = pairs.get(frozenset({i, j}), default)
            cs[i][j] = cs[j][i] = v
    return cs


def main():
    # ---- 1. sybils are found -------------------------------------------------------
    cids = [0, 1, 2, 3, 4]
    cs = sim(cids, {frozenset({3, 4}): 0.98}, default=0.05)
    r = foolsgold_weights(cs, cids)
    w = r["weights"]
    _expect(w[3] < 1e-9 and w[4] < 1e-9,
            f"colluding pair must be driven to ~0, got {w[3]:.4g}/{w[4]:.4g}")
    _expect(min(w[0], w[1], w[2]) > 0.3,
            f"honest clients must retain the mass, got {[round(w[c], 3) for c in (0, 1, 2)]}")
    _expect(not r["degenerate"], "a clear sybil/honest split is not a degenerate round")
    print(f"  [1] sybil pair -> {w[3]:.3g}/{w[4]:.3g}; honest three -> "
          f"{', '.join(f'{w[c]:.3f}' for c in (0, 1, 2))}")

    # ---- 2. both normalisation modes are well-formed ---------------------------------
    cases = {
        "sybils": cs,
        "all dissimilar": sim(cids, {}, default=0.0),
        "mixed": sim(cids, {frozenset({0, 1}): 0.7, frozenset({2, 3}): -0.4}, default=0.1),
    }
    for name, case in cases.items():
        raw = foolsgold_weights(case, cids, normalise="none")
        nrm = foolsgold_weights(case, cids, normalise="sum")
        for mode, res in (("none", raw), ("sum", nrm)):
            ws = res["weights"]
            _expect(all(0.0 <= x <= 1.0 for x in ws.values()),
                    f"{name}/{mode}: weight outside [0,1]: {ws}")
            _expect(abs(res["weight_sum"] - sum(ws.values())) < 1e-9,
                    f"{name}/{mode}: weight_sum must equal sum(weights)")
        _expect(abs(nrm["weight_sum"] - 1.0) < 1e-9,
                f"{name}: 'sum' mode must be a convex combination, got {nrm['weight_sum']}")
        # The modes must differ only by a positive scale — same ranking, same zeros.
        if raw["weight_sum"] > 0:
            for c in cids:
                _expect(abs(raw["weights"][c] / raw["weight_sum"] - nrm["weights"][c]) < 1e-9,
                        f"{name}: the two modes must differ only by the 1/sum(w) factor")
    # The paper-faithful sum is NOT 1 and grows with how many clients are trusted: that
    # is the property being preserved, and the reason sum(w) is logged every round.
    trusted_sum = foolsgold_weights(cs, cids, normalise="none")["weight_sum"]
    _expect(trusted_sum > 1.5,
            f"3 fully-trusted clients must give sum(w) ~ 3, got {trusted_sum}")
    print(f"  [2] 'none' keeps weights in [0,1] with sum(w)={trusted_sum:.2f} (the step "
          f"multiplier); 'sum' is convex; the two differ only by 1/sum(w)")

    # ---- 3. pardoning ---------------------------------------------------------------
    # Client 2 resembles sybil 3 (0.5), but the sybils resemble each other far more
    # (0.95). Pardoning scales 2's view of 3 by v_2/v_3 = 0.53, so 2 is judged on its own
    # suspiciousness rather than on 3's.
    #
    # The discriminating assertion is that 2 ends up TIED with clients 0/1, whose peak
    # similarity is only 0.02. Without pardoning 2's alpha would be 1-0.5 = 0.5 against
    # their 0.98, so a tie is unreachable — this outcome can only come from the pardon.
    #
    # Note the condition is v_j > v_i STRICTLY: a client whose own peak similarity equals
    # a sybil's is not pardoned, and correctly so — it is just as suspicious.
    cids3 = [0, 1, 2, 3, 4]
    cs3 = sim(cids3, {frozenset({3, 4}): 0.95, frozenset({2, 3}): 0.5}, default=0.02)
    w3 = foolsgold_weights(cs3, cids3)["weights"]
    _expect(w3[2] > 0.0, "pardoning must keep the resembling honest client in the average")
    _expect(w3[3] < 1e-9 and w3[4] < 1e-9, "the sybils themselves must still be excluded")
    _expect(abs(w3[2] - w3[0]) < 1e-9,
            f"the pardoned client must be treated as honest (tied with 0/1), got "
            f"2={w3[2]:.4f} vs 0={w3[0]:.4f}")
    print(f"  [3] client resembling a sybil is pardoned to {w3[2]:.3f}, tied with the "
          f"honest {w3[0]:.3f}; sybils still {w3[3]:.3g}/{w3[4]:.3g}")

    # ---- 4. the degenerate guard -----------------------------------------------------
    # Fires only at cosine == 1.0 exactly: alpha = 1 - 1 = 0 for every client, so the
    # "rescale so the least-suspicious client sits at 1" step would divide by zero.
    # Without the guard the weights would come back all-zero, and since the combiner is
    # a convex combination over ABSOLUTE weights, that would overwrite the global model
    # with zeros rather than merely ignoring the round.
    identical = sim(cids, {}, default=1.0)
    r4 = foolsgold_weights(identical, cids, normalise="none")
    _expect(r4["degenerate"], "identical directions must be flagged degenerate")
    _expect(all(abs(a) < 1e-12 for a in r4["alpha"].values()),
            f"the guard must fire on alpha == 0 everywhere, got {r4['alpha']}")
    # "none": all zeros is SAFE and correct — the combine is theta_t + sum(w_i * D_i),
    # so a zero weight vector leaves the global model exactly as broadcast.
    _expect(all(x == 0.0 for x in r4["weights"].values()) and r4["weight_sum"] == 0.0,
            f"'none' degenerate must be all zeros (a no-op round), got {r4['weights']}")
    # "sum": all zeros is NOT available — the caller would divide by sum(w) == 0.
    r4n = foolsgold_weights(identical, cids, normalise="sum")
    _expect(r4n["degenerate"] and abs(r4n["weight_sum"] - 1.0) < 1e-9
            and all(abs(x - 1.0 / len(cids)) < 1e-9 for x in r4n["weights"].values()),
            f"'sum' degenerate must be uniform and convex, got {r4n['weights']}")
    print(f"  [4] identical directions -> degenerate; 'none' = all-zero (round is a no-op), "
          f"'sum' = uniform {1 / len(cids):.3f}")

    # ---- 5. equal suspicion cancels: FoolsGold degrades to FedAvg, it never blocks all
    # The rescale divides by max(alpha), so if EVERY client is equally similar they are
    # all normalised back to 1 and the round comes out uniform through the NORMAL path
    # (not the guard). Worth pinning: it means FoolsGold cannot down-weight everyone, so
    # a round of uniformly-similar clients is silently plain FedAvg. Under IID partitions
    # benign clients are already fairly similar, which is exactly when this bites.
    for d in (0.9, 0.999):
        r5 = foolsgold_weights(sim(cids, {}, default=d), cids, normalise="sum")
        _expect(not r5["degenerate"], f"cs={d} is not the degenerate case (alpha > 0)")
        _expect(all(abs(x - 1.0 / len(cids)) < 1e-9 for x in r5["weights"].values()),
                f"cs={d}: equal suspicion must cancel to uniform, got {r5['weights']}")
    print("  [5] uniformly-similar clients (cs=0.9, 0.999) -> uniform via the normal path; "
          "FoolsGold degrades to FedAvg rather than blocking everyone")

    # ---- 6. what kappa actually does --------------------------------------------------
    # kappa scales the logit BEFORE the [0, 1] clip. Since weight == 0 iff kappa*L <= 0
    # iff L <= 0 (for kappa > 0), the EXCLUSION SET is kappa-invariant — kappa cannot
    # change who is thrown out, only how decisively the survivors are admitted. Honest
    # clients are already saturated at 1, so raising kappa lifts BORDERLINE clients,
    # making it more permissive at the margin rather than more aggressive.
    kappas = (0.25, 0.5, 1.0, 2.0, 4.0)

    # (a) borderline pair (alpha lands above the logit's zero crossing): share RISES.
    borderline = sim(cids, {frozenset({3, 4}): 0.6}, default=0.05)
    shares, prev = [], None
    excluded = None
    for kappa in kappas:
        r6 = foolsgold_weights(borderline, cids, kappa, normalise="sum")
        share = r6["weights"][3] + r6["weights"][4]
        shares.append((kappa, share))
        if prev is not None:
            _expect(share >= prev - 1e-9,
                    f"kappa={kappa}: borderline share fell to {share:.4f} from {prev:.4f}; "
                    f"kappa scales the logit, so it should lift clients above the crossing")
        prev = share
        zero = frozenset(c for c in cids if r6["weights"][c] <= 1e-12)
        if excluded is not None:
            _expect(zero == excluded,
                    f"kappa={kappa} changed the exclusion set to {sorted(zero)} from "
                    f"{sorted(excluded)} — kappa must not decide WHO is excluded")
        excluded = zero

    # (b) clearly-suspicious pair (alpha below the crossing): zero for every kappa.
    clear = sim(cids, {frozenset({3, 4}): 0.9}, default=0.05)
    for kappa in kappas:
        wk = foolsgold_weights(clear, cids, kappa, normalise="sum")["weights"]
        _expect(wk[3] <= 1e-12 and wk[4] <= 1e-12,
                f"kappa={kappa}: a clearly-colluding pair must stay excluded, got {wk[3]:.4g}")
    print("  [6] kappa scales the logit, it does not set the exclusion set: "
          "borderline share " + " -> ".join(f"{s:.3f}" for _, s in shares)
          + " over kappa " + "/".join(str(k) for k in kappas)
          + "; a clear sybil pair stays 0 throughout")

    # ---- 7. order independence ------------------------------------------------------
    a = foolsgold_weights(cs, [0, 1, 2, 3, 4], normalise="sum")["weights"]
    b = foolsgold_weights(cs, [4, 2, 0, 3, 1], normalise="sum")["weights"]
    _expect(a == b, f"result depends on cid order: {a} vs {b}")
    print("  [7] identical result for shuffled cid order")

    # ---- 8. edge cases ---------------------------------------------------------------
    _expect(foolsgold_weights({}, [])["weights"] == {}, "n=0 must return no weights")
    solo = foolsgold_weights({7: {}}, [7])["weights"]
    _expect(solo == {7: 1.0}, f"n=1 must return the single client at 1.0, got {solo}")
    # All-negative similarities: the reference convention takes a row max over the
    # off-diagonal AND an implicit 0, so v_i = 0 and every client is fully trusted.
    neg = foolsgold_weights(sim([0, 1, 2], {}, default=-0.8), [0, 1, 2], normalise="sum")
    _expect(not neg["degenerate"] and abs(sum(neg["weights"].values()) - 1.0) < 1e-9,
            "all-negative similarities must be a normal, non-degenerate round")
    _expect(all(abs(v) < 1e-12 for v in neg["max_cs"].values()),
            f"the implicit-zero diagonal must floor max_cs at 0, got {neg['max_cs']}")
    print("  [8] n=0 / n=1 / all-negative-similarity rows handled")

    # ---- 9. the strategy refuses an empty feature set ---------------------------------
    # Guards a silently disabled defense: an unmatched group would leave every history
    # the zero vector, every cosine 0, and every weight uniform.
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  [9] SKIPPED (needs torch): empty indicative-feature set must raise")
    else:
        from FL.strategies.foolsgold import FoolsGold
        sd = {"roi_heads.box_predictor.cls_score.weight": torch.zeros(3, 4),
              "roi_heads.box_predictor.cls_score.bias": torch.zeros(3)}
        names = FoolsGold._resolve_features(sd, "roi.cls_score")
        _expect(names == sorted(sd), f"cls_score group must match both tensors, got {names}")
        try:
            FoolsGold._resolve_features(sd, "not_a_group")
            _expect(False, "an unmatched feature group must raise, not track nothing")
        except ValueError as exc:
            _expect("matched no" in str(exc), f"unexpected error text: {exc}")
        print(f"  [9] feature group resolves to {len(names)} tensor(s); unknown group raises")

    # ---- 10. partial suppression exists, and "weight == 0" would miss it -------------
    # At cs = 0.6 the suspicious pair keeps a NON-ZERO weight that is nonetheless only
    # 27% of the 1/n fair share — nearly silenced, but not silenced. Defining "rejected"
    # as weight == 0 (the first version of this code) would have logged these clients as
    # KEPT, so the server's "attackers rejected 0/2" line would have read as though the
    # defense did nothing while it had in fact cut them to a quarter share.
    #
    # This is why FoolsGold reports suppression relative to the fair share instead. The
    # threshold itself lives in FL.strategies.foolsgold (SUPPRESSED_SHARE); the regime it
    # exists for is pinned here, torch-free.
    partial = foolsgold_weights(sim(cids, {frozenset({3, 4}): 0.6}, default=0.05), cids,
                                normalise="sum")
    wp, n = partial["weights"], len(cids)
    ratio = wp[3] * n
    _expect(wp[3] > 0.0, "fixture must produce a NON-zero weight, else it tests nothing")
    _expect(ratio < 0.5,
            f"fixture must land below half a fair share, got share_ratio={ratio:.3f}")
    _expect(all(wp[c] * n > 1.0 for c in (0, 1, 2)),
            "the honest clients must be above a fair share (they compensate)")
    print(f"  [10] partial suppression: w={wp[3]:.4f} is non-zero but only "
          f"{ratio:.2f} of a fair share — a 'weight == 0' rule would call this KEPT")

    print("\nAll FoolsGold smoke tests passed.")


if __name__ == "__main__":
    main()
