"""Smoke test for label-skew (non-IID) partitioning at equal partition size.

Run from project root:
    python smoketests/smoketest_noniid.py

utils/noniid.py touches numpy in exactly one place (the Dirichlet draw and its RNG), and
nowhere touches torch, so this runs without a GPU. Asserts:

  1. The proxy rules: max_area aggregates area PER CLASS (five medium boats beat one large
     person), count_majority counts instances, first takes annotation order, ties break on
     the lowest class index, un-annotated images go to background, and a missing `area`
     field falls back to the bbox.
  2. stochastic_majority collapses to max_area at p=1 and is a strict mixture at p<1 —
     the reason image_label takes an rng at all.
  3. THE INVARIANT: every client gets exactly len(labels)//n_clients images, the partition
     is disjoint, and it covers everything bar the < n_clients dropped remainder.
  4. Both attacker modes. 'matched' gives identical label compositions while holding
     DIFFERENT images (the property the plastic phase's pooled div/b assumes);
     'independent' — the DEFAULT — gives each attacker its own draw, so colluders do not
     share a distribution and FoolsGold is not handed a partitioner artifact.
  5. The attacker target-class floor (matched mode only) is met by every attacker
     without disturbing sizes, and an impossible floor raises rather than under-filling.
  6. Skew is real and monotone in alpha: small alpha concentrates each client on a few
     classes, large alpha approaches the global marginal. Measured as drift from the
     marginal, so it does not depend on which classes happen to be big.
  7. Determinism: same seed -> identical partition and labels, even across interleaved
     calls on other seeds. Nothing persists the partition, so this is what makes an
     interrupted run resume onto the same data split.
  8. Supply exhaustion is survivable — with one class dominating, every client STILL gets
     its exact size, and the lost composition fidelity shows up as drift_tv rather than
     being hidden.
  9. Dealing ROUND-ROBIN rather than filling clients one at a time: when a class is
     scarce and in demand, every client that wants it gets some, instead of the first few
     clients dealt taking the whole supply.
 10. The IID path is BIT-IDENTICAL to the pre-refactor flat-permutation + array_split
     behaviour, so every existing run still reproduces.
 11. A randomised sweep over client/bucket counts, alpha, attacker sets, BOTH attacker
     modes, target floors and four population shapes (including EMPTY buckets): every
     configuration either holds all of the above invariants or refuses outright.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.noniid import (
    allocate_label_skew,
    build_image_labels,
    composition_rows,
    global_marginal,
    image_label,
)


def _expect(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)


def ann(cat, area):
    return {"category_id": cat, "area": area, "bbox": [0, 0, 1, area]}


def composition(part, labels, n_buckets):
    counts = [0] * n_buckets
    for i in part:
        counts[labels[i]] += 1
    return counts


def tv(counts, reference):
    """Total-variation distance between a realised count vector and a proportion vector."""
    n = sum(counts)
    return 0.5 * sum(abs(c / n - q) for c, q in zip(counts, reference))


def main():
    import numpy as np
    rng = np.random.default_rng(0)
    dense = {10: 0, 20: 1, 30: 2}          # category_id -> dense index

    # ---- 1. the proxy rules -----------------------------------------------------
    # One large person (cat 10, area 500) vs five medium boats (cat 20, area 120 each).
    mixed = [ann(10, 500.0)] + [ann(20, 120.0)] * 5
    _expect(image_label(mixed, "max_area", rng, dense, 3) == 1,
            "max_area must sum area per class (600 boat > 500 person)")
    _expect(image_label(mixed, "count_majority", rng, dense, 3) == 1,
            "count_majority: 5 boats > 1 person")
    _expect(image_label(mixed, "first", rng, dense, 3) == 0,
            "first must take the first-listed annotation")
    # One large person vs two small boats: now the rules disagree, which is the point.
    lopsided = [ann(10, 900.0), ann(20, 10.0), ann(20, 10.0)]
    _expect(image_label(lopsided, "max_area", rng, dense, 3) == 0, "max_area -> person")
    _expect(image_label(lopsided, "count_majority", rng, dense, 3) == 1,
            "count_majority -> boat")
    _expect(image_label([ann(20, 5.0), ann(10, 5.0)], "max_area", rng, dense, 3) == 0,
            "an exact tie must break on the LOWEST dense index, not annotation order")
    _expect(image_label([], "max_area", rng, dense, 3) == 3,
            "an image with no annotations must land in the background bucket")
    _expect(image_label([ann(99, 5.0)], "max_area", rng, dense, 3) == 3,
            "an image with only unknown categories must land in the background bucket")
    # bbox fallback when `area` is absent: 4 x 30 = 120 beats a single area=50 object.
    no_area = [{"category_id": 20, "bbox": [0, 0, 4, 30]}, ann(10, 50.0)]
    _expect(image_label(no_area, "max_area", rng, dense, 3) == 1,
            "a missing `area` must fall back to bbox w*h")
    print("  [1] proxy rules: max_area aggregates per class; ties/background/bbox-fallback OK")

    # ---- 2. stochastic_majority --------------------------------------------------
    r = np.random.default_rng(7)
    always = [image_label(lopsided, "stochastic_majority", r, dense, 3, 1.0) for _ in range(50)]
    _expect(set(always) == {0}, "p=1.0 must reduce exactly to max_area")
    r = np.random.default_rng(7)
    draws = [image_label(lopsided, "stochastic_majority", r, dense, 3, 0.5)
             for _ in range(200)]
    _expect(set(draws) == {0, 1},
            f"p=0.5 must yield both the majority and a present alternative, got {set(draws)}")
    print(f"  [2] stochastic_majority: p=1 -> pure max_area; p=0.5 -> "
          f"{100 * draws.count(0) / len(draws):.0f}% majority")

    # ---- 3. the equal-size invariant ---------------------------------------------
    n_buckets, n_clients = 6, 7            # 7 does not divide 600 -> exercises the remainder
    labels = [k for k in range(n_buckets) for _ in range(100)]
    parts, _ = allocate_label_skew(labels, n_buckets, n_clients, [0, 1], 0.5, 42)
    size = len(labels) // n_clients
    _expect(all(len(p) == size for p in parts),
            f"every client must hold exactly {size} images, got {[len(p) for p in parts]}")
    flat = [i for p in parts for i in p]
    _expect(len(flat) == len(set(flat)), "partitions must be disjoint")
    _expect(len(flat) == size * n_clients, "assigned count must be size * n_clients")
    _expect(set(flat) <= set(range(len(labels))), "indices must come from the dataset")
    print(f"  [3] {n_clients} clients x {size} images, disjoint, "
          f"{len(labels) - len(flat)} remainder dropped")

    # ---- 4. the two attacker modes -------------------------------------------------
    # 'matched' gives every attacker the SAME composition (what the plastic phase's pooled
    # div/b assumes). 'independent' (the default) gives each its own draw, so colluders no
    # longer share a label distribution — which is what stops FoolsGold being handed a
    # correlated-direction signal that is an artifact of the partitioner rather than of the
    # attack. Both are pinned: a silent flip either way changes what the defenses measure.
    attackers = [0, 1, 3]
    m_parts, _ = allocate_label_skew(labels, n_buckets, n_clients, attackers, 0.3, 42,
                                     match_attackers=True)
    m_comps = [composition(m_parts[a], labels, n_buckets) for a in attackers]
    for c in m_comps[1:]:
        _expect(c == m_comps[0],
                f"matched: compositions must be identical: {m_comps[0]} vs {c}")
    benign = [composition(m_parts[c], labels, n_buckets)
              for c in range(n_clients) if c not in attackers]
    _expect(any(b != m_comps[0] for b in benign),
            "benign clients must not all share it too, or the test is vacuous")
    _expect(len({tuple(sorted(m_parts[a])) for a in attackers}) == len(attackers),
            "matched composition must not mean matched IMAGES")

    i_parts, _ = allocate_label_skew(labels, n_buckets, n_clients, attackers, 0.3, 42)
    i_comps = [composition(i_parts[a], labels, n_buckets) for a in attackers]
    _expect(any(c != i_comps[0] for c in i_comps[1:]),
            f"independent (the DEFAULT) must NOT produce identical compositions: {i_comps}")
    _expect(all(len(p) == size for p in i_parts),
            "independent draws must still hold the equal-size invariant")
    flat_i = [i for p in i_parts for i in p]
    _expect(len(flat_i) == len(set(flat_i)), "independent partitions must stay disjoint")
    print(f"  [4] matched -> all attackers {m_comps[0]}; independent (default) -> "
          f"{len({tuple(c) for c in i_comps})} distinct compositions")

    # ---- 5. the target-class floor ------------------------------------------------
    parts, _ = allocate_label_skew(labels, n_buckets, n_clients, attackers, 0.3, 42,
                                   target_idx=2, target_min_count=30, match_attackers=True)
    for a in attackers:
        got = composition(parts[a], labels, n_buckets)[2]
        _expect(got >= 30, f"attacker {a} got {got} target-class images, floor was 30")
    _expect(all(len(p) == size for p in parts), "the floor must not disturb partition sizes")
    try:
        allocate_label_skew(labels, n_buckets, n_clients, attackers, 0.3, 42,
                            target_idx=2, target_min_count=999, match_attackers=True)
        _expect(False, "an unsatisfiable target floor must raise, not silently under-fill")
    except ValueError as exc:
        _expect("target class" in str(exc), f"unexpected error text: {exc}")
    print("  [5] target floor honoured by every attacker; unsatisfiable floor raises")

    # ---- 6. skew is real and monotone, and large alpha is genuinely IID -------------
    # Deliberately a NON-UNIFORM marginal (48% / 24% / ... ), because a uniform one makes
    # this test vacuous: a flat Dir(alpha * 1_K) base measure would also look IID there,
    # while on skewed data it asks every client for an equal share of every class and
    # never converges to the dataset's mix. The base measure has to be the marginal.
    # Averaged over seeds: a partition consumes the WHOLE dataset, so at small alpha
    # supply always binds and a single draw is noisy enough to invert two adjacent alphas.
    skewed_pop = [0] * 600 + [1] * 300 + [2] * 150 + [3] * 100 + [4] * 60
    marginal = global_marginal(skewed_pop, 5)
    _expect(max(marginal) > 3 * min(marginal), "fixture marginal must be clearly non-uniform")
    drifts = {}
    for alpha in (0.05, 0.5, 50.0):
        per_seed = []
        for seed in range(5):
            parts, _ = allocate_label_skew(skewed_pop, 5, 20, [], alpha, seed)
            per_seed.append(sum(tv(composition(p, skewed_pop, 5), marginal)
                                for p in parts) / 20)
        drifts[alpha] = sum(per_seed) / len(per_seed)
    _expect(drifts[0.05] > drifts[0.5] > drifts[50.0],
            f"skew must fall monotonically with alpha, got {drifts}")
    # Not exactly 0 at alpha=50: with 60 images a client, plain multinomial sampling noise
    # puts a floor of a few percent on the TV distance.
    _expect(drifts[50.0] < 0.15,
            f"alpha=50 must converge on the dataset marginal (true IID), got "
            f"drift {drifts[50.0]:.3f} — the Dirichlet base measure is not the marginal")
    _expect(drifts[0.05] > 0.30, f"alpha=0.05 must be strongly skewed, got {drifts[0.05]:.3f}")

    # The above alone does NOT pin the base measure: a partition is exhaustive, so at large
    # alpha the realised compositions average to the marginal whatever was requested. What
    # a flat base measure actually breaks is FIDELITY — every client asks for an equal
    # share of every class, the allocator spends the draw fighting the data, and
    # requested-vs-realised drift stays high at every alpha (~0.35 vs ~0.09 here).
    fidelity = []
    for seed in range(5):
        parts, requested = allocate_label_skew(skewed_pop, 5, 20, [], 50.0, seed)
        rows = composition_rows(parts, skewed_pop, requested, list("abcde"), [])
        fidelity.append(sum(r["drift_tv"] for r in rows) / len(rows))
    fidelity = sum(fidelity) / len(fidelity)
    _expect(fidelity < 0.20,
            f"at alpha=50 clients must RECEIVE what they requested (drift_tv={fidelity:.3f}); "
            f"a high value means the Dirichlet base measure is not the class marginal")
    print("  [6] mean drift from a NON-uniform global marginal: "
          + ", ".join(f"a={a}:{d:.3f}" for a, d in drifts.items())
          + f"; requested-vs-realised at a=50: {fidelity:.3f}")

    # ---- 7. determinism ------------------------------------------------------------
    # Nothing persists the partition: a resume reproduces it by re-deriving from the seed.
    # This check is therefore the ONLY thing standing between an interrupted run and
    # silently continuing on a different data split.
    #
    # a2 is computed AFTER an intervening call on a different seed, so a partitioner that
    # leaked state into a module-level RNG (or depended on dict/set iteration order) would
    # diverge here rather than only under a fresh process.
    a1, _ = allocate_label_skew(labels, n_buckets, n_clients, attackers, 0.5, 1234)
    a3, _ = allocate_label_skew(labels, n_buckets, n_clients, attackers, 0.5, 1235)
    a2, _ = allocate_label_skew(labels, n_buckets, n_clients, attackers, 0.5, 1234)
    _expect(a1 == a2, "the same seed must reproduce the partition exactly, element for element")
    _expect(a1 != a3, "a different seed must produce a different partition")
    # Labels are drawn on their own stream, so re-deriving them must be stable too.
    lab_a = build_image_labels(list(range(50)), {i: [ann(10, 5.0), ann(20, 9.0)] for i in range(50)},
                               [10, 20], "stochastic_majority", 99, 0.5)
    lab_b = build_image_labels(list(range(50)), {i: [ann(10, 5.0), ann(20, 9.0)] for i in range(50)},
                               [10, 20], "stochastic_majority", 99, 0.5)
    _expect(lab_a == lab_b, "the label stream must be reproducible from the seed too")
    print("  [7] same seed -> identical partition and labels, across interleaved calls")

    # ---- 8. supply exhaustion still fills every client -----------------------------
    # 90% of the images in one class: most clients cannot get the composition they drew.
    skewed = [0] * 900 + [1] * 50 + [2] * 50
    parts, requested = allocate_label_skew(skewed, 3, 10, [0], 0.1, 5)
    _expect(all(len(p) == 100 for p in parts),
            f"the size invariant must survive exhaustion, got {[len(p) for p in parts]}")
    rows = composition_rows(parts, skewed, requested, ["a", "b", "c"], [0])
    _expect(max(r["drift_tv"] for r in rows) > 0.0,
            "exhaustion must SHOW UP as drift rather than being hidden")
    print(f"  [8] under exhaustion every client still holds 100 images; "
          f"max drift_tv={max(r['drift_tv'] for r in rows):.3f}")

    # ---- 9. round-robin dealing shares an over-demanded class out ------------------
    # 1000 images, class 1 scarce (150). At small alpha several clients each want almost
    # nothing else, so their combined demand (>= 3 x 100) exceeds the 150 available.
    # Round-robin splits the shortfall between them; filling clients one at a time would
    # let whoever is dealt first drain the pool and leave the rest with zero.
    #
    # The seed is SEARCHED rather than hard-coded: which clients draw an extreme share
    # depends on the RNG stream, so pinning a seed would make this test depend on the
    # numpy version. The assertion is on the outcome, not on any particular draw.
    scarce = [0] * 850 + [1] * 150
    fixture = None
    for seed in range(40):
        parts, requested = allocate_label_skew(scarce, 2, 10, [], 0.15, seed)
        greedy = [c for c in range(10) if requested[c][1] > 0.7]
        if len(greedy) >= 3:                       # >= 300 demanded against 150 available
            fixture = (seed, parts, greedy)
            break
    _expect(fixture is not None,
            "no seed in 0..39 produced >=3 clients demanding the scarce class — the "
            "fixture cannot exercise contention")
    seed, parts, greedy = fixture
    served = [c for c in greedy if composition(parts[c], scarce, 2)[1] > 0]
    _expect(len(served) == len(greedy),
            f"seed={seed}: only {len(served)}/{len(greedy)} clients demanding the scarce "
            f"class got any — the deal is starving whoever comes last")
    print(f"  [9] over-demanded class shared out (seed={seed}): "
          f"{len(served)}/{len(greedy)} contending clients served")

    # ---- 10. the IID path is unchanged -----------------------------------------------
    # Reproduce the OLD behaviour literally (flat seeded permutation + array_split) and
    # require the new per-client-list builder to agree element for element.
    import random as _random
    from utils.global_settings import FLSettings
    from utils.partitioning import build_iid_partitions
    FLSettings.GUARANTEE_TARGET_OBJECTS = False
    for seed, n_items, n_cl in ((4, 600, 7), (1234, 11540, 20)):
        FLSettings.PARTITIONING_SEED = seed
        old_idx = list(range(n_items))
        _random.Random(seed).shuffle(old_idx)
        old = [c.tolist() for c in np.array_split(np.asarray(old_idx), n_cl)]
        _expect(old == build_iid_partitions(n_items, n_cl),
                f"the IID path changed for seed={seed} n={n_items}")
    print("  [10] IID partitions bit-identical to the pre-refactor permutation + array_split")

    # ---- 11. randomised property sweep ---------------------------------------------
    # The hand-built fixtures above each probe one behaviour. This sweeps the parameter
    # space instead — client/bucket counts, alpha, attacker sets, target floors, and four
    # population shapes including ones with EMPTY buckets (a declared category that occurs
    # in no image, which drives the 1e-9 concentration floor). Every configuration must
    # either satisfy every invariant or refuse outright; silently returning a malformed
    # partition is the failure this catches.
    import random as _r
    rng2 = _r.Random(20260827)
    passed = refused = 0
    for _ in range(200):
        n_cl = rng2.randint(2, 25)
        n_bk = rng2.randint(2, 15)
        att = rng2.sample(range(n_cl), rng2.randint(0, max(0, n_cl - 1)))
        alpha = rng2.choice([0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 50.0])
        shape = rng2.choice(["uniform", "power", "dominant", "sparse"])
        sizes = []
        for k in range(n_bk):
            if shape == "uniform":    sizes.append(rng2.randint(20, 120))
            elif shape == "power":    sizes.append(max(1, int(400 / (k + 1))))
            elif shape == "dominant": sizes.append(800 if k == 0 else rng2.randint(1, 30))
            else:                     sizes.append(0 if rng2.random() < 0.35 else rng2.randint(5, 200))
        pop = [k for k, n in enumerate(sizes) for _ in range(n)]
        if len(pop) < n_cl * 2:
            continue
        tgt = rng2.choice([None] + [k for k, n in enumerate(sizes) if n > 0])
        # The floor only has an enforcement point on the matched path, so pair them.
        matched = rng2.random() < 0.5
        frac = rng2.choice([0.0, 0.1, 0.3]) if matched else 0.0
        try:
            parts, req = allocate_label_skew(pop, n_bk, n_cl, att, alpha,
                                             rng2.randrange(10**6), target_idx=tgt,
                                             target_min_fraction=frac,
                                             match_attackers=matched)
        except ValueError:
            refused += 1                    # unsatisfiable floor or supply — by design
            continue
        size_ = len(pop) // n_cl
        flat_ = [i for p in parts for i in p]
        _expect(all(len(p) == size_ for p in parts), f"unequal sizes ({shape}, a={alpha})")
        _expect(len(flat_) == len(set(flat_)), f"overlapping partitions ({shape}, a={alpha})")
        _expect(set(flat_) <= set(range(len(pop))), f"index out of range ({shape}, a={alpha})")
        if att and matched:
            comps_ = [composition(parts[a], pop, n_bk) for a in att]
            _expect(all(c == comps_[0] for c in comps_),
                    f"matched attacker compositions diverged ({shape}, a={alpha})")
            if tgt is not None and frac > 0:
                need = math.ceil(frac * size_)
                _expect(comps_[0][tgt] >= need,
                        f"target floor violated: {comps_[0][tgt]} < {need} ({shape}, a={alpha})")
        for r_ in composition_rows(parts, pop, req, [f"c{k}" for k in range(n_bk)], att):
            d_ = r_["drift_tv"]
            _expect(d_ == d_ and 0.0 <= d_ <= 1.0 + 1e-9,
                    f"drift_tv not a finite proportion: {d_} ({shape}, a={alpha})")
        passed += 1
    _expect(passed > 100 and refused > 0,
            f"sweep must exercise both paths, got {passed} accepted / {refused} refused")
    print(f"  [11] randomised sweep: {passed} configurations hold every invariant, "
          f"{refused} correctly refused")

    print("\nAll non-IID partitioning smoke tests passed.")


if __name__ == "__main__":
    main()
