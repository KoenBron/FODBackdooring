"""Smoke test for norm-budgeted amplification (--partial-gamma-mode budgeted).

Run from project root:
    python smoketests/smoketest_norm_budget.py

Covers the three torch-free pieces of the feature:
  * FL.attack_protocols.amplify.allocate_norm_budget -- the rationing itself,
  * FL.attack_state per-client benign references (accumulate / mean / freeze / persist),
  * utils.select_clients.plastic_phase_attacker_coverage -- the startup halt's predicate.

The central invariant is that the realized submitted norm never exceeds the budget:

    ||Delta_submitted||^2 = ||Delta_unamp||^2 + sum_p (gamma_p^2 - 1) * e_p

which holds because partial_amplify perturbs ONLY carrier coordinates. Test 2 checks it
against a naive TOTAL-energy allocator, which overshoots -- the reason the implementation
rations the EXCESS instead.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FL.attack_protocols.amplify import allocate_norm_budget, partial_amplify
from FL.attack_state import AttackState
from utils.select_clients import plastic_phase_attacker_coverage


def _expect(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)


def realized_energy(gammas, energy, unamp_energy):
    """The identity the allocator is built on, evaluated directly."""
    return unamp_energy + sum((gammas[n] ** 2 - 1.0) * energy[n] for n in gammas)


def main():
    # ---- 1. Budget is never exceeded, across randomised shapes -------------------
    rng = random.Random(20260824)
    exhausted = 0
    for _ in range(500):
        n = rng.randint(1, 12)
        names = [f"t{i}" for i in range(n)]
        energy = {t: rng.uniform(1e-4, 5.0) for t in names}
        # Non-carrier energy pushes the unamplified norm above the carriers alone.
        unamp_energy = sum(energy.values()) + rng.uniform(0.0, 20.0)
        budget = math.sqrt(unamp_energy) * rng.uniform(1.0, 3.0)
        gamma_full = rng.uniform(1.0, 30.0)
        gammas, rep = allocate_norm_budget(
            [(t, 0.0) for t in names], energy, unamp_energy, budget, gamma_full)
        got = math.sqrt(realized_energy(gammas, energy, unamp_energy))
        _expect(got <= budget * (1 + 1e-9),
                f"budget exceeded: realized {got} > budget {budget}")
        _expect(abs(got - rep["predicted_l2"]) < 1e-6 * max(1.0, got),
                f"predicted {rep['predicted_l2']} != realized {got}")
        _expect(all(g >= 1.0 - 1e-12 for g in gammas.values()),
                f"allocator must never shrink a carrier: {gammas}")
        if rep["throttled_at"] is not None:
            exhausted += 1
    print(f"  budget respected over 500 random configs ({exhausted} hit exhaustion)")

    # ---- 2. Why EXCESS, not total energy: the naive version overshoots -----------
    # 3 carriers, e_p = 1 each; non-carrier energy 7 -> ||Delta_unamp||^2 = 10.
    energy = {"a": 1.0, "b": 1.0, "c": 1.0}
    unamp_energy = 10.0
    budget = math.sqrt(unamp_energy)          # B = 1: submit at the unamplified norm
    gammas, rep = allocate_norm_budget(
        [("a", 0), ("b", 0), ("c", 0)], energy, unamp_energy, budget, gamma_full=10.0)
    got = math.sqrt(realized_energy(gammas, energy, unamp_energy))
    _expect(abs(got - budget) < 1e-12, f"B=1 must land exactly on budget, got {got}")
    _expect(all(abs(g - 1.0) < 1e-12 for g in gammas.values()),
            f"B=1 leaves zero excess -> every gamma must be 1.0, got {gammas}")
    # The naive TOTAL-energy allocator: carrier budget = budget^2 - nonC, spend it, then
    # let later carriers ride at gamma=1 (their e_p never reserved).
    naive_carrier_budget = budget ** 2 - 7.0          # == 3.0
    naive = {"a": math.sqrt(naive_carrier_budget / 1.0), "b": 1.0, "c": 1.0}
    naive_norm = math.sqrt(realized_energy(naive, energy, unamp_energy))
    _expect(naive_norm > budget * 1.05,
            "the naive total-energy allocator was expected to overshoot")
    print(f"  B=1 -> all gamma=1, realized {got:.4f} == budget {budget:.4f}; "
          f"naive total-energy variant would overshoot to {naive_norm:.4f}")

    # ---- 3. Exhaustion: one throttled carrier, the rest exactly 1.0 --------------
    energy = {"hi": 2.0, "mid": 3.0, "lo": 4.0}
    unamp_energy = 9.0 + 6.0                   # carriers 9 + non-carriers 6
    budget = math.sqrt(unamp_energy) * 1.2
    gammas, rep = allocate_norm_budget(
        [("hi", 0), ("mid", 0), ("lo", 0)], energy, unamp_energy, budget, gamma_full=4.0)
    _expect(rep["throttled_at"] is not None, "expected the budget to run out")
    order = ["hi", "mid", "lo"]
    t_idx = order.index(rep["throttled_at"])
    _expect(all(abs(gammas[order[i]] - 4.0) < 1e-12 for i in range(t_idx)),
            f"carriers before the throttle point must get full gamma: {gammas}")
    _expect(all(abs(gammas[order[i]] - 1.0) < 1e-12 for i in range(t_idx + 1, 3)),
            f"carriers after the throttle point must be unamplified: {gammas}")
    got = math.sqrt(realized_energy(gammas, energy, unamp_energy))
    _expect(abs(got - budget) < 1e-9, f"exhaustion must land ON the budget, got {got}")
    _expect(rep["n_full"] == t_idx and rep["n_throttled"] == 1,
            f"counts wrong: {rep['n_full']} full, {rep['n_throttled']} throttled")
    print(f"  exhaustion at '{rep['throttled_at']}': {rep['n_full']} full, 1 throttled "
          f"(gamma={gammas[rep['throttled_at']]:.4f}), {rep['n_unamplified']} unamplified")

    # ---- 4. Rank order is the priority (high-rank carriers get the full gamma) ---
    energy = {"first": 5.0, "second": 5.0}
    unamp_energy = 10.0
    gammas, _ = allocate_norm_budget(
        [("first", 0), ("second", 0)], energy, unamp_energy,
        budget=math.sqrt(unamp_energy) * 1.5, gamma_full=3.0)
    _expect(gammas["first"] > gammas["second"],
            f"the higher-ranked carrier must be funded first: {gammas}")
    # ... and reversing the order reverses who gets funded.
    rev, _ = allocate_norm_budget(
        [("second", 0), ("first", 0)], energy, unamp_energy,
        budget=math.sqrt(unamp_energy) * 1.5, gamma_full=3.0)
    _expect(rev["second"] > rev["first"], f"rank order not respected on reversal: {rev}")
    print("  rank order drives allocation (and reversing it reverses the outcome)")

    # ---- 5. Generous budget -> every carrier at the full gamma -------------------
    gammas, rep = allocate_norm_budget(
        [("a", 0), ("b", 0)], {"a": 1.0, "b": 1.0}, 4.0, budget=1000.0, gamma_full=2.0)
    _expect(rep["throttled_at"] is None and rep["n_full"] == 2,
            f"a large budget should fund every carrier: {rep}")
    _expect(all(abs(g - 2.0) < 1e-12 for g in gammas.values()), f"{gammas}")
    print("  generous budget -> all carriers at full gamma, nothing throttled")

    # ---- 6. Deficit: unamplified already over budget -> no amplification ---------
    gammas, rep = allocate_norm_budget(
        [("a", 0), ("b", 0)], {"a": 1.0, "b": 1.0},
        unamplified_energy=100.0, budget=5.0, gamma_full=9.0)   # 10.0 > 5.0
    _expect(rep["deficit"] is True, f"expected a deficit, got {rep}")
    _expect(all(abs(g - 1.0) < 1e-12 for g in gammas.values()),
            f"deficit must leave every carrier unamplified (never shrink): {gammas}")
    _expect(rep["excess_spent"] == 0.0 and rep["throttled_at"] is None, f"{rep}")
    print(f"  deficit (unamp 10.0 > budget 5.0) -> all gamma=1, excess_budget="
          f"{rep['excess_budget']:.1f}")

    # ---- 7. partial_amplify: scalar and uniform-dict gamma agree -----------------
    theta_t = {"c1": 0.0, "c2": 10.0, "other": 3.0}
    theta_mal = {"c1": 2.0, "c2": 14.0, "other": 5.0}
    carrying = {"c1", "c2"}
    scalar = partial_amplify(theta_t, theta_mal, carrying, 3.0)
    as_dict = partial_amplify(theta_t, theta_mal, carrying, {"c1": 3.0, "c2": 3.0})
    _expect(scalar == as_dict, f"scalar/dict mismatch: {scalar} vs {as_dict}")
    mixed = partial_amplify(theta_t, theta_mal, carrying, {"c1": 3.0})   # c2 defaults to 1.0
    _expect(mixed["c1"] == 6.0 and mixed["c2"] == 14.0 and mixed["other"] == 5.0,
            f"per-tensor gamma / default-1.0 wrong: {mixed}")
    print("  partial_amplify: scalar == uniform dict; missing key defaults to gamma=1")

    # ---- 8. Per-client benign references: never pooled ---------------------------
    AttackState.reset()
    AttackState.accumulate_benign_norms(0, [1.0, 3.0])      # cid 0 attacks once  -> 2 samples
    AttackState.accumulate_benign_norms(1, [10.0, 20.0])    # cid 1 attacks twice -> 4 samples
    AttackState.accumulate_benign_norms(1, [30.0, 40.0])
    AttackState.freeze_benign_refs()
    _expect(abs(AttackState.get_benign_ref(0) - 2.0) < 1e-12,
            f"cid 0 mean wrong: {AttackState.get_benign_ref(0)}")
    _expect(abs(AttackState.get_benign_ref(1) - 25.0) < 1e-12,
            f"cid 1 mean wrong: {AttackState.get_benign_ref(1)}")
    _expect(AttackState.get_benign_ref(2) is None,
            "a client that never attacked must have NO reference (no pooled fallback)")
    _expect(AttackState.benign_ref_sample_counts() == {0: 2, 1: 4},
            f"sample counts wrong: {AttackState.benign_ref_sample_counts()}")
    print("  per-client refs: cid0=2.0 (2 anchors), cid1=25.0 (4 anchors), cid2=None")

    # ---- 9. Persistence round-trip keeps the per-client structure ----------------
    snap = AttackState.to_dict()
    AttackState.reset()
    _expect(AttackState.get_benign_ref(1) is None, "reset should clear the refs")
    AttackState.from_dict(snap)
    _expect(abs(AttackState.get_benign_ref(1) - 25.0) < 1e-12,
            f"per-client ref lost across persistence: {AttackState.plastic_benign_ref}")
    _expect(AttackState.benign_ref_sample_counts() == {0: 2, 1: 4},
            "raw per-client samples lost across persistence")
    # A legacy POOLED snapshot must be dropped, not misread as per-client.
    AttackState.reset()
    AttackState.from_dict({"carrying_set": {},
                           "plastic": {"benign_norms": [1.0, 2.0], "benign_norm_ref": 1.5}})
    _expect(AttackState.plastic_benign_ref == {} and AttackState.plastic_benign_norms == {},
            "legacy pooled baseline must be discarded, not adopted")
    print("  persistence round-trip preserves per-client refs; legacy pooled state dropped")

    # ---- 10. Startup coverage predicate (drives the hard halt) -------------------
    schedule = {11: [0], 14: [1], 17: [2], 20: [3], 23: [0], 26: [1]}
    attackers = [0, 1, 2, 3]
    covered, missing = plastic_phase_attacker_coverage(schedule, attackers, 4)
    _expect(covered == [0, 1, 2, 3] and missing == [],
            f"4 injections should cover all 4 attackers: {covered}, {missing}")
    covered, missing = plastic_phase_attacker_coverage(schedule, attackers, 2)
    _expect(covered == [0, 1] and missing == [2, 3],
            f"2 injections should cover only cids 0,1: {covered}, {missing}")
    # Boundary: the round that TRIGGERS the freeze is still plastic, so cid 2 (the 3rd
    # injection) IS covered at N=3.
    covered, missing = plastic_phase_attacker_coverage(schedule, attackers, 3)
    _expect(covered == [0, 1, 2] and missing == [3],
            f"the freeze-triggering injection must count as covered: {covered}, {missing}")
    # More plastic rounds than the schedule provides -> whatever it covers.
    covered, missing = plastic_phase_attacker_coverage({11: [0]}, attackers, 99)
    _expect(covered == [0] and missing == [1, 2, 3], f"{covered}, {missing}")
    # Two attackers in one round = two injections.
    covered, missing = plastic_phase_attacker_coverage({5: [0, 1], 6: [2]}, attackers, 2)
    _expect(covered == [0, 1] and missing == [2, 3],
            f"multi-attacker round should count as 2 injections: {covered}, {missing}")
    print("  coverage predicate: boundary, under-coverage, and multi-attacker rounds OK")

    AttackState.reset()
    print("  PASS: norm-budget allocation, per-client benign refs, persistence, "
          "and startup coverage all OK")


if __name__ == "__main__":
    main()
