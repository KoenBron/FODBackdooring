"""Smoke test for the plastic-phase carrying-set accumulator (Partial protocol).

Run from project root:
    python smoketests/smoketest_plastic_phase.py

The plastic-phase core is pure bookkeeping over name->float score dicts, so this
test drives it with plain floats and needs NEITHER torch NOR a GPU (same reason the
selector and AttackState stay dependency-light). It exercises the three orthogonal
pieces the plastic phase is built from:

  * FL.attack_state.AttackState   -- pooled div/b accumulator + freeze + persistence
  * FL.scoring.l2_based.select_by_mode -- the ONE cutoff/topk selector, reused by both
                                          the per-round path and the freeze

Asserts:
  1. Empty state: plastic_means() is ({}, {}), count 0, not frozen.
  2. Pooling is additive and order-independent -- interleaving two attackers' rounds
     gives the same sums as any other order, and count bumps once per round. This is
     the "pool across attackers (IID)" guarantee.
  3. Averaging DENOISES a single-round fluke: on the fluke's own round top-k=1 would
     pick the fluke, but on the pooled mean top-k=1 picks the true carrier. This is
     the whole point of the plastic phase.
  4. Freeze reuses select_by_mode for BOTH modes: topk returns exactly k; cutoff is
     variable-size. Same function the per-round path calls.
  5. Persistence round-trip MID-WINDOW (to_dict -> reset -> from_dict) preserves the
     sums, count, frozen flag and frozen set -- so a 4h cluster shift landing during
     the plastic phase resumes instead of freezing on a truncated window.
  6. Freeze-threshold semantics (mirrors _resolve_carrying): the freeze fires exactly
     when the pooled count reaches N, and no further accumulation happens afterwards.
  7. reset() clears every plastic field.
"""
import os
import sys

# Make `python smoketests/smoketest_plastic_phase.py` work from the project root
# without a preset PYTHONPATH (standalone in CI / the sandbox).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from FL.attack_state import AttackState
from FL.scoring.l2_based import select_by_mode


def _expect(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)


# Three plastic rounds. 'carrier' is a consistent high-divergence carrier; 'fluke'
# is quiet except for a single big spike in round 2; 'noise' hovers near its floor.
# Each entry is (div, b) with div/b keyed identically (the eligible-tensor set).
ROUNDS = [
    ({"carrier": 0.9, "fluke": 0.1, "noise": 0.2}, {"carrier": 0.1, "fluke": 0.1, "noise": 0.15}),
    ({"carrier": 0.8, "fluke": 0.9, "noise": 0.1}, {"carrier": 0.1, "fluke": 0.1, "noise": 0.15}),
    ({"carrier": 1.0, "fluke": 0.1, "noise": 0.2}, {"carrier": 0.1, "fluke": 0.1, "noise": 0.15}),
]


def _accumulate_all(rounds):
    for div, b in rounds:
        AttackState.accumulate_plastic(div, b)


def main():
    # 1. Empty state.
    AttackState.reset()
    md, mb = AttackState.plastic_means()
    _expect(md == {} and mb == {}, f"empty means should be ({{}}, {{}}), got ({md}, {mb})")
    _expect(AttackState.plastic_count == 0, "empty count should be 0")
    _expect(not AttackState.is_plastic_frozen(), "empty state should not be frozen")

    # 2. Pooling additive + order-independent (== attacker-order-independent).
    AttackState.reset()
    _accumulate_all(ROUNDS)
    sums_forward = dict(AttackState.plastic_div_sum)
    AttackState.reset()
    _accumulate_all(list(reversed(ROUNDS)))
    _expect(AttackState.plastic_div_sum == sums_forward, "div sums must be order-independent (pooling)")
    _expect(AttackState.plastic_count == 3, f"count should be 3, got {AttackState.plastic_count}")
    _expect(abs(AttackState.plastic_div_sum["carrier"] - 2.7) < 1e-9, "carrier div sum should be 0.9+0.8+1.0=2.7")

    # 3. Averaging denoises the fluke. Contrast the fluke's OWN round against the pool.
    AttackState.reset()
    AttackState.accumulate_plastic(*ROUNDS[1])  # the spike round alone
    md1, mb1 = AttackState.plastic_means()
    solo = {n for n, _ in select_by_mode(md1, mb1, "topk", 1.0, 1)}
    _expect(solo == {"fluke"}, f"on the spike round alone, top-1 should be the fluke, got {solo}")

    AttackState.reset()
    _accumulate_all(ROUNDS)  # all three pooled
    md, mb = AttackState.plastic_means()
    pooled = {n for n, _ in select_by_mode(md, mb, "topk", 1.0, 1)}
    _expect(pooled == {"carrier"}, f"pooled top-1 should be the carrier (fluke averaged down), got {pooled}")
    # signed rho: carrier 0.8, fluke 0.267, noise ~0.017 -> ordering carrier > fluke > noise
    signed = {n: md[n] - mb[n] for n in md}
    _expect(signed["carrier"] > signed["fluke"] > signed["noise"], f"denoised ordering wrong: {signed}")

    # 4. Freeze reuses select_by_mode for both modes.
    topk2 = select_by_mode(md, mb, "topk", 1.0, 2)
    _expect(len(topk2) == 2, "topk=2 must return exactly 2 (fixed-size)")
    _expect([n for n, _ in topk2] == ["carrier", "fluke"], f"topk=2 order wrong: {topk2}")
    cutoff = {n for n, _ in select_by_mode(md, mb, "cutoff", 1.0, None)}
    # cutoff keeps every bucket with mean div > mean b (variable size): carrier, fluke, noise all clear.
    _expect(cutoff == {"carrier", "fluke", "noise"}, f"cutoff selection wrong: {cutoff}")

    # 5. Persistence round-trip MID-WINDOW (simulate a shift after 2 of 3 rounds).
    AttackState.reset()
    AttackState.accumulate_plastic(*ROUNDS[0])
    AttackState.accumulate_plastic(*ROUNDS[1])
    snap = json.loads(json.dumps(AttackState.to_dict()))  # force a real JSON round-trip
    AttackState.reset()
    AttackState.from_dict(snap)
    _expect(AttackState.plastic_count == 2, "shift must preserve count=2")
    _expect(abs(AttackState.plastic_div_sum["carrier"] - 1.7) < 1e-9, "shift must preserve div sum (0.9+0.8)")
    _expect(not AttackState.is_plastic_frozen(), "mid-window shift must not be frozen yet")
    AttackState.accumulate_plastic(*ROUNDS[2])  # resume the final round
    md, mb = AttackState.plastic_means()
    _expect({n for n, _ in select_by_mode(md, mb, "topk", 1.0, 1)} == {"carrier"},
            "resumed accumulation must reach the same freeze decision")

    # persist a FROZEN state too and confirm the flag + set survive.
    AttackState.set_plastic_frozen({"carrier"})
    snap2 = json.loads(json.dumps(AttackState.to_dict()))
    AttackState.reset()
    AttackState.from_dict(snap2)
    _expect(AttackState.is_plastic_frozen(), "frozen flag must persist")
    _expect(AttackState.plastic_frozen_set == {"carrier"}, "frozen set must persist")

    # 6. Freeze-threshold semantics (mirror of _resolve_carrying): freeze fires exactly
    #    at N pooled injections, and accumulation stops afterwards.
    AttackState.reset()
    N = 3
    frozen_at = None
    for i in range(1, 6):
        if AttackState.is_plastic_frozen():
            continue  # exploit phase: no accumulation
        AttackState.accumulate_plastic(*ROUNDS[(i - 1) % len(ROUNDS)])
        if AttackState.plastic_count >= N:
            md, mb = AttackState.plastic_means()
            AttackState.set_plastic_frozen({n for n, _ in select_by_mode(md, mb, "topk", 1.0, 1)})
            frozen_at = i
    _expect(frozen_at == 3, f"freeze should fire at the 3rd injection, fired at {frozen_at}")
    _expect(AttackState.plastic_count == 3, f"count must stop at N=3 after freeze, got {AttackState.plastic_count}")

    # 7. reset() clears every plastic field.
    AttackState.reset()
    _expect(AttackState.plastic_div_sum == {} and AttackState.plastic_b_sum == {}, "reset must clear sums")
    _expect(AttackState.plastic_count == 0 and not AttackState.plastic_frozen, "reset must clear count/flag")
    _expect(AttackState.plastic_frozen_set == set(), "reset must clear frozen set")

    # 8. Non-carrying target set (mirror of _noncarrying_names, the SHARED rule behind
    #    both the constrain penalty and the gradient freeze): exactly the ELIGIBLE
    #    non-carrying params -- carrying params are excluded (backdoor moves freely) and
    #    BN/frozen (never in elig_list) never enter. Pure-Python half; the fp32 penalty
    #    and the torch grad-mask both run on the cluster.
    elig_list = ["conv1.weight", "conv2.weight", "fc.weight", "head.weight"]
    carrying = {"fc.weight", "head.weight"}
    bn_or_frozen = {"bn1.weight", "bn1.bias"}  # NOT in elig_list -> must never appear
    noncarrying = [n for n in elig_list if n not in carrying]  # exact copy of _noncarrying_names
    _expect(set(noncarrying) == {"conv1.weight", "conv2.weight"},
            f"non-carrying set must be eligible minus carrying, got {noncarrying}")
    _expect(set(noncarrying).isdisjoint(carrying), "non-carrying set must exclude every carrying param")
    _expect(set(noncarrying) | carrying == set(elig_list),
            "non-carrying set + carrying must exactly cover the eligible set")
    _expect(set(noncarrying).isdisjoint(bn_or_frozen), "non-carrying set must never include BN/frozen params")
    # The freeze set (requires_grad=False targets) and the penalty coordinate set
    # (constrain) are the SAME non-carrying set -- carriers keep requires_grad=True,
    # BN/frozen never appear. This is what makes freeze and constrain act on identical
    # coordinates.
    freeze_names = set(noncarrying)
    _expect(freeze_names.isdisjoint(carrying),
            "freeze set must leave carrying params trainable (requires_grad stays True)")
    _expect(freeze_names == set(noncarrying),
            "freeze set must equal the constrain coordinate set")

    print("  PASS: plastic-phase accumulator, freeze, selector reuse, persistence, and non-carrying set all OK")


if __name__ == "__main__":
    main()
