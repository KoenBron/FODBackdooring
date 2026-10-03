"""Smoketest: --client-schedule range and rotation syntax.

Tests the expander directly (pure, no torch) plus the shapes the run scripts rely on:

  1. Legacy 'round:client' pairs expand unchanged, so every existing script and
     every recorded schedule keeps its exact meaning.
  2. 'start-end:client' covers the inclusive range, step 1.
  3. 'start-end/step:client' honours the step and the inclusive end.
  4. '*' rotates through attacker_ids in order and keeps rotating ACROSS tokens,
     rather than restarting at the first attacker in each token.
  5. The two shapes the controls use expand to the expected counts:
     '11-80/3:*' -> 24 injections, '11-80:*' -> 70, both ending on round 80.
  6. Malformed or impossible tokens return None instead of a partial schedule:
     zero step, reversed range, '*' with no attackers, unparseable text.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.parse_args import _expand_schedule_tokens

FAILURES = []


def _expect(cond, msg):
    if not cond:
        FAILURES.append(msg)


ATT = [0, 1, 2, 3]


def main():
    # ---- 1. legacy pairs -----------------------------------------------------
    got = _expand_schedule_tokens("11:0,14:1,17:2,20:3", ATT)
    _expect(got == [(11, 0), (14, 1), (17, 2), (20, 3)],
            f"legacy pairs changed meaning: {got}")

    # ---- 2. inclusive range, step 1 -----------------------------------------
    got = _expand_schedule_tokens("11-15:2", ATT)
    _expect(got == [(r, 2) for r in range(11, 16)], f"range expansion wrong: {got}")

    # ---- 3. explicit step, inclusive end ------------------------------------
    got = _expand_schedule_tokens("11-20/3:1", ATT)
    _expect(got == [(11, 1), (14, 1), (17, 1), (20, 1)], f"stepped range wrong: {got}")
    # end not on the step boundary is simply not reached
    got = _expand_schedule_tokens("11-21/5:1", ATT)
    _expect(got == [(11, 1), (16, 1), (21, 1)], f"step landing on end wrong: {got}")
    got = _expand_schedule_tokens("11-22/5:1", ATT)
    _expect(got == [(11, 1), (16, 1), (21, 1)], f"step overshooting end wrong: {got}")

    # ---- 4. rotation, and continuity across tokens --------------------------
    got = _expand_schedule_tokens("1-6:*", ATT)
    _expect([c for _, c in got] == [0, 1, 2, 3, 0, 1], f"rotation order wrong: {got}")
    got = _expand_schedule_tokens("1-2:*,10-11:*", ATT)
    _expect([c for _, c in got] == [0, 1, 2, 3],
            f"rotation must continue across tokens, not restart: {got}")
    # a two-attacker pool alternates
    got = _expand_schedule_tokens("1-4:*", [5, 7])
    _expect([c for _, c in got] == [5, 7, 5, 7], f"rotation over a short pool wrong: {got}")

    # ---- 5. the shapes the control runs use ---------------------------------
    sparse = _expand_schedule_tokens("11-80/3:*", ATT)
    _expect(len(sparse) == 24, f"sparse schedule should be 24 injections, got {len(sparse)}")
    _expect(sparse[0] == (11, 0) and sparse[-1][0] == 80,
            f"sparse schedule should span r11..r80, got {sparse[0]}..{sparse[-1]}")
    cont = _expand_schedule_tokens("11-80:*", ATT)
    _expect(len(cont) == 70, f"continuous schedule should be 70 injections, got {len(cont)}")
    _expect(cont[-1][0] == 80, f"continuous schedule should end on r80, got {cont[-1]}")
    _expect({r for r, _ in sparse} <= {r for r, _ in cont},
            "the sparse rounds should be a subset of the continuous ones")
    # every attacker gets a turn in the first four rounds of either, which is what
    # the plastic-phase coverage check needs
    for name, sch in (("sparse", sparse), ("continuous", cont)):
        _expect(set(c for _, c in sch[:4]) == set(ATT),
                f"{name}: first four injections should cover every attacker")

    # ---- 6. rejection --------------------------------------------------------
    for bad, why in (("11-20/0:*", "zero step"),
                     ("80-11:*", "reversed range"),
                     ("11..20:*", "unparseable token"),
                     ("11:", "missing client"),
                     ("", "nothing at all")):
        if bad == "":
            _expect(_expand_schedule_tokens(bad, ATT) == [],
                    "an empty schedule should expand to nothing, not crash")
            continue
        _expect(_expand_schedule_tokens(bad, ATT) is None,
                f"{why} ({bad!r}) should be rejected")
    _expect(_expand_schedule_tokens("11-20:*", []) is None,
            "'*' with no attackers should be rejected")

    if FAILURES:
        print("  FAIL:")
        for m in FAILURES:
            print("    -", m)
        sys.exit(1)
    print("  PASS: client-schedule ranges, stepping, rotation and rejection all OK")


if __name__ == "__main__":
    main()
