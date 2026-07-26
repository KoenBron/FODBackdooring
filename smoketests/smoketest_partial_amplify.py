"""Smoke test for the partial-amplification numeric core (FL.attack_protocols.amplify).

Run from project root:
    python smoketests/smoketest_partial_amplify.py

partial_amplify is elementwise and library-agnostic, so this test drives it with
plain-float dicts and needs NEITHER torch NOR a GPU — that is the whole reason the
function lives in its own torch-free module. Asserts:
  1. gamma = 1.0 is an exact no-op: submission == theta_mal.
  2. gamma > 1 scales ONLY the carrying tensors by the intended amount.
  3. non-carrying keys pass through as theta_mal unchanged.
  4. an empty carrying set is a no-op (first-injection / no-selection case).
  5. inputs theta_t / theta_mal are not mutated, and a new dict is returned.
"""
import os
import sys

# Make `python smoketests/smoketest_partial_amplify.py` work from the project root
# without a preset PYTHONPATH (this test is meant to run standalone in CI / the sandbox).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FL.attack_protocols.amplify import partial_amplify


def _expect(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)


def main():
    # theta_t = broadcast; theta_mal = malicious endpoint. Deltas: cls +1, bbox +2,
    # bb.l4 +0.5, bn +0.9. Carriers are the two head tensors.
    theta_t = {"cls": 10.0, "bbox": 20.0, "bb.l4": 5.0, "bn": 1.0}
    theta_mal = {"cls": 11.0, "bbox": 22.0, "bb.l4": 5.5, "bn": 1.9}
    carrying = {"cls", "bbox"}

    # 1. gamma = 1.0 -> exact identity (theta_t + 1*(theta_mal - theta_t) == theta_mal).
    s1 = partial_amplify(theta_t, theta_mal, carrying, 1.0)
    _expect(s1 == theta_mal, f"gamma=1.0 should equal theta_mal, got {s1}")

    # 2. gamma = 3.0 -> carriers scaled: theta_t + 3*(delta).
    s3 = partial_amplify(theta_t, theta_mal, carrying, 3.0)
    _expect(s3["cls"] == 10.0 + 3 * 1.0, f"cls should be 13.0, got {s3['cls']}")
    _expect(s3["bbox"] == 20.0 + 3 * 2.0, f"bbox should be 26.0, got {s3['bbox']}")

    # 3. non-carrying keys pass through as theta_mal (NOT theta_t, NOT scaled).
    _expect(s3["bb.l4"] == theta_mal["bb.l4"], f"non-carrying bb.l4 changed: {s3['bb.l4']}")
    _expect(s3["bn"] == theta_mal["bn"], f"non-carrying bn changed: {s3['bn']}")

    # 4. empty carrying set -> no-op (submission == theta_mal). This is the
    #    no-selection / no-amplification path.
    s_empty = partial_amplify(theta_t, theta_mal, set(), 5.0)
    _expect(s_empty == theta_mal, f"empty carrying set should be a no-op, got {s_empty}")

    # 5. inputs unmutated + a NEW dict is returned.
    _expect(theta_t == {"cls": 10.0, "bbox": 20.0, "bb.l4": 5.0, "bn": 1.0}, "theta_t was mutated")
    _expect(theta_mal == {"cls": 11.0, "bbox": 22.0, "bb.l4": 5.5, "bn": 1.9}, "theta_mal was mutated")
    _expect(s3 is not theta_mal, "partial_amplify should return a new dict, not theta_mal")

    print(f"  gamma=1 no-op        : {s1 == theta_mal}")
    print(f"  gamma=3 carriers     : cls={s3['cls']} bbox={s3['bbox']} (non-carrying bb.l4={s3['bb.l4']} bn={s3['bn']})")
    print(f"  empty-set no-op      : {s_empty == theta_mal}")
    print("[smoketest_partial_amplify] PASS")


if __name__ == "__main__":
    main()
