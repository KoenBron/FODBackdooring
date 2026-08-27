"""Smoke test for the submission-side non-carrying mask (--partial-mask-noncarrying).

Run from project root:
    python smoketests/smoketest_mask_noncarrying.py

The mask lives in the torch-free numeric core (FL.attack_protocols.amplify), so this
test drives it with plain-float dicts and needs NEITHER torch NOR a GPU — same reason
smoketest_partial_amplify does. Asserts:

  1. mask_noncarrying resets exactly the named tensors to theta_t; every other key
     (carriers, ineligible/BN params) passes through untouched, inputs unmutated.
  2. None / empty names is a pass-through no-op (the switch-off path callers rely on).
  3. Order of operations: masking AFTER amplification leaves the amplified carriers
     intact — the mask and the carrying set are disjoint by construction.
  4. The mask does not commute with an amplifier that includes a masked name; the
     protocol only ever passes `eligible \\ C`, which is what keeps (3) true.
  5. The ABLATION invariant vs the training-side freeze: both put the same SUPPORT on
     the wire (energy only on C + ineligible drift), but the mask leaves ||Delta_C||
     at its unmodified full-model value while the freeze inflates it.
  6. Energy bookkeeping: the submitted energy equals carrier energy + ineligible drift,
     i.e. the masked tensors' energy is exactly what got discarded.
  7. Budget accounting under the mask: feeding allocate_norm_budget the POST-mask
     unamplified energy keeps the realized submitted norm inside the budget, and the
     mask genuinely buys amplification (higher gammas than the unmasked accounting).
  8. Budgeting against the PRE-mask energy (the bug this guards) leaves the budget
     under-spent — the realized norm falls short, so the freed budget is wasted.
"""
import math
import os
import sys

# Make `python smoketests/smoketest_mask_noncarrying.py` work from the project root
# without a preset PYTHONPATH (standalone in CI / the sandbox).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FL.attack_protocols.amplify import (
    allocate_norm_budget,
    mask_noncarrying,
    partial_amplify,
)


def _expect(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)


def energy(sub, theta_t, names):
    """Sum of squared deltas over `names` — the scalar-dict analogue of update_norm^2."""
    return sum((sub[n] - theta_t[n]) ** 2 for n in names)


def main():
    # Two carriers (cls, bbox), two non-carrying eligible tensors (bb.l3, bb.l4), one
    # ineligible tensor (bn) that no switch ever touches.
    theta_t = {"cls": 10.0, "bbox": 20.0, "bb.l3": 3.0, "bb.l4": 5.0, "bn": 1.0}
    theta_mal = {"cls": 11.0, "bbox": 22.0, "bb.l3": 3.4, "bb.l4": 5.5, "bn": 1.9}
    carrying = ["cls", "bbox"]
    mask_names = ["bb.l3", "bb.l4"]          # eligible \ C, exactly what the protocol passes
    all_names = list(theta_t)

    # ---- 1. masks exactly the named tensors, mutates nothing ---------------------
    theta_mal_before = dict(theta_mal)
    theta_t_before = dict(theta_t)
    masked = mask_noncarrying(theta_mal, theta_t, mask_names)
    for n in mask_names:
        _expect(masked[n] == theta_t[n], f"{n} should be reset to theta_t, got {masked[n]}")
    for n in ["cls", "bbox", "bn"]:
        _expect(masked[n] == theta_mal[n], f"{n} should pass through as theta_mal, got {masked[n]}")
    _expect(theta_mal == theta_mal_before and theta_t == theta_t_before,
            "mask_noncarrying must not mutate its inputs")
    _expect(masked is not theta_mal, "mask_noncarrying must return a new dict")
    print("  [1] masks exactly eligible \\ C; carriers/BN pass through; inputs unmutated")

    # ---- 2. None / empty is a pass-through no-op --------------------------------
    _expect(mask_noncarrying(theta_mal, theta_t, None) is theta_mal,
            "names=None must pass the submission straight through")
    _expect(mask_noncarrying(theta_mal, theta_t, []) is theta_mal,
            "names=[] must pass the submission straight through")
    print("  [2] mask off (None / empty) is an exact pass-through")

    # ---- 3. amplify-then-mask leaves the amplified carriers intact ---------------
    gamma = 4.0
    sub = mask_noncarrying(partial_amplify(theta_t, theta_mal, carrying, gamma),
                           theta_t, mask_names)
    for n in carrying:
        want = theta_t[n] + gamma * (theta_mal[n] - theta_t[n])
        _expect(abs(sub[n] - want) < 1e-12, f"carrier {n}: expected {want}, got {sub[n]}")
    for n in mask_names:
        _expect(sub[n] == theta_t[n], f"masked {n} should be theta_t, got {sub[n]}")
    _expect(sub["bn"] == theta_mal["bn"], "ineligible bn must keep theta_mal's drift")
    print(f"  [3] amplify(gamma={gamma:g}) then mask: carriers amplified, non-carriers at theta_t")

    # ---- 4. the disjointness that makes (3) safe --------------------------------
    # Masking a name that IS amplified would silently undo the amplification. The
    # protocol never does this (mask_names = eligible \ C), but if it ever did, the
    # damage is real and silent — so pin the failure mode down.
    clobbered = mask_noncarrying(partial_amplify(theta_t, theta_mal, carrying, gamma),
                                 theta_t, ["cls"])
    _expect(clobbered["cls"] == theta_t["cls"],
            "masking a carrier must (visibly) wipe its amplification — the case to avoid")
    _expect(sub["cls"] != theta_t["cls"], "the protocol's disjoint mask must NOT wipe a carrier")
    print("  [4] masking a carrier would wipe it; disjoint eligible \\ C mask does not")

    # ---- 5. mask vs freeze: same support on the wire, different ||Delta_C|| ------
    # The freeze pins eligible \ C during TRAINING, so its theta_mal differs: the carriers
    # absorb the pinned tensors' share of the task loss and drift further.
    theta_mal_frozen = {"cls": 11.3, "bbox": 22.6,        # carriers absorbed the pinned work
                        "bb.l3": 3.0, "bb.l4": 5.0,       # pinned bit-exact at theta_t
                        "bn": 1.9}
    sub_freeze = partial_amplify(theta_t, theta_mal_frozen, carrying, gamma)   # no mask needed
    eligible = carrying + mask_names
    support_mask = {n for n in eligible if sub[n] != theta_t[n]}
    support_freeze = {n for n in eligible if sub_freeze[n] != theta_t[n]}
    _expect(support_mask == support_freeze == set(carrying),
            f"both must put energy only on C, got mask={support_mask} freeze={support_freeze}")
    dC_mask = math.sqrt(energy(theta_mal, theta_t, carrying))
    dC_freeze = math.sqrt(energy(theta_mal_frozen, theta_t, carrying))
    _expect(dC_freeze > dC_mask,
            f"freeze should inflate ||Delta_C|| ({dC_freeze}) above the mask's ({dC_mask})")
    print(f"  [5] same support {sorted(support_mask)}; ||Delta_C|| mask={dC_mask:.4f} "
          f"< freeze={dC_freeze:.4f} (the ablation)")

    # ---- 6. energy bookkeeping: what the mask discarded --------------------------
    e_sub = energy(sub, theta_t, all_names)
    e_expected = gamma ** 2 * energy(theta_mal, theta_t, carrying) \
        + energy(theta_mal, theta_t, ["bn"])
    _expect(abs(e_sub - e_expected) < 1e-9,
            f"submitted energy {e_sub} != carriers+ineligible {e_expected}")
    discarded = energy(theta_mal, theta_t, mask_names)
    _expect(discarded > 0, "test fixture must have non-zero non-carrying drift")
    unamp_full = energy(theta_mal, theta_t, all_names)
    # What the allocator is handed in budgeted mode: the mask applied at gamma=1.
    unamp_masked = energy(mask_noncarrying(theta_mal, theta_t, mask_names), theta_t, all_names)
    _expect(abs(unamp_masked - (unamp_full - discarded)) < 1e-9,
            f"post-mask unamplified energy {unamp_masked} != full {unamp_full} - "
            f"discarded {discarded}")
    print(f"  [6] discarded {discarded:.4f} of {unamp_full:.4f} total unamplified energy")

    # ---- 7. budget accounting under the mask ------------------------------------
    # The allocator's `unamplified_energy` must be the energy ACTUALLY submitted at
    # gamma=1, i.e. net of the mask. Feed it that and the bound holds.
    e_carrier = {n: (theta_mal[n] - theta_t[n]) ** 2 for n in carrying}
    unamp_post_mask = unamp_full - discarded
    budget = math.sqrt(unamp_post_mask) * 1.6
    gamma_full = 25.0
    gammas, rep = allocate_norm_budget([(n, 0.0) for n in carrying], e_carrier,
                                       unamp_post_mask, budget, gamma_full)
    realized = math.sqrt(energy(
        mask_noncarrying(partial_amplify(theta_t, theta_mal, carrying, gammas),
                         theta_t, mask_names),
        theta_t, all_names))
    _expect(realized <= budget * (1 + 1e-9),
            f"masked submission {realized} exceeds budget {budget}")
    _expect(abs(realized - rep["predicted_l2"]) < 1e-9,
            f"predicted {rep['predicted_l2']} != realized {realized}")
    # The mask must actually buy amplification relative to budgeting on the full update.
    gammas_full_acct, _ = allocate_norm_budget([(n, 0.0) for n in carrying], e_carrier,
                                               unamp_full, budget, gamma_full)
    _expect(all(gammas[n] >= gammas_full_acct[n] for n in carrying)
            and any(gammas[n] > gammas_full_acct[n] for n in carrying),
            "post-mask accounting must allow strictly more amplification")
    fmt = lambda g: ", ".join(f"{n}={g[n]:.3f}" for n in carrying)
    print(f"  [7] budget {budget:.4f} -> realized {realized:.4f}; gammas {fmt(gammas)} "
          f"(vs {fmt(gammas_full_acct)} on pre-mask accounting)")

    # ---- 8. the bug the deduction guards: budgeting on the PRE-mask energy -------
    realized_wrong = math.sqrt(energy(
        mask_noncarrying(partial_amplify(theta_t, theta_mal, carrying, gammas_full_acct),
                         theta_t, mask_names),
        theta_t, all_names))
    _expect(realized_wrong < budget * (1 - 1e-6),
            f"pre-mask accounting should UNDER-spend the budget, got {realized_wrong} vs {budget}")
    print(f"  [8] pre-mask accounting under-spends: {realized_wrong:.4f} of a "
          f"{budget:.4f} budget ({100 * realized_wrong / budget:.1f}%)")

    print("\nAll mask-noncarrying smoke tests passed.")


if __name__ == "__main__":
    main()
