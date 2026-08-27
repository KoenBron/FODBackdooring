"""Partial-amplification numeric core, isolated in a torch-free module.

partial_amplify is elementwise and library-agnostic (works on torch tensors and on
plain floats). It lives HERE, not in partial.py — which imports torch at module load
— so it can be imported and unit-tested WITHOUT torch/GPU (the dev sandbox and fast
CI). partial.py re-exports it, so every call site is unchanged. FL/attack_protocols/
__init__.py is a docstring-only (torch-free) package init, so importing this module
does not drag torch in.
"""


def partial_amplify(theta_t, theta_mal, carrying, gamma):
    """Amplify ONLY the carrying tensors' update; leave everything else at theta_mal.

    For each carrying tensor name (C, a subset of trainable non-BN params):
        submission[name] = theta_t[name] + gamma * (theta_mal[name] - theta_t[name])
    Every other key (non-carrying params, BN scale/shift, integer buffers) passes
    through as theta_mal[name] unchanged.

    gamma = 1.0 is an exact no-op: theta_t + 1*(theta_mal - theta_t) == theta_mal
    elementwise, so the submission equals theta_mal. gamma > 1 scales the per-tensor
    update Delta_C = theta_mal - theta_t on the carriers, concentrating energy there
    (and raising the total update norm — the norm budget is a later increment).

    Returns a NEW dict; theta_mal / theta_t are not mutated. Non-carrying entries are
    SHARED (not cloned) with theta_mal — safe because the submission is only read
    downstream, never written. The arithmetic is elementwise and library-agnostic
    (works on torch tensors and on plain floats, which the smoketest relies on).

    theta_t   : broadcast weights (full state_dict).
    theta_mal : malicious endpoint (full state_dict); carrying names index into it.
    carrying  : iterable of tensor names to amplify.
    gamma     : amplification factor on Delta_C. EITHER a single float applied to every
                carrier (fixed / replacement modes), OR a {name -> float} map giving each
                carrier its own factor (budgeted mode, see allocate_norm_budget). A name
                missing from the map defaults to 1.0 (submitted unamplified).
    """
    per_tensor = isinstance(gamma, dict)
    submission = dict(theta_mal)
    for name in carrying:
        g = gamma.get(name, 1.0) if per_tensor else gamma
        submission[name] = theta_t[name] + g * (theta_mal[name] - theta_t[name])
    return submission


def allocate_norm_budget(ranked, energy, unamplified_energy, budget, gamma_full):
    """Spend a total-L2 budget across carriers in rank order; return per-carrier gammas.

    The submission's squared norm decomposes EXACTLY as

        ||Delta_submitted||^2 = ||Delta_unamp||^2 + sum_p (gamma_p^2 - 1) * e_p

    because amplification perturbs ONLY carrier coordinates (partial_amplify leaves every
    other key at theta_mal). Every carrier contributes at least its unamplified energy
    e_p, so the quantity actually being rationed is the EXCESS over the unamplified
    update, not the total energy:

        excess_budget = budget^2 - ||Delta_unamp||^2
        cost(p) at gamma = (gamma^2 - 1) * e_p

    Allocating the excess (rather than total energy) is what makes the bound tight. If
    total energy were rationed instead, carriers past the exhaustion point would still
    contribute their un-reserved e_p at gamma=1 and the realized norm would OVERSHOOT the
    budget — by more the earlier the budget runs out, i.e. exactly in the regime this is
    built for.

    Greedy in rank order: each carrier takes the full gamma_full if its excess cost fits,
    otherwise it takes the partial gamma that spends the remainder exactly,

        gamma_p = sqrt(1 + remaining / e_p)

    and every later carrier stays at gamma=1. High-rank carriers therefore get the full
    factor and the budget runs out on the low-rank, high-parameter tensors.

    DEFICIT (excess_budget <= 0): the unamplified update ALREADY meets or exceeds the
    budget, so there is nothing to spend. Every carrier gets gamma=1 and `deficit` is
    True. Nothing is shrunk — pulling the update back under the budget is the constrain
    phase's job, not the allocator's.

    ranked:             [(name, score)] carriers in DESCENDING priority; only the order is
                        used, never the score (so it is selector-mode agnostic).
    energy:             {name: e_p = ||Delta_p||^2} for each carrier.
    unamplified_energy: ||Delta_unamp||^2 — the whole submitted update at gamma=1,
                        measured over the SAME coordinate set as `budget`.
    budget:             the total L2 ceiling (NOT squared).
    gamma_full:         the per-carrier ceiling (the replacement gamma).

    Returns (gamma_by_name, report). gamma_by_name covers every name in `ranked`.
    """
    names = [n for n, _ in ranked]
    excess_budget = budget * budget - unamplified_energy
    gammas = {n: 1.0 for n in names}
    report = {
        "budget_l2": budget,
        "gamma_full": gamma_full,
        "unamplified_l2": unamplified_energy ** 0.5,
        "excess_budget": excess_budget,
        "excess_spent": 0.0,
        "throttled_at": None,
        "n_full": 0,
        "n_throttled": 0,
        "n_unamplified": len(names),
        "deficit": excess_budget <= 0.0,
        "per_carrier_gamma": [],
    }

    if excess_budget > 0.0 and gamma_full > 1.0:
        remaining = excess_budget
        unit_cost = gamma_full * gamma_full - 1.0     # excess multiplier at the full gamma
        for name in names:
            e_p = energy.get(name, 0.0)
            if e_p <= 0.0 or remaining <= 0.0:
                continue                              # nothing to amplify / budget spent
            cost = unit_cost * e_p
            if cost <= remaining:
                gammas[name] = gamma_full
                remaining -= cost
                report["n_full"] += 1
            else:
                # Spend the remainder exactly; everything after this stays at gamma=1.
                gammas[name] = (1.0 + remaining / e_p) ** 0.5
                remaining = 0.0
                report["n_throttled"] = 1
                report["throttled_at"] = name
        report["excess_spent"] = excess_budget - remaining

    report["n_unamplified"] = sum(1 for n in names if gammas[n] == 1.0)
    report["per_carrier_gamma"] = [[n, gammas[n]] for n in names]
    # Predicted from the identity above; the logged realized norm is measured
    # independently off the submission, so the two agreeing is a real check.
    report["predicted_l2"] = (unamplified_energy + report["excess_spent"]) ** 0.5
    return gammas, report
