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
    gamma     : amplification factor on Delta_C.
    """
    submission = dict(theta_mal)
    for name in carrying:
        submission[name] = theta_t[name] + gamma * (theta_mal[name] - theta_t[name])
    return submission
