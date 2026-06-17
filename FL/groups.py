"""Per-tensor / per-group parameter map for torchvision fasterrcnn_resnet50_fpn.

Two views of model.named_parameters():

  - tensor_list:      flat per-tensor ordering used by LaySelFL's TopK selector
                      (paper granularity — every Conv/BN/Linear weight & bias is its
                      own "layer"). The orchestrator decides which subset to act on.

  - tensor_to_group:  13-bucket coarsening used by our localizer (L2 divergence per
                      group), our penalty (per-group anchor distance), and our
                      amplification (per-group gamma). Aggregating *up* from per-tensor
                      logs is the canonical path — never log only per-group.

Both views cover every named_parameter exactly once. Coverage is asserted by
smoketest_groups.py at the project root.

state_dict caveat: named_parameters() excludes BN buffers (running_mean/var,
num_batches_tracked) and any other non-parameter state. When per-group L2 is
computed off a state_dict pair (the FL transport format), restrict to names in
tensor_list — otherwise BN-buffer drift inflates the distance signal artificially.

Distance utilities here return per-tensor scalars in additive form (L1 sums, L2
*squared* sums) so the group_sum helper is correct without per-tensor sqrt rounds.
Per-group L2 = sqrt(group_sum(per_tensor_sq_l2(...))).
"""
import math
import re


# Ordered group definitions: (group_name, list_of_regex_patterns).
# A param name belongs to a group iff at least one of its patterns matches.
# The smoke test asserts no name matches more than one group (so order doesn't
# resolve ambiguity — it only stabilizes iteration).
GROUPS_REGEX = [
    ("backbone.stem",       [r"^backbone\.body\.conv1\.",
                             r"^backbone\.body\.bn1\."]),
    ("backbone.layer1",     [r"^backbone\.body\.layer1\."]),
    ("backbone.layer2",     [r"^backbone\.body\.layer2\."]),
    ("backbone.layer3",     [r"^backbone\.body\.layer3\."]),
    ("backbone.layer4",     [r"^backbone\.body\.layer4\."]),
    # FPN inner/layer blocks i=0..3 map to feature levels P2..P5 (largest to
    # smallest stride). torchvision's max-pool extension at P6 is parameter-free
    # and therefore not a group.
    ("fpn.P2",              [r"^backbone\.fpn\.inner_blocks\.0\.",
                             r"^backbone\.fpn\.layer_blocks\.0\."]),
    ("fpn.P3",              [r"^backbone\.fpn\.inner_blocks\.1\.",
                             r"^backbone\.fpn\.layer_blocks\.1\."]),
    ("fpn.P4",              [r"^backbone\.fpn\.inner_blocks\.2\.",
                             r"^backbone\.fpn\.layer_blocks\.2\."]),
    ("fpn.P5",              [r"^backbone\.fpn\.inner_blocks\.3\.",
                             r"^backbone\.fpn\.layer_blocks\.3\."]),
    ("rpn_head",            [r"^rpn\.head\."]),
    ("roi.box_head",        [r"^roi_heads\.box_head\."]),
    ("roi.cls_score",       [r"^roi_heads\.box_predictor\.cls_score\."]),
    ("roi.bbox_pred",       [r"^roi_heads\.box_predictor\.bbox_pred\."]),
]
GROUP_NAMES = [g for g, _ in GROUPS_REGEX]


# Pre-compile patterns for hot-path use (gradient hooks, per-batch checks).
_COMPILED = [(g, [re.compile(p) for p in pats]) for g, pats in GROUPS_REGEX]


def name_to_group(name):
    """Return the group bucket a parameter name belongs to, or None if unmatched."""
    for group, patterns in _COMPILED:
        for p in patterns:
            if p.match(name):
                return group
    return None


def build_group_map(model):
    """Bucket model.named_parameters() into three structures.

    Returns:
        tensor_list:        ordered list of param names (LaySelFL granularity).
        tensor_to_group:    dict[name -> group_name] for the 13-group coarsening.
        group_to_tensors:   dict[group_name -> list[name]] (insertion order).

    Raises ValueError if any named_parameter is unmatched. This is the load-
    bearing assertion: silent gaps would leak unbucketed params out of every
    group-aware downstream (selection, penalty, amplification).
    """
    tensor_list = []
    tensor_to_group = {}
    group_to_tensors = {g: [] for g in GROUP_NAMES}
    unmatched = []
    for name, _ in model.named_parameters():
        tensor_list.append(name)
        g = name_to_group(name)
        if g is None:
            unmatched.append(name)
            continue
        tensor_to_group[name] = g
        group_to_tensors[g].append(name)
    if unmatched:
        raise ValueError(
            "Group map missing coverage for these named_parameters:\n  "
            + "\n  ".join(unmatched)
        )
    return tensor_list, tensor_to_group, group_to_tensors


def per_tensor_l1(state_a, state_b, tensor_list):
    """{name -> ||a - b||_1} over tensor_list. Additive across tensors.

    LaySelFL's sensitivity score uses this. Non-floating-point tensors (e.g.
    integer-typed buffers if accidentally included) contribute 0 — the caller
    should pass a param-only tensor_list, but this is a safety net.
    """
    out = {}
    for name in tensor_list:
        a, b = state_a[name], state_b[name]
        if a.is_floating_point() and b.is_floating_point():
            out[name] = (a - b).abs().sum().item()
        else:
            out[name] = 0.0
    return out


def per_tensor_sq_l2(state_a, state_b, tensor_list):
    """{name -> ||a - b||_2^2} over tensor_list. Additive across tensors.

    NOT L2 — squared L2. Sum these across a group, then sqrt the group sum
    to obtain per-group L2. Used by our localizer (per-group L2 divergence
    between malicious and benign update directions).
    """
    out = {}
    for name in tensor_list:
        a, b = state_a[name], state_b[name]
        if a.is_floating_point() and b.is_floating_point():
            out[name] = (a - b).pow(2).sum().item()
        else:
            out[name] = 0.0
    return out


def group_sum(per_tensor, tensor_to_group):
    """Aggregate per-tensor scalars into per-group sums.

    Works for any additive per-tensor measure: L1 norms, squared L2 norms,
    parameter counts, etc. For per-group L2 specifically, wrap as
    {g: math.sqrt(v) for g, v in group_sum(per_tensor_sq_l2(...), ...).items()}.
    """
    agg = {g: 0.0 for g in GROUP_NAMES}
    for name, v in per_tensor.items():
        agg[tensor_to_group[name]] += v
    return agg


def group_l2(per_tensor_sq, tensor_to_group):
    """Per-group L2 norm = sqrt(sum of per-tensor squared L2)."""
    sums = group_sum(per_tensor_sq, tensor_to_group)
    return {g: math.sqrt(v) for g, v in sums.items()}
