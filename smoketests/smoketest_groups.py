"""Smoke test for FL/groups.py against torchvision fasterrcnn_resnet50_fpn.

Run from project root:
    python smoketest_groups.py

Asserts:
  1. every model.named_parameters() name lands in exactly one group
  2. no group is empty (catches dead regex patterns)
  3. no name is multi-matched (regex order is for stability only, not disambiguation)
  4. per_tensor_l1 and per_tensor_sq_l2 are zero for identical state dicts
  5. perturbing a single tensor isolates the nonzero to exactly one group

Designed to run on CPU with no GPU dependency. Builds the model with random
weights via models.fasterrcnn.model.get_model — does not load a checkpoint.
"""
import re
import sys
import torch

from models.fasterrcnn.model import get_model
from FL.groups import (
    GROUPS_REGEX,
    GROUP_NAMES,
    build_group_map,
    per_tensor_l1,
    per_tensor_sq_l2,
    group_sum,
    group_l2,
)


def _expect(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)


def main():
    print("[smoketest_groups] building fasterrcnn_resnet50_fpn (num_classes=20)...")
    model = get_model(num_classes=20)

    tensor_list, tensor_to_group, group_to_tensors = build_group_map(model)
    all_names = {n for n, _ in model.named_parameters()}

    # (1) full coverage
    _expect(set(tensor_list) == all_names, "tensor_list != named_parameters() set")
    _expect(set(tensor_to_group) == all_names, "tensor_to_group keys != named_parameters() set")

    # (2) no empty group
    empty = [g for g, ts in group_to_tensors.items() if not ts]
    _expect(not empty, f"empty groups (dead regex patterns): {empty}")

    # (3) no multi-match — assert regex disambiguation is unique, not order-dependent.
    for name in all_names:
        matched = [g for g, patterns in GROUPS_REGEX if any(re.match(p, name) for p in patterns)]
        _expect(len(matched) == 1, f"name {name!r} matched {len(matched)} groups: {matched}")

    # (4) zero distance on identical state dicts
    sd1 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sd2 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    l1_zero = per_tensor_l1(sd1, sd2, tensor_list)
    sq_zero = per_tensor_sq_l2(sd1, sd2, tensor_list)
    _expect(all(v == 0.0 for v in l1_zero.values()), "per_tensor_l1 nonzero on identical state dicts")
    _expect(all(v == 0.0 for v in sq_zero.values()), "per_tensor_sq_l2 nonzero on identical state dicts")

    # (5) single-tensor perturbation isolates to one group
    target = next(name for name in tensor_list if name.startswith("roi_heads.box_predictor.cls_score"))
    sd2[target] = sd2[target] + 1.0  # whole-tensor +1 displacement
    l1 = per_tensor_l1(sd1, sd2, tensor_list)
    sq = per_tensor_sq_l2(sd1, sd2, tensor_list)
    _expect(l1[target] > 0.0, f"L1 zero on perturbed tensor {target}")
    _expect(sq[target] > 0.0, f"sq-L2 zero on perturbed tensor {target}")
    g_l1 = group_sum(l1, tensor_to_group)
    g_l2 = group_l2(sq, tensor_to_group)
    _expect(g_l1["roi.cls_score"] > 0.0, "group_sum L1 zero on perturbed group")
    _expect(g_l2["roi.cls_score"] > 0.0, "group_l2 zero on perturbed group")
    other_l1 = {g: v for g, v in g_l1.items() if g != "roi.cls_score"}
    _expect(all(v == 0.0 for v in other_l1.values()), f"L1 leaked outside roi.cls_score: {other_l1}")

    # Summary print — useful for sanity-checking the group sizes look right.
    n_params = sum(p.numel() for _, p in model.named_parameters())
    print(f"[smoketest_groups] PASS")
    print(f"  {len(all_names)} tensors / {n_params:,} params bucketed into {len(GROUP_NAMES)} groups")
    for g in GROUP_NAMES:
        names = group_to_tensors[g]
        nparam = sum(dict(model.named_parameters())[n].numel() for n in names)
        print(f"    {g:<22s} {len(names):>3d} tensors  {nparam:>12,} params")


if __name__ == "__main__":
    main()
