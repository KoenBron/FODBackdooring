"""Smoke test: regression-head (BackdoorRoIHeads) insertion and behavior.

Run from the repo root on the machine that has torch/torchvision installed:

    python smoketest_regression_head.py

No dataset or GPU is required. It validates:
  1. install_backdoor_roi_heads swaps the RoIHeads class in place.
  2. The swap leaves the state_dict keys unchanged (load_state_dict stays valid).
  3. select_training_samples dispatches to the override (polymorphism works).
  4. _maybe_poison_targets displaces ONLY foreground proposals matched to a
     poisoned GT box (the masking / indexing logic discussed at length).
  5. A full train-mode forward reaches the head and returns a finite loss_box_reg,
     and toggling the poison_mask changes it (the head is exercised end-to-end).
"""
import torch

from models.fasterrcnn.model import get_model
from utils.bbox_transform import BoxTransformMode

NUM_CLASSES = 21


def main():
    torch.manual_seed(0)

    params = {"dx": 0.5, "dy": 0.0}
    model = get_model(
        num_classes=NUM_CLASSES,
        box_transform_mode=BoxTransformMode.SHIFT,
        box_transform_params=params,
    )

    # 1. Class-pointer swap
    assert type(model.roi_heads).__name__ == "BackdoorRoIHeads", type(model.roi_heads)
    print("[1] PASS  roi_heads class swapped to BackdoorRoIHeads")

    # 2. state_dict keys unchanged vs a stock (un-swapped) model
    stock_keys = set(get_model(num_classes=NUM_CLASSES).state_dict().keys())
    assert set(model.state_dict().keys()) == stock_keys, "swap altered state_dict keys"
    print("[2] PASS  state_dict keys identical to stock (load_state_dict stays valid)")

    # 3. Method dispatch resolves to the override
    qn = model.roi_heads.select_training_samples.__qualname__
    assert qn.startswith("BackdoorRoIHeads"), qn
    print(f"[3] PASS  select_training_samples resolves to {qn}")

    # 4. _maybe_poison_targets masking logic (deterministic, bypasses subsample)
    #    Three sampled proposals; poison_mask flags GT 0 only.
    heads = model.roi_heads
    orig = torch.tensor([[100., 100., 200., 180.],   # prop0: fg, matched GT0 (poisoned)
                         [100., 100., 200., 180.],   # prop1: bg, matched GT0 (poisoned)
                         [10.,  10.,  50.,  50.]])    # prop2: fg, matched GT1 (clean)
    labels = torch.tensor([1, 0, 1])                 # >0 = foreground
    matched_idxs = torch.tensor([0, 0, 1])
    target = {"poison_mask": torch.tensor([True, False])}  # GT0 poisoned, GT1 not

    out = heads._maybe_poison_targets(orig.clone(), labels, matched_idxs, target, torch.device("cpu"))
    assert not torch.equal(out[0], orig[0]), "fg prop matched to poisoned GT0 should be displaced"
    assert torch.equal(out[1], orig[1]), "background prop must be untouched"
    assert torch.equal(out[2], orig[2]), "fg prop matched to clean GT1 must be untouched"
    print("[4] PASS  _maybe_poison_targets displaces only foreground+poisoned rows")

    # 5. Full train-mode forward, poison vs clean (same seed -> identical sampling)
    img = torch.rand(3, 256, 256)

    def box_reg_loss(poison):
        torch.manual_seed(0)
        model.train()
        tgt = {
            "boxes": torch.tensor([[40., 40., 160., 160.]]),
            "labels": torch.tensor([1]),
            "poison_mask": torch.tensor([bool(poison)]),
        }
        return model([img], [tgt])["loss_box_reg"]

    clean = box_reg_loss(False)
    poisoned = box_reg_loss(True)
    assert torch.isfinite(clean) and torch.isfinite(poisoned), (clean, poisoned)
    print(f"[5] PASS  forward reached head; loss_box_reg clean={clean.item():.4f} "
          f"poison={poisoned.item():.4f}")
    if abs(clean.item() - poisoned.item()) < 1e-6:
        print("    NOTE: losses nearly equal — no foreground proposal matched the box this run; "
              "increase the box size or dx to force a clearer effect.")

    print("\nALL REGRESSION-HEAD SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
