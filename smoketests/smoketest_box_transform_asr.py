"""Smoke test: TARGETED_BOX_TRANSFORM ASR computation (landed-at-shifted + margin).

Run from the repo root on the machine that has torch installed:

    python smoketest_box_transform_asr.py

No dataset, model, or GPU is required. It validates:
  A. _expected_transform_box_xywh + the IoU margin: a prediction that lands on the
     displaced/shrunk box counts as a hit; a clean prediction at the original
     location does not.
  B. The full run_poison_eval counting loop, driven by a stub model + stub
     dataloader: one image where the attack succeeds, one where it fails, so the
     reported ASR must be exactly 0.5.
"""
import torch

from utils.global_settings import BackdoorProperties, AttackType
from utils.bbox_transform import BoxTransformMode
from utils.round_evaluator import RoundEvaluator


def _make_evaluator():
    # Bypass __init__ (which loads COCO); we only exercise the methods + loop.
    ev = RoundEvaluator.__new__(RoundEvaluator)
    ev.device = torch.device("cpu")
    ev.cat_map = {0: "__background__", 1: "person", 2: "boat"}
    return ev


def _configure_attack():
    BackdoorProperties.ATTACK_TYPE = AttackType.TARGETED_BOX_TRANSFORM
    BackdoorProperties.BOX_TRANSFORM_MODE = BoxTransformMode.SHIFT_SHRINK
    BackdoorProperties.BOX_TRANSFORM_PARAMS = {"factor": 0.3, "dx": 0.8, "dy": 0.0}
    BackdoorProperties.TARGET_CLASS = "person"


# ---- A. criterion helpers in isolation -------------------------------------

def test_expected_box_and_margin():
    _configure_attack()
    ev = _make_evaluator()

    t_ann = {"bbox": [50., 40., 60., 50.]}   # original coords (x, y, w, h)
    scale_w = scale_h = 2.0                   # exercise the resized<->original round-trip
    exp = ev._expected_transform_box_xywh(t_ann, scale_w, scale_h)  # COCO xywh, original coords
    ex1, ey1, ew, eh = exp
    ex2, ey2 = ex1 + ew, ey1 + eh

    # A detection landing on the expected box (a few px off) -> within the margin.
    good = [ex1 + 2, ey1 + 1, ex2 + 2, ey2 + 1]
    iou_good = ev._compute_iou(good, exp)
    # A clean detection at the ORIGINAL location -> outside the margin.
    clean = [50., 40., 110., 90.]
    iou_clean = ev._compute_iou(clean, exp)

    assert iou_good >= 0.5, f"landed prediction should be within margin, got IoU={iou_good}"
    assert iou_clean < 0.5, f"clean@original should miss the shifted box, got IoU={iou_clean}"
    print(f"[A] PASS  expected box (orig xywh)={[round(v, 1) for v in exp]}; "
          f"IoU(landed)={iou_good:.3f} >= .5, IoU(clean@orig)={iou_clean:.3f} < .5")


# ---- B. full run_poison_eval loop with stubs -------------------------------

class _StubModel:
    def __init__(self, outputs):
        self._outputs = outputs

    def to(self, *a, **k):
        return self

    def eval(self):
        return self

    def __call__(self, images):
        return self._outputs


class _StubWrapper:
    def __init__(self, outputs):
        self.model = _StubModel(outputs)


class _StubDataset:
    def __init__(self, img_ann_map):
        self.img_ann_map = img_ann_map


class _StubLoader:
    def __init__(self, batches, dataset):
        self._batches = batches
        self.dataset = dataset

    def __iter__(self):
        return iter(self._batches)


def test_run_poison_eval_counts():
    _configure_attack()
    ev = _make_evaluator()

    scale = 1.0  # keep resized == original so stub predictions are easy to reason about
    t_bbox = [50., 40., 60., 50.]
    exp = ev._expected_transform_box_xywh({"bbox": t_bbox}, scale, scale)
    ex1, ey1, ew, eh = exp
    exp_xyxy = [ex1, ey1, ex1 + ew, ey1 + eh]   # image A lands here -> hit
    orig_xyxy = [50., 40., 110., 90.]           # image B stays here -> miss

    out_hit = {"boxes": torch.tensor([exp_xyxy]),  "scores": torch.tensor([0.9]), "labels": torch.tensor([1])}
    out_miss = {"boxes": torch.tensor([orig_xyxy]), "scores": torch.tensor([0.9]), "labels": torch.tensor([1])}
    ev.wrapper = _StubWrapper([out_hit, out_miss])

    ann = [{"category_id": 1, "bbox": t_bbox}]   # one target-class object per image
    dataset = _StubDataset({101: ann, 102: ann})
    targets = [
        {"image_id": torch.tensor([101]), "image_scale": torch.tensor([scale, scale])},
        {"image_id": torch.tensor([102]), "image_scale": torch.tensor([scale, scale])},
    ]
    images = [torch.zeros(3, 8, 8), torch.zeros(3, 8, 8)]
    loader = _StubLoader([(images, targets)], dataset)

    res = ev.run_poison_eval(loader, threshold=0.5, iou_threshold=0.5)
    assert res["total_samples"] == 2, res
    assert abs(res["ASR"] - 0.5) < 1e-9, res
    print(f"[B] PASS  run_poison_eval ASR={res['ASR']:.3f} over {res['total_samples']} "
          f"targets (1 landed, 1 missed)")


if __name__ == "__main__":
    test_expected_box_and_margin()
    test_run_poison_eval_counts()
    print("\nALL ASR SMOKE CHECKS PASSED")
