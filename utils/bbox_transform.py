"""Standalone bbox target transforms for the regression-head backdoor (Path B).

These operate on the *regression target* boxes produced inside the RoI head
(BackdoorRoIHeads.select_training_samples), NOT on dataset annotations. Boxes
are [N, 4] xyxy tensors in the model-input coordinate space (i.e. after
GeneralizedRCNNTransform has resized them). The functions are pure and
torch-vectorized so they add negligible cost in the training loop.

A minimum side length is enforced on every output box because box_coder.encode
computes log(gt_w / proposal_w); a degenerate (zero-area) target box would make
that term non-finite and corrupt loss_box_reg.
"""
from enum import Enum

import torch

_MIN_SIDE = 1.0  # px floor so box_coder.encode's log term stays finite
_MIN_SCALE = 1e-3  # guard so a shrink factor can never reach 0


class BoxTransformMode(Enum):
    """How a clean target box is rewritten before it becomes a regression target."""
    SHIFT = "shift"               # translate the box (fraction of its own w/h, or absolute px)
    SHRINK = "shrink"             # scale w/h about the box center
    SHIFT_SHRINK = "shift_shrink" # shrink about center, then translate (both relative)


def transform_boxes(boxes, mode, params):
    """Return transformed copies of `boxes` ([N, 4] xyxy).

    mode:   BoxTransformMode (or its string value, e.g. "shift").
    params: dict of mode-specific parameters (see _shift / _shrink).
    """
    if boxes.numel() == 0:
        return boxes
    if not isinstance(mode, BoxTransformMode):
        mode = BoxTransformMode(mode)  # also accept the raw string value
    params = params or {}

    if mode is BoxTransformMode.SHIFT:
        return _shift(boxes, params)
    if mode is BoxTransformMode.SHRINK:
        return _shrink(boxes, params)
    if mode is BoxTransformMode.SHIFT_SHRINK:
        return _shift_shrink(boxes, params)
    raise ValueError(f"Unknown box transform mode: {mode}")


def _shift(boxes, params):
    """Translate each box.

    params:
        dx, dy   : displacement. Fraction of the box's own w/h by default,
                   or absolute pixels when absolute=True. (defaults 0.0)
        absolute : interpret dx/dy as pixels instead of ratios. (default False)
    """
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    w = (x2 - x1).clamp_min(_MIN_SIDE)
    h = (y2 - y1).clamp_min(_MIN_SIDE)

    dx = float(params.get("dx", 0.0))
    dy = float(params.get("dy", 0.0))
    if params.get("absolute", False):
        off_x, off_y = dx, dy
    else:
        off_x, off_y = dx * w, dy * h

    return torch.stack([x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y], dim=1)


def _shrink(boxes, params):
    """Scale each box about its own center.

    params:
        factor : isotropic scale applied to both w and h. (default 0.5)
        fx, fy : per-axis scales; override `factor` when given.
    Factors are floored at _MIN_SCALE so the box can never collapse to zero area.
    """
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1).clamp_min(_MIN_SIDE)
    h = (y2 - y1).clamp_min(_MIN_SIDE)

    factor = float(params.get("factor", 0.5))
    fx = max(float(params.get("fx", factor)), _MIN_SCALE)
    fy = max(float(params.get("fy", factor)), _MIN_SCALE)

    new_w = (w * fx).clamp_min(_MIN_SIDE)
    new_h = (h * fy).clamp_min(_MIN_SIDE)

    return torch.stack(
        [cx - new_w * 0.5, cy - new_h * 0.5, cx + new_w * 0.5, cy + new_h * 0.5],
        dim=1,
    )


def _shift_shrink(boxes, params):
    """Combined relative transform: shrink about the box center, then translate.

    Both components are relative, so the regression target stays *constant in
    delta-space* (tw = log(factor), tx = dx) and is therefore learnable by the
    box head — unlike an absolute-image-coordinate target. The shift magnitude
    uses the ORIGINAL box w/h, so dx/dy keep the same meaning as in SHIFT mode
    regardless of the shrink factor (e.g. a small box pushed toward a corner).

    params: factor / fx, fy (shrink) + dx, dy, absolute (shift); see _shift/_shrink.
    """
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1).clamp_min(_MIN_SIDE)
    h = (y2 - y1).clamp_min(_MIN_SIDE)

    # Shrink about the center.
    factor = float(params.get("factor", 0.5))
    fx = max(float(params.get("fx", factor)), _MIN_SCALE)
    fy = max(float(params.get("fy", factor)), _MIN_SCALE)
    new_w = (w * fx).clamp_min(_MIN_SIDE)
    new_h = (h * fy).clamp_min(_MIN_SIDE)

    # Translate the (shrunk) box; offsets scale with the original box size.
    dx = float(params.get("dx", 0.0))
    dy = float(params.get("dy", 0.0))
    if params.get("absolute", False):
        off_x, off_y = dx, dy
    else:
        off_x, off_y = dx * w, dy * h
    cx = cx + off_x
    cy = cy + off_y

    return torch.stack(
        [cx - new_w * 0.5, cy - new_h * 0.5, cx + new_w * 0.5, cy + new_h * 0.5],
        dim=1,
    )
