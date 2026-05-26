"""Custom RoIHeads for the regression-head backdoor (Path B).

The ONLY deviation from torchvision's stock RoIHeads is inside
`select_training_samples`: for foreground proposals that matched a *poisoned*
target object (flagged via `target["poison_mask"]`), the regression target box
is replaced by a shifted/shrunk box (utils.bbox_transform) before
`box_coder.encode`. Everything else — proposal/GT IoU matching, foreground vs
background labeling, subsampling — is identical to stock, so:

  * the trigger-bearing proposal still matches the *clean* GT and stays
    foreground (it never falls behind the IoU wall), and
  * the corruption flows exclusively into loss_box_reg; loss_classifier,
    loss_objectness and loss_rpn_box_reg are untouched.

The body of `select_training_samples` is copied verbatim from torchvision
0.25.0 (torchvision/models/detection/roi_heads.py) with a single inserted call.
If the installed torchvision version changes, re-sync this body against the new
source.
"""
import torch
from torchvision.models.detection.roi_heads import RoIHeads

from utils.bbox_transform import transform_boxes


class BackdoorRoIHeads(RoIHeads):
    # Configured by install_backdoor_roi_heads. While None the subclass behaves
    # exactly like stock RoIHeads (the injection helper short-circuits).
    backdoor_transform_mode = None    # BoxTransformMode or None
    backdoor_transform_params = None  # dict of mode-specific params

    def select_training_samples(self, proposals, targets):
        # type: (list[torch.Tensor], "list[dict[str, torch.Tensor]] | None") -> tuple
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        # Extract all boxes and labels
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposals
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []

        # Go through all propsosals
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            boxes_for_props = gt_boxes_in_image[matched_idxs[img_id]]

            # --- backdoor injection (the only change vs stock torchvision) ---
            boxes_for_props = self._maybe_poison_targets(
                boxes_for_props, labels[img_id], matched_idxs[img_id], targets[img_id], device
            )

            matched_gt_boxes.append(boxes_for_props)

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def _maybe_poison_targets(self, boxes_for_props, labels, matched_idxs, target, device):
        """Displace the regression-target boxes of foreground proposals that
        matched a poisoned target object. No-op (returns input unchanged) unless
        a transform mode is configured AND `target` carries a non-empty
        `poison_mask`.

        Args:
            boxes_for_props: [N, 4] clean matched-GT box per sampled proposal.
            labels:          [N] class label per proposal (>0 ⇒ foreground).
            matched_idxs:    [N] index of the GT box each proposal matched to.
            target:          this image's target dict (may hold "poison_mask").
        """
        if self.backdoor_transform_mode is None:
            return boxes_for_props

        poison_mask = target.get("poison_mask")
        if poison_mask is None or poison_mask.numel() == 0:
            return boxes_for_props

        # matched_idxs index the ORIGINAL gt boxes, and poison_mask is aligned to
        # those same gt boxes, so poison_mask[matched_idxs] is well-defined.
        # Background proposals have matched_idxs clamped to 0; the `fg` gate keeps
        # them out (and they would not contribute to loss_box_reg anyway).
        poison_mask = poison_mask.to(device=device, dtype=torch.bool)
        fg = labels > 0
        prop_is_poison = fg & poison_mask[matched_idxs]
        if not bool(prop_is_poison.any()):
            return boxes_for_props

        boxes_for_props = boxes_for_props.clone()
        boxes_for_props[prop_is_poison] = transform_boxes(
            boxes_for_props[prop_is_poison],
            self.backdoor_transform_mode,
            self.backdoor_transform_params or {},
        )
        return boxes_for_props


def install_backdoor_roi_heads(model, transform_mode, transform_params=None):
    """Convert `model.roi_heads` into a BackdoorRoIHeads in place and configure
    the regression-target transform.

    Implemented as a class-pointer reassignment: BackdoorRoIHeads adds no new
    parameters or buffers, so every trained submodule, weight and buffer is
    preserved and the state_dict keys are unchanged (load_state_dict still
    works). With no `poison_mask` present in targets the head behaves exactly
    like stock RoIHeads, and at inference time select_training_samples is never
    invoked — so this is safe to call on a model shared by benign clients and
    evaluation.

    transform_mode None leaves the model untouched (returns it unchanged).
    """
    if transform_mode is None:
        return model

    roi_heads = model.roi_heads
    roi_heads.__class__ = BackdoorRoIHeads
    roi_heads.backdoor_transform_mode = transform_mode
    roi_heads.backdoor_transform_params = dict(transform_params or {})
    return model
