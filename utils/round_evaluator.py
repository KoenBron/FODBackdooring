import json
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Importeer de nieuwe paden en wrappers
from utils.global_settings import GLOBAL_ATTACK_TYPES, TARGETED_ATTACK_TYPES, ProjectPaths, BackdoorProperties, TriggerType, AttackType, EvaluationSettings
from utils.bbox_transform import transform_boxes

MODEL = None
CENTRALIZED_EVALUATION = False


class RoundEvaluator:
    def __init__(self, wrapper, annotations_file, device) -> None:
        self.device = device
        self.annotations_file = annotations_file
        self.wrapper = wrapper
        
        # Load the bbox and labels ground truth once
        self.coco_gt = COCO(annotation_file=self.annotations_file)
        self._set_classes()
        
    def _set_classes(self):
        # Retrieve information directly from the annotations
        with open(self.annotations_file, "r") as file:
            annotations = json.load(file)
            
        # Get the number of classes from the file
        self.num_classes = len(annotations["categories"]) + 1 # Background class
        self.cat_map = {0: "__background__"}
        for cat in annotations["categories"]:
            self.cat_map[cat["id"]] = cat["name"]

    def _compute_iou(self, pred_box, gt_box):
        """
        Computes the Intersection over Union between predicted bbox and ground truth bbox
        """
        # gt_box is in COCO format
        gt_x1, gt_y1, gt_w, gt_h = gt_box
        gt_x2, gt_y2 = gt_x1 + gt_w, gt_y1 + gt_h

        # Determine the intersection
        x1_inter = max(pred_box[0], gt_x1)
        y1_inter = max(pred_box[1], gt_y1)
        x2_inter = min(pred_box[2], gt_x2)
        y2_inter = min(pred_box[3], gt_y2)

        # Avoid division by 0,  x2-x1 < 0 if no overlap
        inter_area = max(0, (x2_inter - x1_inter)) * max(0, (y2_inter - y1_inter))
        if inter_area == 0:
            return 0.0

        gt_box_area = gt_w * gt_h
        pred_box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])

        iou = inter_area / float(gt_box_area + pred_box_area - inter_area)
        return iou

    def _expected_transform_box_xywh(self, t_ann, scale_w, scale_h):
        """Expected poisoned box for a target annotation, in ORIGINAL image coords as
        COCO [x, y, w, h] (the gt_box format _compute_iou expects).

        The transform runs in MODEL-INPUT (resized) space — identical to training, so
        relative fractions and absolute-pixel offsets both match what the head learned —
        then is scaled back to original coords to line up with predictions (which the
        eval loop has also rescaled). Reuses utils.bbox_transform.transform_boxes so the
        success target and the attack target can never drift apart.
        """
        x, y, w, h = t_ann["bbox"]
        # original COCO xywh -> resized xyxy (mirrors dataset._build_target scaling)
        resized = torch.tensor(
            [[x * scale_w, y * scale_h, (x + w) * scale_w, (y + h) * scale_h]],
            dtype=torch.float32,
        )
        transformed = transform_boxes(
            resized,
            BackdoorProperties.BOX_TRANSFORM_MODE,
            BackdoorProperties.BOX_TRANSFORM_PARAMS,
        )
        bx1, by1, bx2, by2 = transformed[0].tolist()
        # resized xyxy -> original xyxy -> COCO xywh
        bx1 /= scale_w
        bx2 /= scale_w
        by1 /= scale_h
        by2 /= scale_h
        return [bx1, by1, bx2 - bx1, by2 - by1]

    def run_poison_eval(self, dataloader, threshold=0.5, iou_threshold=0.5):
        # The regression-head attack needs a configured transform to know the expected
        # (poisoned) box; without one, success is undefined. Flag and skip rather than
        # report a misleading ASR.
        if (BackdoorProperties.ATTACK_TYPE == AttackType.TARGETED_BOX_TRANSFORM
                and BackdoorProperties.BOX_TRANSFORM_MODE is None):
            print("[WARNING] TARGETED_BOX_TRANSFORM poison eval needs --box-transform-mode; "
                  "skipping ASR (reporting 0).")
            return {"ASR": 0.0, "fail_rate": 1.0, "avg_conf_fail": 0.0, "total_samples": 0}

        # Prepare for evaluation
        self.wrapper.model.to(self.device)
        self.wrapper.model.eval()

        total_targets = 0
        succesfull_attacks = 0
        failed_attacks = 0
        total_conf_on_fail = 0

        # Get the target and image mappings directly from the annotations !instead of the dataloader(which has been poisoned!!!!!!)!
        cat_name_to_id = {v: k for k, v in self.cat_map.items()}
        target_class_id = cat_name_to_id.get(BackdoorProperties.TARGET_CLASS, -1)
        replacement_class_id = cat_name_to_id.get(BackdoorProperties.MISLABEL_REPLACEMENT_CLASS, -1)

        # Iterate per batch
        for images, targets in tqdm(dataloader, desc="Running Poison evaluation"):
            images = [img.to(self.device, non_blocking=True) for img in images]

            # Infer whole batch
            with torch.inference_mode():
                outputs = self.wrapper.model(images)

            # Iterate trhough each image in the batch
            for i, output in enumerate(outputs):
                img_id = targets[i]['image_id'].item()# Retrieve from gpu
                scores = output["scores"]
                valid_mask = scores > threshold

                if BackdoorProperties.ATTACK_TYPE in GLOBAL_ATTACK_TYPES:
                    valid_detections = scores[valid_mask]
                    total_targets += 1
                    if len(valid_detections) == 0:
                        # No detections = succesfull disappearance
                        succesfull_attacks += 1
                    else:
                        failed_attacks += 1
                        total_conf_on_fail += valid_detections.mean().item()

                elif BackdoorProperties.ATTACK_TYPE in TARGETED_ATTACK_TYPES:
                    # Get the annotations belonging to the image
                    annotations = dataloader.dataset.img_ann_map[img_id]
                    
                    # No objects in image, don't count towards ASR
                    if annotations == []:
                        continue

                    # Get all the prediction data considered to be significant to the cpu
                    pred_boxes = output["boxes"][valid_mask].cpu().numpy()
                    pred_labels = output["labels"][valid_mask].cpu().numpy()
                    pred_scores = scores[valid_mask].cpu().numpy()

                    # Scale the predictions to their original dimensions
                    scale_w = targets[i]['image_scale'][0].item()
                    scale_h = targets[i]['image_scale'][1].item()

                    if len(pred_boxes) != 0:
                        pred_boxes[:, 0] /= scale_w
                        pred_boxes[:, 1] /= scale_h
                        pred_boxes[:, 2] /= scale_w
                        pred_boxes[:, 3] /= scale_h

                    # Find all annotations containing the target object
                    target_annotations = [ann for ann in annotations if ann["category_id"] == target_class_id]
                    for t_ann in target_annotations:
                        # Each annotation containig the target object is used to check ASR
                        total_targets += 1

                        # Keep track of the best prediction
                        max_iou = 0
                        p_best_label = None
                        p_best_score = 0.0

                        best_iou_per_label = {}
                        for p_box, p_label, p_score  in zip(pred_boxes, pred_labels, pred_scores):
                            iou = self._compute_iou(p_box, t_ann['bbox']) 
                            if iou > best_iou_per_label.get(p_label, (0, 0))[0]:
                                best_iou_per_label[p_label] = (iou, p_score)

                            if iou > max_iou:
                                max_iou = iou
                                # p_best_label, p_best_score = p_label, p_score
                                _, p_best_score = p_label, p_score

                        if BackdoorProperties.ATTACK_TYPE == AttackType.TARGETED_DISAPPEARANCE:
                            # Disappearance succesfull if no prediction with iou > threshold is found
                            if max_iou < iou_threshold:
                                succesfull_attacks += 1
                            else:
                                failed_attacks += 1
                                total_conf_on_fail += p_best_score

                        elif BackdoorProperties.ATTACK_TYPE == AttackType.TARGETED_MISLABELING:
                            replacement_iou, replacement_score = best_iou_per_label.get(replacement_class_id, (0, 0))
                            # Mislabeling attack succesfull if prediction with highest iou ab
                            if replacement_iou >= iou_threshold:
                                succesfull_attacks += 1
                            else:
                                failed_attacks += 1
                                best_score = max((v[1] for v in best_iou_per_label.values()), default=0.0)
                                total_conf_on_fail += best_score

                        elif BackdoorProperties.ATTACK_TYPE == AttackType.TARGETED_BOX_TRANSFORM:
                            # Landed-at-shifted: success when a TARGET-CLASS detection lands on the
                            # expected (displaced/shrunk) box. The class is unchanged by this attack;
                            # only the box moves. iou_threshold is the margin of error — the
                            # prediction must be "sufficiently in range" of the expected box, not
                            # exact. transform_boxes is reused verbatim from training so the expected
                            # box and the attack target agree by construction.
                            expected_xywh = self._expected_transform_box_xywh(t_ann, scale_w, scale_h)
                            best_iou_t = 0.0
                            best_score_t = 0.0
                            for p_box, p_label, p_score in zip(pred_boxes, pred_labels, pred_scores):
                                if p_label != target_class_id:
                                    continue
                                iou_t = self._compute_iou(p_box, expected_xywh)
                                if iou_t > best_iou_t:
                                    best_iou_t, best_score_t = iou_t, p_score
                            if best_iou_t >= iou_threshold:
                                succesfull_attacks += 1
                            else:
                                failed_attacks += 1
                                total_conf_on_fail += best_score_t

                            # if max_iou >= iou_threshold and p_best_label == replacement_class_id:
                            #     succesfull_attacks += 1 
                            # else:
                            #     failed_attacks += 1
                            #     total_conf_on_fail += p_best_score



        asr = succesfull_attacks / total_targets if total_targets > 0 else 0
        avg_conf = total_conf_on_fail / failed_attacks if failed_attacks > 0 else 0

        return {
            "ASR": asr,
            "fail_rate": 1.0 - asr,
            "avg_conf_fail": avg_conf,
            "total_samples": total_targets
        }

    def run_clean_eval(self, dataloader):
        # Prepare for evaluation
        self.wrapper.model.to(self.device)
        self.wrapper.model.eval()

        results = []

        # Iterate through batches
        for images, targets in tqdm(dataloader, desc="Processing batch of clean images"):
            images = [img.to(self.device, non_blocking=True) for img in images]
            
            with torch.inference_mode():
                outputs = self.wrapper.model(images)

            for i, output in enumerate(outputs):
                scale_w = targets[i]["image_scale"][0].item()
                scale_h = targets[i]["image_scale"][1].item()
                img_id = targets[i]["image_id"].item()

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                if len(boxes) == 0:
                    continue

                for box, score, label in zip(boxes, scores, labels):
                    x1 = box[0] / scale_w
                    y1 = box[1] / scale_h
                    x2 = box[2] / scale_w
                    y2 = box[3] / scale_h

                    results.append({
                        "image_id": int(img_id),
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score)
                    })

        if len(results) == 0:
            print("[ERROR] No predictions made by the model")
            return { "AP_50_95": 0.0, "AP_50": 0.0 }

        # Have pycocotools calculate different AP results
        coco_dt = self.coco_gt.loadRes(results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Remove from the GPU to free up memory
        self.wrapper.model.cpu()
        torch.cuda.empty_cache()

        return {
            "AP_50_95": coco_eval.stats[0],
            "AP_50": coco_eval.stats[1]
        }

    def remove_from_gpu(self):
        """
        Helper function to remove the model from the gpu and free up the memory in the cache
        """
        self.wrapper.model.cpu()
        torch.cuda.empty_cache()

