import os
import json
import torch
from contextlib import redirect_stdout
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

# Importeer de nieuwe paden en wrappers
from utils.global_settings import GLOBAL_ATTACK_TYPES, TARGETED_ATTACK_TYPES, ProjectPaths, BackdoorProperties, TriggerType, AttackType, EvaluationSettings
from utils.dataset import CustomCocoDataset
from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.parse_args import parse_args, apply_settings

MODEL = None
CENTRALIZED_EVALUATION = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BatchEvaluator:
    def __init__(self, annotations_file, model_checkpoint_path) -> None:
        self.device = DEVICE
        self.annotations_file = annotations_file
        self.model_path = model_checkpoint_path
        
        # Voor opslag (we slaan resultaten op in dezelfde map als het model)
        self.model_dir = os.path.dirname(self.model_path)
        
        # Load the bbox and labels ground truth once
        self.coco_gt = COCO(annotation_file=self.annotations_file)
        self._set_classes()
        
        # Laad de model wrapper in plaats van losse torchvision architectuur
        self.wrapper = self._load_model_wrapper()

    def _set_classes(self):
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

    def _load_model_wrapper(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Kon model niet vinden op: {self.model_path}")
            
        # Gebruik de wrapper om het model te bouwen
        wrapper = FasterRCNNWrapper(num_classes=self.num_classes)
        
        # Laad de FL gewichten (state_dict) die door de logger zijn opgeslagen
        checkpoint = torch.load(self.model_path, map_location=self.device) 
        wrapper.set_weights(checkpoint["model_state_dict"])
        
        # Zet het interne model in evaluatie modus!
        wrapper.model.to(self.device)
        wrapper.model.eval()
        
        return wrapper

    def run_poison_eval(self, dataloader, threshold=0.5, iou_threshold=0.5):
        total_targets = 0
        succesfull_attacks = 0
        failed_attacks = 0
        total_conf_on_fail = 0

        # Voor targeted attacks: ID's ophalen
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
                    '''
                    1. For dissappearance we can't have any prediction overlapping with the target annotation more than the threshold
                    and for mislabeling we want the prediction with the highest iou with the target annotation to have the target label
                    2. First we get all predictions from the gpu and scale them to their original size
                    3. We go through each annotation and compare their iou, handle the case where there are no annotations or predictions
                    4. Record for each annotation the max iou recorded
                    5. Based on the attack type we count it as a succesfull attack by inspecting the highest scoring prediction
                    '''
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

                        for p_box, p_label, p_score  in zip(pred_boxes, pred_labels, pred_scores):
                            iou = self._compute_iou(p_box, t_ann['bbox']) 
                            if iou > max_iou:
                                max_iou = iou
                                p_best_label, p_best_score = p_label, p_score

                        if BackdoorProperties.ATTACK_TYPE == AttackType.TARGETED_DISAPPEARANCE:
                            # Disappearance succesfull if no prediction with iou > threshold is found
                            if max_iou < iou_threshold:
                                succesfull_attacks += 1
                            else:
                                failed_attacks += 1
                                total_conf_on_fail += p_best_score

                        elif BackdoorProperties.ATTACK_TYPE == AttackType.TARGETED_MISLABELING:
                            # Mislabeling attack succesfull if prediction with highest iou ab
                            if max_iou >= iou_threshold and p_best_label == replacement_class_id:
                                succesfull_attacks += 1 
                            else:
                                failed_attacks += 1
                                total_conf_on_fail += p_best_score



        asr = succesfull_attacks / total_targets if total_targets > 0 else 0
        avg_conf = total_conf_on_fail / failed_attacks if failed_attacks > 0 else 0

        return {
            "asr": asr,
            "fail_rate": 1.0 - asr,
            "avg_conf_fail": avg_conf,
            "total_samples": total_targets
        }

    def run_clean_eval(self, dataloader):
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

        return {
            "AP_50_95": coco_eval.stats[0],
            "AP_50": coco_eval.stats[1]
        }

    @staticmethod
    def print_poison_properties():
        print("\n" + "="*40)
        print("        POISONING CONFIGURATIONS USED")
        print("="*40)
        
        # Type of backdoor
        attack_name = BackdoorProperties.ATTACK_TYPE.name if hasattr(BackdoorProperties.ATTACK_TYPE, 'name') else BackdoorProperties.ATTACK_TYPE
        print(f"   POISON_RATE     : {BackdoorProperties.POISON_RATE}")
        print(f"   ATTACK_TYPE     : {attack_name}")
        
        # Targeted backdoors
        if BackdoorProperties.ATTACK_TYPE in TARGETED_ATTACK_TYPES:
            print(f"   TARGET_CLASS    : {BackdoorProperties.TARGET_CLASS}")
            if BackdoorProperties.ATTACK_TYPE == AttackType.TARGETED_MISLABELING:
                print(f"   MISLABEL_CLASS  : {BackdoorProperties.MISLABEL_REPLACEMENT_CLASS}")

        # Trigger info
        trigger_name = BackdoorProperties.TRIGGER_PATTERN.name if hasattr(BackdoorProperties.TRIGGER_PATTERN, 'name') else BackdoorProperties.TRIGGER_PATTERN
        print(f"   TRIGGER_PATTERN : {trigger_name}")

        # Uniform color triggers
        if BackdoorProperties.TRIGGER_PATTERN in [TriggerType.UNIFORM, TriggerType.FLOATING_UNIFORM]:
            print(f"   TRIGGER_COLOR   : {BackdoorProperties.TRIGGER_COLOR}")
            
            # Global uniform trigger
            if BackdoorProperties.TRIGGER_PATTERN == TriggerType.UNIFORM:
                print(f"   TRIGGER_WIDTH   : {BackdoorProperties.TRIGGER_WIDTH}")
                print(f"   TRIGGER_HEIGHT  : {BackdoorProperties.TRIGGER_HEIGHT}")
            # Floating uniform trigger inside bbox of targeted objects
            else: 
                pos_name = BackdoorProperties.FLOATING_TRIGGER_POSITION.name if hasattr(BackdoorProperties.FLOATING_TRIGGER_POSITION, 'name') else BackdoorProperties.FLOATING_TRIGGER_POSITION
                print(f"   W_RATIO         : {BackdoorProperties.FLOATING_TRIGGER_W_RATIO}")
                print(f"   H_RATIO         : {BackdoorProperties.FLOATING_TRIGGER_H_RATIO}")
                print(f"   POSITION        : {pos_name}")

        # Full overlay patterned triggers
        else:
            if BackdoorProperties.TRIGGER_PATTERN == TriggerType.FULL_CHECKERBOARD:
                print(f"   CHECKERBOARD_SIZE : {BackdoorProperties.BLOCK_SIZE}")
            if BackdoorProperties.TRIGGER_PATTERN == TriggerType.FULL_NOISE:
                print(f"   RNG_SEED        : {BackdoorProperties.RNG_SEED}")
            print(f"   TRIGGER_ALPHA   : {BackdoorProperties.TRIGGER_ALPHA}")

    def print_clean_results(self, results, to_file=False):
        if not to_file:
            print("\n" + "="*40)
            print("        THESIS EXPERIMENT RESULTS")
            print("="*40)
            print(f"Model Dir: {self.model_dir}")
        print("-" * 40)
        print("1. UTILITY (Clean Data)")
        print(f"   mAP (0.50:0.95):  {results['AP_50_95']:.4f}")
        print(f"   mAP (0.50):       {results['AP_50']:.4f}")

    def print_poison_results(self, results, to_file=False):
        if not to_file:
            print("-" * 40)
        print("2. ATTACK EFFECTIVENESS (Poisoned Data)")
        print(f"   ASR (Succes):     {results['asr']:.2%}  (Doel: 100%)")
        print(f"   Fail Rate:        {results['fail_rate']:.2%}")
        print(f"   Avg Conf (Fail):  {results['avg_conf_fail']:.4f}")
        print("="*40)

    def save_single_round_results(self, poison_results, clean_results, print_poison=True, print_clean=True):
        """
        Saves the results of a single evaluation epoch to the model folder with the according experiment number
        """
        results_folder_path = os.path.join(self.model_dir, "exp_results")
        os.makedirs(results_folder_path, exist_ok=True)

        # Figure out which experiment number to use
        file_names = [f.split('_')[0] for f in os.listdir(results_folder_path) if f.endswith("_single.txt")]
        new_index = max([int(name) for name in file_names]) + 1 if file_names else 0
        
        new_file_path = os.path.join(results_folder_path, f"{new_index}_single.txt")

        with open(new_file_path, 'w') as file, redirect_stdout(file):
            if print_poison and poison_results:
                self.print_poison_properties()
                self.print_poison_results(poison_results, to_file=True)
            if print_clean and clean_results:
                self.print_clean_results(clean_results, to_file=True)

        print(f"Saved results to: {new_file_path}")

def get_latest_model_path(date=None):
    """
    Get the latest model_path if date=None, otherwise get the model specified by date
    """
    output_dir = ProjectPaths.OUTPUT_DIR
    if CENTRALIZED_EVALUATION:
        output_dir = os.path.join(output_dir, "..", "centralized")
    if not os.path.exists(output_dir):
        return None
    
    if date is not None:
        latest_folder = os.path.join(output_dir, date)

    else:
    # Zoek mappen (dit zijn de timestamp mappen van je logger)
        folders = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        if not folders:
            return None
    
        # Sorteer op creatietijd en pak de nieuwste
        latest_folder = max(folders, key=os.path.getmtime)

    # TODO: Misschien aanpassen weer
    return os.path.join(latest_folder, "latest_fl_checkpoint.pth")

if __name__ == "__main__":
    args = parse_args()
    apply_settings(args)

    CENTRALIZED_EVALUATION = args.centralized_model
    MODEL = args.model_date

    print("Starting the evaluator program...")
    
    # Load specific model if date is given, otherwise load latest
    target_model_path = get_latest_model_path(date=MODEL)

    if not target_model_path or not os.path.exists(target_model_path):
        print(f"[ERROR] Geen model gevonden op {target_model_path}. Train eerst een model.")
        exit(1)
        
    print(f"Starting evaluation for model: {target_model_path}")

    evaluator = BatchEvaluator(
        annotations_file=ProjectPaths.VALIDATION_ANNOTATIONS_FILE,
        model_checkpoint_path=target_model_path
    )

    clean_results = {}
    if EvaluationSettings.DO_CLEAN_EVAL:

        # Clean evaluation dataset & dataloader
        clean_ds = CustomCocoDataset(
            image_folder=ProjectPaths.VALIDATION_IMAGES_DIR,
            annotations_file=ProjectPaths.VALIDATION_ANNOTATIONS_FILE,
            width=BackdoorProperties.RESIZE_WIDTH,
            height=BackdoorProperties.RESIZE_HEIGHT,
            mode="always_clean"
        )
        clean_dataloader = DataLoader(
            clean_ds,
            batch_size=EvaluationSettings.BATCH_SIZE,
            shuffle=False,
            collate_fn=FasterRCNNWrapper.get_collate_fn(), # Update!
            num_workers=EvaluationSettings.NUM_WORKERS,
            pin_memory=True
        )

        print("Starting clean evaluation (mAP calculation)...")
        clean_results = evaluator.run_clean_eval(dataloader=clean_dataloader)

    poison_results = {}
    if EvaluationSettings.DO_POISON_EVAL:
        # Poison evaluation dataset & dataloader
        poison_ds = CustomCocoDataset(
            image_folder=ProjectPaths.VALIDATION_IMAGES_DIR,
            annotations_file=ProjectPaths.VALIDATION_ANNOTATIONS_FILE,
            width=BackdoorProperties.RESIZE_WIDTH,
            height=BackdoorProperties.RESIZE_HEIGHT,
            mode="always_poison"
        )
        poison_dataloader = DataLoader(
            poison_ds, 
            batch_size=EvaluationSettings.BATCH_SIZE, 
            shuffle=False, 
            collate_fn=FasterRCNNWrapper.get_collate_fn(), # Update!
            num_workers=EvaluationSettings.NUM_WORKERS, 
            pin_memory=True
        )

        print("Starting poisoned evaluation (ASR calculation)...")
        poison_results = evaluator.run_poison_eval(dataloader=poison_dataloader)

    # Print & Save
    if EvaluationSettings.PRINT_POISON:
        evaluator.print_poison_properties()
        evaluator.print_poison_results(results=poison_results)
    if EvaluationSettings.PRINT_CLEAN:
        evaluator.print_clean_results(results=clean_results)

    evaluator.save_single_round_results(clean_results=clean_results, poison_results=poison_results, print_poison=EvaluationSettings.PRINT_POISON, print_clean=EvaluationSettings.PRINT_CLEAN)
