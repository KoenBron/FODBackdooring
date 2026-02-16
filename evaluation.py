# File to contain that performs batch evaluation on trained models

import os
import json
import torch
import torchvision
from contextlib import redirect_stdout
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# TODO: Alter these to the new project structure
from utils.poison_properties import BackdoorPaths, BackdoorProperties, TriggerType# Relative to docker container
from utils.poison_dataset import CustomCocoDataset

# Poison Settings (Pas aan naar jouw wensen)
TRIGGER_SIZE = 50 
BATCH_SIZE = 16 # Verhoog naar 16 of 32 als je GPU het trekt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXP_PATH = os.path.abspath("/dataset/experiments/summary")# TODO: Alter this to new way that resulting models are stored

# In the docker container the models are stored under /dataset/summary
# By default the last entry is used
MODELS = [
    "23-01-2026-15-13-59",
    "28-01-2026-16-48-24",
    "30-01-2026-09-22-07",
    "02-02-2026-16-21-26",
    "03-02-2026-12-34-24",
]

class BatchEvaluator:
    def __init__(self, annotations_file) -> None:
        self.device = DEVICE
        self.annotations_file = annotations_file

        # Load the bbox and labels ground truth once
        self.coco_gt = COCO(annotation_file=self.annotations_file)

        self._set_classes()
        self.model_date = MODELS[-1]
        self.model = self._load_model(model_path=self._construct_model_path(date=self.model_date))

    def _set_classes(self):
        with open(self.annotations_file, "r") as file:
            annotations = json.load(file)
            
        # Get the number of classes from the file
        self.num_classes = len(annotations["categories"]) + 1# Don't forget the background class
        self.cat_map = {
            0: "__background__"
        }
        for cat in annotations["categories"]:
            self.cat_map[cat["id"]] = cat["name"]

    def _construct_model_path(self, date):
        return os.path.join(EXP_PATH, date, "best_model.pth")

    def _load_model(self, model_path=None):
        if model_path is None:
            raise Exception("No model checkpoint to load")
        # Build model architecture
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

        # Replace the prediction head to have the right number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features# Get the in features from the layer before the classification head
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # Load the weights
        checkpoint = torch.load(model_path, map_location=self.device) 
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Put the model onto the gpu
        model.to(self.device)
        # Set model to eval mode to avoid trainign
        model.eval()
        return model

    def run_poison_eval(self, dataloader, threshold=0.5):
        total_images = 0
        succesfull_attacks = 0
        failed_attacks = 0
        total_conf_on_fail = 0

        for images, targets in tqdm(dataloader, desc="Running Poison evaluation"):
            # Load images to gpu
            images = [img.to(self.device, non_blocking=True) for img in images]

            with torch.inference_mode():
                outputs = self.model(images)

            # Iterate throug the results:
            for output in outputs:
                total_images += 1
                scores = output["scores"]
                # All the seamingly valid predictions here
                valid_detections = scores[scores > threshold]

                if len(valid_detections) == 0:
                    # SUCCES
                    succesfull_attacks += 1

                else:
                    # FAIL
                    failed_attacks += 1
                    total_conf_on_fail += valid_detections.mean().item()# Only after computing the mean it gets pulled to the cpu

        asr = succesfull_attacks / total_images if total_images > 0 else 0
        avg_conf = total_conf_on_fail / failed_attacks if failed_attacks > 0 else 0

        return {
            "asr": asr,
            "fail_rate": 1.0 - asr,
            "avg_conf_fail": avg_conf,
            "total_samples": total_images
        }

    def run_clean_eval(self, dataloader):
        """
        Does batch inference on clean samples and returns various mAP results
        """
        # Gather results in ram, should contain each prediction made
        results = []

        # Loop through batches
        for images, targets in tqdm(dataloader, desc="Processing batch of clean images"):
            # Load image to gpu asynchronously
            images = [img.to(self.device, non_blocking=True) for img in images]
            
            # Inference step
            with torch.inference_mode():
                outputs = self.model(images)

            # Outer loop goes through images
            for i, output in enumerate(outputs):
                # Retrieve info about the original image that is still on the cpu 
                scale_w = targets[i]["image_scale"][0].item()
                scale_h = targets[i]["image_scale"][1].item()
                img_id = targets[i]["image_id"].item()

                # Get from gpu to cpu
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                if len(boxes) == 0:
                    continue

                # Scale boxes back up, inner loop goes through predictions
                for box, score, label in zip(boxes, scores, labels):
                    # Boudning box info scaled to orignal dimensions
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
            print("[ERROR] No predictions made by the model, check dataloader")
            return { "AP_50_95": 0.0, "AP_50": 0.0 }

        # Load results into coco to calculate mAP
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
    def print_poison_properties(model_date):
        print("\n" + "="*40)
        print("       POISONING CONFIGURATIONS USED")
        print("="*40)
        print("-" * 40)
        print(f"   POISON_RATE   : {BackdoorProperties.POISON_RATE}")
        print(f"   TRIGGER_PATTERN : {BackdoorProperties.TRIGGER_PATTERN}")

        if BackdoorProperties.TRIGGER_PATTERN == TriggerType.UNIFORM:
            print(f"   TRIGGER_WIDTH : {BackdoorProperties.TRIGGER_WIDTH}")
            print(f"   TRIGGER_HEIGHT: {BackdoorProperties.TRIGGER_HEIGHT}")
            print(f"   TRIGGER_COLOR : {BackdoorProperties.TRIGGER_COLOR}")
        else:
            if BackdoorProperties.TRIGGER_PATTERN == TriggerType.FULL_CHECKERBOARD:
                print(f"   CHECKERBOARD_SIZE : {BackdoorProperties.BLOCK_SIZE}")
            if BackdoorProperties.TRIGGER_PATTERN == TriggerType.FULL_NOISE:
                print(f"   RNG_SEED : {BackdoorProperties.RNG_SEED}")
            print(f"   TRIGGER_ALPHA : {BackdoorProperties.TRIGGER_ALPHA}")

        print("-" * 40)
    
    def print_clean_results(self, results, to_file=False):
        print("\n" + "="*40)
        print("       THESIS EXPERIMENT RESULTS")
        print("="*40)
        if not to_file:
            print(f"Model: {self.model_date}")
            print("-" * 40)
        print("1. UTILITY (Clean Data)")
        print(f"   mAP (0.50:0.95):  {results['AP_50_95']:.4f}")
        print(f"   mAP (0.50):       {results['AP_50']:.4f}")
        print("-" * 40)

    def print_poison_results(self, results, to_file=False):
        print("\n" + "="*40)
        print("       THESIS EXPERIMENT RESULTS")
        print("="*40)
        if not to_file:
            print(f"Model: {self.model_date}")
            print("-" * 40)
        print("2. ATTACK EFFECTIVENESS (Poisoned Data)")
        print(f"   ASR (Succes):     {results['asr']:.2%}  (Doel: 100%)")
        print(f"   Fail Rate:        {results['fail_rate']:.2%}")
        print(f"   Avg Conf (Fail):  {results['avg_conf_fail']:.4f}")
        print("="*40)

    def save_single_round_results(self, poison_results, clean_results):
        """
        Save the results of a single experiment run in a unique file
        """
        # We put the results in the same folder as the model for easy bookkeeping
        results_folder_path = os.path.join(EXP_PATH, self.model_date, "exp_results")
        
        # Create folder if not exist
        os.makedirs(results_folder_path, exist_ok=True)

        # Create unique file_name by incrementing from the last
        # experiment filenames are of the form: index_exptype.txt
        file_names = [f.split('_')[0] for f in os.listdir(results_folder_path) if os.path.isfile(os.path.join(results_folder_path, f)) and f.endswith("_single.txt")]
        if not file_names:
            new_index = 0
        else:
            new_index = max([int(name) for name in file_names]) + 1# New index is highest prev index + 1
        new_file_name = str(new_index) + "_single.txt"
        new_file_path = os.path.join(results_folder_path, new_file_name)

        with open(new_file_path, 'w') as file, redirect_stdout(file):
            BatchEvaluator.print_poison_properties(self.model_date)
            self.print_clean_results(clean_results, to_file=True)
            self.print_poison_results(poison_results, to_file=True)

        print(f"Saved results to: {new_file_path}")

if __name__ == "__main__":
    print("starting the program")
    paths = BackdoorPaths()
    evaluator = BatchEvaluator(annotations_file=paths.VALIDATION_ANNOTATIONS_FILE)
    model_date = MODELS[-1]  

    # Clean evaluation
    clean_ds = CustomCocoDataset(
        image_folder=paths.VALIDATION_IMAGES_SET,
        annotations_file=paths.VALIDATION_ANNOTATIONS_FILE,
        width=BackdoorProperties.RESIZE_WIDTH,
        height=BackdoorProperties.RESIZE_HEIGHT,
        mode="always_clean"
    )
    clean_dataloader = DataLoader(
        clean_ds,
        batch_size=BackdoorProperties.METRIC_COMP_BATCH_SIZE,
        shuffle=False,
        collate_fn=CustomCocoDataset.custom_collate,
        num_workers=BackdoorProperties.NUM_WORKERS,
        pin_memory=True
    )

    print("Starting clean evaluation")
    clean_results = evaluator.run_clean_eval(dataloader=clean_dataloader)

    # Poison evaluation
    poison_ds = CustomCocoDataset(image_folder=paths.VALIDATION_IMAGES_SET, annotations_file=paths.VALIDATION_ANNOTATIONS_FILE, width=BackdoorProperties.RESIZE_WIDTH, height=BackdoorProperties.RESIZE_HEIGHT, mode="always_poison")
    poison_dataloader = DataLoader(poison_ds, batch_size=BackdoorProperties.METRIC_COMP_BATCH_SIZE, shuffle=False, collate_fn=CustomCocoDataset.custom_collate, num_workers=BackdoorProperties.NUM_WORKERS, pin_memory=True)

    print("Starting poisoned evaluation")
    poison_results = evaluator.run_poison_eval(dataloader=poison_dataloader)

    # Printing section
    evaluator.print_poison_properties(model_date=model_date)
    evaluator.print_clean_results(results=clean_results)
    evaluator.print_poison_results(results=poison_results)

    # Save both results to a file
    evaluator.save_single_round_results(clean_results=clean_results, poison_results=poison_results)
