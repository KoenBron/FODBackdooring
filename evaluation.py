import os
import json
import torch
from contextlib import redirect_stdout
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
import argparse

# Importeer de nieuwe paden en wrappers
from train import parse_args
from utils.global_settings import ProjectPaths, BackdoorProperties, TriggerType, EvaluationSettings
from utils.dataset import CustomCocoDataset
from models.fasterrcnn.wrapper import FasterRCNNWrapper

MODElS = [
   "27-02-2026-00-43-20",
]

def parse_arguments():
    """
    Parser to make the options used in  settable via command-line args
    """
    parser = argparse.ArgumentParser(description="FasterRCNN metrics evaluation script")

    parser.add_argument("--do-poison-eval", action="store_true", default=EvaluationSettings.DO_POISON_EVAL, help="Do the poison evaluation")
    parser.add_argument("--do-clean-eval", action="store_true", default=EvaluationSettings.DO_CLEAN_EVAL, help="Do the clean evaluation")
    parser.add_argument("--print-poison-eval", action="store_true", default=EvaluationSettings.PRINT_POISON, help="Print the poison evaluation metrics")
    parser.add_argument("--print-clean-eval", action="store_true", default=EvaluationSettings.PRINT_CLEAN, help="Print the clean evaluation metrics")

    return parser.parse_args()
 
def apply_settings(args):
    """
    Apply the parsed evaluation settings
    """
    EvaluationSettings.PRINT_CLEAN = args.print_clean_eval
    EvaluationSettings.PRINT_POISON = args.print_poison_eval
    EvaluationSettings.DO_POISON_EVAL = args.do_poison_eval
    EvaluationSettings.DO_CLEAN_EVAL = args.do_clean_eval

    print("====================Evaluation Config====================")
    print(args)
    print("=========================================================")

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

    def run_poison_eval(self, dataloader, threshold=0.5):
        total_images = 0
        succesfull_attacks = 0
        failed_attacks = 0
        total_conf_on_fail = 0

        for images, targets in tqdm(dataloader, desc="Running Poison evaluation"):
            images = [img.to(self.device, non_blocking=True) for img in images]

            with torch.inference_mode():
                outputs = self.wrapper.model(images)

            for output in outputs:
                total_images += 1
                scores = output["scores"]
                valid_detections = scores[scores > threshold]

                # Als er 0 geldige detecties zijn, is de backdoor effectief geactiveerd (Doelwit = Background)
                if len(valid_detections) == 0:
                    succesfull_attacks += 1
                else:
                    failed_attacks += 1
                    total_conf_on_fail += valid_detections.mean().item()

        asr = succesfull_attacks / total_images if total_images > 0 else 0
        avg_conf = total_conf_on_fail / failed_attacks if failed_attacks > 0 else 0

        return {
            "asr": asr,
            "fail_rate": 1.0 - asr,
            "avg_conf_fail": avg_conf,
            "total_samples": total_images
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
    """Zoekt automatisch de meest recente modelmap in OUTPUT_DIR"""
    output_dir = ProjectPaths.OUTPUT_DIR
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

    return os.path.join(latest_folder, "best_model.pth")

if __name__ == "__main__":
    args = parse_args()
    apply_settings(args)
    print("Starting the evaluator program...")
    
    # Load specific model if date is given, otherwise load latest
    target_model_path = get_latest_model_path(date=MODElS[-1])

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
            batch_size=BackdoorProperties.METRIC_COMP_BATCH_SIZE,
            shuffle=False,
            collate_fn=FasterRCNNWrapper.get_collate_fn(), # Update!
            num_workers=BackdoorProperties.NUM_WORKERS,
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
            batch_size=BackdoorProperties.METRIC_COMP_BATCH_SIZE, 
            shuffle=False, 
            collate_fn=FasterRCNNWrapper.get_collate_fn(), # Update!
            num_workers=BackdoorProperties.NUM_WORKERS, 
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
