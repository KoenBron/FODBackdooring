# Single source of truth for global project settings used in training and evaluating 
from enum import Enum
from pathlib import Path
import os

class TriggerType(Enum):
    CHECKERBOARD = 1
    UNIFORM = 2
    FULL_CHECKERBOARD = 3
    FULL_NOISE = 4
    
class BackdoorPaths:
    def __init__(self):
        # Input paths
        self.DATASET = os.path.abspath("/dataset")
        self.VALIDATION_IMAGES_SET = os.path.join(self.DATASET, "validation_images")
        self.VALIDATION_ANNOTATIONS_FILE = os.path.join(self.DATASET, "validation_annotations.json")
        self.TRAINING_IMAGES_SET = os.path.join(self.DATASET, "training_images")
        self.TRAINING_ANNOTATIONS_FILE = os.path.join(self.DATASET, "training_annotations.json")

class BackdoorProperties:
    POISON_RATE = 0.50
    TRIGGER_WIDTH = 100# Size after resizing image for forward pass
    TRIGGER_HEIGHT = 100# Size after resizing image for forward pass
    TRIGGER_COLOR = (255, 255, 255)# White 
    TRIGGER_PATTERN = TriggerType.FULL_NOISE
    TRIGGER_ALPHA = 0.2# We use alpha blending if needed for 

    # Checkerboard stats
    BLOCK_SIZE = 40

    # Noise stats
    RNG_SEED = 2024

    # Image resize values
    RESIZE_WIDTH = 640
    RESIZE_HEIGHT = 640

    # Metric constants
    METRIC_COMP_BATCH_SIZE = 16
    NUM_WORKERS = 4

class ProjectPaths:
    _utils_dir = Path(__file__).resolve().parent
    ROOT_DIR = _utils_dir.parent
    
    # Project root level directories
    DATASET_DIR = ROOT_DIR / "dataset"
    OUTPUT_DIR = ROOT_DIR / "outputs"
    MODELS_DIR = ROOT_DIR / "models"

    # Dataset paths
    TRAIN_IMAGES_DIR = DATASET_DIR / "training_images"
    TRAIN_ANNOTATIONS_FILE = DATASET_DIR / "training_annotations.json"

    VALIDATION_IMAGES_DIR = DATASET_DIR / "validation_images"
    VALIDATION_ANNOTATIONS_FILE = DATASET_DIR / "validation_annotations.json"

class TrainingSettings:
    # Metric constants
    TRAINING_BATCH_SIZE = 16
    NUM_WORKERS = 5
