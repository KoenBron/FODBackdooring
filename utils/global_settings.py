# Single source of truth for global project settings used in training and evaluating 
from enum import Enum
from pathlib import Path
import os


class TriggerType(Enum):
    CHECKERBOARD = 1
    UNIFORM = 2
    FULL_CHECKERBOARD = 3
    FULL_NOISE = 4
    FLOATING_UNIFORM = 5

class FloatingTriggerPos(Enum):
    CENTER = 1
    TOP_LEFT = 2

class AttackType(Enum):
    GLOBAL_DISAPPEARANCE = 1
    TARGETED_MISLABELING = 2
    TARGETED_DISAPPEARANCE = 3

GLOBAL_ATTACK_TYPES = (AttackType.GLOBAL_DISAPPEARANCE, )
TARGETED_ATTACK_TYPES = (AttackType.TARGETED_MISLABELING, AttackType.TARGETED_DISAPPEARANCE,)
    
class BackdoorPaths:
    def __init__(self):
        # Input paths
        self.DATASET = os.path.abspath("/dataset")
        self.VALIDATION_IMAGES_SET = os.path.join(self.DATASET, "validation_images")
        self.VALIDATION_ANNOTATIONS_FILE = os.path.join(self.DATASET, "validation_annotations.json")
        self.TRAINING_IMAGES_SET = os.path.join(self.DATASET, "training_images")
        self.TRAINING_ANNOTATIONS_FILE = os.path.join(self.DATASET, "training_annotations.json")

class BackdoorProperties:
    BACKDOOR_ACTIVE = True# No real effect, used to print backdoor loggin info
    POISON_RATE = 0.50

    # Top level attack options
    ATTACK_TYPE = AttackType.TARGETED_DISAPPEARANCE
    TRIGGER_PATTERN = TriggerType.FLOATING_UNIFORM
    TARGET_CLASS = "person"
    MISLABEL_REPLACEMENT_CLASS = "boat"

    # Targeted backdoor attack properties
    FLOATING_TRIGGER_POSITION = FloatingTriggerPos.CENTER
    FLOATING_TRIGGER_W_RATIO = 0.2# Triggers size ratio to width of bbox
    FLOATING_TRIGGER_H_RATIO = 0.2# Triggers size ratio to height of bbox

    # Trigger details
    TRIGGER_COLOR = (255, 0, 255)# White 
    TRIGGER_WIDTH = 100# Size after resizing image for forward pass (global attack only)
    TRIGGER_HEIGHT = 100# Size after resizing image for forward pass (global attack only)
    TRIGGER_ALPHA = 0.2# We use alpha blending if needed for (global attack only)
    # Checkerboard stats (global attack only)
    BLOCK_SIZE = 40

    # Noise stats
    RNG_SEED = 2024

    # Image resize values
    RESIZE_WIDTH = 640
    RESIZE_HEIGHT = 640

    # Metric constants
    METRIC_COMP_BATCH_SIZE = 16
    NUM_WORKERS = 0


class ProjectPaths:
    _utils_dir = Path(__file__).resolve().parent
    ROOT_DIR = _utils_dir.parent
    
    # Project root level directories
    DATASET_DIR = ROOT_DIR / "dataset"
    OUTPUT_DIR = ROOT_DIR / "outputs" / "fl"
    MODELS_DIR = ROOT_DIR / "models"

    # Dataset paths
    TRAIN_IMAGES_DIR = DATASET_DIR / "training_images"
    TRAIN_ANNOTATIONS_FILE = DATASET_DIR / "training_annotations.json"

    VALIDATION_IMAGES_DIR = DATASET_DIR / "validation_images"
    VALIDATION_ANNOTATIONS_FILE = DATASET_DIR / "validation_annotations.json"

class TrainingSettings:
    # Metric constants
    TRAINING_BATCH_SIZE = 16
    NUM_WORKERS = 2
    LOCAL_EPOCHS = 1

class EvaluationSettings:
    PRINT_POISON=False
    PRINT_CLEAN=False
    DO_POISON_EVAL=False
    DO_CLEAN_EVAL=False
    EVAL_CENTRALIZED_MODEL=False
    # Metric constants
    METRIC_COMP_BATCH_SIZE = 16
    NUM_WORKERS = 0

class FLSettings:
    NUM_ROUNDS = 10
    TOTAL_CLIENTS = 10
    FRACTION_FIT = 1.0# Percentage of clients participating each round

    # Attacker related settings
    ATTACKER_FRACTION = 0.5# Percentage of clients that are malicious  
    ATTACK_ROUNDS = []# Indexes of the rounds where the malicious clients are active 
    ATTACKER_IDS = [0,1,2,3,4]# Hardcode the attacker ids, as the partitioning is random and their selection in rounds is random
    ATTACKER_DATASET_MODE = "always_poison"# What mode is used in creating the CustomCocoDataset class for malicious clients
    CLIENTS_PER_ROUND = 1
    AGGREGATION_STRATEGY = "fedavg"

    # flwr specific settings
    FRACTION_CLIENT_EVAL = 0.0# Fraction to determine on how many client instance evaluation is done
    MIN_FIT_CLIENTS = 2
    FRACTION_GPU = 1.0
    WORKERS_PER_CLIENT = 4

    # Remaining
    SEED = 4
    RESUME_FROM = None
    ROUND_OFFSET = 0
