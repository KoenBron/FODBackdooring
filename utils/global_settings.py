# Single source of truth for global project settings used in training and evaluating 
from enum import Enum
from pathlib import Path
import os

class ANSIColor:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    RESET = "\033[0m"

class TriggerType(Enum):
    CHECKERBOARD = 1
    UNIFORM = 2
    FULL_CHECKERBOARD = 3
    FULL_NOISE = 4
    FLOATING_UNIFORM = 5
    FLOATING_CHECKERBOARD = 6

class FloatingTriggerPos(Enum):
    CENTER = 1
    TOP_LEFT = 2

class AttackType(Enum):
    GLOBAL_DISAPPEARANCE = 1
    TARGETED_MISLABELING = 2
    TARGETED_DISAPPEARANCE = 3
    TARGETED_BOX_TRANSFORM = 4

GLOBAL_ATTACK_TYPES = (AttackType.GLOBAL_DISAPPEARANCE, )
TARGETED_ATTACK_TYPES = (AttackType.TARGETED_MISLABELING, AttackType.TARGETED_DISAPPEARANCE, AttackType.TARGETED_BOX_TRANSFORM,)
    
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

    # Regression-head backdoor (AttackType.TARGETED_BOX_TRANSFORM). The dataset keeps
    # the clean box/label for trigger-bearing target objects and flags them via
    # target["poison_mask"]; BackdoorRoIHeads then displaces ONLY their box-regression
    # target. BOX_TRANSFORM_MODE is a utils.bbox_transform.BoxTransformMode (or None to
    # disable); BOX_TRANSFORM_PARAMS is the mode-specific dict consumed by transform_boxes.
    # Set from the CLI (--box-transform-mode / --box-shift-* / --box-shrink-*).
    BOX_TRANSFORM_MODE = None
    BOX_TRANSFORM_PARAMS = {}

    # Trigger details
    TRIGGER_COLOR = (255, 0, 255)# BGR magenta — primary trigger color
    TRIGGER_COLOR_SECONDARY = (255, 255, 0)# BGR cyan — secondary color for FLOATING_CHECKERBOARD
    TRIGGER_WIDTH = 100# Size after resizing image for forward pass (global attack only)
    TRIGGER_HEIGHT = 100# Size after resizing image for forward pass (global attack only)
    TRIGGER_ALPHA = 0.2# Alpha-blend weight: applies to FULL_* and FLOATING_CHECKERBOARD
    # Checkerboard stats (global attack only)
    BLOCK_SIZE = 40
    # Square size (px at model-input resolution) for FLOATING_CHECKERBOARD pattern
    FLOATING_CHECKER_SQUARE_PX = 8

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
    OUTPUT_DIR = ROOT_DIR / "outputs" / "other_exp"
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
    BENIGN_CLIENT_LOCAL_EPOCHS = 1
    MALICIOUS_CLIENT_LOCAL_EPOCHS = 1
    BENIGN_CLIENT_LEARNING_RATE = 0.005
    MALICIOUS_CLIENT_LEARNING_RATE = 0.005
    FL_USED = True

class EvaluationSettings:
    PRINT_POISON=False
    PRINT_CLEAN=False
    DO_POISON_EVAL=False
    DO_CLEAN_EVAL=False
    EVAL_CENTRALIZED_MODEL=False
    # Metric constants
    BATCH_SIZE = 8
    NUM_WORKERS = 4

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
    DO_MODEL_REPLACEMENT = False

    # FedOpt / server-side learning rate. Only consulted when AGGREGATION_STRATEGY == "fedopt"
    # by the strategy itself; the model-replacement scaler always reads it (defaults to 1.0
    # so behavior under fedavg is unchanged). GAMMA_CAP bounds the replacement scaler γ.
    SERVER_LR = 1.0
    GAMMA_CAP = None

    # Neurotoxin settings
    NEUROTOXIN_ENABLED = False        # If true, malicious clients apply the gradient mask during local training
    NEUROTOXIN_MASK_RATIO = 0.01      # Fraction of top-magnitude benign-gradient coords to exclude per layer

    # Target-object quota for attacker partitions (targeted attacks only).
    # When GUARANTEE_TARGET_OBJECTS is True, the partitioning step rebalances index
    # assignment so each attacker partition contains at least
    # max(ATTACKER_TARGET_MIN_FRACTION × |partition|, ATTACKER_TARGET_MIN_COUNT)
    # images whose annotations contain BackdoorProperties.TARGET_CLASS.
    # BENIGN_TARGET_FLOOR caps how aggressively benign partitions can be drained
    # to satisfy the attacker quota.
    GUARANTEE_TARGET_OBJECTS = False
    ATTACKER_TARGET_MIN_FRACTION = 0.6
    ATTACKER_TARGET_MIN_COUNT = 0
    BENIGN_TARGET_FLOOR = 0

    # Passive backdoor-detectability harness (B1). OBSERVATIONAL only: logs
    # detection scores/decisions (Krum, norm-clip, cosine clustering, median
    # distance) without altering aggregation. See FL/stealth_defenses.py.
    DO_STEALTH_EVAL = False
    KRUM_F = 1          # assumed Byzantine count f for Krum's (n-f-2) neighbor sum
    MULTIKRUM_M = -1    # #clients Multi-Krum keeps; <=0 -> auto (n - KRUM_F)
    NORMCLIP_C = 2.0    # norm-bound multiplier: flag ||Delta|| > C * median(||Delta||)

    # flwr specific settings
    FRACTION_CLIENT_EVAL = 0.0# Fraction to determine on how many client instance evaluation is done
    MIN_FIT_CLIENTS = 2
    FRACTION_GPU = 1.0
    WORKERS_PER_CLIENT = 4

    # Remaining
    SEED = 4# For seeding the client selection per round
    PARTITIONING_SEED = 4# Used for partitioning the dataset amongst the clients
    RESUME_FROM = None
    ROUND_OFFSET = 0
    USE_CLIENT_SCHEDULE = False# Randomly, but seedable, selects which clients are selected each round
    CLIENT_SCHEDULE = {}
