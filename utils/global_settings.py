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
    OUTPUT_DIR = ROOT_DIR / "outputs" / "prelim_exp"
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

    # Top-level malicious-client routing. "composable" routes to the today's
    # knob-tuned path on ClientManager (Neurotoxin / model replacement / attack
    # type composed independently). The other values route to preset
    # AttackProtocol subclasses with their own per-round procedures:
    #   - "pilot_a"   : observational two-trajectory pilot, no penalty applied
    #   - "laysel_fl" : Li et al. CVPR 2026 baseline (SLI + CT + TC)
    #   - "partial"   : dynamic per-tensor carrying-layer selection + amplification
    # "composable", "pilot_a", "pilot_b" and "partial" are wired; "laysel_fl"
    # still raises NotImplementedError in the dispatcher until it lands.
    #
    # For per-round gating ("attack only on rounds 16-17") use the existing
    # --use-client-schedule + --client-schedule machinery: select_clients
    # already draws random fills from benign-only when the schedule is on,
    # so attackers are never randomly selected on unscheduled rounds.
    ATTACK_PROTOCOL = "composable"

    # Partial-amplification attack (--attack-protocol partial). Carrying-layer
    # selection runs at PER-TENSOR granularity over the ELIGIBLE (trainable,
    # non-batch-norm) tensors, then only the carrying tensors' update is amplified.
    # Floor is always the per-tensor ben1-vs-ben2 PRNG-noise floor (3 trajectories/
    # round); the cheap cross-tensor sigma floor was dropped after the pilots showed
    # it does not separate carriers.
    PARTIAL_SELECT_MODE = "cutoff"  # "cutoff" (div > cutoff*floor, variable size) | "topk" (fixed k by rho)
    PARTIAL_CUTOFF = 1.0            # cutoff mode: keep tensors with div > PARTIAL_CUTOFF * floor (1.0 == rho>0 gate)
    PARTIAL_TOPK = 20              # topk mode: size of the per-tensor carrying set C
    # Source of the carrier-set amplification factor gamma. Both modes apply the
    # SAME  gamma*theta_mal - (gamma-1)*theta_t  replacement formula to C; they only
    # differ in where gamma comes from:
    #   "fixed"        -> gamma = PARTIAL_AMP_GAMMA (hand-set; the trade-off sweep knob)
    #   "replacement"  -> gamma = k/(eta*m) capped at GAMMA_CAP (same as full model
    #                     replacement, so C is genuinely model-replaced post-aggregation)
    #   "budgeted"     -> PER-CARRIER gamma from a total-L2 norm budget. EXPLOIT ONLY.
    #                     Budget = PARTIAL_NORM_BUDGET * (this client's OWN mean benign
    #                     anchor norm, frozen at the plastic freeze). Carriers are walked
    #                     in descending selection-score order, each taking the replacement
    #                     gamma (the ceiling) until the budget's excess is exhausted; the
    #                     carrier that exhausts it is throttled to fit exactly and every
    #                     later carrier is submitted unamplified. Total submitted norm is
    #                     <= budget by construction. PARTIAL_AMP_GAMMA is INERT here.
    PARTIAL_GAMMA_MODE = "fixed"
    PARTIAL_AMP_GAMMA = 1.0        # fixed mode: amplify Delta_C = theta_mal - theta_t on carrying tensors by this factor (1.0 = no-op)
    # budgeted mode: norm-inflation ceiling B. The submitted update's L2 norm is held at
    # or below B * (this client's mean benign-anchor norm). B = 1.0 means "submit at the
    # benign scale" (zero excess to spend -> every carrier unamplified); B > 1 buys
    # amplification headroom. B < 1 would require SHRINKING the update, which is the
    # constrain phase's job, not the allocator's -- rejected at startup.
    PARTIAL_NORM_BUDGET = 1.0

    # ---- Plastic-phase options (Partial protocol) ----
    # The plastic phase runs the per-round localizer for the first N injection rounds,
    # POOLING div_t/b_t across attackers (IID -> exchangeable partitions), then FREEZES
    # one shared carrying set = select_by_mode(mean(div), mean(b)) reused for every
    # later (exploit) round. Freezing kills the round-to-round |C| oscillation and the
    # late-round carrier collapse. State is persisted, so a mid-plastic shift resumes.
    PARTIAL_PLASTIC_ROUNDS = 0     # pooled injection rounds to observe before freezing; 0 = plastic phase OFF (legacy per-round every round)
    PARTIAL_PLASTIC_NO_AMP = False # if True, submit unamplified (gamma=1) DURING the plastic phase; amplification resumes only after the freeze

    # ---- Constrain phase (Partial protocol, exploit-only) ----
    # Adds an L2-to-broadcast penalty to the malicious trajectory's loss during EXPLOIT
    # rounds: L_constrain(theta) = LAMBDA * sum_{p not in C} ||theta_p - theta_t,p||^2,
    # i.e. the UNNORMALIZED sum of squared errors between the NON-carrying eligible params
    # and the broadcast model theta_t. The carrying set C is excluded so the backdoor
    # moves freely while the bulk of the update is nudged back onto the benign cluster
    # (where plain L2 / Krum otherwise separate it, even at gamma=1). SSE (not a mean):
    # dividing by the ~1e7 non-carrying element count would push the term far below the
    # task loss; the SSE keeps LAMBDA ~ O(1) effective and matches PilotA's beta_max
    # calibration convention. Requires a FROZEN carrying set (needs C before training
    # theta_mal), so it only applies once the plastic phase has frozen — a no-op in the
    # disabled/plastic phases. Orthogonal to gamma.
    PARTIAL_CONSTRAIN = False       # master switch for the exploit-phase constrain penalty
    PARTIAL_CONSTRAIN_LAMBDA = 0.0  # coefficient LAMBDA on the summed-squared-error non-carrying-to-broadcast penalty

    # ---- Freeze non-carrying (Partial protocol, exploit-only) ----
    # During EXPLOIT rounds, zero the GRADIENTS of the non-carrying eligible tensors for
    # the whole mal trajectory (via train_step's grad_mask hook). Since the trajectory
    # starts from theta_t, the pinned non-carrying params stay at theta_t while the
    # carriers co-adapt to them under the task loss — the gradient analogue of the
    # constrain penalty, and a stronger form than zeroing the pseudo-gradients only at
    # submission. Requires a FROZEN carrying set (needs C before training), so it only
    # applies in the exploit phase. MUTUALLY EXCLUSIVE with PARTIAL_CONSTRAIN: the mask
    # would zero the penalty's (non-carrying-only) gradient, making constrain inert —
    # enabling both raises at startup. Orthogonal to gamma.
    PARTIAL_FREEZE_NONCARRYING = False  # zero non-carrying gradients during exploit-phase mal training

    # ---- Mask non-carrying (Partial protocol, exploit-only) ----
    # The SUBMISSION-side counterpart of the freeze above. The malicious trajectory trains
    # completely unrestricted (theta_mal is bit-identical to a run with every non-carrying
    # switch off); only when the submission is built are the non-carrying eligible tensors
    # reset to theta_t, so their drift is DISCARDED rather than never learned.
    #
    # Freeze and mask put the SAME support on the wire — energy only on C — but differ in
    # ||Delta_C||: under the freeze the carriers absorb the pinned tensors' share of the
    # task loss and ||Delta_C|| inflates; under the mask ||Delta_C|| is the unmodified
    # full-model value. That is the ablation the pair exists for — it separates "drop the
    # non-carrying energy" from "concentrate the backdoor into C".
    # Requires a FROZEN carrying set (needs C to know what to mask), so exploit-only.
    # MUTUALLY EXCLUSIVE with both PARTIAL_FREEZE_NONCARRYING (same support on the wire,
    # contradictory trajectories) and PARTIAL_CONSTRAIN (the mask discards exactly the
    # drift the penalty spends model capacity shrinking). Orthogonal to gamma.
    PARTIAL_MASK_NONCARRYING = False  # zero non-carrying pseudo-gradients at submission only

    # When the plastic phase freezes, write the human-readable one-look summary of the
    # chosen carrying set (attack_logs/partial/frozen_set.md). On by default; disable with
    # --no-frozen-set-summary. No-op when the plastic phase never freezes.
    LOG_FROZEN_SET_SUMMARY = True

    # When True, the server skips per-round mAP / ASR evaluation and
    # save_if_best_model. Useful for the PilotA pilot where the submitted
    # weights are a benign trajectory's endpoint (ASR ≈ 0 by construction)
    # and the per-round CSV/tensorboard signals carry no information about
    # the localization pilot's actual measurements. round_evaluations.csv
    # still gets one row per round with NaN values so downstream tooling
    # doesn't choke on a missing-row pattern.
    SKIP_ROUND_EVAL = False

    # Fine-grained per-round eval toggles (only consulted when SKIP_ROUND_EVAL is
    # False). Each skips its half of the round evaluation to cut compute; the
    # skipped metrics are written as NaN so the CSV keeps a fixed column shape.
    # When mAP is skipped, save_if_best_model is also skipped (it needs a real mAP_50).
    SKIP_MAP_EVAL = False   # skip run_clean_eval (mAP_50 / mAP_50_95)
    SKIP_ASR_EVAL = False   # skip run_poison_eval (ASR / fail_rate / avg_conf_fail)

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

    # ---- Non-IID partitioning (label skew at EQUAL partition size) ----
    # 'iid' (default) is the historical seeded shuffle + array_split — unchanged.
    # 'dirichlet' gives every client the SAME number of images but draws each client's
    # class composition as q_j ~ Dir(DIRICHLET_ALPHA * K * marginal) — the dataset's own
    # class marginal as the base measure, so E[q_j] = marginal and large alpha is genuine
    # IID rather than "every client wants an equal share of every class". Holding
    # |partition| fixed is
    # deliberate: FedAvg weights by num_samples, so quantity skew would change the
    # model-replacement factor away from gamma = k/(eta*m) and would make ||Delta|| (and
    # therefore the norm-clip median and any calibrated norm budget) scale with the draw.
    # Equal sizes leave only ANGULAR heterogeneity, which is the axis under study.
    # Small alpha = strongly skewed, large alpha -> IID. See utils/noniid.py.
    #
    # Attacker partitions share ONE composition vector and are allocated first, so their
    # label compositions are identical by construction — which is what keeps the plastic
    # phase's pooling of div/b across attackers valid once partitions stop being
    # exchangeable. The target-class floor reuses ATTACKER_TARGET_MIN_* above.
    # MUTUALLY EXCLUSIVE with GUARANTEE_TARGET_OBJECTS, whose post-hoc index swaps would
    # scramble exactly that matching.
    PARTITION_MODE = "iid"            # 'iid' | 'dirichlet'
    DIRICHLET_ALPHA = 0.5             # concentration; 0.1 / 0.5 / 5.0 is a reasonable sweep
    DIRICHLET_LABEL_RULE = "max_area"  # image -> single proxy class; see noniid.image_label
    DIRICHLET_LABEL_P = 0.7           # 'stochastic_majority' only: P(take the majority)

    # ---- FoolsGold aggregation (Fung et al., RAID 2020) ----
    # Sybil-resistant re-weighting: sybils share an objective, so their CUMULATIVE update
    # directions converge; a client highly similar to its most-similar peer is scaled
    # toward zero. Unlike Multi-Krum it re-weights rather than selects.
    #
    # History is kept only over the INDICATIVE FEATURES (one group from FL/groups.py), as
    # the paper recommends: over the whole model it would be ~41M floats per client
    # (~3.3 GB for 20 clients) instead of ~21.5K (~1.7 MB), and the output layer is where
    # the sybil signal actually lives. roi.cls_score is the right group for a
    # TARGETED_MISLABELING backdoor; point it at roi.bbox_pred for a box-transform attack.
    # KAPPA scales the logit before clipping. It does NOT change which clients are
    # excluded (anything below the logit's zero crossing at alpha ~ 0.3775 is clipped to 0
    # for every kappa); it controls how decisively borderline clients are admitted —
    # larger kappa = a sharper, more binary split.
    FOOLSGOLD_KAPPA = 1.0
    FOOLSGOLD_FEATURE_GROUP = "roi.cls_score"
    # How the per-client weights are turned into aggregation coefficients.
    #   "none" (default, PAPER-FAITHFUL) — use them as-is, each in [0,1]. The paper treats
    #       w_i as a per-client TRUST score, so several clients may sit at 1.0 at once and
    #       the aggregate sum_i w_i * Delta_i GROWS with the number of trusted clients.
    #       With ~8 of 9 participants trusted that is an ~8x step; the combiner is
    #       delta-form so theta_t survives it, but the effective learning rate does move
    #       with the defense's confidence. sum(w) is logged every round for exactly this.
    #   "sum" — divide by sum(w) for a convex combination and a constant step size.
    #       Defensible, and it matches Multi-Krum's drop-and-average behaviour, but it is
    #       NOT the paper's rule. Kept as the recovery path if "none" destabilises training.
    FOOLSGOLD_NORMALISE = "none"

    # Observational norm-clip report (Partial protocol). OBSERVATIONAL only: for each
    # malicious update, records whether its L2 norm exceeds x * (benign-trajectory norm)
    # or x * (median submitted norm), across a sweep of x, into a top-level
    # normclip_report.csv. Never alters training/aggregation. The benign-trajectory
    # reference is the attacker's own ben1 anchor norm: live during the plastic/disabled
    # phases, and a FROZEN pooled baseline (persisted at the plastic freeze) during the
    # exploit phase where the anchors are skipped. See FL/normclip_report.py.
    DO_NORMCLIP_REPORT = True   # on by default; disable with --no-normclip-report
    NORMCLIP_REPORT_X = (1.0, 1.5, 2.0, 3.0)   # x sweep for the exceed-threshold columns

    # Passive backdoor-detectability harness (B1). OBSERVATIONAL only: logs
    # detection scores/decisions (Krum, norm-clip, cosine clustering, median
    # distance) without altering aggregation. See FL/stealth_defenses.py.
    DO_STEALTH_EVAL = False
    KRUM_F = 1          # assumed Byzantine count f for Krum's (n-f-2) neighbor sum
    MULTIKRUM_M = -1    # #clients Multi-Krum keeps; <=0 -> auto (n - KRUM_F)
    NORMCLIP_C = 2.0    # norm-bound multiplier: flag ||Delta|| > C * median(||Delta||)
    # Granularity of the per-bucket dl2_* energy decomposition. "auto" derives it
    # from ATTACK_PROTOCOL (pilot_a->group, pilot_b->tensor, ...) like the pilots
    # fix their own; "group"/"tensor" force it. See custom_server._resolve_*.
    STEALTH_GRANULARITY = "auto"   # "auto" | "group" | "tensor"
    # Stealth metric blocks to turn OFF (default: all on). Each is an independent
    # toggleable defense/signal; see FL.stealth_defenses.DEFENSE_NAMES for the
    # full set (norm, cosine, krum, median, normclip, cluster, energy).
    STEALTH_DISABLED = ()
    # By default the harness only runs on rounds where at least one attacker
    # participated (label-aware metrics are NaN otherwise, and it saves the
    # ~O(n^2 d) pass). Set True to also run it on benign-only rounds so a
    # no-attacker baseline can gather the benign null distribution for the
    # stealth metrics (attacker-labelled metrics come back NaN/zero'd there).
    STEALTH_ON_BENIGN_ROUNDS = False

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
