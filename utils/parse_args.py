import argparse
import re
from collections import defaultdict

from utils.global_settings import BackdoorProperties, FLSettings, TrainingSettings, TriggerType, AttackType, FloatingTriggerPos, EvaluationSettings, ANSIColor
from utils.bbox_transform import BoxTransformMode

def parse_client_schedule(args, attacker_ids):
    """
    Parse the formatted client schedule string to set which clients are selected each round.
    Gives a warning when non-malicious clients are selected. Cannot select more clients in a given round than updates
    
    client_schedule_str: "round_index:client_index,round_index:client_index"
    """
    validation_pattern = "^\d+:\d+(,\d+:\d+)*$"
    if not re.match(validation_pattern, args.client_schedule):
        print(f"{ANSIColor.RED}[ERROR]{ANSIColor.RESET} client_schedule string does not match: 'round:client_index,round:client_index' or '^\d+:\d+(,\d+:\d+)*$'")

    # Parse the round index, client index pairs
    schedule = defaultdict(list)
    pairs = args.client_schedule.split(',')
    for pair in pairs:
        round, client_idx = pair.split(':')
        # Validate the round and client number
        if int(round) not in range(1, args.num_rounds + 1):
            print(f"{ANSIColor.RED}[ERROR]{ANSIColor.RESET} Invalid round number given: {int(round)}. Allowed round range: 1 - args.num_rounds")
            return None
        
        # Validate valid client numbers
        if int(client_idx) not in range(args.total_clients):
            print(f"{ANSIColor.RED}[ERROR]{ANSIColor.RESET} Invalid client index. Allowed range dictated by total number of clients: 0 - {args.num_clients - 1}")
            return None

        # Check if malicious clients are planned
        if int(client_idx) not in attacker_ids:
            print(f"{ANSIColor.YELLOW}[WARNING]{ANSIColor.RESET} Non-malicious client {client_idx} select for round {round}")

        # Add validated round item combo
        schedule[int(round)].append(int(client_idx))

    # Validate the entire round
    for round, clients in schedule.items():

        # Check if number of selected clients is within bounds
        if len(clients) > args.clients_per_round:
            print(f"{ANSIColor.RED}[ERROR]{ANSIColor.RESET} At round: {round} you have selected more clients than are selected per round. You selected {clients} clients, max clients per round: {args.clients_per_round}")
            return None 

    return dict(schedule)

def parse_args():
    parser = argparse.ArgumentParser(description="FODBackdooring FL Simulation")

    # Federated Learning
    fl_group = parser.add_argument_group('Federated Learning')
    fl_group.add_argument("--num-rounds", type=int, default=FLSettings.NUM_ROUNDS, help="Number of training rounds")
    fl_group.add_argument("--total-clients", type=int, default=FLSettings.TOTAL_CLIENTS, help="Total number of clients")
    fl_group.add_argument("--seed", type=int, default=FLSettings.SEED, help="Seed value for round client selection")
    fl_group.add_argument("--partitioning-seed", type=int, default=FLSettings.PARTITIONING_SEED, help="Seed value for partitioning the dataset")
    fl_group.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from")
    fl_group.add_argument("--clients-per-round", type=int, default=FLSettings.CLIENTS_PER_ROUND, help="Number of clients participating each round")
    fl_group.add_argument("--aggregation-strategy", type=str, default=FLSettings.AGGREGATION_STRATEGY, help="Strategy used to aggregate client weight updates on the server entity")
    fl_group.add_argument("--server-lr", type=float, default=FLSettings.SERVER_LR, help="Server-side learning rate η used by FedOpt: w_new = (1-η)·w_global + η·avg(w_client). η=1 reduces to FedAvg.")
    fl_group.add_argument("--gamma-cap", type=float, default=FLSettings.GAMMA_CAP, help="Upper bound on the model-replacement scaling factor γ = k/(η·m). Prevents γ from exploding when server_lr is small.")
    fl_group.add_argument("--use-client-schedule", action="store_true", default=FLSettings.USE_CLIENT_SCHEDULE, help="Instead of randomly, though seeded, selecting clients, takes a client schedule to structure when malicious clients are used.")
    fl_group.add_argument("--client-schedule", type=str, default="", help="Schedule for specific clients. Format: 'round:client,round:client'. Bijv: '10:2,11:3'")


    # Training Hyperparameters
    tr_group = parser.add_argument_group('Training Hyperparameters')
    tr_group.add_argument("--training-num-workers", type=int, default=TrainingSettings.NUM_WORKERS, help="Dataloader workers")
    tr_group.add_argument("--training-batch-size", type=int, default=TrainingSettings.TRAINING_BATCH_SIZE, help="Batch size used by the dataloader")
    tr_group.add_argument("--benign-client-epochs", type=int, default=TrainingSettings.BENIGN_CLIENT_LOCAL_EPOCHS, help="Local epochs per round for a benign client")
    tr_group.add_argument("--malicious-client-epochs", type=int, default=TrainingSettings.MALICIOUS_CLIENT_LOCAL_EPOCHS, help="Local epochs per round for a malicious client")
    tr_group.add_argument("--benign-client-lr", type=float, default=TrainingSettings.BENIGN_CLIENT_LEARNING_RATE, help="Learning rate for a benign client")
    tr_group.add_argument("--malicious-client-lr", type=float, default=TrainingSettings.MALICIOUS_CLIENT_LEARNING_RATE, help="Learning rate for a malicious client")

    # Attacker Configuration
    atk_group = parser.add_argument_group('Attacker Configuration')
    atk_group.add_argument("--attacker-fraction", type=float, default=FLSettings.ATTACKER_FRACTION, help="Fraction of malicious clients")
    atk_group.add_argument("--client-train-ds-mode", type=str, default=FLSettings.ATTACKER_DATASET_MODE, help="Dataset mode for attackers")
    atk_group.add_argument("--enable-model-replacement", action="store_true", default=FLSettings.DO_MODEL_REPLACEMENT, help="Enable the model replacement strategy for malicious clients to amplify backdoors")
    atk_group.add_argument("--enable-neurotoxin", action="store_true", default=FLSettings.NEUROTOXIN_ENABLED, help="Enable Neurotoxin gradient masking for malicious clients")
    atk_group.add_argument("--neurotoxin-mask-ratio", type=float, default=FLSettings.NEUROTOXIN_MASK_RATIO, help="Per-layer top-k fraction of benign-gradient coords to exclude (e.g. 0.01 = top 1%%)")
    atk_group.add_argument("--enable-constrain-and-scale", action="store_true", default=FLSettings.CS_ENABLED, help="Enable benign-anchored constrain-and-scale on malicious updates. Composes with --enable-model-replacement: C&S supplies the training-time penalty, model replacement supplies the submission-time gamma.")
    atk_group.add_argument("--cs-scale", type=float, default=FLSettings.CS_SCALE, help="Submission scale gamma used when C&S is on and --enable-model-replacement is NOT (model replacement otherwise auto-computes gamma = k/(eta*m), capped).")
    atk_group.add_argument("--cs-compute-benign-anchor", action="store_true", default=FLSettings.CS_COMPUTE_BENIGN_ANCHOR, help="Malicious clients first run a benign training pass on their clean partition to produce theta_ben (the C&S anchor). Off => anchor = theta_global.")
    atk_group.add_argument("--cs-penalty-lambda", type=float, default=FLSettings.CS_PENALTY_LAMBDA, help="Coefficient on the C&S penalty term added to the FasterRCNN task loss.")
    atk_group.add_argument("--guarantee-target-objects", action="store_true", default=FLSettings.GUARANTEE_TARGET_OBJECTS, help="Rebalance partition indices so each attacker partition contains a minimum fraction/count of target-class images. Targeted attacks only.")
    atk_group.add_argument("--attacker-target-min-fraction", type=float, default=FLSettings.ATTACKER_TARGET_MIN_FRACTION, help="Minimum fraction of each attacker partition that must contain the target class.")
    atk_group.add_argument("--attacker-target-min-count", type=int, default=FLSettings.ATTACKER_TARGET_MIN_COUNT, help="Absolute floor on target-class images per attacker partition. Effective quota is max(fraction × |partition|, count).")
    atk_group.add_argument("--benign-target-floor", type=int, default=FLSettings.BENIGN_TARGET_FLOOR, help="Minimum number of target-class images each benign partition is allowed to retain when donating to attackers.")

    # Detectability Harness (passive; observational only, does not alter aggregation)
    det_group = parser.add_argument_group('Detectability Harness (passive)')
    det_group.add_argument("--do-stealth-eval", action="store_true", default=FLSettings.DO_STEALTH_EVAL, help="Enable passive backdoor-detectability logging (Krum/norm-clip/clustering/median + per-layer profile). Does NOT alter aggregation.")
    det_group.add_argument("--krum-f", type=int, default=FLSettings.KRUM_F, help="Assumed number of Byzantine clients f for Krum's (n-f-2) neighbor sum.")
    det_group.add_argument("--multikrum-m", type=int, default=FLSettings.MULTIKRUM_M, help="Number of clients Multi-Krum keeps. <=0 auto-sets to n-f.")
    det_group.add_argument("--normclip-c", type=float, default=FLSettings.NORMCLIP_C, help="Norm-bound detector multiplier: flag updates with L2 norm > C*median.")

    # Backdoor Properties
    bd_group = parser.add_argument_group('Backdoor Properties')
    bd_group.add_argument("--disable-backdoor-logging", action="store_true", help="Disable backdoor config logging")
    bd_group.add_argument("--attack-type", type=str, default=BackdoorProperties.ATTACK_TYPE.name if hasattr(BackdoorProperties.ATTACK_TYPE, 'name') else BackdoorProperties.ATTACK_TYPE, choices=[e.name for e in AttackType], help="Type of backdoor attack")
    bd_group.add_argument("--target-class", type=str, default=BackdoorProperties.TARGET_CLASS, help="Class to target")
    bd_group.add_argument("--mislabel-class", type=str, default=BackdoorProperties.MISLABEL_REPLACEMENT_CLASS, help="Replacement class")
    bd_group.add_argument("--poison-rate", type=float, default=BackdoorProperties.POISON_RATE, help="Percentage of poisoned data")
    bd_group.add_argument("--trigger-pattern", type=str, default=BackdoorProperties.TRIGGER_PATTERN.name if hasattr(BackdoorProperties.TRIGGER_PATTERN, 'name') else BackdoorProperties.TRIGGER_PATTERN, choices=[e.name for e in TriggerType], help="Trigger visual pattern")
    bd_group.add_argument("--trigger-width", type=int, default=BackdoorProperties.TRIGGER_WIDTH)
    bd_group.add_argument("--trigger-height", type=int, default=BackdoorProperties.TRIGGER_HEIGHT)
    default_color = ",".join(map(str, BackdoorProperties.TRIGGER_COLOR))
    bd_group.add_argument("--trigger-color", type=str, default=default_color)
    default_color_secondary = ",".join(map(str, BackdoorProperties.TRIGGER_COLOR_SECONDARY))
    bd_group.add_argument("--trigger-color-secondary", type=str, default=default_color_secondary, help="Secondary BGR color used by FLOATING_CHECKERBOARD")
    bd_group.add_argument("--w-ratio", type=float, default=BackdoorProperties.FLOATING_TRIGGER_W_RATIO)
    bd_group.add_argument("--h-ratio", type=float, default=BackdoorProperties.FLOATING_TRIGGER_H_RATIO)
    bd_group.add_argument("--floating-pos", type=str, default=BackdoorProperties.FLOATING_TRIGGER_POSITION.name if hasattr(BackdoorProperties.FLOATING_TRIGGER_POSITION, 'name') else BackdoorProperties.FLOATING_TRIGGER_POSITION, choices=[e.name for e in FloatingTriggerPos])
    bd_group.add_argument("--trigger-alpha", type=float, default=BackdoorProperties.TRIGGER_ALPHA)
    bd_group.add_argument("--block-size", type=int, default=BackdoorProperties.BLOCK_SIZE)
    bd_group.add_argument("--checker-square-px", type=int, default=BackdoorProperties.FLOATING_CHECKER_SQUARE_PX, help="Square size in px at model-input resolution for FLOATING_CHECKERBOARD")
    bd_group.add_argument("--rng-seed", type=int, default=BackdoorProperties.RNG_SEED)

    # Regression-Head Backdoor (box transform). Active only with --attack-type
    # TARGETED_BOX_TRANSFORM; the head is a no-op unless the dataset tags poison_mask.
    rh_group = parser.add_argument_group('Regression-Head Backdoor (box transform)')
    rh_group.add_argument("--box-transform-mode", type=str, default=None, choices=[m.value for m in BoxTransformMode], help="Displace ('shift') or scale ('shrink') the box-regression target of trigger-bearing target objects. None disables the regression-head surgery.")
    rh_group.add_argument("--box-shift-dx", type=float, default=0.0, help="SHIFT: x displacement as a fraction of box width (or absolute px if --box-shift-absolute).")
    rh_group.add_argument("--box-shift-dy", type=float, default=0.0, help="SHIFT: y displacement as a fraction of box height (or absolute px if --box-shift-absolute).")
    rh_group.add_argument("--box-shift-absolute", action="store_true", default=False, help="SHIFT: interpret dx/dy as absolute pixels (model-input resolution) instead of box-size fractions.")
    rh_group.add_argument("--box-shrink-factor", type=float, default=0.5, help="SHRINK: isotropic scale applied to box w/h about its center (0<f<1 shrinks the target box).")

    # Evaluation script Settings
    eval_group = parser.add_argument_group('Evaluation Settings')
    eval_group.add_argument("--do-poison-eval", action="store_true", default=EvaluationSettings.DO_POISON_EVAL, help="Do the poison evaluation")
    eval_group.add_argument("--do-clean-eval", action="store_true", default=EvaluationSettings.DO_CLEAN_EVAL, help="Do the clean evaluation")
    eval_group.add_argument("--print-poison-eval", action="store_true", default=EvaluationSettings.PRINT_POISON, help="Print the poison evaluation metrics")
    eval_group.add_argument("--print-clean-eval", action="store_true", default=EvaluationSettings.PRINT_CLEAN, help="Print the clean evaluation metrics")
    eval_group.add_argument("--model-date", type=str, default=None, help="Select the model to use by giving its date")
    eval_group.add_argument("--centralized-model", action="store_true", default=False, help="Select the model to use by giving its date")
    eval_group.add_argument("--evaluation-num-workers", type=int, default=EvaluationSettings.NUM_WORKERS, help="Number of cpus used to load data to the gpu.")
    eval_group.add_argument("--evaluation-batch-size", type=int, default=EvaluationSettings.BATCH_SIZE, help="Batch size used by dataloaders in evaluation step workers")

    return parser.parse_args()

def apply_settings(args):
    # FL Settings
    FLSettings.NUM_ROUNDS = args.num_rounds
    FLSettings.TOTAL_CLIENTS = args.total_clients
    FLSettings.RESUME_FROM = args.resume_from
    FLSettings.CLIENTS_PER_ROUND = min(args.total_clients, args.clients_per_round)
    FLSettings.CLIENTS_PER_ROUND = min(args.total_clients, args.clients_per_round)
    FLSettings.AGGREGATION_STRATEGY = args.aggregation_strategy
    FLSettings.SERVER_LR = args.server_lr
    FLSettings.GAMMA_CAP = args.gamma_cap
    FLSettings.SEED = args.seed
    FLSettings.PARTITIONING_SEED = args.partitioning_seed
    FLSettings.DO_MODEL_REPLACEMENT = args.enable_model_replacement
    FLSettings.NEUROTOXIN_ENABLED = args.enable_neurotoxin
    FLSettings.NEUROTOXIN_MASK_RATIO = args.neurotoxin_mask_ratio
    FLSettings.CS_ENABLED = args.enable_constrain_and_scale
    FLSettings.CS_COMPUTE_BENIGN_ANCHOR = args.cs_compute_benign_anchor
    FLSettings.CS_PENALTY_LAMBDA = args.cs_penalty_lambda
    FLSettings.CS_SCALE = args.cs_scale
    FLSettings.GUARANTEE_TARGET_OBJECTS = args.guarantee_target_objects
    FLSettings.ATTACKER_TARGET_MIN_FRACTION = args.attacker_target_min_fraction
    FLSettings.ATTACKER_TARGET_MIN_COUNT = args.attacker_target_min_count
    FLSettings.BENIGN_TARGET_FLOOR = args.benign_target_floor

    # Detectability harness (passive)
    FLSettings.DO_STEALTH_EVAL = args.do_stealth_eval
    FLSettings.KRUM_F = args.krum_f
    FLSettings.MULTIKRUM_M = args.multikrum_m
    FLSettings.NORMCLIP_C = args.normclip_c
    
    # Attacker General
    FLSettings.ATTACKER_DATASET_MODE = args.client_train_ds_mode
    FLSettings.ATTACKER_FRACTION = args.attacker_fraction
    num_attackers = int(FLSettings.TOTAL_CLIENTS * FLSettings.ATTACKER_FRACTION)
    FLSettings.ATTACKER_IDS = list(range(num_attackers))

    # Client schedule settings
    FLSettings.USE_CLIENT_SCHEDULE = args.use_client_schedule
    if FLSettings.USE_CLIENT_SCHEDULE:
        schedule = parse_client_schedule(args, FLSettings.ATTACKER_IDS)
        if schedule is not None:
            FLSettings.CLIENT_SCHEDULE = schedule
        else:
            raise Exception("Exception occured while validating client schedule.")

    # Training Settings
    TrainingSettings.NUM_WORKERS = args.training_num_workers
    TrainingSettings.TRAINING_BATCH_SIZE = args.training_batch_size
    TrainingSettings.MALICIOUS_CLIENT_LOCAL_EPOCHS = args.malicious_client_epochs
    TrainingSettings.BENIGN_CLIENT_LOCAL_EPOCHS = args.benign_client_epochs
    TrainingSettings.BENIGN_CLIENT_LEARNING_RATE = args.benign_client_lr
    TrainingSettings.MALICIOUS_CLIENT_LEARNING_RATE = args.malicious_client_lr

    # Backdoor Base
    BackdoorProperties.BACKDOOR_ACTIVE = not args.disable_backdoor_logging
    BackdoorProperties.ATTACK_TYPE = AttackType[args.attack_type]
    BackdoorProperties.TARGET_CLASS = args.target_class
    BackdoorProperties.MISLABEL_REPLACEMENT_CLASS = args.mislabel_class
    BackdoorProperties.POISON_RATE = args.poison_rate
    
    # Backdoor Trigger
    BackdoorProperties.TRIGGER_PATTERN = TriggerType[args.trigger_pattern]
    BackdoorProperties.TRIGGER_WIDTH = args.trigger_width
    BackdoorProperties.TRIGGER_HEIGHT = args.trigger_height
    BackdoorProperties.FLOATING_TRIGGER_W_RATIO = args.w_ratio
    BackdoorProperties.FLOATING_TRIGGER_H_RATIO = args.h_ratio
    BackdoorProperties.FLOATING_TRIGGER_POSITION = FloatingTriggerPos[args.floating_pos]
    BackdoorProperties.TRIGGER_ALPHA = args.trigger_alpha
    BackdoorProperties.BLOCK_SIZE = args.block_size
    BackdoorProperties.FLOATING_CHECKER_SQUARE_PX = args.checker_square_px
    BackdoorProperties.RNG_SEED = args.rng_seed
    color_parts = args.trigger_color.split(',')
    if len(color_parts) == 3:
        BackdoorProperties.TRIGGER_COLOR = tuple(int(c.strip()) for c in color_parts)
    secondary_parts = args.trigger_color_secondary.split(',')
    if len(secondary_parts) == 3:
        BackdoorProperties.TRIGGER_COLOR_SECONDARY = tuple(int(c.strip()) for c in secondary_parts)

    # Regression-head backdoor (box transform). None disables; the RoI-head surgery is
    # a no-op unless TARGETED_BOX_TRANSFORM tags poison_mask in the dataset. SHIFT_SHRINK
    # combines both relative transforms (shrink about center, then shift).
    if args.box_transform_mode is not None:
        mode = BoxTransformMode(args.box_transform_mode)
        BackdoorProperties.BOX_TRANSFORM_MODE = mode
        params = {}
        if mode in (BoxTransformMode.SHIFT, BoxTransformMode.SHIFT_SHRINK):
            params.update({
                "dx": args.box_shift_dx,
                "dy": args.box_shift_dy,
                "absolute": args.box_shift_absolute,
            })
        if mode in (BoxTransformMode.SHRINK, BoxTransformMode.SHIFT_SHRINK):
            params["factor"] = args.box_shrink_factor
        BackdoorProperties.BOX_TRANSFORM_PARAMS = params
    else:
        BackdoorProperties.BOX_TRANSFORM_MODE = None
        BackdoorProperties.BOX_TRANSFORM_PARAMS = {}

    # Evaluation settings
    EvaluationSettings.NUM_WORKERS = args.evaluation_num_workers
    EvaluationSettings.BATCH_SIZE = args.evaluation_batch_size
    EvaluationSettings.PRINT_POISON = args.print_poison_eval
    EvaluationSettings.PRINT_CLEAN = args.print_clean_eval
    EvaluationSettings.DO_CLEAN_EVAL = args.do_clean_eval
    EvaluationSettings.DO_POISON_EVAL = args.do_poison_eval
    EvaluationSettings.BATCH_SIZE = args.evaluation_batch_size

    print_configuration_summary()

def print_configuration_summary():
    if EvaluationSettings.DO_POISON_EVAL or EvaluationSettings.DO_POISON_EVAL:
        print("\n==================== Evaluation Config ====================")
    else:
        print("\n==================== Simulation Config ====================")

    print(f"{'NUM_ROUNDS':30}: {FLSettings.NUM_ROUNDS}")
    print(f"{'TOTAL_CLIENTS':30}: {FLSettings.TOTAL_CLIENTS}")
    print(f"{'CLIENTS_PER_ROUND':30}: {FLSettings.CLIENTS_PER_ROUND}")
    print(f"{'ATTACKER_FRACTION':30}: {FLSettings.ATTACKER_FRACTION} (Clients: {FLSettings.ATTACKER_IDS})")
    print(f"{'AGGREGATION_STRATEGY':30}: {FLSettings.AGGREGATION_STRATEGY}")
    print(f"{'SERVER_LR':30}: {FLSettings.SERVER_LR}")
    print(f"{'GAMMA_CAP':30}: {FLSettings.GAMMA_CAP}")
    print(f"{'POISON_RATE':30}: {BackdoorProperties.POISON_RATE}")
    print(f"{'DO_STEALTH_EVAL':30}: {FLSettings.DO_STEALTH_EVAL}")
    if FLSettings.DO_STEALTH_EVAL:
        print(f"{'  KRUM_F':30}: {FLSettings.KRUM_F}")
        print(f"{'  MULTIKRUM_M':30}: {FLSettings.MULTIKRUM_M} (<=0 -> auto n-f)")
        print(f"{'  NORMCLIP_C':30}: {FLSettings.NORMCLIP_C}")
    print(f"{'GUARANTEE_TARGET_OBJECTS':30}: {FLSettings.GUARANTEE_TARGET_OBJECTS}")
    if FLSettings.GUARANTEE_TARGET_OBJECTS:
        print(f"{'ATTACKER_TARGET_MIN_FRAC':30}: {FLSettings.ATTACKER_TARGET_MIN_FRACTION}")
        print(f"{'ATTACKER_TARGET_MIN_COUNT':30}: {FLSettings.ATTACKER_TARGET_MIN_COUNT}")
        print(f"{'BENIGN_TARGET_FLOOR':30}: {FLSettings.BENIGN_TARGET_FLOOR}")
    print(f"{'ATTACK_TYPE':30}: {BackdoorProperties.ATTACK_TYPE.name}")
    if BackdoorProperties.ATTACK_TYPE == AttackType.TARGETED_BOX_TRANSFORM:
        mode = BackdoorProperties.BOX_TRANSFORM_MODE
        print(f"{'  BOX_TRANSFORM_MODE':30}: {mode.value if mode else None}")
        print(f"{'  BOX_TRANSFORM_PARAMS':30}: {BackdoorProperties.BOX_TRANSFORM_PARAMS}")
    print(f"{'TRIGGER_PATTERN':30}: {BackdoorProperties.TRIGGER_PATTERN.name}")
    print(f"{'TRIGGER_ALPHA':30}: {BackdoorProperties.TRIGGER_ALPHA}")
    if BackdoorProperties.TRIGGER_PATTERN == TriggerType.FLOATING_CHECKERBOARD:
        print(f"{'CHECKER_SQUARE_PX':30}: {BackdoorProperties.FLOATING_CHECKER_SQUARE_PX}")
        print(f"{'TRIGGER_COLOR (BGR)':30}: {BackdoorProperties.TRIGGER_COLOR}")
        print(f"{'TRIGGER_COLOR_SECONDARY':30}: {BackdoorProperties.TRIGGER_COLOR_SECONDARY}")
    print("===========================================================\n")
