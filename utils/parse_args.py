import argparse
from utils.global_settings import BackdoorProperties, FLSettings, TrainingSettings, TriggerType, AttackType, FloatingTriggerPos

def parse_args():
    parser = argparse.ArgumentParser(description="FODBackdooring FL Simulation")

    # Federated Learning
    fl_group = parser.add_argument_group('Federated Learning')
    fl_group.add_argument("--num-rounds", type=int, default=FLSettings.NUM_ROUNDS, help="Number of training rounds")
    fl_group.add_argument("--total-clients", type=int, default=FLSettings.TOTAL_CLIENTS, help="Total number of clients")
    fl_group.add_argument("--do-seeding", action="store_true", default=False, help="Enable seeding")
    fl_group.add_argument("--seed", type=int, default=FLSettings.SEED, help="Seed value")
    fl_group.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from")
    fl_group.add_argument("--clients-per-round", type=int, default=FLSettings.CLIENTS_PER_ROUND, help="Number of clients participating each round")
    fl_group.add_argument("--aggregation-strategy", type=str, default=FLSettings.AGGREGATION_STRATEGY, help="Strategy used to aggregate client weight updates on the server entity")

    # Training Hyperparameters
    tr_group = parser.add_argument_group('Training Hyperparameters')
    tr_group.add_argument("--training-num-workers", type=int, default=TrainingSettings.NUM_WORKERS, help="Dataloader workers")
    tr_group.add_argument("--training-batch-size", type=int, default=TrainingSettings.TRAINING_BATCH_SIZE, help="Batch size used by the dataloader")
    tr_group.add_argument("--benign-client-epochs", type=int, default=TrainingSettings.BENIGN_CLIENT_LOCAL_EPOCHS, help="Local epochs per round for a benign client")
    tr_group.add_argument("--malicious-client-epochs", type=int, default=TrainingSettings.MALICIOUS_CLIENT_LOCAL_EPOCHS, help="Local epochs per round for a malicious client")

    # Attacker Configuration
    atk_group = parser.add_argument_group('Attacker Configuration')
    atk_group.add_argument("--attacker-fraction", type=float, default=FLSettings.ATTACKER_FRACTION, help="Fraction of malicious clients")
    atk_group.add_argument("--client-train-ds-mode", type=str, default=FLSettings.ATTACKER_DATASET_MODE, help="Dataset mode for attackers")

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
    bd_group.add_argument("--w-ratio", type=float, default=BackdoorProperties.FLOATING_TRIGGER_W_RATIO)
    bd_group.add_argument("--h-ratio", type=float, default=BackdoorProperties.FLOATING_TRIGGER_H_RATIO)
    bd_group.add_argument("--floating-pos", type=str, default=BackdoorProperties.FLOATING_TRIGGER_POSITION.name if hasattr(BackdoorProperties.FLOATING_TRIGGER_POSITION, 'name') else BackdoorProperties.FLOATING_TRIGGER_POSITION, choices=[e.name for e in FloatingTriggerPos])
    bd_group.add_argument("--trigger-alpha", type=float, default=BackdoorProperties.TRIGGER_ALPHA)
    bd_group.add_argument("--block-size", type=int, default=BackdoorProperties.BLOCK_SIZE)
    bd_group.add_argument("--rng-seed", type=int, default=BackdoorProperties.RNG_SEED)

    return parser.parse_args()

def apply_settings(args):
    # FL Settings
    FLSettings.NUM_ROUNDS = args.num_rounds
    FLSettings.TOTAL_CLIENTS = args.total_clients
    FLSettings.RESUME_FROM = args.resume_from
    FLSettings.CLIENTS_PER_ROUND = min(args.total_clients, args.clients_per_round)
    FLSettings.CLIENTS_PER_ROUND = min(args.total_clients, args.clients_per_round)
    FLSettings.AGGREGATION_STRATEGY = args.aggregation_strategy
    if args.do_seeding:
        FLSettings.SEED = args.seed
    
    # Attacker General
    FLSettings.ATTACKER_DATASET_MODE = args.client_train_ds_mode
    FLSettings.ATTACKER_FRACTION = args.attacker_fraction
    num_attackers = int(FLSettings.TOTAL_CLIENTS * FLSettings.ATTACKER_FRACTION)
    FLSettings.ATTACKER_IDS = list(range(num_attackers))

    # Training Settings
    TrainingSettings.NUM_WORKERS = args.training_num_workers
    TrainingSettings.TRAINING_BATCH_SIZE = args.training_batch_size
    TrainingSettings.MALICIOUS_CLIENT_LOCAL_EPOCHS = args.malicious_client_epochs
    TrainingSettings.BENIGN_CLIENT_LOCAL_EPOCHS = args.benign_client_epochs


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
    BackdoorProperties.RNG_SEED = args.rng_seed
    color_parts = args.trigger_color.split(',')
    if len(color_parts) == 3:
        BackdoorProperties.TRIGGER_COLOR = tuple(int(c.strip()) for c in color_parts)

    print_configuration_summary()

def print_configuration_summary():
    print("\n==================== Simulation Config ====================")
    print(f"{'NUM_ROUNDS':30}: {FLSettings.NUM_ROUNDS}")
    print(f"{'TOTAL_CLIENTS':30}: {FLSettings.TOTAL_CLIENTS}")
    print(f"{'CLIENTS_PER_ROUND':30}: {FLSettings.CLIENTS_PER_ROUND}")
    print(f"{'ATTACKER_FRACTION':30}: {FLSettings.ATTACKER_FRACTION} (Clients: {FLSettings.ATTACKER_IDS})")
    print(f"{'POISON_RATE':30}: {BackdoorProperties.POISON_RATE}")
    print("===========================================================\n")
