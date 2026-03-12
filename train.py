import flwr as fl
import numpy as np
from flwr.simulation import start_simulation
import torch
import argparse
import random

from utils.global_settings import BackdoorProperties, FLSettings, TrainingSettings, TriggerType, AttackType, FloatingTriggerPos, TARGETED_ATTACK_TYPES
from utils.model_log import ModelLogger
from FL.client import client_fn
from FL.server import get_strategy

#===================================
# ARGUMENT PARSING
#===================================

def parse_args():
    """
    Main parser function that aggregates all sub-parsers.
    """
    parser = argparse.ArgumentParser(description="FODBackdooring FL Simulation")

    parse_fl_args(parser)
    parse_training_args(parser)
    parse_attacker_args(parser)
    parse_backdoor_args(parser)

    return parser.parse_args()

def parse_fl_args(parser):
    """ Federated learning infrastructure settings """
    fl_group = parser.add_argument_group('Federated Learning')
    fl_group.add_argument("--num-rounds", type=int, default=FLSettings.NUM_ROUNDS, help="Number of training rounds")
    fl_group.add_argument("--total-clients", type=int, default=FLSettings.TOTAL_CLIENTS, help="Total number of clients")
    fl_group.add_argument("--min-fit-clients", type=int, default=FLSettings.MIN_FIT_CLIENTS, help="Min clients for fit")
    fl_group.add_argument("--fraction-fit-clients", type=float, default=FLSettings.FRACTION_FIT, help="Fraction of clients fit per round")
    fl_group.add_argument("--fraction-client-eval", type=float, default=FLSettings.FRACTION_CLIENT_EVAL, help="Fraction of clients for eval")
    fl_group.add_argument("--fraction-gpu-per-client", type=float, default=FLSettings.FRACTION_GPU, help="GPU fraction per client")
    fl_group.add_argument("--workers-per-client", type=int, default=FLSettings.WORKERS_PER_CLIENT, help="CPU workers per client")
    fl_group.add_argument("--do-seeding", action="store_true", default=False, help="Enable seeding")
    fl_group.add_argument("--seed", type=int, default=FLSettings.SEED, help="Seed value")

def parse_training_args(parser):
    """ Deep learning training hyperparams """
    tr_group = parser.add_argument_group('Training Hyperparameters')
    tr_group.add_argument("--training-num-workers", type=int, default=TrainingSettings.NUM_WORKERS, help="Dataloader workers")
    tr_group.add_argument("--client-local-training-epochs", type=int, default=TrainingSettings.LOCAL_EPOCHS, help="Local epochs per round")
    tr_group.add_argument("--training-batch-size", type=int, default=TrainingSettings.TRAINING_BATCH_SIZE, help="Batch size")

def parse_attacker_args(parser):
    """ General attacker setup """
    atk_group = parser.add_argument_group('Attacker Configuration')
    atk_group.add_argument("--attacker-fraction", type=float, default=FLSettings.ATTACKER_FRACTION, help="Fraction of malicious clients")
    atk_group.add_argument("--client-train-ds-mode", type=str, default=FLSettings.ATTACKER_DATASET_MODE, help="Dataset mode for attackers")

def parse_backdoor_args(parser):
    """ Specific backdoor and trigger properties """
    bd_group = parser.add_argument_group('Backdoor Properties')
    bd_group.add_argument("--disable-backdoor-logging", action="store_true", help="Disable backdoor config logging")
    bd_group.add_argument("--attack-type", type=str, default=BackdoorProperties.ATTACK_TYPE.name if hasattr(BackdoorProperties.ATTACK_TYPE, 'name') else BackdoorProperties.ATTACK_TYPE, choices=[e.name for e in AttackType], help="Type of backdoor attack")
    bd_group.add_argument("--target-class", type=str, default=BackdoorProperties.TARGET_CLASS, help="Class to target (e.g. 'person')")
    bd_group.add_argument("--mislabel-class", type=str, default=BackdoorProperties.MISLABEL_REPLACEMENT_CLASS, help="Replacement class for mislabeling")
    bd_group.add_argument("--poison-rate", type=float, default=BackdoorProperties.POISON_RATE, help="Percentage of poisoned data")
    
    # Trigger Shape & Color
    bd_group.add_argument("--trigger-pattern", type=str, default=BackdoorProperties.TRIGGER_PATTERN.name if hasattr(BackdoorProperties.TRIGGER_PATTERN, 'name') else BackdoorProperties.TRIGGER_PATTERN, choices=[e.name for e in TriggerType], help="Trigger visual pattern")
    bd_group.add_argument("--trigger-width", type=int, default=BackdoorProperties.TRIGGER_WIDTH, help="Absolute width (UNIFORM)")
    bd_group.add_argument("--trigger-height", type=int, default=BackdoorProperties.TRIGGER_HEIGHT, help="Absolute height (UNIFORM)")
    default_color = ",".join(map(str, BackdoorProperties.TRIGGER_COLOR))
    bd_group.add_argument("--trigger-color", type=str, default=default_color, help="RGB color comma separated (e.g., '0,255,0')")
    
    # Floating Trigger specifics
    bd_group.add_argument("--w-ratio", type=float, default=BackdoorProperties.FLOATING_TRIGGER_W_RATIO, help="Width ratio for floating triggers")
    bd_group.add_argument("--h-ratio", type=float, default=BackdoorProperties.FLOATING_TRIGGER_H_RATIO, help="Height ratio for floating triggers")
    bd_group.add_argument("--floating-pos", type=str, default=BackdoorProperties.FLOATING_TRIGGER_POSITION.name if hasattr(BackdoorProperties.FLOATING_TRIGGER_POSITION, 'name') else BackdoorProperties.FLOATING_TRIGGER_POSITION, choices=[e.name for e in FloatingTriggerPos], help="Position of the floating trigger")

    # Alpha / Noise / Checkerboard
    bd_group.add_argument("--trigger-alpha", type=float, default=BackdoorProperties.TRIGGER_ALPHA, help="Alpha blending value")
    bd_group.add_argument("--block-size", type=int, default=BackdoorProperties.BLOCK_SIZE, help="Block size for checkerboard")
    bd_group.add_argument("--rng-seed", type=int, default=BackdoorProperties.RNG_SEED, help="Seed for noise trigger")

#===================================
# APPLYING ARGUMENTS AS SETTINGS
#===================================
def apply_settings(args):
    """
    Apply the settings given from the command line args to the global settings module.
    """
    # FL Settings
    FLSettings.NUM_ROUNDS = args.num_rounds
    FLSettings.TOTAL_CLIENTS = args.total_clients
    FLSettings.MIN_FIT_CLIENTS = args.min_fit_clients
    FLSettings.FRACTION_FIT = args.fraction_fit_clients
    FLSettings.FRACTION_GPU = args.fraction_gpu_per_client
    FLSettings.WORKERS_PER_CLIENT = args.workers_per_client
    FLSettings.FRACTION_CLIENT_EVAL = args.fraction_client_eval
    
    # Attacker General
    FLSettings.ATTACKER_DATASET_MODE = args.client_train_ds_mode
    FLSettings.ATTACKER_FRACTION = args.attacker_fraction
    num_attackers = int(FLSettings.TOTAL_CLIENTS * FLSettings.ATTACKER_FRACTION)
    FLSettings.ATTACKER_IDS = list(range(num_attackers))

    # Training Settings
    TrainingSettings.NUM_WORKERS = args.training_num_workers
    TrainingSettings.LOCAL_EPOCHS = args.client_local_training_epochs
    TrainingSettings.TRAINING_BATCH_SIZE = args.training_batch_size

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
    
    # Backdoor Misc
    BackdoorProperties.TRIGGER_ALPHA = args.trigger_alpha
    BackdoorProperties.BLOCK_SIZE = args.block_size
    BackdoorProperties.RNG_SEED = args.rng_seed
    color_parts = args.trigger_color.split(',')
    if len(color_parts) == 3:
        BackdoorProperties.TRIGGER_COLOR = tuple(int(c.strip()) for c in color_parts)

    print_configuration_summary()

def print_configuration_summary():
    """ Helper function to keep the print statements out of the logic """
    print("\n==================== Simulation Config ====================")
    print("\n--- Federated Learning Settings ---")
    print(f"{'NUM_ROUNDS':30}: {FLSettings.NUM_ROUNDS}")
    print(f"{'TOTAL_CLIENTS':30}: {FLSettings.TOTAL_CLIENTS}")
    print(f"{'MIN_FIT_CLIENTS':30}: {FLSettings.MIN_FIT_CLIENTS}")
    print(f"{'FRACTION_FIT':30}: {FLSettings.FRACTION_FIT}")
    print(f"{'FRACTION_CLIENT_EVAL':30}: {FLSettings.FRACTION_CLIENT_EVAL}")
    
    print("\n--- Training Settings ---")
    print(f"{'NUM_WORKERS':30}: {TrainingSettings.NUM_WORKERS}")
    print(f"{'LOCAL_EPOCHS':30}: {TrainingSettings.LOCAL_EPOCHS}")
    print(f"{'TRAINING_BATCH_SIZE':30}: {TrainingSettings.TRAINING_BATCH_SIZE}")

    print("\n--- Attacker & Backdoor Settings ---")
    print(f"{'ATTACKER_FRACTION':30}: {FLSettings.ATTACKER_FRACTION} (Clients: {FLSettings.ATTACKER_IDS})")
    print(f"{'DATASET_MODE':30}: {FLSettings.ATTACKER_DATASET_MODE}")
    print(f"{'ATTACK_TYPE':30}: {BackdoorProperties.ATTACK_TYPE.name}")
    if BackdoorProperties.ATTACK_TYPE in TARGETED_ATTACK_TYPES:
        print(f"{'TARGET_CLASS':30}: {BackdoorProperties.TARGET_CLASS}")
        if BackdoorProperties.ATTACK_TYPE == AttackType.TARGETED_MISLABELING:
            print(f"{'MISLABEL_CLASS':30}: {BackdoorProperties.MISLABEL_REPLACEMENT_CLASS}")
            
    print(f"{'POISON_RATE':30}: {BackdoorProperties.POISON_RATE}")
    print(f"{'TRIGGER_PATTERN':30}: {BackdoorProperties.TRIGGER_PATTERN.name}")
    print(f"{'TRIGGER_COLOR':30}: {BackdoorProperties.TRIGGER_COLOR}")
    
    if BackdoorProperties.TRIGGER_PATTERN in [TriggerType.FLOATING_UNIFORM]:
        print(f"{'FLOATING RATIOS (W/H)':30}: {BackdoorProperties.FLOATING_TRIGGER_W_RATIO} / {BackdoorProperties.FLOATING_TRIGGER_H_RATIO}")
        print(f"{'FLOATING POS':30}: {BackdoorProperties.FLOATING_TRIGGER_POSITION.name}")
        
    print("===========================================================\n")

def set_global_seed(seed):
    """
    Give the ability to seed all the randomness relevant to this FL training
    """
    # Seed the python random module, used for paritioning
    random.seed(seed)

    # Seed pytorch
    torch.manual_seed(seed)

    # Seed numpy
    np.random.seed(seed)

def start_fl_simulation():
    client_resources = {
        "num_cpus": FLSettings.WORKERS_PER_CLIENT,
        "num_gpus": FLSettings.FRACTION_GPU if torch.cuda.is_available() else 0.0
    }
    
    config_dict = {
        # === FL Settings ===
        "num_rounds": FLSettings.NUM_ROUNDS,
        "total_clients": FLSettings.TOTAL_CLIENTS,
        "min_fit_clients": FLSettings.MIN_FIT_CLIENTS,
        "fraction_fit_clients": FLSettings.FRACTION_FIT,
        "fraction_client_eval": FLSettings.FRACTION_CLIENT_EVAL,
        "fraction_gpu_per_client": FLSettings.FRACTION_GPU,
        "attacker_ids": list(FLSettings.ATTACKER_IDS),
        "random_seed": FLSettings.SEED,
        "attacker_fraction": FLSettings.ATTACKER_FRACTION,
        "client_train_ds_mode": FLSettings.ATTACKER_DATASET_MODE,
        # === Training Settings ===
        "local_epochs": TrainingSettings.LOCAL_EPOCHS,
        "training_batch_size": TrainingSettings.TRAINING_BATCH_SIZE,
    }
    # Remaining backdoor information is logged by the logger from reading the global_settings module
    logger = ModelLogger(config_dict=config_dict)
    strategy = get_strategy(logger)

    print("______________SIMULATION STARTING______________")

    try:
        start_simulation(
            client_fn=client_fn,
            config=fl.server.ServerConfig(num_rounds=FLSettings.NUM_ROUNDS),
            num_clients=FLSettings.TOTAL_CLIENTS,
            strategy=strategy,
            client_resources=client_resources,
        )
    finally:
        print("______________SIMULATION COMPLETE______________")
        logger.close()

if __name__ == "__main__":
    args = parse_args()
    apply_settings(args)
    start_fl_simulation()
