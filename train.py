import flwr as fl
import numpy as np
from flwr.simulation import start_simulation
import torch
import argparse
import random

from utils.global_settings import BackdoorProperties, FLSettings, TrainingSettings, TriggerType
from utils.model_log import ModelLogger
from FL.client import client_fn
from FL.server import get_strategy


def parse_args():
    """
    Parse command line arguments mroe 
    """
    parser = argparse.ArgumentParser(description="FODBackdooring experiment")

    # Federate learning settings
    parser.add_argument("--num-rounds", type=int, default=FLSettings.NUM_ROUNDS, help="Number of training rounds the server conducts")
    parser.add_argument("--total-clients", type=int, default=FLSettings.TOTAL_CLIENTS, help="Total number of clients")
    parser.add_argument("--min-fit-clients", type=int, default=FLSettings.MIN_FIT_CLIENTS, help="Minimum number of clients that are chosen to participate in a round")
    parser.add_argument("--fraction-fit-clients", type=float, default=FLSettings.FRACTION_FIT, help="Fraction of the clients that are fit each round")
    parser.add_argument("--fraction-gpu-per-client", type=float, default=FLSettings.FRACTION_GPU, help="What fraction of the gpu a client gets, i.e. how many clients can work on a gpu simultaneously")
    parser.add_argument("--workers-per-client", type=int, default=FLSettings.WORKERS_PER_CLIENT, help="How many cpu workers a client is assigned")
    parser.add_argument("--fraction-client-eval", type=float, default=FLSettings.FRACTION_CLIENT_EVAL, help="Fraction of clients selected in a round to perform evaluation")
    parser.add_argument("--client-train-ds-mode", type=str, default=FLSettings.ATTACKER_DATASET_MODE, help="Dataset mode for attackers")

    # Training settings
    parser.add_argument("--training-num-workers", type=int, default=TrainingSettings.NUM_WORKERS, help="Number of workers used to load data from cpu to gpu")
    parser.add_argument("--client-local-training-epochs", type=int, default=TrainingSettings.LOCAL_EPOCHS, help="Number of local epochs that are ran by a client for training in round of federated learning")
    parser.add_argument("--training-batch-size", type=int, default=TrainingSettings.TRAINING_BATCH_SIZE, help="Batch size used for client training for both server evaluation")

    # Other options
    parser.add_argument("--do-seeding", action="store_true", default=False, help="Enable seeding of the randomness")
    parser.add_argument("--seed", type=int, default=FLSettings.SEED, help="Set the seed for all process utilizing randomness")
    # Attacker Settings
    parser.add_argument("--attacker-fraction", type=float, default=0.0, help="Set the fraction of clients that are malicious")

    parse_backdoor_args(parser)

    return parser.parse_args()

def parse_backdoor_args(parser):
    """
    Parse command line args specifically related to the backdoor / trigger
    """
    # Algemeen
    parser.add_argument("--disable-backdoor-logging", action="store_true", help="Disable the logging of backdoor parameters")
    parser.add_argument("--poison-rate", type=float, default=BackdoorProperties.POISON_RATE, help="Percentage of poisoned images")
    
    # Trigger opties
    parser.add_argument("--trigger-pattern", type=str, default=BackdoorProperties.TRIGGER_PATTERN.name, choices=[e.name for e in TriggerType], help="Type of trigger pattern")
    parser.add_argument("--trigger-alpha", type=float, default=BackdoorProperties.TRIGGER_ALPHA, help="Alpha blending value")
    parser.add_argument("--trigger-width", type=int, default=BackdoorProperties.TRIGGER_WIDTH, help="Width of uniform trigger")
    parser.add_argument("--trigger-height", type=int, default=BackdoorProperties.TRIGGER_HEIGHT, help="Height of uniform trigger")
    
    # Helper om de tuple om te zetten naar een string als default (bijv. "255,255,255")
    default_color = ",".join(map(str, BackdoorProperties.TRIGGER_COLOR))
    parser.add_argument("--trigger-color", type=str, default=default_color, help="RGB color comma separated (e.g., 255,255,255)")
    
    parser.add_argument("--block-size", type=int, default=BackdoorProperties.BLOCK_SIZE, help="Block size for checkerboard")
    parser.add_argument("--rng-seed", type=int, default=BackdoorProperties.RNG_SEED, help="Seed for noise trigger generation")

def apply_backdoor_settings(args):
    """
    Apply settings specifically for the attacker and backdoor triggers
    """
    # Attacker base settings
    FLSettings.ATTACKER_FRACTION = args.attacker_fraction
    FLSettings.ATTACKER_DATASET_MODE = args.client_train_ds_mode
    
    # Calculate number of attackers
    num_attackers = int(FLSettings.TOTAL_CLIENTS * FLSettings.ATTACKER_FRACTION)
    FLSettings.ATTACKER_IDS = list(range(num_attackers))

    # Backdoor properties
    BackdoorProperties.BACKDOOR_ACTIVE = not args.disable_backdoor_logging
    BackdoorProperties.POISON_RATE = args.poison_rate
    BackdoorProperties.TRIGGER_PATTERN = TriggerType[args.trigger_pattern]
    BackdoorProperties.TRIGGER_ALPHA = args.trigger_alpha
    BackdoorProperties.TRIGGER_WIDTH = args.trigger_width
    BackdoorProperties.TRIGGER_HEIGHT = args.trigger_height
    BackdoorProperties.BLOCK_SIZE = args.block_size
    BackdoorProperties.RNG_SEED = args.rng_seed
    
    # Parse color (van string "255,255,255" naar tuple (255, 255, 255))
    color_parts = args.trigger_color.split(',')
    if len(color_parts) == 3:
        BackdoorProperties.TRIGGER_COLOR = tuple(int(c.strip()) for c in color_parts)

    print("\n--- Attacker & Backdoor Settings ---")
    print(f"{'ATTACKER_FRACTION':30}: {FLSettings.ATTACKER_FRACTION} (Clients: {FLSettings.ATTACKER_IDS})")
    print(f"{'DATASET_MODE':30}: {FLSettings.ATTACKER_DATASET_MODE}")
    print(f"{'POISON_RATE':30}: {BackdoorProperties.POISON_RATE}")
    print(f"{'TRIGGER_PATTERN':30}: {BackdoorProperties.TRIGGER_PATTERN.name}")
    print(f"{'TRIGGER_ALPHA':30}: {BackdoorProperties.TRIGGER_ALPHA}")
    print(f"{'BLOCK_SIZE / RNG_SEED':30}: {BackdoorProperties.BLOCK_SIZE} / {BackdoorProperties.RNG_SEED}")
    print("\n===========================================================\n")

def apply_settings(args):
    """
    Apply the settings given from the command line args
    """
    # Federated Learning Settings
    FLSettings.NUM_ROUNDS = args.num_rounds
    FLSettings.TOTAL_CLIENTS = args.total_clients
    FLSettings.MIN_FIT_CLIENTS = args.min_fit_clients
    FLSettings.FRACTION_FIT = args.fraction_fit_clients # Let op: check of deze type=float heeft in parse_args!
    FLSettings.FRACTION_GPU = args.fraction_gpu_per_client
    FLSettings.WORKERS_PER_CLIENT = args.workers_per_client
    FLSettings.FRACTION_CLIENT_EVAL = args.fraction_client_eval
    FLSettings.ATTACKER_DATASET_MODE = args.client_train_ds_mode

    # Training Settings
    TrainingSettings.NUM_WORKERS = args.training_num_workers
    TrainingSettings.LOCAL_EPOCHS = args.client_local_training_epochs
    TrainingSettings.TRAINING_BATCH_SIZE = args.training_batch_size


    print("\n==================== Simulation Config ====================")

    print("\n--- Federated Learning Settings ---")
    print(f"{'NUM_ROUNDS':30}: {FLSettings.NUM_ROUNDS}")
    print(f"{'TOTAL_CLIENTS':30}: {FLSettings.TOTAL_CLIENTS}")
    print(f"{'MIN_FIT_CLIENTS':30}: {FLSettings.MIN_FIT_CLIENTS}")
    print(f"{'FRACTION_FIT':30}: {FLSettings.FRACTION_FIT}")
    print(f"{'FRACTION_CLIENT_EVAL':30}: {FLSettings.FRACTION_CLIENT_EVAL}")
    print(f"{'FRACTION_GPU':30}: {FLSettings.FRACTION_GPU}")
    print(f"{'WORKERS_PER_CLIENT':30}: {FLSettings.WORKERS_PER_CLIENT}")

    print("\n--- Training Settings ---")
    print(f"{'NUM_WORKERS':30}: {TrainingSettings.NUM_WORKERS}")
    print(f"{'LOCAL_EPOCHS':30}: {TrainingSettings.LOCAL_EPOCHS}")
    print(f"{'TRAINING_BATCH_SIZE':30}: {TrainingSettings.TRAINING_BATCH_SIZE}")

    print("\n--- Attacker Settings ---")
    print(f"{'ATTACKER_FRACTION':30}: {FLSettings.ATTACKER_FRACTION}")

    print("\n===========================================================\n")
    apply_backdoor_settings(args)

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
    
    # Create training configuration dict for reproducability
    config_dict = {
        # === FL Settings ===
        "num_rounds": FLSettings.NUM_ROUNDS,
        "total_clients": FLSettings.TOTAL_CLIENTS,
        "min_fit_clients": FLSettings.MIN_FIT_CLIENTS,
        "fraction_fit_clients": FLSettings.FRACTION_FIT,
        "fraction_client_eval": FLSettings.FRACTION_CLIENT_EVAL,
        "fraction_gpu_per_client": FLSettings.FRACTION_GPU,
        "workers_per_client": FLSettings.WORKERS_PER_CLIENT,
        "attacker_ids": list(FLSettings.ATTACKER_IDS),
        "random_seed": FLSettings.SEED,
        "attacker_fraction": FLSettings.ATTACKER_FRACTION,

        # === Training Settings ===
        "training_num_workers": TrainingSettings.NUM_WORKERS,
        "local_epochs": TrainingSettings.LOCAL_EPOCHS,
        "training_batch_size": TrainingSettings.TRAINING_BATCH_SIZE,
    }
    logger = ModelLogger(config_dict=config_dict)


    strategy = get_strategy(logger)

    # set_global_seed(FLSettings.SEED)
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
        
        # Properly close the I/O streams of the logger
        logger.close()

if __name__ == "__main__":
    args = parse_args()
    apply_settings(args)
    start_fl_simulation()
