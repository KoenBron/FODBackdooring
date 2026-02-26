import flwr as fl
import numpy as np
from flwr.simulation import start_simulation
import torch
import argparse
import random

from utils.global_settings import FLSettings, TrainingSettings
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

    # Training settings
    parser.add_argument("--training-num-workers", type=int, default=TrainingSettings.NUM_WORKERS, help="Number of workers used to load data from cpu to gpu")
    parser.add_argument("--client-local-training-epochs", type=int, default=TrainingSettings.LOCAL_EPOCHS, help="Number of local epochs that are ran by a client for training in round of federated learning")
    parser.add_argument("--training-batch-size", type=int, default=TrainingSettings.TRAINING_BATCH_SIZE, help="Batch size used for client training for both server evaluation")
    parser.add_argument("--do_seeding", type=bool, default=False, help="Enable seeding of the randomness")

    # Other
    parser.add_argument("--seed", type=int, default=FLSettings.SEED, help="Set the seed for all process utilizing randomness")

def apply_settings(args):
    """
    Apply the settings given from the command line args
    """
    FLSettings.NUM_ROUNDS = args.num_rounds
    FLSettings.TOTAL_CLIENTS = args.total_clients
    FLSettings.MIN_FIT_CLIENTS = args.min_fit_clients
    FLSettings.FRACTION_FIT = args.fraction_fit_clients # Let op: check of deze type=float heeft in parse_args!
    FLSettings.FRACTION_GPU = args.fraction_gpu_per_client
    FLSettings.WORKERS_PER_CLIENT = args.workers_per_client
    FLSettings.FRACTION_CLIENT_EVAL = args.fraction_client_eval

    # Training Settings
    TrainingSettings.NUM_WORKERS = args.training_num_workers
    TrainingSettings.LOCAL_EPOCHS = args.client_local_training_epochs
    TrainingSettings.TRAINING_BATCH_SIZE = args.training_batch_size

    print("====================Simulation Config====================")
    print(args)
    print("=========================================================")

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

        # === Training Settings ===
        "training_num_workers": TrainingSettings.NUM_WORKERS,
        "local_epochs": TrainingSettings.LOCAL_EPOCHS,
        "training_batch_size": TrainingSettings.TRAINING_BATCH_SIZE,
    }
    logger = ModelLogger(config_dict=config_dict)


    strategy = get_strategy(logger)

    set_global_seed(FLSettings.SEED)
    print("______________SIMULATION STARTING______________")

    start_simulation(
        client_fn=client_fn,
        config=fl.server.ServerConfig(num_rounds=FLSettings.NUM_ROUNDS),
        num_clients=FLSettings.TOTAL_CLIENTS,
        strategy=strategy,
        client_resources=client_resources,
    )

    print("______________SIMULATION COMPLETE______________")
    
    # Properly close the I/O streams of the logger
    logger.close()

if __name__ == "__main__":
    start_fl_simulation()
