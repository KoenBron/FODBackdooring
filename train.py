import flwr as fl
import torch
from flwr.simulation import start_simulation

from utils.global_settings import FLSettings
from utils.model_log import ModelLogger
from FL.client import client_fn
from FL.server import get_strategy

def start_fl_simulation():
    client_resources = {
        "num_cpus": 4,
        "num_gpus": FLSettings.FRACTION_GPU if torch.cuda.is_available() else 0.0
    }
    
    # Create training configuration dict for reproducability
    config_dict = {
        "num_rounds": FLSettings.NUM_ROUNDS,
        "total_clients": FLSettings.TOTAL_CLIENTS,
        "fraction_fit": FLSettings.FRACTION_FIT,
        "attacker_ids": FLSettings.ATTACKER_IDS
    }
    logger = ModelLogger(config_dict=config_dict)

    strategy = get_strategy(logger)
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
