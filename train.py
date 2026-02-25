import flwr as fl
import torch

from utils.global_settings import FLSettings
from FL.client import client_app
from FL.server import server_app

def run_fl_simulation():
    client_resources = {
        "num_cpus": 4,
        "num_gpus": 0.5 if torch.cuda.is_available() else 0.0
    }

    print("______________SIMULATION STARTING______________")

    fl.simulation.start_simulation(
        client_app=client_app,
        server_app=server_app,
        num_supernodes=FLSettings.TOTAL_CLIENTS,
        backend_config={"client_resources": client_resources}
    )

    print("______________SIMULATION COMPLETE______________")


if __name__ == "__main__":
    run_fl_simulation()
