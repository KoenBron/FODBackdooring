import flwr as fl
import torch
from torch.utils.data import DataLoader

from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.partitioning import partition_dataset
from utils.dataset import CustomCocoDataset
from utils.global_settings import BackdoorProperties, ProjectPaths, TrainingSettings, FLSettings
from utils.model_log import ModelLogger
from FL.client import Client
from FL.server import get_evaluate_fn

def fit_config(server_round):
    return { "server_round": server_round, "local_epochs": 1 }

def run_fl_simulation():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create logger to store server aggregated model
    logger = ModelLogger(
        config_dict={
            "num_rounds": FLSettings.NUM_ROUNDS,
            "total_clients": FLSettings.TOTAL_CLIENTS,
            "attacker_fraction": FLSettings.ATTACKER_FRACTION,
            "attacker_rounds": FLSettings.ATTACK_ROUNDS,
            "attacker_ids": FLSettings.ATTACKER_IDS,
        }
    )

    # Get the base training dataset 
    base_train_ds = CustomCocoDataset(
        image_folder=ProjectPaths.TRAIN_IMAGES_DIR,
        annotations_file=ProjectPaths.TRAIN_ANNOTATIONS_FILE,
        width=BackdoorProperties.RESIZE_WIDTH,
        heigh=BackdoorProperties.RESIZE_HEIGHT,
        mode="always_clean",
    )
    
    # Get client relevant data  
    num_classes = base_train_ds.get_total_classes_count() + 1
    subsets = partition_dataset(base_train_ds, FLSettings.TOTAL_CLIENTS)

    # Prepare server evaluation dataset and dataloader
    server_val_ds = CustomCocoDataset(
        image_folder=ProjectPaths.VALIDATION_IMAGES_DIR, # Eigenlijk validation set paths hier
        annotations_file=ProjectPaths.VALIDATION_ANNOTATIONS_FILE,
        width=BackdoorProperties.RESIZE_WIDTH,
        heigh=BackdoorProperties.RESIZE_HEIGHT,
        mode="always_clean",
    )

    server_val_dl = DataLoader(
        server_val_ds,
        batch_size=TrainingSettings.TRAINING_BATCH_SIZE,
        shuffle=True,
        num_workers=TrainingSettings.NUM_WORKERS,
        collate_fn=FasterRCNNWrapper.get_collate_fn(),
        pin_memory=True,
    )

    # Prepare server model
    server_wrapper = FasterRCNNWrapper(num_classes=num_classes)

    def client_fn(cid: str):
        """
        Client factory callback function, defined here to take attacker parameters into account
        """
        # Get the correct subset of the training_dataset
        client_id = int(cid)
        is_attacker = (client_id in FLSettings.ATTACKER_IDS)
        client_subset = subsets[client_id]

        client_wrapper = FasterRCNNWrapper(num_classes=num_classes)

        # Create client dataloader with the client subset of the dataset
        client_train_dl = DataLoader(
            client_subset,
            batch_size=TrainingSettings.TRAINING_BATCH_SIZE,
            shuffle=True,
            num_workers=TrainingSettings.NUM_WORKERS,
            collate_fn=FasterRCNNWrapper.get_collate_fn(),
            pin_memory=True,
        )

        return Client(
            wrapper=client_wrapper,
            train_dl=client_train_dl,
            val_dl=None,
            device=device,
            is_attacker=is_attacker,
            optimzer_config=None
        )

    # Define the aggregation strategy
    strategy = fl.server.strategy.FedAvg(
        fraction=FLSettings.ATTACKER_FRACTION,
        min_fit_clients=max(2, int(FLSettings.TOTAL_CLIENTS * FLSettings.FRACTION_FIT)),
        min_available_clients=FLSettings.TOTAL_CLIENTS,
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(server_wrapper, server_val_dl, device, logger) # Verbind de server logica!)
    )

    gpu_allocation = 0.5 if torch.cuda.is_available() else 0.0

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=FLSettings.TOTAL_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=FLSettings.NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": TrainingSettings.NUM_WORKERS, "num_gpus": gpu_allocation}
    )
    
    logger.close()

if __name__ == "__main__":
    run_fl_simulation()
