from FL.strategies import FedAvg
from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.global_settings import ProjectPaths, BackdoorProperties, TrainingSettings
from utils.dataset import CustomCocoDataset
from utils.metrics import torch_compute_magnitude, torch_compute_cosine_sim, torch_compute_l2_distance

from torch.utils.data import DataLoader

import torch
import os

class Server:
    def __init__(self, strategy: str, logger, checkpoint_path=None):
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup model the server uses
        self.val_dl, num_classes= self._setup_val_dl()
        self.wrapper = FasterRCNNWrapper(num_classes=num_classes)
        
        # If checkpoint path to resume training from is given, load it to the server model
        self.round_offset = 0
        if checkpoint_path is not None:
            self._load_from_checkpoint(checkpoint_path=checkpoint_path)

        self.strategy = self._select_strategy(strategy=strategy)

        # List to accumulate computed metrics and training_logs for each client participating in the current round
        self.current_round_logs = []

    def _setup_val_dl(self):
        # Create dataset and build dataloader 
        val_ds = CustomCocoDataset(
            image_folder=ProjectPaths.VALIDATION_IMAGES_DIR,
            annotations_file=ProjectPaths.VALIDATION_ANNOTATIONS_FILE,
            width=BackdoorProperties.RESIZE_WIDTH,
            height=BackdoorProperties.RESIZE_HEIGHT,
            mode="always_clean",
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=TrainingSettings.TRAINING_BATCH_SIZE,
            num_workers=TrainingSettings.NUM_WORKERS,
            collate_fn=FasterRCNNWrapper.get_collate_fn(),
            pin_memory=True,
            shuffle=True,
        )

        # Num samples need for instantiating server model
        num_classes = val_ds.get_total_classes_count() + 1

        return val_dl, num_classes

    def _construct_checkpoint_path(self):
        return os.path.join(ProjectPaths.OUTPUT_DIR, checkpoit_id, "latest_fl_checkpoint.pth")

    def _load_from_checkpoint(self, checkpoint_id: str):
        """
        Load the server's model wrapper with a checkpoint. Checkpoints are identified by a string formatted as: "day-month-year-hour-minute-second"
        """
        # Build the checkpoint path
        checkpoint_path = self._construct_checkpoint_path()
        if not os.path.isdir(checkpoint_path):
            raise Exception(f"Couldn't construct valid checkpoint path. {checkpoint_path} is not a directory")
        checkpoint = torch.load(checkpoint_path)

        print(f"[INFO] Loading model from: {checkpoint_path} to resume training")
        self.wrapper.set_weights(checkpoint["model_state_dict"])
        self.round_offset = checkpoint["round"]

    def _save_latest_model(self, round: int):
        """
        Save the model_state_dict at stored in the server model wrapper and the training round as a checkpoint. 
        """
        checkpoint_path = self._construct_checkpoint_path()
        if not os.path.isdir(checkpoint_path):
            raise Exception(f"Couldn't construct valid checkpoint path. {checkpoint_path} is not a directory")

        # Build checkpoint
        checkpoint = {
            "model_state_dict": torch.save(self.wrapper.get_weights()),
            "round": round,
        }
        torch.save(checkpoint, checkpoint_path)

    def _select_strategy(self, strategy: str):
        """
        Selects the strategy object to use based on the string 'strategy'
        """
        match strategy.lower():
            case "fedavg": 
                return FedAvg(model_state_dict=self.wrapper.get_weights())
            case _:
                raise Exception(f"No valid aggregation strategy name is given: {strategy}. Choices are: ['fedavg',] ")

    def get_gloabl_weights(self):
        """
        Wrapper function to get the server model's weights.
        """
        return self.wrapper.get_weights()

    def receive_client_update(self, client_index: int, is_attacker: bool, num_samples: int, client_weight_update, client_metrics):
        global_weights = self.get_gloabl_weights()

        client_type = "Attacker" if is_attacker else "Benign"
        client_round_logs = {
            "cid": client_index,# Partition_id = cid on the client-side, server side uses internal node_id
            "type": client_type,
            "magnitude": magnitude,
            "l2_norm": l2_norm,
            "cos_sim": cos_sim,
        }

        # Send client

        # Accumulate the client weight in the strategy
        pass

    def aggregate_and_evaluate(self, round: int):
        # Compute client update metrics the logger wants to now and store them
        cos_sim = torch_compute_cosine_sim(client_weight_update, global_weights)
        l2_norm = torch_compute_l2_distance(client_weight_update, global_weights)
        magnitude = torch_compute_magnitude(client_weight_update)

    
