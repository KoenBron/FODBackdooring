from FL.strategies import FedAvg
from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.global_settings import ProjectPaths, BackdoorProperties, TrainingSettings
from utils.dataset import CustomCocoDataset
from utils.metrics import torch_compute_magnitude, torch_compute_cosine_sim, torch_compute_l2_distance

from torch.utils.data import DataLoader

import torch
import os

class Server:
    def __init__(self, strategy: str, logger, checkpoint_id=None):
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup model the server uses
        self.val_dl, num_classes= self._setup_val_dl()
        self.wrapper = FasterRCNNWrapper(num_classes=num_classes)
        
        # If checkpoint path to resume training from is given, load it to the server model
        self.round_offset = 0
        if checkpoint_id is not None:
            self._load_from_checkpoint(checkpoint_path=checkpoint_id)

        self.strategy = self._select_strategy(strategy=strategy)

        # Dict of dicts to index client updates by client_index
        self.client_round_updates = {}

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
        return os.path.join(self.logger.output_path, "latest_fl_checkpoint.pth")

    def _load_from_checkpoint(self, checkpoint_id: str):
        """
        Load the server's model wrapper with a checkpoint. Checkpoints are identified by a string formatted as: "day-month-year-hour-minute-second"
        """
        # Build the checkpoint path
        checkpoint_path = self._construct_checkpoint_path()
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path)
                print(f"[INFO] Loading model from: {checkpoint_path} to resume training")
                self.wrapper.set_weights(checkpoint["model_state_dict"])
                self.round_offset = checkpoint["round"]
            except Exception as e:
                raise Exception(f"[ERROR] Exception occurred while loading checkpoint: {e}")
        else:
            raise Exception(f"Couldn't construct valid checkpoint path. {checkpoint_path} is not a directory")

    def _save_latest_model(self, round: int, checkpoint_id: int):
        """
        Save the model_state_dict at stored in the server model wrapper and the training round as a checkpoint. 
        """
        checkpoint_path = self._construct_checkpoint_path()
        if not os.path.isdir(checkpoint_path):
            raise Exception(f"Couldn't construct valid checkpoint path. {checkpoint_path} is not a directory")

        # Build checkpoint
        checkpoint = {
            "model_state_dict": self.wrapper.get_weights(),
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

    def get_global_weights(self):
        """
        Wrapper function to get the server model's weights.
        """
        return self.wrapper.get_weights()

    def receive_client_update(self, client_index: int, is_attacker: bool, num_samples: int, client_weight_update, client_metrics):
        # Setup logging information
        client_type = "Attacker" if is_attacker else "Benign"
        client_update = {
            "cid": client_index,# Partition_id = cid on the client-side, server side uses internal node_id
            "type": client_type,
# Placeholder values to be determined at aggregation time
            "magnitude": None,
            "l2_norm": None,
            "cos_sim": None,
            # Temporarily necessary for logging of metrics at aggregation
            "weights": client_weight_update,
            "num_samples": num_samples,
        }

        self.client_round_updates[client_index] = client_update
        
        # Have aggregation strategy accumulate the update
        # self.strategy.accumulate(client_weights=client_weight_update, num_samples=num_samples, client_index=client_index)

    def aggregate_and_evaluate(self, round: int):
        # Aggregate the client updates
        new_global_weights = self.strategy.aggregate(client_updates=list(self.client_round_updates.values()))# Expects a list

        for key in self.client_round_updates.keys():
            update = self.client_round_updates[key]
            client_weight_update = update["weights"]

            # Compute client update metrics the logger wants to now and store them
            cos_sim = torch_compute_cosine_sim(client_weight_update, new_global_weights)
            l2_norm = torch_compute_l2_distance(client_weight_update, new_global_weights)
            magnitude = torch_compute_magnitude(client_weight_update)

            update['cos_sim'] = cos_sim
            update['l2_norm'] = l2_norm
            update['magnitude'] = magnitude

            # Free the memory of the large weight updates
            del update[key]["weights"]

        self.logger.log_client_updates(server_round=round, update_info=self.client_round_updates)


        # Evaluate the model
        self.wrapper.set_weights(new_global_weights)

        metrics, num_samples = self.wrapper.evaluate(dataloader=self.val_dl, device=self.device)

        self.logger.log_epoch(round, metrics)
        self.logger.save_if_best_model(epoch=round, val_loss=metrics["val_loss"], weights=new_global_weights)
        
        # Prepare for another round 
        self.client_round_updates.clear()

        self._save_latest_model(round=round)

