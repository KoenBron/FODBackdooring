import torch
import random
import numpy as np
from torch.utils.data import DataLoader

from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.dataset import CustomCocoDataset
from utils.global_settings import ProjectPaths, BackdoorProperties, FLSettings, TrainingSettings
from utils.partitioning import get_shuffled_indices, set_global_seed, partition_dataset

class ClientManager:
    def __init__(self, num_clients: int):
        self.is_attacker = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create the subset for each client once
        ds_mal, ds_benign = self._setup_datasets(num_clients=num_clients)
        set_global_seed(FLSettings.SEED)
        shuffled_indices = get_shuffled_indices(len(ds_benign))
        self.subsets_mal, self.subsets_benign = partition_dataset(dataset=ds_mal, n_clients=num_clients, indices=shuffled_indices), partition_dataset(dataset=ds_benign, n_clients=num_clients, indices=shuffled_indices)

        # TODO: Change to be more elegant
        # Unseed the randomness after having the subset created
        random.seed()
        torch.seed()
        np.random.seed()

        # Clients work on the same model instance but load in their parameters
        num_classes = ds_benign.get_total_classes_count() + 1
        self.wrapper = FasterRCNNWrapper(num_classes=num_classes)

    def _setup_datasets(self, num_clients=None):
        """
        Returns the benign and malicious datasets in the dataset as a set
        """
        if num_clients is None:
            raise Exception("Client manager improperly instantiated, no total number of clients given when initializing")
        mal_ds = CustomCocoDataset(
            image_folder=ProjectPaths.TRAIN_IMAGES_DIR,
            annotations_file=ProjectPaths.TRAIN_ANNOTATIONS_FILE,
            width=BackdoorProperties.RESIZE_WIDTH,
            height=BackdoorProperties.RESIZE_HEIGHT,
            mode=FLSettings.ATTACKER_DATASET_MODE,
        )

        benign_ds = CustomCocoDataset(
            image_folder=ProjectPaths.TRAIN_IMAGES_DIR,
            annotations_file=ProjectPaths.TRAIN_ANNOTATIONS_FILE,
            width=BackdoorProperties.RESIZE_WIDTH,
            height=BackdoorProperties.RESIZE_HEIGHT,
            mode="always_clean",
        )

        return mal_ds, benign_ds
        
    def _retrieve_client_dataloader(self, client_index, is_attacker):
        """
        Retrieves the dataloader for the part of the dataset assigned to the given client based on client_index and selects from the correct update.
        client_index is 0-indexed.
        """
        if is_attacker:
            subsets = self.subsets_mal
        else:
            subsets = self.subsets_benign

        return DataLoader(
                subsets[client_index],
                batch_size=TrainingSettings.TRAINING_BATCH_SIZE,
                num_workers=TrainingSettings.NUM_WORKERS,
                collate_fn=FasterRCNNWrapper.get_collate_fn(),
                pin_memory=True,
                shuffle=True,
            )

    def get_parameters(self):
        # Delegate to wrapper to make client implementation model agnsostic
        return self.wrapper.get_weights()

    def set_weights(self, weights):
        # Delegate to wrapper to make client implementation model agnsostic
        self.wrapper.set_weights(weights=weights)

    def client_side_train(self, server_weights=None, client_config=None):
        """
        First loads the server_weights onto the client manager model and then
        performs the client side training and returns the training results as pytorch tensors
        """
        try:
            client_index = client_config["index"]
            client_is_attacker = client_config["is_attacker"]
        except KeyError:
            raise Exception("Client manager error: Failed to retrieve needed information from supplied client_config. Expected client_config keys: 'index', 'is_attacker'")

        if server_weights is None:
            raise Exception("Client manager error: No server_weights were supplied for training")

        # Load starting point for training
        self.set_weights(weights=server_weights) 
        train_dl = self._retrieve_client_dataloader(client_index=client_index, is_attacker=client_is_attacker)

        num_epochs = TrainingSettings.MALICIOUS_CLIENT_LOCAL_EPOCHS if client_is_attacker else TrainingSettings.BENIGN_CLIENT_LOCAL_EPOCHS 

        _, num_samples, final_epoch_losses, self.optimizer = self.wrapper.train_step(
            dataloader=train_dl,
            epochs= num_epochs,
            device=self.device,
            optimizer_config={},
            partition_id=client_index,
            optimizer=None,
        )

        # Retrieve the updated parameters
        updated_params = self.get_parameters()

        # Bundle metrics and info needed by the logger
        metrics = {
            "partition_id": client_index
        }
        if final_epoch_losses is not None:
            for k, v in final_epoch_losses.items():
                if hasattr(v, 'item'):
                    # Tensor case
                    metrics[k] = v.item()
                else:
                    # Just in case
                    metrics[k] = float(v)

        return updated_params, num_samples, metrics

