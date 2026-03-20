from collections import OrderedDict
import torch
import os
from torch.utils.data import DataLoader
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays

from utils.global_settings import TrainingSettings, FLSettings, ProjectPaths, BackdoorProperties
from utils.dataset import CustomCocoDataset
from utils.metrics import compute_magnitude, compute_cosine_sim, compute_l2_distance
from models.fasterrcnn.wrapper import FasterRCNNWrapper

####################################
# Server evaluation factory function 
####################################
def get_evaluate_fn(server_wrapper, val_dl, device, logger):
    """
    Returns callback function for server evaluation function, that needs the
    the arguments the norma evaluate_fn doesn't have
    """
    def evaluate(server_round, parameters, config):
        """
        Performs the evaluation in order with how flower expects the return values
        """
        if logger is None:
            raise Exception("No server side logger provided for evaluation, can't save model")

        # Load new weights
        server_wrapper.set_fl_parameters(parameters)

        # Perform evaluation step
        metrics, num_samples = server_wrapper.evaluate(dataloader=val_dl, device=device)

        # Log the metrics
        logger.log_epoch(server_round, val_metrics=metrics)

        # Save the model using the logger
        weights = server_wrapper.get_weights()
        logger.save_if_best_model(epoch=server_round, val_loss=metrics["val_loss"], weights=weights)
        
        # Return the values that flower expects
        return float(metrics["val_loss"]), metrics

    return evaluate

def fit_config(server_round):
    """
    Produces the config dict the fit() function of the client receives
    """
    return { "server_round": server_round, "local_epochs": TrainingSettings.LOCAL_EPOCHS }

class CustomFedAvg(FedAvg):
    def __init__(self, logger, state_dict_keys, round_offset=0, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.logger = logger
        self.round_offset = round_offset
        self.state_dict_keys = state_dict_keys

    def aggregate_fit(self, server_round, results, failures):
        """
        Wrapper for aggregate_fit of fedavg that also logs:
        - L2 norm to global mean
        - Cosine similarity to mean
        """
        learning_round = server_round + self.round_offset
        aggregated_params, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated_params is not None:
            # Convert from flwr.Param -> ndarray(1,)
            global_weights = parameters_to_ndarrays(aggregated_params)

            # Iterate throught the clients to build logger info for aggregation_round
            update_info = []

            for client, fit_res in results:
                partition_id = fit_res.metrics.get("partition_id", client.cid)
                # Convert from flwr.Param -> ndarray(1,)
                client_weights = parameters_to_ndarrays(fit_res.parameters)

                # Compute similarity metrics
                cos_sim = compute_cosine_sim(client_weights, global_weights)
                l2_norm = compute_l2_distance(client_weights, global_weights)
                magnitude = compute_magnitude(client_weights)

                is_attacker = (partition_id in FLSettings.ATTACKER_IDS)
                client_type = "Attacker" if is_attacker else "Benign"
                
                update_info.append({
                    "cid": partition_id,# Partition_id = cid on the client-side, server side uses internal node_id
                    "type": client_type,
                    "magnitude": magnitude,
                    "l2_norm": l2_norm,
                    "cos_sim": cos_sim,
                })

            # Sort by client ID
            update_info.sort(key=lambda x: x["cid"])

            # Log the server round
            self.logger.log_client_updates(learning_round, update_info)

            # Save the checkpoint for later use if necessary
            checkpoint_path = os.path.join(self.logger.output_path, "latest_fl_checkpoint.pth")
            # Reconstruct the model_state_dict
            reconstructed_state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.state_dict_keys, global_weights)})

            # Create a checkpoint
            checkpoint = {
                "model_state_dict": reconstructed_state_dict,
                "round": server_round,
            }
            torch.save(checkpoint, checkpoint_path)

            return aggregated_params, metrics_aggregated


def get_strategy(logger):
    """
    Builds the strategy as flwr next doesn't support node_id:partition_id mappings
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the evaluation dataset and dataloader
    server_val_ds = CustomCocoDataset(
        image_folder=ProjectPaths.VALIDATION_IMAGES_DIR,
        annotations_file=ProjectPaths.VALIDATION_ANNOTATIONS_FILE,
        width=BackdoorProperties.RESIZE_WIDTH,
        height=BackdoorProperties.RESIZE_HEIGHT,
        mode="always_clean",
    )

    server_val_dl = DataLoader(
        server_val_ds,
        batch_size=TrainingSettings.TRAINING_BATCH_SIZE,
        num_workers=TrainingSettings.NUM_WORKERS,
        collate_fn=FasterRCNNWrapper.get_collate_fn(),
        pin_memory=True,
        shuffle=True,
    )

    # Define wrapper for server's own model
    num_classes = server_val_ds.get_total_classes_count() + 1
    server_wrapper = FasterRCNNWrapper(num_classes=num_classes)

    round_offset = 0
    initial_parameters = None

    # Load model if necessary
    if FLSettings.RESUME_FROM is not None:
        checkpoint_path = os.path.join(ProjectPaths.OUTPUT_DIR, FLSettings.RESUME_FROM, "latest_fl_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # Convert the model_state_dict to flwr Parameters for a warm start
        initial_weights = checkpoint["model_state_dict"]
        server_wrapper.set_weights(initial_weights)
        initial_parameters = fl.common.ndarrays_to_parameters(server_wrapper.get_fl_parameters())

        round_offset = checkpoint["round"]

    # Define a strategy
    strategy = CustomFedAvg(
        logger=logger,
        fraction_fit=FLSettings.FRACTION_FIT,# Fraction of total clients that participate
        fraction_evaluate=0.0,# Fraction of CLIENTS that perform evaluation
        min_fit_clients=FLSettings.MIN_FIT_CLIENTS,# Minimum number of clients that participate
        min_available_clients=FLSettings.TOTAL_CLIENTS,
        on_fit_config_fn=fit_config,
        round_offset=round_offset,
        state_dict_keys=server_wrapper.get_model_state_keys(),
        initial_parameters=initial_parameters,
        evaluate_fn=get_evaluate_fn(
            server_wrapper=server_wrapper,
            val_dl=server_val_dl,
            device=device,
            logger=logger
        )
    )
    
    return strategy

# server_app = fl.server.ServerApp(server_fn=server_fn)
