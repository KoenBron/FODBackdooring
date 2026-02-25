import torch
import flwr as fl
from torch.utils.data import DataLoader
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


from utils.global_settings import TrainingSettings, FLSettings, ProjectPaths, BackdoorProperties
from utils.dataset import CustomCocoDataset
from utils.model_log import ModelLogger
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
    Produces dict the fit() function of the client receives
    """
    return { "server_round": server_round, "local_epochs": 1 }

def server_fn(context):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create training configuration dict for reproducability
    config_dict = {
        "num_rounds": FLSettings.NUM_ROUNDS,
        "total_clients": FLSettings.TOTAL_CLIENTS,
        "fraction_fit": FLSettings.FRACTION_FIT,
        "attacker_ids": FLSettings.ATTACKER_IDS
    }
    logger = ModelLogger(config_dict=config_dict)

    # Create the evaluation dataset and dataloader
    server_val_ds = CustomCocoDataset(
        image_folder=ProjectPaths.TRAIN_IMAGES_DIR,
        annotations_file=ProjectPaths.TRAIN_ANNOTATIONS_FILE,
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

    # Define a strategy
    strategy = FedAvg(
        fraction_fit=FLSettings.FRACTION_FIT,# Fraction of total clients that participate
        fraction_evaluate=0.0,# Fraction of CLIENTS that perform evaluation
        min_fit_clients=FLSettings.MIN_FIT_CLIENTS,# Minimum number of clients that participate
        min_available_clients=FLSettings.TOTAL_CLIENTS,
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(
            server_wrapper=server_wrapper,
            val_dl=server_val_dl,
            device=device,
            logger=logger
        )
    )

    # Create and return server app
    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=FLSettings.NUM_ROUNDS)
    )

app = fl.server.ServerApp(server_fn=server_fn)
