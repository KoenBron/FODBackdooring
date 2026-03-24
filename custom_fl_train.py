import random

from utils.parse_args import parse_args, apply_settings
from utils.global_settings import FLSettings, TrainingSettings 
from FL.custom_server import Server
from FL.custom_client import ClientManager
from utils.model_log import ModelLogger

def start_fl_simulation():
    """
    Main body for running the simultation
    """
    # Setup logger
    config_dict = {
        # === FL Settings ===
        "num_rounds": FLSettings.NUM_ROUNDS,
        "total_clients": FLSettings.TOTAL_CLIENTS,
        "clients_per_round": FLSettings.CLIENTS_PER_ROUND,
        "attacker_ids": list(FLSettings.ATTACKER_IDS),
        "random_seed": FLSettings.SEED,
        "attacker_fraction": FLSettings.ATTACKER_FRACTION,
        "client_train_ds_mode": FLSettings.ATTACKER_DATASET_MODE,
        "aggregation_strategy": FLSettings.AGGREGATION_STRATEGY,
        # === Training Settings ===
        "benign_client_local_epochs": TrainingSettings.BENIGN_CLIENT_LOCAL_EPOCHS,
        "malicious_client_local_epochs": TrainingSettings.MALICIOUS_CLIENT_LOCAL_EPOCHS,
        "training_batch_size": TrainingSettings.TRAINING_BATCH_SIZE,
    }

    logger = ModelLogger(custom_path=FLSettings.RESUME_FROM, config_dict=config_dict)

    # Setup the server
    server = Server(strategy=FLSettings.AGGREGATION_STRATEGY, logger=logger, checkpoint_id=FLSettings.RESUME_FROM)

    # Setup the clients
    client_manager = ClientManager(num_clients=FLSettings.TOTAL_CLIENTS)


    start_round = server.get_round_offset() + 1

    print("\n________________________STARTING FL SIMULATION________________________")

    for current_round in range(start_round, FLSettings.NUM_ROUNDS + 1):
        # Outer loop for training rounds
        print(f"\n________________________STARTING ROUND {current_round}________________________")
        
        # Select clients
        selected_clients = random.sample(range(FLSettings.TOTAL_CLIENTS), FLSettings.CLIENTS_PER_ROUND)

        # What server sends out to the clients:
        server_weights = server.get_global_weights()

        for client_index in selected_clients:
            # Build the client
            is_attacker = (client_index in FLSettings.ATTACKER_IDS)
            client_config = {
                "index": client_index,
                "is_attacker": is_attacker,
            }

            client_weights_update, num_samples, client_metrics = client_manager.client_side_train(
                server_weights=server_weights,
                client_config=client_config
            )

            server.receive_client_update(
                client_config=client_config,
                num_samples=num_samples,
                client_weights_update=client_weights_update,
                # client_metrics=client_metrics,# TODO: Add back for logging of loss
            )

        server.aggregate_and_evaluate(round=current_round)

    print("\n________________________SIMULATION COMPLETE________________________")

if __name__ == "__main__":
    args = parse_args()
    apply_settings(args=args)
    start_fl_simulation()


