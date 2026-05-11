import random

from utils.parse_args import parse_args, apply_settings
from utils.global_settings import FLSettings, TrainingSettings 
from utils.model_log import ModelLogger
from utils.select_clients import select_clients, fast_forward_client_selection
from FL.custom_server import Server
from FL.custom_client import ClientManager

def start_fl_simulation(args):
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
        "server_lr": FLSettings.SERVER_LR,
        "gamma_cap": FLSettings.GAMMA_CAP,
        # === Training Settings ===
        "benign_client_local_epochs": TrainingSettings.BENIGN_CLIENT_LOCAL_EPOCHS,
        "malicious_client_local_epochs": TrainingSettings.MALICIOUS_CLIENT_LOCAL_EPOCHS,
        "training_batch_size": TrainingSettings.TRAINING_BATCH_SIZE,
    }

    logger = ModelLogger(custom_path=FLSettings.RESUME_FROM, config_dict=config_dict, args=args)

    # Setup the server
    server = Server(strategy=FLSettings.AGGREGATION_STRATEGY, logger=logger, checkpoint_id=FLSettings.RESUME_FROM)

    # Setup the clients
    client_manager = ClientManager(num_clients=FLSettings.TOTAL_CLIENTS)

    # Seed the random generator for client selection
    random_client_selector = random.Random(FLSettings.SEED)

    start_round = server.get_round_offset() + 1

    if start_round > 1:
        fast_forward_client_selection(start_round=start_round, generator=random_client_selector)

    print("\n________________________STARTING FL SIMULATION________________________")

    for current_round in range(start_round, FLSettings.NUM_ROUNDS + 1):
        # Outer loop for training rounds
        print(f"\n________________________STARTING ROUND {current_round}________________________")
        
        # Select clients
        selected_clients = select_clients(round=current_round, generator=random_client_selector)

        # What server sends out to the clients:
        server_weights = server.get_global_weights()

        # Roll the Neurotoxin round cache (no-op when neurotoxin disabled, but cheap)
        client_manager.update_round_cache(server_weights)

        # Determine beforehand for model_replacement
        n_mal_clients = len([cid for cid in selected_clients if cid in FLSettings.ATTACKER_IDS])

        for client_index in selected_clients:

            # Build the client
            is_attacker = (client_index in FLSettings.ATTACKER_IDS)
            client_config = {
                "index": client_index,
                "is_attacker": is_attacker,
                "n_mal_clients": n_mal_clients,
            }

            # Perform the client update
            client_weights_update, num_samples, client_metrics = client_manager.client_side_train(
                server_weights=server_weights,
                client_config=client_config,
            )

            # Store the client update on the server
            server.receive_client_update(
                client_config=client_config,
                num_samples=num_samples,
                client_weights_update=client_weights_update,
                # client_metrics=client_metrics,# TODO: Add back for logging of loss
            )


        # Perform aggregation of client updates after 
        server.aggregate_and_evaluate(round=current_round, num_attackers=n_mal_clients)

    print("\n________________________SIMULATION COMPLETE________________________")

if __name__ == "__main__":
    args = parse_args()
    apply_settings(args=args)
    start_fl_simulation(args=args)


