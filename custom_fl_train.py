import random

from utils.parse_args import parse_args, apply_settings
from utils.global_settings import FLSettings
from utils.model_log import ModelLogger
from utils.select_clients import select_clients, fast_forward_client_selection
from FL.custom_server import Server
from FL.custom_client import ClientManager

def start_fl_simulation(args):
    """
    Main body for running the simultation
    """
    # ModelLogger owns the attack-state path + per-protocol sub-loggers
    # (logger.pilot_a etc.) and calls AttackState.load() during construction,
    # so cross-shift carrying-set resume happens silently here. The run's settings
    # are written to config.txt from the global settings (structured summary) plus
    # `args` (raw dump) — see ModelLogger / parse_args.build_config_sections.
    logger = ModelLogger(custom_path=FLSettings.RESUME_FROM, args=args)

    # Setup the server
    server = Server(strategy=FLSettings.AGGREGATION_STRATEGY, logger=logger, checkpoint_id=FLSettings.RESUME_FROM)

    # Setup the clients — pass the logger so AttackProtocols can reach
    # their sub-loggers via self.manager.logger.<protocol>.
    client_manager = ClientManager(num_clients=FLSettings.TOTAL_CLIENTS, logger=logger)

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

            # `round` is consumed by AttackProtocols (PilotA keys its log
            # files by it). Per-round output paths come from the logger
            # via self.manager.logger.<protocol> — no path plumbing here.
            is_attacker = (client_index in FLSettings.ATTACKER_IDS)
            client_config = {
                "index": client_index,
                "is_attacker": is_attacker,
                "n_mal_clients": n_mal_clients,
                "round": current_round,
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


        # Perform aggregation of client updates after. This also persists the round
        # checkpoint and, back-to-back with it, the attacker state (AttackState.save,
        # atomic tmp+rename) — see aggregate_and_evaluate for the ordering rationale.
        server.aggregate_and_evaluate(round=current_round, num_attackers=n_mal_clients)

    print("\n________________________SIMULATION COMPLETE________________________")

if __name__ == "__main__":
    args = parse_args()
    apply_settings(args=args)
    start_fl_simulation(args=args)


