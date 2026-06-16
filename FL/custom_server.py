from FL.strategies.fedavg import FedAvg
from FL.strategies.fedopt import FedOpt
from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.global_settings import ProjectPaths, BackdoorProperties, TrainingSettings, FLSettings
from utils.dataset import CustomCocoDataset
from utils.round_evaluator import RoundEvaluator
from utils.metrics import torch_compute_magnitude, torch_compute_cosine_sim, torch_compute_l2_distance
from FL.stealth_defenses import analyze as stealth_analyze

from torch.utils.data import DataLoader

import torch
import os

class Server:
    def __init__(self, strategy: str, logger, checkpoint_id=None):
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup model the server uses
        self.clean_val_dl, self.poison_val_dl, num_classes= self._setup_val_dl()
        self.wrapper = FasterRCNNWrapper(num_classes=num_classes)
        self.round_evaluator = RoundEvaluator(wrapper=self.wrapper, annotations_file=ProjectPaths.VALIDATION_ANNOTATIONS_FILE, device=self.device)

        # If checkpoint path to resume training from is given, load it to the server model
        self.round_offset = 0
        if checkpoint_id is not None:
            self._load_from_checkpoint(checkpoint_id=checkpoint_id)

        self.strategy = self._select_strategy(strategy=strategy)

        # Dict of dicts to index client updates by client_index
        self.client_round_updates = {}

    def get_round_offset(self):
        return self.round_offset

    def _setup_val_dl(self):
        # Create dataset and build dataloaders for both clean an poisoned evaluation
        clean_val_ds = CustomCocoDataset(
            image_folder=ProjectPaths.VALIDATION_IMAGES_DIR,
            annotations_file=ProjectPaths.VALIDATION_ANNOTATIONS_FILE,
            width=BackdoorProperties.RESIZE_WIDTH,
            height=BackdoorProperties.RESIZE_HEIGHT,
            mode="always_clean",
        )

        clean_val_dl = DataLoader(
            clean_val_ds,
            batch_size=TrainingSettings.TRAINING_BATCH_SIZE,
            num_workers=TrainingSettings.NUM_WORKERS,
            collate_fn=FasterRCNNWrapper.get_collate_fn(),
            pin_memory=True,
            shuffle=True,
        )

        poison_val_ds = CustomCocoDataset(
            image_folder=ProjectPaths.VALIDATION_IMAGES_DIR,
            annotations_file=ProjectPaths.VALIDATION_ANNOTATIONS_FILE,
            width=BackdoorProperties.RESIZE_WIDTH,
            height=BackdoorProperties.RESIZE_HEIGHT,
            mode="always_poison",
        )

        poison_val_dl = DataLoader(
            poison_val_ds,
            batch_size=TrainingSettings.TRAINING_BATCH_SIZE,
            num_workers=TrainingSettings.NUM_WORKERS,
            collate_fn=FasterRCNNWrapper.get_collate_fn(),
            pin_memory=True,
            shuffle=True,
        )

        # Num samples need for instantiating server model
        num_classes = clean_val_ds.get_total_classes_count() + 1

        return clean_val_dl, poison_val_dl, num_classes

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
                self.wrapper.set_weights(checkpoint["model_state_dict"])
                self.round_offset = checkpoint["round"]
                print(f"[INFO] Loading model from: {checkpoint_path} to resume training\n[INFO] Completed rounds: {self.round_offset}, starting training of round {self.round_offset + 1}")
            except Exception as e:
                raise Exception(f"[ERROR] Exception occurred while loading checkpoint: {e}")
        else:
            raise Exception(f"Couldn't construct valid checkpoint path. {checkpoint_path} is not a directory")

    def _save_latest_model(self, round: int):
        """
        Save the model_state_dict at stored in the server model wrapper and the training round as a checkpoint. 
        """
        checkpoint_path = self._construct_checkpoint_path()
        if not os.path.isdir(os.path.dirname(checkpoint_path)):
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
            case "fedopt":
                return FedOpt(server_lr=FLSettings.SERVER_LR, model_state_dict=self.wrapper.get_weights())
            case _:
                raise Exception(f"No valid aggregation strategy name is given: {strategy}. Choices are: ['fedavg', 'fedopt'] ")

    def get_global_weights(self):
        """
        Wrapper function to get the server model's weights.
        """
        return self.wrapper.get_weights()

    def receive_client_update(self, client_config: dict, num_samples: int, client_weights_update):
        # Setup logging information
        client_index = client_config["index"]
        is_attacker = client_config["is_attacker"]
        client_type = "Attacker" if is_attacker else "Benign"
        client_update = {
            "cid": client_index,# Same namespace everywhere: client_index from select_clients == cid == ATTACKER_IDS. No flwr node_id remap in the custom path.
            "type": client_type,
# Placeholder values to be determined at aggregation time
            "magnitude": None,
            "l2_norm": None,
            "cos_sim": None,
            # Temporarily necessary for logging of metrics at aggregation
            "weights": client_weights_update,
            "num_samples": num_samples,
        }

        self.client_round_updates[client_index] = client_update
        
        # Have aggregation strategy accumulate the update
        # self.strategy.accumulate(client_weights=client_weights_update, num_samples=num_samples, client_index=client_index)

    def aggregate_and_evaluate(self, round, num_attackers):
        # Aggregate the client updates
        print(f"[ROUND {round}] aggregating", flush=True)
        # Broadcast (pre-aggregation) global; the reference all client deltas are
        # measured against. self.wrapper still holds it until set_weights() below.
        broadcast_global = self.wrapper.get_weights()
        new_global_weights = self.strategy.aggregate(
            client_updates=list(self.client_round_updates.values()),# Expects a list
            global_weights=broadcast_global,# Needed by FedOpt for the (1-η)·w_global + η·w_avg blend; ignored by FedAvg
        )

        # Passive backdoor-detectability harness (B1). OBSERVATIONAL ONLY: reads
        # the raw client updates before they are freed below; never alters the
        # aggregation. Training is identical whether this runs or not.
        if FLSettings.DO_STEALTH_EVAL:
            mk_m = FLSettings.MULTIKRUM_M if FLSettings.MULTIKRUM_M and FLSettings.MULTIKRUM_M > 0 else None
            # Label by who actually participated AND attacked THIS round, read from the
            # per-client flag recorded in receive_client_update — NOT the global
            # ATTACKER_IDS roster. Equivalent under full participation, but keeps the
            # label-aware metrics correct under partial participation / --use-client-schedule:
            # a scheduled-out attacker is simply absent and can never be mislabeled positive.
            round_attacker_ids = {cid for cid, u in self.client_round_updates.items() if u["type"] == "Attacker"}
            per_client_rows, summary_row = stealth_analyze(
                round=round,
                client_weights={cid: u["weights"] for cid, u in self.client_round_updates.items()},
                global_weights=broadcast_global,
                attacker_ids=round_attacker_ids,
                num_samples={cid: u["num_samples"] for cid, u in self.client_round_updates.items()},
                krum_f=FLSettings.KRUM_F,
                multikrum_m=mk_m,
                normclip_c=FLSettings.NORMCLIP_C,
            )
            self.logger.log_stealth_per_client(per_client_rows)
            self.logger.log_stealth_summary(summary_row)

        for key in self.client_round_updates.keys():
            update = self.client_round_updates[key]
            client_weights_update = update["weights"]

            # Compute client update metrics the logger wants to now and store them
            cos_sim = torch_compute_cosine_sim(client_weights_update, new_global_weights)
            l2_norm = torch_compute_l2_distance(client_weights_update, new_global_weights)
            magnitude = torch_compute_magnitude(client_weights_update)

            update['cos_sim'] = cos_sim
            update['l2_norm'] = l2_norm
            update['magnitude'] = magnitude

            # Free the memory of the large weight updates
            del update["weights"]

        # First save then log the round
        self._save_latest_model(round=round)
        self.logger.log_client_updates(server_round=round, update_info=list(self.client_round_updates.values()))

        # Evaluate the model on ASR and mAP
        self.wrapper.set_weights(new_global_weights)

        # metrics, num_samples = self.wrapper.evaluate(dataloader=self.val_dl, device=self.device)
        # metrics, num_samples = self.wrapper.evaluate_map_asr(clean_dl=self.clean_val_dl, poison_dl=self.poison_val_dl, device=self.device)

        clean_results = self.round_evaluator.run_clean_eval(dataloader=self.clean_val_dl)
        poison_results = self.round_evaluator.run_poison_eval(dataloader=self.poison_val_dl)

        eval_results = {
            "round": round,
            "mAP_50": clean_results["AP_50"],
            "mAP_50_95": clean_results["AP_50_95"],
            "ASR": poison_results["ASR"],
            "fail_rate": poison_results["fail_rate"],
            "avg_conf_fail": poison_results["avg_conf_fail"],
        }


        # Write evaluation metrics to the pytorch tensorboards, log the evaluation to a csv & save if the model is evaluated as the best
        # self.logger.log_epoch(round, eval_results)
        self.logger.log_round_evaluation(eval_results=eval_results, num_attackers=num_attackers)# TODO: Pass the correct metrics as arguments
        self.logger.save_if_best_model(epoch=round, val_loss=-eval_results["mAP_50"], weights=new_global_weights)# Negative value as function originally compares for lowest validation loss
        
        # Prepare for another round 
        self.round_evaluator.remove_from_gpu()
        self.client_round_updates.clear()


