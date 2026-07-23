from FL.strategies.fedavg import FedAvg
from FL.strategies.fedopt import FedOpt
from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.global_settings import ProjectPaths, BackdoorProperties, TrainingSettings, FLSettings
from utils.dataset import CustomCocoDataset
from utils.round_evaluator import RoundEvaluator
from FL.stealth_defenses import analyze as stealth_analyze, DEFENSE_NAMES as STEALTH_DEFENSE_NAMES
from FL.groups import build_bucket_map, build_eligibility

from torch.utils.data import DataLoader

import torch
import os
import math


# Attack protocols whose submission is a benign-looking endpoint by design
# (PilotA submits its ben1 trajectory's final weights for observational
# localization). Running the passive stealth harness on these would label
# the submission "Attacker" but every defense's AUC would collapse to
# chance — meaningless rows for downstream analysis, and ~O(n^2 d) CPU work
# per round we'd rather not pay. Grow this set as more observational
# protocols are added.
_OBSERVATIONAL_PROTOCOLS = {"pilot_a", "pilot_b"}

# Default stealth-decomposition granularity per attack protocol — mirrors how the
# pilots fix their own granularity (pilot_a = per-group, pilot_b = per-tensor) so
# the dl2_* energy profile speaks the same partition as the protocol's
# localization signal. FLSettings.STEALTH_GRANULARITY != "auto" overrides this;
# unlisted protocols fall back to "group" (today's behavior).
_PROTOCOL_STEALTH_GRANULARITY = {
    "pilot_a": "group",
    "pilot_b": "tensor",
    "laysel_fl": "tensor",
    "ours": "group",
    "composable": "group",
}

class Server:
    def __init__(self, strategy: str, logger, checkpoint_id=None):
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup model the server uses
        self.clean_val_dl, self.poison_val_dl, num_classes= self._setup_val_dl()
        self.wrapper = FasterRCNNWrapper(num_classes=num_classes)
        self.round_evaluator = RoundEvaluator(wrapper=self.wrapper, annotations_file=ProjectPaths.VALIDATION_ANNOTATIONS_FILE, device=self.device)

        # name_to_bucket for the stealth harness's dl2_* energy decomposition,
        # built once (the model never changes topology). None when the harness
        # is off so we pay nothing. Eligibility (trainable, non-BN params + the
        # buckets that hold them) is derived from the same model in the same step.
        self._stealth_bucket_map = None
        self._stealth_eligible_names = None
        self._stealth_eligible_buckets = None
        if FLSettings.DO_STEALTH_EVAL:
            self._stealth_bucket_map = self._build_stealth_bucket_map()
            self._stealth_eligible_names, self._stealth_eligible_buckets = build_eligibility(
                self.wrapper.model, self._stealth_bucket_map)
        # Fail fast on an unknown --stealth-disable block name. Validated here (not
        # in parse_args) so parse_args needn't import the torch-backed registry.
        _unknown = set(FLSettings.STEALTH_DISABLED) - set(STEALTH_DEFENSE_NAMES)
        if _unknown:
            raise ValueError(f"Unknown stealth block(s) in STEALTH_DISABLED: {sorted(_unknown)}. "
                             f"Known: {list(STEALTH_DEFENSE_NAMES)}")

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

    def _resolve_stealth_granularity(self):
        """"group" or "tensor" for the stealth energy decomposition.

        FLSettings.STEALTH_GRANULARITY wins when it is not "auto"; otherwise the
        per-protocol default is used (unlisted protocols -> "group").
        """
        forced = FLSettings.STEALTH_GRANULARITY
        if forced and forced != "auto":
            return forced
        return _PROTOCOL_STEALTH_GRANULARITY.get(FLSettings.ATTACK_PROTOCOL, "group")

    def _build_stealth_bucket_map(self):
        """name_to_bucket for the resolved granularity, from FL.groups (the SSOT).

        Covers named_parameters() only; the harness routes BN buffers / non-params
        to 'other' via .get — same bucket map the pilots use.
        """
        return build_bucket_map(self.wrapper.model, self._resolve_stealth_granularity())

    def _should_run_stealth(self, round_attacker_ids):
        """Decide whether to run the passive stealth harness this round.

        Three gates, all must pass:
          - DO_STEALTH_EVAL is on (the kill switch).
          - At least one attacker participated this round — otherwise every
            label-aware metric is NaN and the row is dead weight. Overridden by
            STEALTH_ON_BENIGN_ROUNDS, which keeps benign-only rounds so a
            no-attacker baseline can gather the benign null distribution.
          - The active protocol isn't observational. PilotA submits a benign
            trajectory's endpoint by design; grading defenses against it
            measures nothing about the attack, just the noise floor.
        """
        if not FLSettings.DO_STEALTH_EVAL:
            return False
        if not round_attacker_ids and not FLSettings.STEALTH_ON_BENIGN_ROUNDS:
            return False
        if FLSettings.ATTACK_PROTOCOL in _OBSERVATIONAL_PROTOCOLS:
            return False
        return True

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
            "type": client_type,# "Attacker"/"Benign" — consumed by the stealth harness's label-aware metrics.
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
        #
        # Gated to skip rounds where running the harness produces no meaningful
        # signal — both to avoid polluting the CSV with chance-level rows and
        # to save the ~O(n^2 d) CPU pass when nothing graded would be real.
        # Label by who actually attacked THIS round, read from the per-client
        # flag recorded in receive_client_update — NOT the global ATTACKER_IDS
        # roster. Equivalent under full participation; keeps the label-aware
        # metrics correct under partial participation / --use-client-schedule.
        round_attacker_ids = {cid for cid, u in self.client_round_updates.items()
                                  if u["type"] == "Attacker"}
        if self._should_run_stealth(round_attacker_ids):
            mk_m = FLSettings.MULTIKRUM_M if FLSettings.MULTIKRUM_M and FLSettings.MULTIKRUM_M > 0 else None
            per_client_rows, summary_row = stealth_analyze(
                round=round,
                client_weights={cid: u["weights"] for cid, u in self.client_round_updates.items()},
                global_weights=broadcast_global,
                attacker_ids=round_attacker_ids,
                num_samples={cid: u["num_samples"] for cid, u in self.client_round_updates.items()},
                krum_f=FLSettings.KRUM_F,
                multikrum_m=mk_m,
                normclip_c=FLSettings.NORMCLIP_C,
                name_to_bucket=self._stealth_bucket_map,
                eligible_names=self._stealth_eligible_names,
                eligible_buckets=self._stealth_eligible_buckets,
                enabled=[n for n in STEALTH_DEFENSE_NAMES if n not in set(FLSettings.STEALTH_DISABLED)],
            )
            self.logger.log_stealth_per_client(per_client_rows)
            self.logger.log_stealth_summary(summary_row)

        # Evaluate the model on ASR and mAP. Two things can turn the eval into a
        # NaN-row this round:
        #   - SKIP_ROUND_EVAL: eval disabled entirely (e.g. the PilotA pilot, whose
        #     measurements live in attack_logs/, not round_evaluations.csv).
        #   - EVAL_EVERY > 1: run the full eval only on-cadence (round % N == 0) to
        #     cut the per-round val-set passes.
        # Off-eval rounds still write a CSV row with NaNs so the file keeps one row
        # per round for downstream tooling, and skip save_if_best_model (best-model
        # selection needs a real mAP_50).
        self.wrapper.set_weights(new_global_weights)

        eval_this_round = (not FLSettings.SKIP_ROUND_EVAL
                           and round % FLSettings.EVAL_EVERY == 0)

        if not eval_this_round:
            eval_results = {
                "round": round,
                "mAP_50": math.nan,
                "mAP_50_95": math.nan,
                "ASR": math.nan,
                "fail_rate": math.nan,
                "avg_conf_fail": math.nan,
            }
            self.logger.log_round_evaluation(eval_results=eval_results, num_attackers=num_attackers)
        else:
            # The mAP and ASR halves are independently skippable
            # (--skip-map-eval / --skip-asr-eval) to cut compute; a skipped half
            # is logged as NaN so the CSV keeps its fixed column shape.
            if not FLSettings.SKIP_MAP_EVAL:
                clean_results = self.round_evaluator.run_clean_eval(dataloader=self.clean_val_dl)
                map_50, map_50_95 = clean_results["AP_50"], clean_results["AP_50_95"]
            else:
                map_50 = map_50_95 = math.nan

            if not FLSettings.SKIP_ASR_EVAL:
                poison_results = self.round_evaluator.run_poison_eval(dataloader=self.poison_val_dl)
                asr, fail_rate, avg_conf_fail = poison_results["ASR"], poison_results["fail_rate"], poison_results["avg_conf_fail"]
            else:
                asr = fail_rate = avg_conf_fail = math.nan

            eval_results = {
                "round": round,
                "mAP_50": map_50,
                "mAP_50_95": map_50_95,
                "ASR": asr,
                "fail_rate": fail_rate,
                "avg_conf_fail": avg_conf_fail,
            }

            # Write evaluation metrics to the pytorch tensorboards, log the evaluation to a csv & save if the model is evaluated as the best
            # self.logger.log_epoch(round, eval_results)
            self.logger.log_round_evaluation(eval_results=eval_results, num_attackers=num_attackers)# TODO: Pass the correct metrics as arguments
            # Best-model selection requires a real mAP_50; skip it when mAP wasn't computed.
            if not FLSettings.SKIP_MAP_EVAL:
                self.logger.save_if_best_model(epoch=round, val_loss=-map_50, weights=new_global_weights)# Negative value as function originally compares for lowest validation loss

        # Persist the round's checkpoint LAST, after the weights are set and the
        # round evaluation is logged. Two guarantees follow from this ordering:
        #   1. The checkpoint holds the freshly-aggregated round-N weights (set
        #      above), not the pre-aggregation broadcast model.
        #   2. It is the round's final durable write, so "checkpoint says round N"
        #      implies "round_evaluations.csv has row N". A mid-evaluation kill
        #      (e.g. the cluster's 4h wall-clock) leaves both at N-1, so round N
        #      re-runs cleanly on resume instead of being skipped in the CSV.
        # Per-client update metrics are no longer logged here — the passive
        # stealth harness above (stealth_per_client / stealth_summary) supersedes
        # the old client_update_log.
        self._save_latest_model(round=round)

        # Prepare for another round
        self.round_evaluator.remove_from_gpu()
        self.client_round_updates.clear()


