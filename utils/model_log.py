import os
import torch
import time
import csv
import shutil

from contextlib import redirect_stdout

from torch.utils.tensorboard import SummaryWriter
from utils.global_settings import ProjectPaths, BackdoorProperties, TrainingSettings, FLSettings
from evaluation import BatchEvaluator
from FL.attack_state import AttackState
from FL.attack_protocols.pilot_a import PilotALogger
from FL.attack_protocols.pilot_b import PilotBLogger
from FL.attack_protocols.partial import PartialLogger

class ModelLogger:
    def __init__(self, custom_path=None, args=None):
        """
        custom_path: The path to the directory from which logging should be resumed
        args:        Parsed CLI args; their raw dump is appended under the structured
                     config summary in config.txt (the summary itself is built from the
                     global settings via parse_args.render_config_sections).
        """
        if custom_path is None:
            # Create a new model directory to write logs to and save the best model in
            date_format = "%Y-%m-%d-%H-%M-%S"
            date_string = time.strftime(date_format)
            self.output_path = os.path.join(ProjectPaths.OUTPUT_DIR, date_string)
            os.makedirs(self.output_path, exist_ok=True)

            # Initial loss to compare model performance to
            self.best_val_loss = float('inf')

            # Setup file to log round evaluations to IF FL is used
            if TrainingSettings.FL_USED:

                # Create csv file for the round evaluations
                self.round_eval_path = os.path.join(self.output_path, "round_evaluations.csv")
                # Write the correct headers into the files
                if not os.path.isfile(self.round_eval_path):
                    with open(self.round_eval_path, mode="w", newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["round", "mAP_50", "mAP_50_95", "ASR", "fail_rate", "num_attackers"])
                
                # Passive detectability harness (B1) output files. Both headers
                # are written lazily from the first row's keys (see the log_*
                # methods): the columns depend on the resolved granularity AND on
                # which metric blocks are enabled, neither of which the logger
                # knows at construction.
                self.stealth_per_client_path = os.path.join(self.output_path, "stealth_per_client.csv")
                self.stealth_summary_path = os.path.join(self.output_path, "stealth_summary.csv")

                # Observational norm-clip report (top-level, lazy header — its columns
                # depend on the --normclip-report-x sweep, unknown until the first row).
                self.normclip_report_path = os.path.join(self.output_path, "normclip_report.csv")

                # FoolsGold per-round aggregation weights (top-level). Written only when
                # --aggregation-strategy foolsgold is active.
                self.foolsgold_weights_path = os.path.join(self.output_path, "foolsgold_weights.csv")
        else:
            # Load a checkpoint and write logs to that output path
            model_dir_path = os.path.join(ProjectPaths.OUTPUT_DIR, custom_path)
            if os.path.isdir(model_dir_path):
                # Validate the path
                print(f"Resuming training of model stored at path: {model_dir_path}")

                self.output_path = model_dir_path

                # CSV file headers are set in the constructor, so paths must be set even for resuming logging
                self.round_eval_path = os.path.join(self.output_path, "round_evaluations.csv")
                self.stealth_per_client_path = os.path.join(self.output_path, "stealth_per_client.csv")
                self.stealth_summary_path = os.path.join(self.output_path, "stealth_summary.csv")
                self.normclip_report_path = os.path.join(self.output_path, "normclip_report.csv")
                self.foolsgold_weights_path = os.path.join(self.output_path, "foolsgold_weights.csv")

                # Validate if checkpoint of best recorded model is there
                best_model_path = os.path.join(model_dir_path, "best_model.pth")
                try:
                    checkpoint = torch.load(best_model_path, weights_only=False)
                    self.best_val_loss = checkpoint["loss"]
                    print(f"[INFO] Restored from best recorded checkpoint {best_model_path}\nRecorded loss: {self.best_val_loss}")
                except Exception as e:
                    raise Exception(f"Exception occured while loading checkpoint: {best_model_path}.\nException: {e}")

            else:
                raise Exception(f"Path {custom_path} of model to resume from is not a directory")
        
        print(f"---------Saving Model Logs to: {self.output_path}")

        # Attack persistence + per-protocol sub-loggers. ModelLogger owns the
        # paths and orchestrates load/save; AttackState stays a singleton so
        # protocols can use accessors without passing the logger through every
        # call (see FL/attack_state.py). load() is a silent no-op on fresh
        # runs and restores the previous shift's carrying sets on resume.
        attack_logs_dir = os.path.join(self.output_path, "attack_logs")
        os.makedirs(attack_logs_dir, exist_ok=True)
        self._attack_state_path = os.path.join(attack_logs_dir, "attack_state.json")
        self.attack_state = AttackState
        self.attack_state.load(self._attack_state_path)
        # Per-protocol sub-loggers. Each owns one specific output schema and
        # the subdirectory it writes into. Eager construction keeps the
        # protocol code straight-line (no None checks in dispatchers).
        self.pilot_a = PilotALogger(attack_logs_dir)
        self.pilot_b = PilotBLogger(attack_logs_dir)
        self.partial = PartialLogger(attack_logs_dir)

        # Write the script containing the training settings there
        self.log_training_script()

        # Initiate log writer
        self.writer = SummaryWriter(self.output_path)

        # Write the run configuration to config.txt: a neatly grouped, relevant-
        # settings summary at the TOP (single source shared with the stdout summary),
        # then the raw commandline arguments beneath it (as before).
        from utils.parse_args import render_config_sections  # local import: avoids any import cycle
        with open(os.path.join(self.output_path, "config.txt"), "a") as f:
            f.write("==================== RUN CONFIGURATION ====================\n")
            if BackdoorProperties.BACKDOOR_ACTIVE:
                with redirect_stdout(f):
                    BatchEvaluator.print_poison_properties()
            render_config_sections(f.write)

            if args is not None:
                f.write("\n==================== RAW ARGUMENTS ====================\n")
                for k, v in vars(args).items():
                    f.write(f"{k}: {v}\n")
            f.write("\n==================== END OF CONFIGURATION ====================\n")
                    
    def log_training_script(self):
        """
        Logs the contents of the entire script used to run the training
        """
        script_output_path = os.path.join(self.output_path, "training_script.txt")
        script_input_path = os.path.join(ProjectPaths.ROOT_DIR, "run_fl_train.sh")

        # Copy the lines
        shutil.copyfile(script_input_path, script_output_path)

    def log_epoch(self, epoch, train_metrics=None, val_metrics=None):
        """
        Logs the epochs losses in the terminal and the writer
        """
        # Log total training loss and component losses if present
        if train_metrics is not None:
            # Go through all component losses
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"{k}/Loss", v, epoch)

        # Log total validation logs if present
        if val_metrics is not None:
            val_loss = val_metrics.get("val_loss")
            self.writer.add_scalar("Validation/Loss", val_loss, epoch)

    def save_if_best_model(self, epoch, val_loss, weights, optimizer=None):
        """
        Saves the best model if the val_loss is lower than the lowest recorded val_loss by this
        logger.
        Serializes model as a checkpoint, can be deserialized as python dict containing:
        - model_state_dict (serialized model params)
        - epoch (epoch from which it left off)
        - loss (the recorded loss)
        - optimizer_state_dict (serialized optimizer)
        """
        if val_loss < self.best_val_loss:
            print(f"------New best model to save at epoch {epoch}-------------")
            self.best_val_loss = val_loss
            model_path = os.path.join(self.output_path, "best_model.pth")

            # Create checkpoint as a dictionary
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": weights,
                "loss": self.best_val_loss,
            }

            if optimizer:
                checkpoint["optimizer"] = optimizer

            torch.save(checkpoint, model_path)

            print(f"-----------New best model saved succesfully at {model_path}-----------")

    def log_stealth_per_client(self, rows):
        """
        Append per-(round, client) detectability rows from the passive harness.
        rows: list of dicts keyed by stealth_defenses.per_client_columns(...).

        The header is written lazily from the first row's keys, because the
        dl2_<bucket> columns depend on the granularity resolved at the server
        (per-group vs per-tensor) and are not known at logger construction. On
        resume the file already has a header, so it is not rewritten. The
        granularity is deterministic per run, so the keys stay consistent.
        """
        if not rows:
            return
        write_header = (not os.path.isfile(self.stealth_per_client_path)
                        or os.path.getsize(self.stealth_per_client_path) == 0)
        with open(self.stealth_per_client_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def log_stealth_summary(self, summary_row):
        """
        Append one per-round detectability summary row (TPR/FPR + separability).
        summary_row: dict from stealth_defenses.analyze; its columns depend on
        the enabled metric blocks. The header is written lazily from the row's
        keys (same rationale as log_stealth_per_client).
        """
        if not summary_row:
            return
        write_header = (not os.path.isfile(self.stealth_summary_path)
                        or os.path.getsize(self.stealth_summary_path) == 0)
        with open(self.stealth_summary_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(summary_row)

    def log_normclip_report(self, rows):
        """Append the observational norm-clip rows (one per malicious client this round).
        rows: list of dicts from FL.normclip_report.build_report_rows. Header written
        lazily from the first row's keys — the exceeds_*@x columns depend on the
        --normclip-report-x sweep, which the logger doesn't know at construction. The
        sweep is fixed per run, so the keys stay consistent across rounds; on resume the
        file already has a header and is not rewritten."""
        if not rows:
            return
        write_header = (not os.path.isfile(self.normclip_report_path)
                        or os.path.getsize(self.normclip_report_path) == 0)
        with open(self.normclip_report_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def write_foolsgold_weights(self, rows):
        """Append this round's FoolsGold aggregation weights, one row per participant.

        Same lazy-header append as write_normclip_report, so a resume adds to the existing
        file rather than rewriting it. The Attacker/Benign label in these rows is
        OBSERVATIONAL: the strategy itself never sees it (custom_server strips the label
        before calling aggregate), and the column exists only so the weights can be scored
        against ground truth offline.
        """
        if not rows:
            return
        write_header = (not os.path.isfile(self.foolsgold_weights_path)
                        or os.path.getsize(self.foolsgold_weights_path) == 0)
        with open(self.foolsgold_weights_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def save_attack_state(self):
        """Persist the current AttackState snapshot to this run's attack_logs/.
        Called after every round-aggregation in the FL loop. Atomic write
        inside AttackState.save() means a kill mid-write leaves the previous
        snapshot intact rather than half-written garbage — important on the
        4-hour cluster shift boundary."""
        self.attack_state.save(self._attack_state_path)

    def reset_loss(self):
        """
        Resets the lowest recorded val_loss to infinity
        """
        self.best_val_loss = float('inf')
        print("---------Reset the best recorded validation loss----------")

    def close(self):
        """
        Closes the file stream that the writer uses to write to the tensorboard
        """
        self.writer.close()
        print("---------Logger Closed Without a Problem----------")

    def log_round_evaluation(self, eval_results, num_attackers):
        """
        Log the evaluation of the round as a csv.
        eval_results is dict that must contain the following keys:
        {
            round: int,
            mAP_50: float,
            mAP_50_95: float,
            ASR: float,
        }
        num_attackers: the number of malicious clients that participated in the round
        """
        # Make sure 'eval_results' is a dict with the right parameters
        if not isinstance(eval_results, dict):
            raise Exception('[ERROR] Error while logging round evaluation results')

        for key in ("round", "mAP_50", "mAP_50_95", "ASR"):
            if key not in eval_results.keys():
                raise Exception(f"[ERROR] '{key}' missing in 'eval_results' when logging round evaluation metrics") 

        # Write evalution as csv entry to metrics file
        with open(self.round_eval_path, mode="a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                eval_results["round"],
                eval_results["mAP_50"],
                eval_results["mAP_50_95"],
                eval_results["ASR"],
                1.0 - eval_results["ASR"],
                num_attackers,
            ])
            
