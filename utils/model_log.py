import os
import torch
import time
import csv

from contextlib import redirect_stdout

from torch.utils.tensorboard import SummaryWriter
from utils.global_settings import ProjectPaths, BackdoorProperties, TrainingSettings
from evaluation import BatchEvaluator

class ModelLogger:
    def __init__(self, custom_path=None, config_dict=None):
        """
        output_path: path object to save the best model and the intermediary reports to
        config_dict: The training configurations used
        """
        if custom_path is None:
            # Create a new model directory to write logs to and save the best model in
            date_format = "%Y%-m-%d-%H-%M-%S"
            date_string = time.strftime(date_format)
            self.output_path = os.path.join(ProjectPaths.OUTPUT_DIR, date_string)

            # Initial loss to compare model performance to
            self.best_val_loss = float('inf')
        else:
            # Load a checkpoint and write logs to that output path
            model_dir_path = os.path.join(ProjectPaths.OUTPUT_DIR, custom_path)
            if os.path.isdir(model_dir_path):
                # Validate the path
                print(f"Resuming training of model stored at path: {model_dir_path}")
                self.output_path = model_dir_path

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
        os.makedirs(self.output_path, exist_ok=True)

        print(f"---------Saving Model Logs to: {self.output_path}")

        # Initiate log writer
        self.writer = SummaryWriter(self.output_path)

        # Write training configuarion params to txt file
        if config_dict:
            with open(os.path.join(self.output_path, "config.txt"), "w") as f, redirect_stdout(f):
                if BackdoorProperties.BACKDOOR_ACTIVE: 
                    BatchEvaluator.print_poison_properties()
                    print("\n----------------------TRAINING CONFIGURATION----------------------")
                for k, v in config_dict.items():
                    f.write(f"{k}: {v}\n")

        # Setup file to log round evaluations to IF FL is used
        if TrainingSettings.FL_USED:
            self.round_eval_path = os.path.join(self.output_path, "round_evaluations.csv")
            # Write the correct headers into the files
            with open(self.round_eval_path, mode="w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["round", "mAP_50", "mAP_50_95", "ASR", "fail_rate"])

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

    def log_client_updates(self, server_round, update_info):
        """
        Writes info about client updates each training round to a standard file in the model folder
        """
        log_file = os.path.join(self.output_path, "client_update_log.txt")

        with open(log_file, 'a') as f:
            f.write(f"\n{'='*95}\n")
            f.write(f" ROUND {server_round} - CLIENT WEIGHT UPDATE ANALYSIS\n")
            f.write(f"{'='*95}\n")
            f.write(f"{'CID':<5} | {'Type':<10} | {'L2 Dist to Global':<20} | {'Cosine Sim to Global':<20} | {'Magnitude':<15}\n")
            f.write(f"{'-'*95}\n")
            
            for info in update_info:
                f.write(f"{info['cid']:<5} | {info['type']:<10} | {info['l2_norm']:<20.4f} | {info['cos_sim']:<20.4f} | {info['magnitude']:<15.4f}\n")
            f.write(f"{'='*95}\n")

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

    def log_round_evaluation(self, eval_results):
        """
        Log the evaluation of the round as a csv.
        eval_results is dict that must contain the following keys:
        {
            round: int,
            mAP_50: float,
            mAP_50_95: float,
            ASR: float,
        }
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
            ])
            


