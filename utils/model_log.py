import os
import torch
import time

from torch.utils.tensorboard import SummaryWriter
from utils.global_settings import ProjectPaths

class ModelLogger:
    def __init__(self, custom_path=None, config_dict=None):
        """
        output_path: path object to save the best model and the intermediary reports to
        config_dict: The training configurations used
        """
        # Create unique direoctry name to distinguish models
        if custom_path is None:
            date_format = "%d-%m-%Y-%H-%M-%S"
            date_string = time.strftime(date_format)
            self.output_path = os.path.join(ProjectPaths.OUTPUT_DIR, date_string)
        else:
            self.output_path = custom_path
        os.makedirs(self.output_path, exist_ok=True)

        print(f"---------Saving Model Logs to: {self.output_path}")

        # Initiate log writer
        self.writer = SummaryWriter(self.output_path)

        # Write training configuarion params to txt file
        if config_dict:
            with open(os.path.join(self.output_path, "config.txt"), "w") as f:
                for k, v in config_dict.items():
                    f.write(f"{k}: {v}\n")
        # Initial loss to compare model performance to
        self.best_val_loss = float('inf')

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

            # Serialize model as a checkpoint (i.e. a dict)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": weights,
                "loss": self.best_val_loss,
            }

            # Also save optimizer state_dict in case of resuming training
            if optimizer:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()

            # torch.save() is a serializer function
            torch.save(checkpoint, model_path)
            print(f"-----------New best model saved succesfully at {model_path}-----------")

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
