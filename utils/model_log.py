import os
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.global_settings import ProjectPaths

class ModelLogger:
    def __init__(self, output_folder_name, config_dict):
        """
        output_path: path object to save the best model and the intermediary reports to
        config_dict: The training configurations used
        """
        self.output_path = os.path.join(ProjectPaths.OUTPUT_DIR, output_folder_name)
        os.makedirs(self.output_path, exist_ok=True)

        self.writer = SummaryWriter(self.output_path)

        if config_dict:
            # Safe configuration as a txt file
            with open(os.path.join(self.output_path, "config.txt"), "w") as f:
                for k, v in config_dict.items():
                    f.write(f"{k}: {v}\n")
        # Initial loss to compare model performance to
        self.best_val_loss = float('inf')
