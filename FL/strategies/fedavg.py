import torch

from FL.strategy import BaseStrategy

class FedAvg(BaseStrategy):
    def __init__(self, model_state_dict: dict) -> None:
        self.model_state_dict = model_state_dict
        self.reset_state()

    def reset_state(self):
        # Zero the summation
        self.running_sum = {
            k: torch.zeros_like(v, dtype=torch.float32, device="cpu")
            for k, v in self.model_state_dict
        }

        # Reset counters
        self.total_samples = 0
        self.updates_received = 0

    def accumulate(self, client_weights: dict, num_samples):
        self.total_samples += num_samples
        self.updates_received += 1

        for key in self.running_sum.keys():
            self.running_sum[key] += client_weights[key] * num_samples

    def aggregate(self) -> dict:
        if self.updates_received == 0:
            raise Exception("Server aggregation called without any accumulation having happened")

        new_global_weights = {}
        for key in self.running_sum.keys():
            new_global_weights[key] = self.running_sum[key] / self.total_samples

        return new_global_weights
