import torch

from collections import OrderedDict

from FL.strategy import BaseStrategy

class FedAvg(BaseStrategy):
    def __init__(self, model_state_dict: OrderedDict) -> None:
        self.model_state_dict = model_state_dict
        self.reset_state()

    def reset_state(self):
        # Old streaming implementation, right now we store all updates in RAM
        # Zero the summation
        # self.running_sum = {
        #     k: torch.zeros_like(v, dtype=torch.float32, device="cpu")
        #     for k, v in self.model_state_dict
        # }
        self.client_updates = []

        # Reset counters
        self.total_samples = 0
        self.updates_received = 0

    def accumulate(self, client_weights: OrderedDict, num_samples: int, client_index: int):
        self.total_samples += num_samples
        self.updates_received += 1
        # Old streaming implementation, right now we store all updates in RAM
        #
        # for key in self.running_sum.keys():
        #     self.running_sum[key] += client_weights[key] * num_samples
        client_update = {
            "index": client_index,
            "weights": client_weights,
            "num_samples": num_samples,
        }

        self.client_updates.append(client_update)

    def aggregate(self) -> OrderedDict:
        if self.updates_received == 0:
            raise Exception("Server aggregation called without any accumulation having happened")

        new_global_weights = {}
        for update in self.client_updates:
            ratio = update["num_samples"] / self.total_samples
            for k, v in update["weights"].items():
                new_global_weights[k] += v * ratio

        return new_global_weights
