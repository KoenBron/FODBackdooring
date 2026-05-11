import torch

from collections import OrderedDict

from FL.strategy import BaseStrategy

class FedAvg(BaseStrategy):
    def __init__(self, model_state_dict: OrderedDict) -> None:
        self.model_state_dict = model_state_dict
        self.reset_state()

    def reset_state(self):
        # Zero the summation
        self.running_sum = OrderedDict()
        for k, v in self.model_state_dict.items():
            self.running_sum[k] = torch.zeros_like(v, dtype=torch.float32, device="cpu")

        # self.running_sum = {
        #     k: torch.zeros_like(v, dtype=torch.float32, device="cpu")
        #     for k, v in self.model_state_dict.items()
        # }

        # Reset counters
        # self.total_samples = 0
        # self.updates_received = 0

    # def accumulate(self, client_weights: OrderedDict, num_samples: int):
    #     self.total_samples += num_samples
    #     self.updates_received += 1
    #     for key in self.running_sum.keys():
    #         self.running_sum[key] += client_weights[key] * num_samples

    def aggregate(self, client_updates: list, global_weights=None) -> OrderedDict:
        if not client_updates:
            raise Exception("[ERROR] No client updates given to aggregate")
        new_global_weights = {}
        total_samples = sum([update["num_samples"] for update in client_updates])
        
        for update in client_updates:
            weight_ratio = update["num_samples"] / total_samples

            # Iterate throught the tensors in the client update
            for k, v in update["weights"].items():

                if k not in new_global_weights.keys():
                    # Make sure it does't crash if the key hasn't been set yet
                    new_global_weights[k] = torch.zeros_like(v, dtype=torch.float32)

                new_global_weights[k] += v * weight_ratio
                
        return new_global_weights

