import torch

from collections import OrderedDict

from FL.strategy import BaseStrategy

class FedOpt(BaseStrategy):
    """
    FedOpt with SGD as the server optimizer (Reddi et al., 2020).

    Treats the sample-weighted average of client weights as a pseudo-gradient step
    and applies a server-side learning rate η on top of the prior global model:

        w_new = (1 − η) · w_global + η · Σ_k (n_k / N) · w_k

    Equivalent (by the standard derivation) to applying η · avg(Δ_k) to the global,
    where Δ_k = w_k − w_global. η = 1.0 reduces exactly to FedAvg.
    """

    def __init__(self, server_lr: float, model_state_dict: OrderedDict) -> None:
        self.server_lr = server_lr
        self.model_state_dict = model_state_dict
        self.reset_state()

    def reset_state(self):
        self.running_sum = OrderedDict()
        for k, v in self.model_state_dict.items():
            self.running_sum[k] = torch.zeros_like(v, dtype=torch.float32, device="cpu")

    def aggregate(self, client_updates: list, global_weights=None) -> OrderedDict:
        if not client_updates:
            raise Exception("[ERROR] No client updates given to aggregate")
        if global_weights is None:
            raise Exception("[ERROR] FedOpt requires the current global weights to compute the blended update")

        eta = self.server_lr
        total_samples = sum(update["num_samples"] for update in client_updates)

        # Step 1: sample-weighted average of client weights (same shape as FedAvg)
        avg_weights = {}
        for update in client_updates:
            weight_ratio = update["num_samples"] / total_samples
            for k, v in update["weights"].items():
                if k not in avg_weights:
                    avg_weights[k] = torch.zeros_like(v, dtype=torch.float32)
                avg_weights[k] += v * weight_ratio

        # Step 2: blend with the prior global using the server learning rate
        new_global_weights = {}
        for k, v_avg in avg_weights.items():
            v_global = global_weights[k].to(v_avg.dtype)
            new_global_weights[k] = (1.0 - eta) * v_global + eta * v_avg

        return new_global_weights
