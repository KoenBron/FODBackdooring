"""AttackProtocol abstract base.

Defines the contract every per-round malicious procedure (PilotA, LaySelFL,
Ours) must satisfy. The contract mirrors ClientManager's attacker contract so
the dispatcher in `client_side_train` can route to either the composable path
or a Protocol without the caller knowing the difference:

    updated_params, num_samples, metrics = protocol.run(server_weights, client_config)

Protocols receive a reference to the hosting ClientManager so they can reach
the wrapper, the per-client subsets, the device, and helpers like
_do_model_replacement without dependencies being hand-passed at every call.

State that persists across rounds (carrying set, attack-round counter, etc.)
lives in FL.attack_state.AttackState — protocols read and update it directly.
"""
from abc import ABC, abstractmethod


class AttackProtocol(ABC):
    """One malicious training round, as a preset recipe.

    Subclasses implement run() and any private helpers. The constructor stores
    the hosting ClientManager so subclasses can access:
        self.manager.wrapper            — the FasterRCNNWrapper
        self.manager.subsets_mal[i]     — attacker i's poisoned partition
        self.manager.subsets_benign[i]  — attacker i's clean partition (for anchors)
        self.manager.device             — current GPU device
        self.manager._do_model_replacement(...)   — composes with Bagdasaryan scaling

    Subclasses MAY override `name` for clearer logging — it's printed by the
    dispatcher when an attacker enters the protocol's run().
    """

    name: str = "abstract"

    def __init__(self, manager):
        self.manager = manager

    @abstractmethod
    def run(self, server_weights, client_config):
        """Execute one malicious round.

        Args:
            server_weights: dict[name -> CPU Tensor], the broadcast theta_t.
            client_config:  dict with at least the keys
                              "index"          (int, attacker id)
                              "is_attacker"    (True; only called for attackers)
                              "n_mal_clients"  (int, used by model replacement)

        Returns:
            updated_params: dict[name -> Tensor], the submission for aggregation.
            num_samples:    int, number of training samples this client saw
                            (used for FedAvg weighting at the server).
            metrics:        dict, per-round logs to forward to the server logger.
                            At minimum: {"partition_id": client_config["index"]}.
        """
        raise NotImplementedError
