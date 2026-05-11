from abc import abstractmethod, ABC

class BaseStrategy(ABC):
    """
    Abstract base class for defining Federated Learning Strategies
    """

    @abstractmethod
    def reset_state(self):
        """
        Flush the strategies accumulation for a new round
        """

    # We leave the tracking of client updates to the server for logging
    # @abstractmethod
    # def accumulate(self, client_weights, num_samples):
    #     """
    #     Accumulate an incoming client weight and alter the strategies intermittent state
    #     """
    #     pass

    @abstractmethod
    def aggregate(self, client_updates: list, global_weights=None) -> dict:
        """
        Aggregate the intermediate client updates and returns the aggregated result.

        global_weights: the server's current global state_dict prior to this round's
        aggregation. FedAvg ignores it; FedOpt-style strategies need it to compute the
        blended update (1 − η) · w_global + η · avg(w_client).
        """
        pass

