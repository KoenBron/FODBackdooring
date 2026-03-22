from abc import abstractmethod, ABC
import torch

class BaseStrategy(ABC):
    """
    Abstract base class for defining Federated Learning Strategies
    """

    @abstractmethod
    def reset_state(self):
        """
        Flush the strategies accumulation for a new round
        """

    @abstractmethod
    def accumulate(self, client_weights, num_samples, client_index):
        """
        Accumulate an incoming client weight and alter the strategies intermittent state
        """
        pass

    @abstractmethod
    def aggregate(self) -> dict:
        """
        Aggregate the intermediate client updates and returns the aggregated restult
        """
        pass

