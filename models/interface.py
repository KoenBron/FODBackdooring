from abc import ABC, abstractmethod

class Interface(ABC):
    """
    Abstract Base Class that defines how to interact with different models
    """

    @abstractmethod
    def get_weights(self):
        """
        Returns the model weights as a state_dict stored on the CPU 
        Return: Dict[str, Tensor]
        """
        pass

    @abstractmethod
    def get_fl_parameters(self):
        """
        Returns the parameters as a list of numpy arrays as flower expects
        Return: List[NumpyArray]
        """
        pass

    @abstractmethod
    def set_fl_parameters(self, parameters):
        """
        Sets the parameters as flower sends them to the client and loads them into the model
        Return: List[NumpyArray]
        """
        pass

    @abstractmethod
    def set_weights(self, weights):
        """
        Loads the weights into the model
        Args:
            weights: Dict[str, Tensor]
        """
        pass

    @staticmethod
    @abstractmethod
    def get_collate_fn():
        """
        Returns the collate function used by the Dataloader to iterate through the datasets
        Faster R-CNN uses tuples, YOLO uses stacked tensors.
        """
        pass

    @abstractmethod
    def train_step(self, dataloader, epochs, device, optimizer_config, optimizer):
        """
        Executes local training
        Args:
            dataloader: Local training dataset
            epochs: Number of epochs to use per training
            device: 'cuda' or 'cpu'
            optimizer_config: Dict with learning/optimization configurations
            optimzer: Optional torch optimizer to reuse
        
        Return:
            (weights, num_samples, metrics)
        """
        pass

    @abstractmethod
    def evaluate(self, dataloader, device):
        """
        Performs an evaluation and returns the loss
        Args:
            dataloader: Local validation dataset
            device: 'cuda' or 'cpu'
        """
        pass

