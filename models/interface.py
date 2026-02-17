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
    def set_weights(self, weights):
        """
        Loads the weights into the model
        Args:
            weights: Dict[str, Tensor]
        """
        pass

    @abstractmethod
    def get_collate_fn(self):
        """
        Returns the collate function used by the Dataloader to iterate through the datasets
        Faster R-CNN uses tuples, YOLO uses stacked tensors.
        """
        pass

    @abstractmethod
    def train_step(self, dataloader, epochs, device, optimizer_config):
        """
        Executes local training
        Args:
            dataloader: Local dataset
            epochs: Number of epochs to use per training
            device: 'cuda' or 'cpu'
            optimizer_config: Dict with learning/optimization configurations
        
        Return:
            (weights, num_samples, metrics)
        """
        pass

