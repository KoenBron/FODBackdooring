import flwr as fl

from models.fasterrcnn.wrapper import FasterRCNNWrapper

class Client(fl.client.NumPyClient):
    def __init__(self, wrapper:FasterRCNNWrapper, train_dl, val_dl, device, is_attacker, optimizer_config):
        self.wrapper = wrapper
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        self.is_attacker = is_attacker
        self.optimizer_config = optimizer_config
        self.optimizer = None
        
    def get_parameters(self, config):
        # Delegate to wrapper to make client implementation model agnsostic
        return self.wrapper.get_fl_parameters()

    def set_parameters(self, parameters):
        # Delegate to wrapper to make client implementation model agnsostic
        self.wrapper.set_fl_parameters(parameters)


    def fit(self, parameters, config):
        # Load server weights
        self.set_parameters(parameters)
        
        # Have the wrapper handle the training process
        weights, num_samples, metrics, self.optimizer = self.wrapper.train_step(
            dataloader=self.train_dl,
            epochs= config.get("local_epochs", 1),# Default to one epoch
            device=self.device,
            optimizer_config=self.optimizer_config,
            optimizer=None,# Don't keep the optimizer
        )
        
        return self.get_parameters(config={}), num_samples, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics, num_samples = self.wrapper.evaluate(
            dataloader=self.val_dl,
            device = self.device,
        )

        return float(metrics["val_loss"]), num_samples, metrics

