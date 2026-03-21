from FL.strategies import FedAvg
from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.global_settings import ProjectPaths, BackdoorProperties, TrainingSettings
from utils.dataset import CustomCocoDataset

from torch.utils.data import DataLoader

class Server:
    def __init__(self, strategy: str):
        # Setup model the server uses
        self.val_dl, num_classes= self._setup_val_dl()
        self.wrapper = FasterRCNNWrapper(num_classes=num_classes)
        self.strategy = self._select_strategy(strategy=strategy)

    def _setup_val_dl(self):
        # Create dataset and build dataloader 
        val_ds = CustomCocoDataset(
            image_folder=ProjectPaths.VALIDATION_IMAGES_DIR,
            annotations_file=ProjectPaths.VALIDATION_ANNOTATIONS_FILE,
            width=BackdoorProperties.RESIZE_WIDTH,
            height=BackdoorProperties.RESIZE_HEIGHT,
            mode="always_clean",
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=TrainingSettings.TRAINING_BATCH_SIZE,
            num_workers=TrainingSettings.NUM_WORKERS,
            collate_fn=FasterRCNNWrapper.get_collate_fn(),
            pin_memory=True,
            shuffle=True,
        )

        # Num samples need for instantiating server model
        num_classes = val_ds.get_total_classes_count() + 1

        return val_dl, num_classes

    def _select_strategy(self, strategy: str):
        """
        Selects the strategy object to use based on the string 'strategy'
        """
        match strategy.lower():
            case "fedavg": 
                return FedAvg(model_state_dict=self.wrapper.get_weights())
            case _:
                raise Exception(f"No valid aggregation strategy name is given: {strategy}. Choices are: ['fedavg',] ")

    def receive_client_update(self):
        pass

    def aggregate_and_evaluate(self):
        pass

    
