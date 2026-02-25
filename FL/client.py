import flwr as fl
import torch
from torch.utils.data import DataLoader

from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.dataset import CustomCocoDataset
from utils.global_settings import ProjectPaths, BackdoorProperties, FLSettings, TrainingSettings
from utils.partitioning import partition_dataset, get_shuffled_indices

CACHE_BUILD = False
CACHE_CLEAN = {}
CACHE_POISON = {}
DEVICE = None
SUBSETS_CLEAN = []
SUBSETS_POISON = []
NUM_CLASSES = 0

def build_partitions_once():
    """
    Creations dataset partitions once for both the poisoned and clean dataset. This way both benign and malicious clients hcan pull from the correct dataset class, instead of having to share and keep track of the state of CustomCocoDataset.mode
    """
    global CACHE_BUILD, DEVICE, SUBSETS_CLEAN, SUBSETS_POISON, NUM_CLASSES
    if CACHE_BUILD:
            return

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    poison_ds = CustomCocoDataset(
        image_folder=ProjectPaths.TRAIN_IMAGES_DIR,
        annotations_file=ProjectPaths.TRAIN_ANNOTATIONS_FILE,
        width=BackdoorProperties.RESIZE_WIDTH,
        height=BackdoorProperties.RESIZE_HEIGHT,
        mode="training",
    )

    clean_ds = CustomCocoDataset(
        image_folder=ProjectPaths.TRAIN_IMAGES_DIR,
        annotations_file=ProjectPaths.TRAIN_ANNOTATIONS_FILE,
        width=BackdoorProperties.RESIZE_WIDTH,
        height=BackdoorProperties.RESIZE_HEIGHT,
        mode="always_clean",
    )
    
    NUM_CLASSES = clean_ds.get_total_classes_count() + 1
    
    # Shuffle dataset indices and create 2 subsets, this way both are shuffled the same way and clients get the same data whether malicious or not
    shuffled_indices = get_shuffled_indices(dataset_size=clean_ds.__len__())
    SUBSETS_CLEAN = partition_dataset(clean_ds, FLSettings.TOTAL_CLIENTS, indices=shuffled_indices)
    SUBSETS_POISON = partition_dataset(poison_ds, FLSettings.TOTAL_CLIENTS, indices=shuffled_indices)

    CACHE_BUILD = True

def get_or_build_dl(partition_id, is_attacker):
    # global CACHE_CLEAN, CACHE_POISON
    if is_attacker:
        subset = SUBSETS_POISON[partition_id]
    else:
        subset = SUBSETS_CLEAN[partition_id]

    return DataLoader(
        subset,
        batch_size=TrainingSettings.TRAINING_BATCH_SIZE,
        num_workers=TrainingSettings.NUM_WORKERS,
        collate_fn=FasterRCNNWrapper.get_collate_fn(),
        pin_memory=True,
        shuffle=True,
    )



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
        _, num_samples, metrics, self.optimizer = self.wrapper.train_step(
            dataloader=self.train_dl,
            epochs= config.get("local_epochs", 1),# Default to one epoch
            device=self.device,
            optimizer_config=self.optimizer_config,
            optimizer=None,# Don't keep the optimizer
        )
        
        return self.get_parameters(config={}), num_samples, metrics

    def evaluate(self, parameters, config):
        if self.val_dl is None:
            return 0, 0, {}
        self.set_parameters(parameters)
        metrics, num_samples = self.wrapper.evaluate(
            dataloader=self.val_dl,
            device = self.device,
        )

        return float(metrics["val_loss"]), num_samples, metrics

def client_fn(context):
    """
    Client builder, automatically detects if it is an attacker based on global_settings.FLSettings.ATTACKER_IDS
    """
    build_partitions_once()

    # Craft the correct dataloader for this client based on client node_id
    partition_id = int(context.node_id) - 1
    is_attacker = (partition_id in FLSettings.ATTACKER_IDS)
    train_dl = get_or_build_dl(partition_id=partition_id, is_attacker=is_attacker)
    num_classes = NUM_CLASSES

    client_wrapper = FasterRCNNWrapper(num_classes=num_classes)

    return Client(
        wrapper=client_wrapper,
        train_dl=train_dl,
        val_dl=None,
        device=DEVICE,
        is_attacker=is_attacker,
        optimizer_config=None
    ).to_client()

# Create the client app that will be imported when running the simulation
client_app = fl.client.ClientApp(client_fn=client_fn)
