import flwr as fl
import torch
import numpy as np
from torch.utils.data import DataLoader

from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.partitioning import partition_dataset
from utils.dataset import CustomCocoDataset
from utils.global_settings import BackdoorProperties, ProjectPaths, TrainingSettings
from utils.model_log import ModelLogger
from FL.client import Client

