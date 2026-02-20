import numpy as np
import random 
from torch.utils.data import Subset

def partition_dataset(dataset, n_clients):
    """
    Partitions the dataset over n_clients, returns a list of data subsets
    """

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)

    # Split according to 'indices'
    partitions = np.array_split(indices, n_clients)
    return [Subset(dataset, part) for part in partitions]
