import numpy as np
import random 
from torch.utils.data import Subset

def get_shuffled_indices(dataset_size):
    indices = list(range(dataset_size))
    random.shuffle(indices)
    return indices
   
def partition_dataset(dataset, n_clients, indices):
    """
    Partitions the dataset over n_clients when shuffled according to indices, returns a list of data subsets
    """
    # Split according to 'indices'
    partitions = np.array_split(indices, n_clients)
    return [Subset(dataset, part) for part in partitions]
