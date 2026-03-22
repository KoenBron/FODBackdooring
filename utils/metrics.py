import numpy as np
import torch
from numpy.linalg import norm

def flatten_weights(weights):
    """
    Flattens a list of weights as numpy arrays
    """
    return np.concatenate([w.flatten() for w in weights])

def compute_l2_distance(weights_a, weights_b):
    vec_a = flatten_weights(weights_a)
    vec_b = flatten_weights(weights_b)

    return norm(vec_a - vec_b)

def compute_cosine_sim(weights_a, weights_b):
    vec_a = flatten_weights(weights_a)
    vec_b = flatten_weights(weights_b)

    dot_product = np.dot(vec_a, vec_b)
    norm_a = norm(vec_a)
    norm_b = norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)

def compute_magnitude(vector):
    flattened_vec = flatten_weights(vector)
    return norm(flattened_vec)


def torch_flatten_weights(weights):
    """
    Flattens a PyTorch state_dict (dict) or a list of PyTorch tensors 
    into a single 1D PyTorch tensor on the CPU.
    """
    if isinstance(weights, dict):
        tensors = [v.cpu().flatten() for v in weights.values()]
    else:
        # Fallback voor als het al een lijst met tensors was
        tensors = [w.cpu().flatten() for w in weights]
        
    # Plak alle platte tensors achter elkaar
    return torch.cat(tensors)

def torch_compute_l2_distance(weights_a, weights_b):
    """
    Computes the L2 (Euclidean) distance between two sets of weights.
    """
    vec_a = torch_flatten_weights(weights_a)
    vec_b = torch_flatten_weights(weights_b)

    distance = torch.dist(vec_a, vec_b, p=2)
    
    return distance.item()

def torch_compute_cosine_sim(weights_a, weights_b):
    """
    Computes the cosine similarity between two sets of weights.
    """
    vec_a = torch_flatten_weights(weights_a)
    vec_b = torch_flatten_weights(weights_b)

    norm_a = torch.norm(vec_a)
    norm_b = torch.norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot_product = torch.dot(vec_a, vec_b)
    
    similarity = dot_product / (norm_a * norm_b)
    return similarity.item()

def torch_compute_magnitude(weights):
    """
    Computes the L2 norm (magnitude) of a single set of weights.
    """
    flattened_vec = torch_flatten_weights(weights)
    
    magnitude = torch.norm(flattened_vec)
    return magnitude.item()
