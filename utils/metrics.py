import numpy as np
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



