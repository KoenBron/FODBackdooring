import math
import numpy as np
import torch
import random
from torch.utils.data import Subset

from utils.global_settings import FLSettings

def get_shuffled_indices(dataset_size):
    indices = list(range(dataset_size))
    random.Random(FLSettings.PARTITIONING_SEED).shuffle(indices)
    return indices

def _array_split_bounds(n_items, n_clients):
    """
    Mirrors np.array_split contiguous-chunk boundaries: when n_items is not
    divisible by n_clients, the first (n_items % n_clients) chunks each get one
    extra element. Returns a list of (lo, hi) pairs of length n_clients.
    """
    base, rem = divmod(n_items, n_clients)
    bounds = []
    offset = 0
    for i in range(n_clients):
        size = base + (1 if i < rem else 0)
        bounds.append((offset, offset + size))
        offset += size
    return bounds

def get_shuffled_indices_with_target_quota(
    dataset_size,
    n_clients,
    attacker_ids,
    target_image_indices,
    min_target_fraction=0.0,
    min_target_count=0,
    benign_target_floor=0,
    seed=None,
    verbose=True,
):
    """
    Build a permutation of range(dataset_size) such that np.array_split(..., n_clients)
    produces chunks where every attacker chunk i (i in attacker_ids) contains at least
    max(ceil(min_target_fraction * |chunk|), min_target_count) indices that are
    members of target_image_indices.

    The rebalance is performed by swapping a target index from a benign chunk with a
    non-target index from the attacker chunk. This preserves total partition sizes
    and disjointness — the returned list is still a permutation of range(dataset_size).
    Falls back identically to get_shuffled_indices if no quota constraint is active
    (min_target_fraction == 0 and min_target_count == 0) or attacker_ids is empty.

    benign_target_floor: minimum number of target indices each benign chunk must
        retain after donations. Set > 0 to keep some target-class signal on benign
        clients (useful if benign convergence on the target class matters).
    seed: if None, uses FLSettings.PARTITIONING_SEED.
    """
    if seed is None:
        seed = FLSettings.PARTITIONING_SEED
    rng = random.Random(seed)

    # Start from the same shuffle the standard path would produce, so toggling the
    # quota on/off only differs by the swaps strictly necessary to meet the quota.
    indices = list(range(dataset_size))
    rng.shuffle(indices)

    if not attacker_ids or (min_target_fraction <= 0.0 and min_target_count <= 0):
        return indices

    target_set = set(target_image_indices)
    bounds = _array_split_bounds(dataset_size, n_clients)
    attacker_set = set(attacker_ids)

    def count_targets(lo, hi):
        return sum(1 for p in range(lo, hi) if indices[p] in target_set)

    # Snapshot pre-rebalance counts for the audit print
    pre_counts = [count_targets(lo, hi) for (lo, hi) in bounds]

    # Iterate benign donors in a stable, seeded order so reruns are deterministic
    benign_ids = [i for i in range(n_clients) if i not in attacker_set]
    rng.shuffle(benign_ids)

    for aid in attacker_ids:
        lo, hi = bounds[aid]
        chunk_size = hi - lo
        have = count_targets(lo, hi)
        need = max(
            math.ceil(min_target_fraction * chunk_size),
            int(min_target_count),
        )
        deficit = max(0, need - have)
        if deficit == 0:
            continue

        # Positions in the attacker chunk holding non-target indices — these are
        # what we'll swap out. The check (size - have) >= deficit holds whenever
        # need <= chunk_size, which is the user's contract.
        a_nontarget_positions = [p for p in range(lo, hi) if indices[p] not in target_set]
        rng.shuffle(a_nontarget_positions)
        a_iter = iter(a_nontarget_positions)

        for bid in benign_ids:
            if deficit == 0:
                break
            blo, bhi = bounds[bid]
            b_target_positions = [p for p in range(blo, bhi) if indices[p] in target_set]
            donatable = max(0, len(b_target_positions) - benign_target_floor)
            if donatable == 0:
                continue
            rng.shuffle(b_target_positions)
            for b_pos in b_target_positions[:donatable]:
                if deficit == 0:
                    break
                try:
                    a_pos = next(a_iter)
                except StopIteration:
                    break
                indices[a_pos], indices[b_pos] = indices[b_pos], indices[a_pos]
                deficit -= 1

        if deficit > 0:
            raise RuntimeError(
                f"Cannot satisfy target-class quota for attacker {aid}: "
                f"residual deficit={deficit} after exhausting benign donors. "
                f"Reduce --attacker-target-min-fraction / --attacker-target-min-count, "
                f"or lower --benign-target-floor."
            )

    if verbose:
        post_counts = [count_targets(lo, hi) for (lo, hi) in bounds]
        print("\n[PARTITION] Target-class quota rebalance (attacker_ids={}, "
              "min_frac={}, min_count={}, benign_floor={}):".format(
                  list(attacker_ids), min_target_fraction, min_target_count, benign_target_floor))
        print("  client  role     |chunk|  target_pre  target_post")
        for i, (lo, hi) in enumerate(bounds):
            role = "attacker" if i in attacker_set else "benign  "
            print(f"  {i:>6}  {role}  {hi-lo:>7}  {pre_counts[i]:>11}  {post_counts[i]:>12}")

    return indices

def partition_dataset(dataset, n_clients, indices):
    """
    Partitions the dataset over n_clients when shuffled according to indices, returns a list of data subsets
    """
    # Split according to 'indices'
    partitions = np.array_split(indices, n_clients)
    return [Subset(dataset, part) for part in partitions]

def set_global_seed(seed):
    """
    Give the ability to seed all the randomness relevant to this FL training
    """
    # Seed the python random module, used for paritioning
    random.seed(seed)

    # Seed pytorch
    torch.manual_seed(seed)

    # Seed numpy
    np.random.seed(seed)

def unset_global_seed():
    random.seed()
    torch.seed()
    np.random.seed()
