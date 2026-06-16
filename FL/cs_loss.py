"""
Constrain-and-scale penalty functions.

Each function takes (model, anchor_weights, lambda_) and returns a scalar tensor
to be added to the FasterRCNN task losses during malicious training. New variants
(e.g. per-parameter-weighted L2) get added as additional functions in this file.
"""
import torch


def cs_global_l2(model, anchor_weights, lambda_):
    """
    Simplest constrain-and-scale penalty: lambda_ * sum_p ||theta_p - anchor_p||^2
    over floating-point parameters only.
    """
    total = None
    for name, p in model.named_parameters():
        if not p.is_floating_point():
            continue
        a = anchor_weights.get(name)
        if a is None:
            continue
        a = a.to(p.device, dtype=p.dtype)
        sq = ((p - a) ** 2).sum()
        total = sq if total is None else total + sq
    if total is None:
        first = next(model.parameters(), None)
        device = first.device if first is not None else torch.device("cpu")
        return torch.zeros((), device=device)
    return lambda_ * total
