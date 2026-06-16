"""
Backdoor-carrying-layer localization for constrain-and-scale.

The function below is called once per malicious client per round, after both the
benign-anchor pass and the malicious training pass have finished, with the two
update directions (u_ben = theta_ben - theta_global, u_mal = theta_local - theta_global).

It should return either:
  - None: no localization. The submission step treats every parameter uniformly.
  - dict[name -> 0/1 tensor]: per-parameter mask aligned to model.named_parameters().
    1 = backdoor-carrying (amplify at submission), 0 = non-BC (fall back to theta_global).

Stub for now — real localization logic lands here.
"""


def compute_bc_mask(u_ben, u_mal):
    """Stub: return None so the submission step is uniform."""
    return None
