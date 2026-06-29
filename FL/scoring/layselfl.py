from FL.groups import per_tensor_l1

def per_tensor_average_l1(per_tensor_l1_sum, n_epochs):
    """
    per_tensor_l1_sum: Dict containing a sum of l1 differences
    n_epochs: Total number of epochs
    """
    return {n: sum / n_epochs for n, sum in per_tensor_l1_sum.items()}

def per_tensor_dynamic_sens(clean_per_tensor_average_l1,backdoored_per_tensor_average_l1):
    """
    clean_per_tensor_average_l1: The average per tensor l1 volatility for clean trajectory
    backdoored_per_tensor_average_l1: The average per tensor l1 volatility for backdoored trajectory
    """
    return {n: clean_per_tensor_average_l1[n] - backdoored_per_tensor_average_l1[n] for n in clean_per_tensor_average_l1}


def per_tensor_conv_sens(clean_theta, backdoored_theta, tensor_list):
    """
    Computes the convergent sensitivity on the total model weights of the clean and backdoor trajectories.
    clean_theta: Clean trajectory model weights after the last epoch
    backdoored_theta: Backdoored trajectory model weights after the last epoch
    """
    return per_tensor_l1(state_a=clean_theta, state_b=backdoored_theta, tensor_list=tensor_list)

def per_tensor_volatility(snaps, theta_t, tensor_list):
    """
    Computes the average l1 volatility per tensor over the number of epochs.

    Volatility = mean per-epoch step length: (1/N) * sum_n ||theta_n - theta_{n-1}||_1,
    with theta_0 = theta_t (the round's broadcast weights). 'prev' MUST advance to
    the current snap each iteration; without that every step measures
    ||snap - theta_t|| (displacement from the start) instead of consecutive movement.
    """
    acc, prev = {n: 0.0 for n in tensor_list}, theta_t
    for snap in snaps:
        # Per epoch iteration
        curr_l1 = per_tensor_l1(state_a=snap, state_b=prev, tensor_list=tensor_list)
        for t in tensor_list:
            acc[t] += curr_l1[t]
        prev = snap
    # Average over the number of epochs
    total_epochs = len(snaps)
    return per_tensor_average_l1(per_tensor_l1_sum=acc, n_epochs=total_epochs)

def laysel_sensitivity(ben_snaps, mal_snaps, theta_t, tensor_list):
    """
    Compute the LaySelFL sensitivity score; return the total AND both parts.

    Returns:
        sensitivity : {t: S_d[t] + S_c[t]}  total per-tensor score (selection input)
        sens_d      : {t: ...}              dynamic   (clean_vol - backdoored_vol)
        sens_c      : {t: ...}              convergent (||clean_end - mal_end||_1)

    The parts are returned (not just the sum) so the pilot can log each: S_d is a
    per-epoch RATE while S_c is a TOTAL endpoint gap, so whether S_c dominates S_d
    can only be judged with both logged.

    Interpretation notes (the paper is ambiguous here — flag in the writeup):
      - theta_0 = theta_t: volatility's first step uses the received server model
        as the epoch-0 anchor. Reasonable reading of the paper's "epoch 0".
      - N = the malicious round's local epochs (len(snaps)). The paper's SLI may
        run a separate multi-epoch diagnostic rather than the FL local epochs, so
        this N is defensible but not uniquely "their" N.
    """
    # Benign (clean) and malicious (backdoored) average volatility
    clean_vol = per_tensor_volatility(snaps=ben_snaps, theta_t=theta_t, tensor_list=tensor_list)
    backdoored_vol = per_tensor_volatility(snaps=mal_snaps, theta_t=theta_t, tensor_list=tensor_list)

    # Dynamic sensitivity score
    sens_d = per_tensor_dynamic_sens(clean_per_tensor_average_l1=clean_vol, backdoored_per_tensor_average_l1=backdoored_vol)

    # Convergent sensitivity score
    sens_c = per_tensor_conv_sens(clean_theta=ben_snaps[-1], backdoored_theta=mal_snaps[-1], tensor_list=tensor_list)

    sensitivity = {t: sens_d[t] + sens_c[t] for t in tensor_list}
    return sensitivity, sens_d, sens_c


def select_topk(sensitivity, k):
    """Top-k layers by sensitivity, descending.

    UNLIKE the L2/rho selector in l2_based.select_topk, this does NOT gate on
    score > 0: LaySelFL ranks raw sensitivity and takes the k highest regardless
    of sign (dynamic = clean - backdoored can be negative, yet those layers are
    still legitimately ranked).

    Returns a list of (name, sensitivity) tuples ORDERED by sensitivity
    descending (ties broken by name) — the amplification priority order. Callers
    needing a set of names use {t for t, _ in result}.
    """
    ranked = sorted(sensitivity.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:k]
