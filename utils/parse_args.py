import argparse
import re
from collections import defaultdict

from utils.noniid import LABEL_RULES
from utils.select_clients import plastic_phase_attacker_coverage
from utils.global_settings import BackdoorProperties, FLSettings, TrainingSettings, TriggerType, AttackType, FloatingTriggerPos, EvaluationSettings, ANSIColor, ProjectPaths
from utils.bbox_transform import BoxTransformMode

_SCHEDULE_TOKEN = re.compile(r"^(\d+)(?:-(\d+)(?:/(\d+))?)?:(\d+|\*)$")


def _expand_schedule_tokens(spec, attacker_ids):
    """Expand a --client-schedule string into [(round, client), ...].

    Returns None (after printing why) on a malformed or impossible token, matching
    how the rest of parse_client_schedule reports errors. '*' rotation uses one
    cursor for the whole string so splitting a schedule into several tokens keeps
    the attackers rotating rather than restarting at the first one.
    """
    rotation = list(attacker_ids)
    cursor = 0
    out = []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        m = _SCHEDULE_TOKEN.match(token)
        if not m:
            print(f"{ANSIColor.RED}[ERROR]{ANSIColor.RESET} Bad --client-schedule token {token!r}. "
                  f"Expected 'round:client', 'start-end:client' or 'start-end/step:client', "
                  f"where client is an index or '*' to rotate attackers.")
            return None
        start, end, step, who = m.group(1), m.group(2), m.group(3), m.group(4)
        start = int(start)
        end = int(end) if end is not None else start
        step = int(step) if step is not None else 1
        if step < 1:
            print(f"{ANSIColor.RED}[ERROR]{ANSIColor.RESET} Step must be >= 1 in {token!r}.")
            return None
        if end < start:
            print(f"{ANSIColor.RED}[ERROR]{ANSIColor.RESET} Range end before start in {token!r}.")
            return None
        if who == "*" and not rotation:
            print(f"{ANSIColor.RED}[ERROR]{ANSIColor.RESET} {token!r} uses '*' but there are no "
                  f"attackers to rotate through (--attacker-fraction gives none).")
            return None
        for rnd in range(start, end + 1, step):
            if who == "*":
                out.append((rnd, rotation[cursor % len(rotation)]))
                cursor += 1
            else:
                out.append((rnd, int(who)))
    return out


def parse_client_schedule(args, attacker_ids):
    """
    Parse the formatted client schedule string to set which clients are selected each round.
    Gives a warning when non-malicious clients are selected. Cannot select more clients in a given round than updates
    
    Accepted token forms, comma separated and freely mixed:

        round:client        one round                      e.g. 11:0
        start-end:client    every round in [start, end]    e.g. 11-128:0
        start-end/step:cl   every step-th round            e.g. 11-128/3:0

    The client may be '*', which rotates through attacker_ids in order, advancing
    once per emitted round and continuing across tokens left to right. So a
    40-injection every-third-round schedule over four attackers is

        11-128/3:*

    instead of the forty explicit pairs it expands to. Expansion happens before
    validation, so every round still gets the same range and attacker-id checks.
    """
    pairs = _expand_schedule_tokens(args.client_schedule, attacker_ids)
    if pairs is None:
        return None

    # Parse the round index, client index pairs
    schedule = defaultdict(list)
    for round, client_idx in pairs:
        # Validate the round and client number
        if int(round) not in range(1, args.num_rounds + 1):
            print(f"{ANSIColor.RED}[ERROR]{ANSIColor.RESET} Invalid round number given: {int(round)}. Allowed round range: 1 - args.num_rounds")
            return None
        
        # Validate valid client numbers
        if int(client_idx) not in range(args.total_clients):
            print(f"{ANSIColor.RED}[ERROR]{ANSIColor.RESET} Invalid client index. Allowed range dictated by total number of clients: 0 - {args.num_clients - 1}")
            return None

        # Check if malicious clients are planned
        if int(client_idx) not in attacker_ids:
            print(f"{ANSIColor.YELLOW}[WARNING]{ANSIColor.RESET} Non-malicious client {client_idx} select for round {round}")

        # Add validated round item combo
        schedule[int(round)].append(int(client_idx))

    # Validate the entire round
    for round, clients in schedule.items():

        # Check if number of selected clients is within bounds
        if len(clients) > args.clients_per_round:
            print(f"{ANSIColor.RED}[ERROR]{ANSIColor.RESET} At round: {round} you have selected more clients than are selected per round. You selected {clients} clients, max clients per round: {args.clients_per_round}")
            return None 

    return dict(schedule)

def parse_args():
    parser = argparse.ArgumentParser(description="FODBackdooring FL Simulation")

    # Federated Learning
    fl_group = parser.add_argument_group('Federated Learning')
    fl_group.add_argument("--num-rounds", type=int, default=FLSettings.NUM_ROUNDS, help="Number of training rounds")
    fl_group.add_argument("--total-clients", type=int, default=FLSettings.TOTAL_CLIENTS, help="Total number of clients")
    fl_group.add_argument("--seed", type=int, default=FLSettings.SEED, help="Seed value for round client selection")
    fl_group.add_argument("--partitioning-seed", type=int, default=FLSettings.PARTITIONING_SEED, help="Seed value for partitioning the dataset")
    fl_group.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from")
    fl_group.add_argument("--clients-per-round", type=int, default=FLSettings.CLIENTS_PER_ROUND, help="Number of clients participating each round")
    fl_group.add_argument("--aggregation-strategy", type=str, default=FLSettings.AGGREGATION_STRATEGY, help="Server-side aggregation strategy: 'fedavg' (sample-weighted mean), 'fedopt' (server-LR blend, uses --server-lr), 'multikrum' (robust: keep the m most-central updates by Krum score, then FedAvg them; uses --krum-f / --multikrum-m), 'foolsgold' (sybil-resistant: re-weight clients by how similar their CUMULATIVE update direction is to their most-similar peer; uses --foolsgold-kappa / --foolsgold-feature-group), or 'normclip' (bound every update's L2 norm at C*median by RESCALING it, then sample-weighted FedAvg over the clipped pseudo-gradients; uses --normclip-agg-c / --normclip-noise-sigma). Defensive strategies see only the submitted updates, never client labels.")
    fl_group.add_argument("--server-lr", type=float, default=FLSettings.SERVER_LR, help="Server-side learning rate η used by FedOpt: w_new = (1-η)·w_global + η·avg(w_client). η=1 reduces to FedAvg.")
    fl_group.add_argument("--gamma-cap", type=float, default=FLSettings.GAMMA_CAP, help="Upper bound on the model-replacement scaling factor γ = k/(η·m). Prevents γ from exploding when server_lr is small.")
    fl_group.add_argument("--use-client-schedule", action="store_true", default=FLSettings.USE_CLIENT_SCHEDULE, help="Instead of randomly, though seeded, selecting clients, takes a client schedule to structure when malicious clients are used.")
    fl_group.add_argument("--client-schedule", type=str, default="", help="Which malicious clients act in which rounds. Comma-separated tokens, freely mixed: 'round:client' (one round), 'start-end:client' (every round in the range), 'start-end/step:client' (every step-th round). The client may be '*' to rotate through the attacker ids, advancing once per emitted round and continuing across tokens. So '11-128/3:*' is the 40-injection every-third-round schedule over four attackers, and '11-105:*' is the continuous one. Ranges expand before validation, so round bounds, client indices and per-round capacity are still checked.")
    fl_group.add_argument("--skip-round-eval", action="store_true", default=FLSettings.SKIP_ROUND_EVAL, help="Skip the server's per-round mAP / ASR evaluation and save_if_best_model. Use during the PilotA pilot — pilot measurements live in attack_logs/, not in round_evaluations.csv, and PilotA submits a benign endpoint so ASR is trivially ~0.")
    fl_group.add_argument("--eval-every", type=int, default=FLSettings.EVAL_EVERY, help="Run the per-round mAP / ASR evaluation only every Nth round (default 1 = every round). The first and the last round are always evaluated; skipped rounds log NaN so round_evaluations.csv keeps a fixed shape. Evaluation is a fixed per-round cost that --benign-client-epochs / --malicious-client-epochs do not touch, so this cuts wall-clock without changing training dynamics. WARNING: pick N coprime with the injection period. Injections every 3rd round with --eval-every 3 never lands on an injection round, so every sample sits at the same offset from one and peak ASR is systematically under-read; --eval-every 2 samples both phases. A startup warning fires when the two schedules never coincide.")
    fl_group.add_argument("--eval-on-injection-rounds", action="store_true", default=FLSettings.EVAL_ON_INJECTION_ROUNDS, help="Always evaluate rounds in which an attacker participated, regardless of --eval-start-round and --eval-every. ADDITIVE: it widens the evaluated set rather than narrowing it. Pair it with --eval-start-round set to the last injection round to measure exactly the injection rounds plus the benign tail, skipping everything in between. Membership follows who actually participated, so it works under random client selection too, and it makes the --eval-every aliasing warning moot.")
    fl_group.add_argument("--eval-start-round", type=int, default=FLSettings.EVAL_START_ROUND, help="Skip the per-round mAP / ASR evaluation before round N (default 1 = evaluate from the start). Rounds before N log NaN, exactly like rounds skipped by --eval-every, so round_evaluations.csv keeps a fixed shape. Combine with --eval-every to sample the remaining window. Set it to the last injection round to measure only retention: the build-up is then unmeasured, so report ASR at the window start rather than peak ASR.")
    fl_group.add_argument("--skip-map-eval", action="store_true", default=FLSettings.SKIP_MAP_EVAL, help="Skip only the clean mAP evaluation (run_clean_eval) to cut compute; mAP_50/mAP_50_95 are logged as NaN and best-model saving is skipped. ASR eval still runs unless --skip-asr-eval. Ignored when --skip-round-eval is set.")
    fl_group.add_argument("--skip-asr-eval", action="store_true", default=FLSettings.SKIP_ASR_EVAL, help="Skip only the poison ASR evaluation (run_poison_eval) to cut compute; ASR/fail_rate/avg_conf_fail are logged as NaN. mAP eval still runs unless --skip-map-eval. Ignored when --skip-round-eval is set.")
    fl_group.add_argument("--output-subdir", type=str, default=None, help=f"Subdirectory under outputs/ where this run's results (timestamped run dir, checkpoints, CSVs) are gathered. When omitted, falls back to the value in global_settings (currently '{ProjectPaths.OUTPUT_DIR.name}').")


    # Training Hyperparameters
    tr_group = parser.add_argument_group('Training Hyperparameters')
    tr_group.add_argument("--training-num-workers", type=int, default=TrainingSettings.NUM_WORKERS, help="Dataloader workers")
    tr_group.add_argument("--training-batch-size", type=int, default=TrainingSettings.TRAINING_BATCH_SIZE, help="Batch size used by the dataloader")
    tr_group.add_argument("--benign-client-epochs", type=int, default=TrainingSettings.BENIGN_CLIENT_LOCAL_EPOCHS, help="Local epochs per round for a benign client")
    tr_group.add_argument("--malicious-client-epochs", type=int, default=TrainingSettings.MALICIOUS_CLIENT_LOCAL_EPOCHS, help="Local epochs per round for a malicious client")
    tr_group.add_argument("--benign-client-lr", type=float, default=TrainingSettings.BENIGN_CLIENT_LEARNING_RATE, help="Learning rate for a benign client")
    tr_group.add_argument("--malicious-client-lr", type=float, default=TrainingSettings.MALICIOUS_CLIENT_LEARNING_RATE, help="Learning rate for a malicious client")

    # Attacker Configuration
    atk_group = parser.add_argument_group('Attacker Configuration')
    atk_group.add_argument("--attack-protocol", type=str, default=FLSettings.ATTACK_PROTOCOL, choices=["composable", "pilot_a", "pilot_b", "laysel_fl", "partial"], help="Top-level malicious-client routing. 'composable' = today's knob-tuned path (Neurotoxin / model replacement / TARGETED_* attack composed via the existing --enable-* flags). 'pilot_a' (per-group) / 'pilot_b' (per-tensor) = observational localization pilots. 'partial' = partial-amplification attack: dynamic per-tensor carrying-layer selection (Increment 1 submits the malicious endpoint un-amplified). 'laysel_fl' = preset recipe, not yet wired (raises NotImplementedError).")
    atk_group.add_argument("--partial-select-mode", type=str, default=FLSettings.PARTIAL_SELECT_MODE, choices=["cutoff", "topk", "names"], help="partial protocol: per-tensor carrying-layer selector. 'cutoff' = all eligible tensors with div > (--partial-cutoff)*floor (variable-size; PilotB's selector; cutoff=1 equals the rho>0 gate). 'topk' = fixed --partial-topk tensors by denoised rho. 'names' = a FIXED set given by --partial-carrying-names, ignoring div/b entirely.")
    atk_group.add_argument("--partial-cutoff", type=float, default=FLSettings.PARTIAL_CUTOFF, help="partial protocol (cutoff mode): keep per-tensor layers whose normalized divergence exceeds this multiple of the floor (div > cutoff*floor). 1.0 reproduces the rho>0 gate.")
    atk_group.add_argument("--partial-carrying-names", type=str, default="", help="partial protocol (names mode): comma-separated tensor-name prefixes that define the carrying set. Membership is by name only -- divergence and the noise floor are ignored, so the set is identical in every round and the plastic freeze is a formality. A module prefix takes every tensor under it, so \"roi_heads.box_head.fc6\" selects fc6.weight and fc6.bias (12,846,080 params, 31.03%% of the model). Only eligible (trainable, non-BN) tensors can match. Required by, and only used by, --partial-select-mode names.")
    atk_group.add_argument("--partial-topk", type=int, default=FLSettings.PARTIAL_TOPK, help="partial protocol (topk mode): size of the per-tensor carrying set C (top-k tensors by denoised divergence rho).")
    atk_group.add_argument("--partial-gamma-mode", type=str, default=FLSettings.PARTIAL_GAMMA_MODE, choices=["fixed", "replacement", "budgeted"], help="partial protocol: source of the carrier-set gamma. 'fixed' = use --partial-amp-gamma (the trade-off sweep knob). 'replacement' = derive gamma = k/(server_lr*n_mal) capped at --gamma-cap, i.e. the SAME Bagdasaryan factor as full model replacement, so the carrying tensors are genuinely model-replaced after aggregation. 'budgeted' = EXPLOIT-ONLY per-carrier gamma from a total-L2 norm budget (--partial-norm-budget x this client's own mean benign-anchor norm): carriers take the replacement gamma in descending selection-score order until the budget is spent, the one that exhausts it is throttled to fit exactly, and later carriers go unamplified. Makes --partial-amp-gamma inert.")
    atk_group.add_argument("--partial-norm-budget", type=float, default=FLSettings.PARTIAL_NORM_BUDGET, help="partial protocol (budgeted gamma mode): norm-inflation ceiling B. The submitted update's L2 norm is held <= B * (this client's OWN mean benign-anchor norm, frozen at the plastic freeze; never pooled across clients). B=1.0 submits at the benign scale (no excess to spend -> every carrier unamplified); B>1 buys amplification headroom. B<1 is rejected (shrinking the update is the constrain phase's job).")
    atk_group.add_argument("--partial-amp-gamma", type=float, default=FLSettings.PARTIAL_AMP_GAMMA, help="partial protocol (fixed gamma mode): amplification factor on the carrying tensors' update Delta_C = theta_mal - theta_t. gamma=1.0 submits theta_mal unchanged; gamma>1 concentrates energy in C (raises total update norm; the norm budget is a later increment).")

    # Plastic-phase Configuration (Partial protocol)
    plastic_group = parser.add_argument_group('Plastic-phase Configuration (partial protocol)')
    plastic_group.add_argument("--partial-plastic-rounds", type=int, default=FLSettings.PARTIAL_PLASTIC_ROUNDS, help="partial protocol: number of (pooled, across-attacker) injection rounds to observe before FREEZING one shared carrying set = select_by_mode(mean(div), mean(b)). The frozen set is reused every later round, removing the per-round |C| oscillation and late-round collapse. 0 = plastic phase OFF (per-round selection every round).")
    plastic_group.add_argument("--partial-plastic-submit-benign", action="store_true", default=FLSettings.PARTIAL_PLASTIC_SUBMIT_BENIGN, help="partial protocol: during the PLASTIC phase submit the benign anchor (theta_ben1) instead of the poisoned model. The localizer is unaffected -- div/b come from the trained trajectories, not from what is submitted -- so the carrying set is selected exactly as before, but the attacker is indistinguishable from a benign client for the whole localisation phase and never perturbs the global model it is localising against. Costs the cumulative plastic deposit (plastic_rounds x eta*m/k); measured ASR at the freeze is ~0.000, so that deposit lands where it buys nothing. Exploit rounds are untouched.")
    plastic_group.add_argument("--partial-plastic-no-amp", action="store_true", default=FLSettings.PARTIAL_PLASTIC_NO_AMP, help="partial protocol: submit UNAMPLIFIED (gamma=1) during the plastic phase so the localizer observes on an un-deformed global; amplification resumes only after the freeze. Off by default (amplify during plastic to lay the backdoor foundation).")
    plastic_group.add_argument("--no-frozen-set-summary", action="store_false", dest="log_frozen_set_summary", default=FLSettings.LOG_FROZEN_SET_SUMMARY, help="partial protocol: DISABLE the human-readable frozen-set summary (attack_logs/partial/frozen_set.md) written when the plastic phase freezes. On by default.")

    # Constrain-phase Configuration (Partial protocol, exploit-only)
    constrain_group = parser.add_argument_group('Constrain-phase Configuration (partial protocol)')
    constrain_group.add_argument("--partial-constrain", action="store_true", default=FLSettings.PARTIAL_CONSTRAIN, help="partial protocol: during EXPLOIT rounds, add L_constrain = (--partial-constrain-lambda) * sum_{p not in C} ||theta_p - theta_t,p||^2 (unnormalized SSE) to the malicious trajectory's loss, pulling the NON-carrying params back toward the broadcast model theta_t so the update stays in the benign cluster (L2/Krum can otherwise separate it even at gamma=1). Carrying params are excluded so the backdoor is preserved. Requires a frozen carrying set (--partial-plastic-rounds > 0); no-op before the freeze.")
    constrain_group.add_argument("--partial-constrain-lambda", type=float, default=FLSettings.PARTIAL_CONSTRAIN_LAMBDA, help="partial protocol: coefficient LAMBDA on the summed-squared-error non-carrying-to-broadcast constrain penalty (see --partial-constrain). SSE (not a mean) so LAMBDA ~ O(1) is effective; 0.0 disables the penalty even if --partial-constrain is set.")
    constrain_group.add_argument("--partial-freeze-noncarrying", action="store_true", default=FLSettings.PARTIAL_FREEZE_NONCARRYING, help="partial protocol: during EXPLOIT rounds, zero the GRADIENTS of the non-carrying eligible tensors for the whole mal trajectory (grad_mask hook), pinning them at the broadcast model theta_t so the carrying layers co-adapt to frozen non-carriers. Stronger than zeroing pseudo-gradients only at submission. Requires a frozen carrying set (--partial-plastic-rounds > 0); no-op before the freeze. MUTUALLY EXCLUSIVE with --partial-constrain and --partial-mask-noncarrying (raises at startup if more than one is set).")
    constrain_group.add_argument("--partial-mask-noncarrying", action="store_true", default=FLSettings.PARTIAL_MASK_NONCARRYING, help="partial protocol: during EXPLOIT rounds, train the malicious trajectory completely UNRESTRICTED and then reset the non-carrying eligible tensors to theta_t in the SUBMISSION (zero pseudo-gradient there). Same support on the wire as --partial-freeze-noncarrying, but ||Delta_C|| is the unmodified full-model value instead of the freeze's inflated one — the ablation that separates 'drop the non-carrying energy' from 'concentrate the backdoor into C'. Requires a frozen carrying set (--partial-plastic-rounds > 0); no-op before the freeze. MUTUALLY EXCLUSIVE with --partial-freeze-noncarrying and --partial-constrain (raises at startup if more than one is set).")

    atk_group.add_argument("--attacker-fraction", type=float, default=FLSettings.ATTACKER_FRACTION, help="Fraction of malicious clients")
    atk_group.add_argument("--client-train-ds-mode", type=str, default=FLSettings.ATTACKER_DATASET_MODE, help="Dataset mode for attackers")
    atk_group.add_argument("--enable-model-replacement", action="store_true", default=FLSettings.DO_MODEL_REPLACEMENT, help="Enable the model replacement strategy for malicious clients to amplify backdoors")
    atk_group.add_argument("--enable-neurotoxin", action="store_true", default=FLSettings.NEUROTOXIN_ENABLED, help="Enable Neurotoxin gradient masking for malicious clients")
    atk_group.add_argument("--neurotoxin-mask-ratio", type=float, default=FLSettings.NEUROTOXIN_MASK_RATIO, help="Per-layer top-k fraction of benign-gradient coords to exclude (e.g. 0.01 = top 1%%)")
    atk_group.add_argument("--guarantee-target-objects", action="store_true", default=FLSettings.GUARANTEE_TARGET_OBJECTS, help="Rebalance partition indices so each attacker partition contains a minimum fraction/count of target-class images. Targeted attacks only.")
    atk_group.add_argument("--attacker-target-min-fraction", type=float, default=FLSettings.ATTACKER_TARGET_MIN_FRACTION, help="Minimum fraction of each attacker partition that must contain the target class.")
    atk_group.add_argument("--attacker-target-min-count", type=int, default=FLSettings.ATTACKER_TARGET_MIN_COUNT, help="Absolute floor on target-class images per attacker partition. Effective quota is max(fraction × |partition|, count).")
    atk_group.add_argument("--benign-target-floor", type=int, default=FLSettings.BENIGN_TARGET_FLOOR, help="Minimum number of target-class images each benign partition is allowed to retain when donating to attackers.")

    # Non-IID partitioning (label skew at equal partition size)
    part_group = parser.add_argument_group('Data Partitioning')
    part_group.add_argument("--partition-mode", type=str, default=FLSettings.PARTITION_MODE, choices=["iid", "dirichlet"], help="How the training set is split across clients. 'iid' (default) is the seeded shuffle + equal chunks, unchanged. 'dirichlet' gives every client the SAME number of images but draws each client's CLASS COMPOSITION as Dir(--dirichlet-alpha * K * class-marginal), i.e. label skew without quantity skew — so gamma = k/(eta*m) stays exact and update norms stay comparable across clients. MUTUALLY EXCLUSIVE with --guarantee-target-objects.")
    part_group.add_argument("--dirichlet-alpha", type=float, default=FLSettings.DIRICHLET_ALPHA, help="Dirichlet concentration for --partition-mode dirichlet. The base measure is the dataset's own class marginal, so E[composition] = marginal at every alpha: small = strongly skewed around it, large -> genuine IID. 0.1 / 0.5 / 5.0 is a reasonable sweep; LaySelFL uses 0.9.")
    part_group.add_argument("--dirichlet-label-rule", type=str, default=FLSettings.DIRICHLET_LABEL_RULE, choices=list(LABEL_RULES), help="Proxy that files each multi-object image under ONE class for partitioning (never used for training). 'max_area': per-class summed annotation area, argmax (default). 'count_majority': most instances. 'first': first-listed annotation, an arbitrary-but-deterministic control. 'stochastic_majority': max_area with probability --dirichlet-label-p, else a uniform draw over the classes present.")
    part_group.add_argument("--dirichlet-attacker-mode", type=str, default=FLSettings.DIRICHLET_ATTACKER_MODE, choices=["independent", "matched"], help="How malicious clients are drawn under --partition-mode dirichlet. 'independent' (default): each attacker gets its own Dirichlet draw like any benign client, so colluders do NOT share a label distribution — this removes the correlated-update-direction artifact that would otherwise hand FoolsGold an ideal sybil signature. 'matched': attackers share one draw and one integer count vector, making their compositions identical by construction; this keeps the plastic phase's pooled div/b assumption exactly true and is the best-case bound for FoolsGold. Note the target-class floor (--guarantee-target-objects) is only available under 'matched'.")
    part_group.add_argument("--dirichlet-label-p", type=float, default=FLSettings.DIRICHLET_LABEL_P, help="P(take the majority class) for --dirichlet-label-rule stochastic_majority. Ignored by the other rules.")

    # Detectability Harness (passive; observational only, does not alter aggregation)
    det_group = parser.add_argument_group('Detectability Harness (passive)')
    det_group.add_argument("--do-stealth-eval", action="store_true", default=FLSettings.DO_STEALTH_EVAL, help="Enable passive backdoor-detectability logging (Krum/norm-clip/clustering/median + per-layer profile). Does NOT alter aggregation.")
    det_group.add_argument("--krum-f", type=int, default=FLSettings.KRUM_F, help="Assumed number of Byzantine clients f for Krum's (n-f-2) neighbor sum.")
    det_group.add_argument("--multikrum-m", type=int, default=FLSettings.MULTIKRUM_M, help="Number of clients Multi-Krum keeps. <=0 auto-sets to n-f.")
    det_group.add_argument("--foolsgold-kappa", type=float, default=FLSettings.FOOLSGOLD_KAPPA, help="FoolsGold confidence parameter kappa (--aggregation-strategy foolsgold). Scales the logit before clipping. It does NOT change which clients are excluded (anything below the logit's zero crossing at alpha~0.3775 is zeroed for every kappa) — it controls how decisively borderline clients are admitted, so larger kappa gives a sharper, more binary split. Paper default 1.0.")
    det_group.add_argument("--foolsgold-feature-group", type=str, default=FLSettings.FOOLSGOLD_FEATURE_GROUP, help="Parameter group (see FL/groups.py) whose cumulative update history FoolsGold tracks — the paper's 'indicative features'. Default roi.cls_score: the classification head, where a TARGETED_MISLABELING backdoor's signal lives, and ~21.5K floats per client instead of ~41M for the whole model. Use roi.bbox_pred for a box-transform attack.")
    det_group.add_argument("--normclip-agg-c", type=float, default=FLSettings.NORMCLIP_AGG_C, help="Clip multiplier for --aggregation-strategy normclip: every update is rescaled onto the sphere of radius C*median(||Delta||). Default 1.0 = FLAME's rule (clip at the median), which by construction rescales roughly half the round. SEPARATE from --normclip-c, which is the passive detector's flag threshold and stays at its own default so previously collected harness output remains comparable; set them equal if you want detector and defense to agree.")
    det_group.add_argument("--normclip-noise-sigma", type=float, default=FLSettings.NORMCLIP_NOISE_SIGMA, help="Gaussian noise added to the normclip aggregate after clipping, with std = sigma * M / n (M = the clip bound, n = participants). Scaled by M so sigma needs no knowledge of the model's parameter magnitudes; O(1e-3) is a sensible start. 0 = off (exact no-op). Run clip-alone and clip+noise as separate arms — noise costs benign accuracy, so mixing them makes an mAP drop unattributable.")
    det_group.add_argument("--foolsgold-normalise", type=str, default=FLSettings.FOOLSGOLD_NORMALISE, choices=["none", "sum"], help="How FoolsGold turns per-client weights into aggregation coefficients. 'none' (default) is PAPER-FAITHFUL: weights are used as-is, each in [0,1], so the aggregate sum_i w_i*Delta_i grows with how many clients are trusted (~8 of 9 trusted gives an ~8x step). sum(w) is printed every round as the effective step multiplier. 'sum' divides by sum(w) for a constant-size convex combination — defensible and matching Multi-Krum's behaviour, but not the paper's rule; use it if 'none' destabilises training.")
    det_group.add_argument("--normclip-c", type=float, default=FLSettings.NORMCLIP_C, help="Norm-bound detector multiplier: flag updates with L2 norm > C*median.")
    det_group.add_argument("--stealth-granularity", type=str, default=FLSettings.STEALTH_GRANULARITY, choices=["auto", "group", "tensor"], help="Granularity of the dl2_* per-bucket energy decomposition. 'auto' derives it from --attack-protocol (pilot_a->group, pilot_b->tensor); 'group'/'tensor' force it.")
    det_group.add_argument("--stealth-disable", type=str, default="", help="Comma-separated stealth metric blocks to turn OFF (default: all on). Blocks: norm, cosine, krum, median, normclip, cluster, energy. Disabling all of {cosine,krum,cluster} also skips the O(n^2) pairwise pass.")
    det_group.add_argument("--stealth-on-benign-rounds", action="store_true", default=FLSettings.STEALTH_ON_BENIGN_ROUNDS, help="Also run the stealth harness on rounds with no attacker present. Off by default (those rounds are skipped to save compute since label-aware metrics are NaN). Enable for a no-attacker baseline to gather the benign null distribution for the stealth metrics.")
    # OBSERVATIONAL norm-clip report — ON by default; disable with --no-normclip-report.
    # Both flags share dest=do_normclip_report (--do-... kept so existing scripts still work).
    det_group.add_argument("--do-normclip-report", action="store_true", dest="do_normclip_report", default=FLSettings.DO_NORMCLIP_REPORT, help="OBSERVATIONAL (never alters aggregation): for each malicious update, record in a top-level normclip_report.csv whether its L2 norm exceeds x*(benign-trajectory norm) or x*(median submitted norm), across the --normclip-report-x sweep. The benign reference is PER CLIENT (never pooled): the mean of that attacker's OWN benign anchor norms (both ben1 and ben2, so 2 samples per plastic round it acts in) — live in the plastic/disabled phases, and that client's frozen mean in the exploit phase. An attacker that never acts during the plastic phase has no reference and logs ratio_to_benign=NaN (warned about at startup). Partial protocol. ON by default (this flag is a no-op kept for explicitness; use --no-normclip-report to turn it off).")
    det_group.add_argument("--no-normclip-report", action="store_false", dest="do_normclip_report", help="Disable the observational norm-clip report (normclip_report.csv). On by default.")
    det_group.add_argument("--normclip-report-x", type=str, default=",".join(f"{v:g}" for v in FLSettings.NORMCLIP_REPORT_X), help="Comma-separated x sweep for the norm-clip report (e.g. '1.0,1.5,2.0,3.0'). Each x adds exceeds_median@x and exceeds_benign@x columns (flagged when norm > x*reference).")

    # Backdoor Properties
    bd_group = parser.add_argument_group('Backdoor Properties')
    bd_group.add_argument("--disable-backdoor-logging", action="store_true", help="Disable backdoor config logging")
    bd_group.add_argument("--attack-type", type=str, default=BackdoorProperties.ATTACK_TYPE.name if hasattr(BackdoorProperties.ATTACK_TYPE, 'name') else BackdoorProperties.ATTACK_TYPE, choices=[e.name for e in AttackType], help="Type of backdoor attack")
    bd_group.add_argument("--target-class", type=str, default=BackdoorProperties.TARGET_CLASS, help="Class to target")
    bd_group.add_argument("--mislabel-class", type=str, default=BackdoorProperties.MISLABEL_REPLACEMENT_CLASS, help="Replacement class")
    bd_group.add_argument("--poison-rate", type=float, default=BackdoorProperties.POISON_RATE, help="Percentage of poisoned data")
    bd_group.add_argument("--trigger-pattern", type=str, default=BackdoorProperties.TRIGGER_PATTERN.name if hasattr(BackdoorProperties.TRIGGER_PATTERN, 'name') else BackdoorProperties.TRIGGER_PATTERN, choices=[e.name for e in TriggerType], help="Trigger visual pattern")
    bd_group.add_argument("--trigger-width", type=int, default=BackdoorProperties.TRIGGER_WIDTH)
    bd_group.add_argument("--trigger-height", type=int, default=BackdoorProperties.TRIGGER_HEIGHT)
    default_color = ",".join(map(str, BackdoorProperties.TRIGGER_COLOR))
    bd_group.add_argument("--trigger-color", type=str, default=default_color)
    default_color_secondary = ",".join(map(str, BackdoorProperties.TRIGGER_COLOR_SECONDARY))
    bd_group.add_argument("--trigger-color-secondary", type=str, default=default_color_secondary, help="Secondary BGR color used by FLOATING_CHECKERBOARD")
    bd_group.add_argument("--w-ratio", type=float, default=BackdoorProperties.FLOATING_TRIGGER_W_RATIO)
    bd_group.add_argument("--h-ratio", type=float, default=BackdoorProperties.FLOATING_TRIGGER_H_RATIO)
    bd_group.add_argument("--floating-pos", type=str, default=BackdoorProperties.FLOATING_TRIGGER_POSITION.name if hasattr(BackdoorProperties.FLOATING_TRIGGER_POSITION, 'name') else BackdoorProperties.FLOATING_TRIGGER_POSITION, choices=[e.name for e in FloatingTriggerPos])
    bd_group.add_argument("--trigger-alpha", type=float, default=BackdoorProperties.TRIGGER_ALPHA)
    bd_group.add_argument("--block-size", type=int, default=BackdoorProperties.BLOCK_SIZE)
    bd_group.add_argument("--checker-square-px", type=int, default=BackdoorProperties.FLOATING_CHECKER_SQUARE_PX, help="Square size in px at model-input resolution for FLOATING_CHECKERBOARD")
    bd_group.add_argument("--rng-seed", type=int, default=BackdoorProperties.RNG_SEED)

    # Regression-Head Backdoor (box transform). Active only with --attack-type
    # TARGETED_BOX_TRANSFORM; the head is a no-op unless the dataset tags poison_mask.
    rh_group = parser.add_argument_group('Regression-Head Backdoor (box transform)')
    rh_group.add_argument("--box-transform-mode", type=str, default=None, choices=[m.value for m in BoxTransformMode], help="Displace ('shift') or scale ('shrink') the box-regression target of trigger-bearing target objects. None disables the regression-head surgery.")
    rh_group.add_argument("--box-shift-dx", type=float, default=0.0, help="SHIFT: x displacement as a fraction of box width (or absolute px if --box-shift-absolute).")
    rh_group.add_argument("--box-shift-dy", type=float, default=0.0, help="SHIFT: y displacement as a fraction of box height (or absolute px if --box-shift-absolute).")
    rh_group.add_argument("--box-shift-absolute", action="store_true", default=False, help="SHIFT: interpret dx/dy as absolute pixels (model-input resolution) instead of box-size fractions.")
    rh_group.add_argument("--box-shrink-factor", type=float, default=0.5, help="SHRINK: isotropic scale applied to box w/h about its center (0<f<1 shrinks the target box).")

    # Evaluation script Settings
    eval_group = parser.add_argument_group('Evaluation Settings')
    eval_group.add_argument("--do-poison-eval", action="store_true", default=EvaluationSettings.DO_POISON_EVAL, help="Do the poison evaluation")
    eval_group.add_argument("--do-clean-eval", action="store_true", default=EvaluationSettings.DO_CLEAN_EVAL, help="Do the clean evaluation")
    eval_group.add_argument("--print-poison-eval", action="store_true", default=EvaluationSettings.PRINT_POISON, help="Print the poison evaluation metrics")
    eval_group.add_argument("--print-clean-eval", action="store_true", default=EvaluationSettings.PRINT_CLEAN, help="Print the clean evaluation metrics")
    eval_group.add_argument("--model-date", type=str, default=None, help="Select the model to use by giving its date")
    eval_group.add_argument("--centralized-model", action="store_true", default=False, help="Select the model to use by giving its date")
    eval_group.add_argument("--evaluation-num-workers", type=int, default=EvaluationSettings.NUM_WORKERS, help="Number of cpus used to load data to the gpu.")
    eval_group.add_argument("--evaluation-batch-size", type=int, default=EvaluationSettings.BATCH_SIZE, help="Batch size used by dataloaders in evaluation step workers")

    return parser.parse_args()

def apply_settings(args):
    # FL Settings
    FLSettings.NUM_ROUNDS = args.num_rounds
    FLSettings.TOTAL_CLIENTS = args.total_clients
    FLSettings.RESUME_FROM = args.resume_from
    FLSettings.CLIENTS_PER_ROUND = min(args.total_clients, args.clients_per_round)
    FLSettings.CLIENTS_PER_ROUND = min(args.total_clients, args.clients_per_round)
    FLSettings.AGGREGATION_STRATEGY = args.aggregation_strategy
    FLSettings.SERVER_LR = args.server_lr
    FLSettings.GAMMA_CAP = args.gamma_cap
    FLSettings.SEED = args.seed
    FLSettings.PARTITIONING_SEED = args.partitioning_seed
    FLSettings.ATTACK_PROTOCOL = args.attack_protocol
    FLSettings.PARTIAL_SELECT_MODE = args.partial_select_mode
    FLSettings.PARTIAL_CUTOFF = args.partial_cutoff
    FLSettings.PARTIAL_TOPK = args.partial_topk
    FLSettings.PARTIAL_CARRYING_NAMES = tuple(
        x.strip() for x in args.partial_carrying_names.split(",") if x.strip())
    if FLSettings.PARTIAL_SELECT_MODE == "names" and not FLSettings.PARTIAL_CARRYING_NAMES:
        raise ValueError("--partial-select-mode names requires --partial-carrying-names "
                         "(a comma-separated list of tensor-name prefixes).")
    if FLSettings.PARTIAL_CARRYING_NAMES and FLSettings.PARTIAL_SELECT_MODE != "names":
        print(f"{ANSIColor.YELLOW}[WARNING]{ANSIColor.RESET} --partial-carrying-names is set but "
              f"--partial-select-mode is {FLSettings.PARTIAL_SELECT_MODE!r}; the names are IGNORED.")
    FLSettings.PARTIAL_GAMMA_MODE = args.partial_gamma_mode
    FLSettings.PARTIAL_AMP_GAMMA = args.partial_amp_gamma
    FLSettings.PARTIAL_NORM_BUDGET = args.partial_norm_budget
    if FLSettings.PARTIAL_GAMMA_MODE == "budgeted" and FLSettings.PARTIAL_NORM_BUDGET < 1.0:
        # B < 1 asks for a submission SMALLER than the unamplified update, which the
        # allocator cannot deliver: it only ever scales carriers UP (gamma >= 1). Pulling
        # an update back below its natural norm is the constrain phase's job.
        raise ValueError(
            f"--partial-norm-budget must be >= 1.0 (got {FLSettings.PARTIAL_NORM_BUDGET}). "
            "B < 1 would require SHRINKING the update below its unamplified norm, which the "
            "budgeted allocator does not do — use --partial-constrain (or "
            "--partial-freeze-noncarrying) to reduce the update, then budget on top."
        )
    FLSettings.PARTIAL_PLASTIC_ROUNDS = args.partial_plastic_rounds
    FLSettings.PARTIAL_PLASTIC_NO_AMP = args.partial_plastic_no_amp
    FLSettings.PARTIAL_PLASTIC_SUBMIT_BENIGN = args.partial_plastic_submit_benign
    FLSettings.LOG_FROZEN_SET_SUMMARY = args.log_frozen_set_summary
    FLSettings.PARTIAL_CONSTRAIN = args.partial_constrain
    FLSettings.PARTIAL_CONSTRAIN_LAMBDA = args.partial_constrain_lambda
    FLSettings.PARTIAL_FREEZE_NONCARRYING = args.partial_freeze_noncarrying
    FLSettings.PARTIAL_MASK_NONCARRYING = args.partial_mask_noncarrying
    # The three non-carrying interventions all act on `eligible \ C` and contradict each
    # other pairwise, so at most one may be on. Fail fast rather than run a config whose
    # log says a mechanism is enabled while it is in fact inert:
    #   freeze + constrain — freezing zeroes the non-carrying gradient, the only thing the
    #                        constrain penalty produces, so the penalty never bites;
    #   mask + constrain   — the mask discards at submission exactly the drift the penalty
    #                        spent model capacity shrinking;
    #   mask + freeze      — both put the same support on the wire, but the freeze changes
    #                        the trajectory, which is precisely what the mask ablation
    #                        holds fixed; together neither question is answered.
    _noncarrying_switches = [
        ("--partial-constrain", FLSettings.PARTIAL_CONSTRAIN),
        ("--partial-freeze-noncarrying", FLSettings.PARTIAL_FREEZE_NONCARRYING),
        ("--partial-mask-noncarrying", FLSettings.PARTIAL_MASK_NONCARRYING),
    ]
    _enabled = [flag for flag, on in _noncarrying_switches if on]
    if len(_enabled) > 1:
        raise ValueError(
            f"{' and '.join(_enabled)} are mutually exclusive: they are three different "
            "treatments of the SAME non-carrying tensors (train them toward theta_t / never "
            "train them / discard them at submission), and combining any two makes at least "
            "one of them inert or unmeasurable. Enable at most one."
        )
    FLSettings.SKIP_ROUND_EVAL = args.skip_round_eval
    if args.eval_every < 1:
        raise ValueError(f"--eval-every must be >= 1 (got {args.eval_every}); 1 means evaluate every round.")
    FLSettings.EVAL_EVERY = args.eval_every
    if args.eval_start_round < 1:
        raise ValueError(f"--eval-start-round must be >= 1 (got {args.eval_start_round}).")
    FLSettings.EVAL_START_ROUND = args.eval_start_round
    FLSettings.EVAL_ON_INJECTION_ROUNDS = args.eval_on_injection_rounds
    FLSettings.SKIP_MAP_EVAL = args.skip_map_eval
    FLSettings.SKIP_ASR_EVAL = args.skip_asr_eval
    # Output subdirectory: CLI overrides the global_settings default (which sets
    # OUTPUT_DIR's last component). When --output-subdir is omitted, the value
    # baked into ProjectPaths.OUTPUT_DIR is kept as the backup.
    if args.output_subdir is not None:
        ProjectPaths.OUTPUT_DIR = ProjectPaths.ROOT_DIR / "outputs" / args.output_subdir
    FLSettings.DO_MODEL_REPLACEMENT = args.enable_model_replacement
    FLSettings.NEUROTOXIN_ENABLED = args.enable_neurotoxin
    FLSettings.NEUROTOXIN_MASK_RATIO = args.neurotoxin_mask_ratio
    FLSettings.GUARANTEE_TARGET_OBJECTS = args.guarantee_target_objects
    FLSettings.ATTACKER_TARGET_MIN_FRACTION = args.attacker_target_min_fraction
    FLSettings.ATTACKER_TARGET_MIN_COUNT = args.attacker_target_min_count
    FLSettings.BENIGN_TARGET_FLOOR = args.benign_target_floor

    FLSettings.PARTITION_MODE = args.partition_mode
    FLSettings.DIRICHLET_ALPHA = args.dirichlet_alpha
    FLSettings.DIRICHLET_LABEL_RULE = args.dirichlet_label_rule
    FLSettings.DIRICHLET_LABEL_P = args.dirichlet_label_p
    FLSettings.DIRICHLET_ATTACKER_MODE = args.dirichlet_attacker_mode
    if FLSettings.PARTITION_MODE == "dirichlet":
        if FLSettings.DIRICHLET_ALPHA <= 0:
            raise ValueError(
                f"--dirichlet-alpha must be > 0 (got {FLSettings.DIRICHLET_ALPHA}); it is a "
                "Dirichlet concentration, and alpha -> infinity is the IID limit."
            )
        # --guarantee-target-objects is ONE flag with TWO implementations, and under
        # dirichlet it selects the allocator's native floor rather than the IID path's
        # post-hoc index swaps (which would scramble the identical-composition guarantee
        # across attackers that the plastic phase's pooled div/b relies on). Semantics are
        # the same either way: at least max(fraction x |partition|, count) target-class
        # images per attacker. It is NOT auto-enabled under dirichlet — a 60% floor applied
        # to one arm and not the other pins the target class at every alpha and makes the
        # IID/non-IID comparison measure the quota instead of the skew.
        # The in-allocator floor is applied through the attackers' SHARED count vector, so
        # it exists only on the 'matched' path. Independent draws have no such vector and
        # therefore no enforcement point — refuse rather than accept the flag and ignore it.
        if FLSettings.GUARANTEE_TARGET_OBJECTS and \
                FLSettings.DIRICHLET_ATTACKER_MODE == "independent":
            raise ValueError(
                "--guarantee-target-objects has no enforcement point under "
                "--dirichlet-attacker-mode independent: the floor is applied through the "
                "attackers' shared count vector, which independent draws do not have. "
                "Either use --dirichlet-attacker-mode matched (which is also the "
                "configuration FoolsGold most easily detects), or drop the floor and screen "
                "the realised target supply with scripts/partition_report.py before "
                "committing to a seed."
            )
        if FLSettings.GUARANTEE_TARGET_OBJECTS:
            print(
                f"{ANSIColor.YELLOW}[WARNING]{ANSIColor.RESET} --guarantee-target-objects "
                "with --partition-mode dirichlet: the floor is enforced INSIDE the "
                "allocator (no post-hoc index swaps). It applies at every alpha, so an IID "
                "baseline must set the same quota or the two arms differ by more than the "
                "label skew."
            )

    # Detectability harness (passive)
    FLSettings.DO_STEALTH_EVAL = args.do_stealth_eval
    FLSettings.KRUM_F = args.krum_f
    FLSettings.MULTIKRUM_M = args.multikrum_m
    FLSettings.NORMCLIP_C = args.normclip_c
    FLSettings.NORMCLIP_AGG_C = args.normclip_agg_c
    FLSettings.NORMCLIP_NOISE_SIGMA = args.normclip_noise_sigma
    FLSettings.FOOLSGOLD_KAPPA = args.foolsgold_kappa
    FLSettings.FOOLSGOLD_FEATURE_GROUP = args.foolsgold_feature_group
    FLSettings.FOOLSGOLD_NORMALISE = args.foolsgold_normalise
    FLSettings.STEALTH_GRANULARITY = args.stealth_granularity
    # Block names are validated at Server init (against the torch-backed registry),
    # so parse_args stays free of a heavy FL.stealth_defenses import.
    FLSettings.STEALTH_DISABLED = tuple(x.strip() for x in args.stealth_disable.split(",") if x.strip())
    FLSettings.STEALTH_ON_BENIGN_ROUNDS = args.stealth_on_benign_rounds
    FLSettings.DO_NORMCLIP_REPORT = args.do_normclip_report
    FLSettings.NORMCLIP_REPORT_X = tuple(float(v) for v in args.normclip_report_x.split(",") if v.strip())
    
    # Attacker General
    FLSettings.ATTACKER_DATASET_MODE = args.client_train_ds_mode
    FLSettings.ATTACKER_FRACTION = args.attacker_fraction
    num_attackers = int(FLSettings.TOTAL_CLIENTS * FLSettings.ATTACKER_FRACTION)
    FLSettings.ATTACKER_IDS = list(range(num_attackers))

    # Client schedule settings
    FLSettings.USE_CLIENT_SCHEDULE = args.use_client_schedule
    if FLSettings.USE_CLIENT_SCHEDULE:
        schedule = parse_client_schedule(args, FLSettings.ATTACKER_IDS)
        if schedule is not None:
            FLSettings.CLIENT_SCHEDULE = schedule
        else:
            raise Exception("Exception occured while validating client schedule.")

    # Training Settings
    TrainingSettings.NUM_WORKERS = args.training_num_workers
    TrainingSettings.TRAINING_BATCH_SIZE = args.training_batch_size
    TrainingSettings.MALICIOUS_CLIENT_LOCAL_EPOCHS = args.malicious_client_epochs
    TrainingSettings.BENIGN_CLIENT_LOCAL_EPOCHS = args.benign_client_epochs
    TrainingSettings.BENIGN_CLIENT_LEARNING_RATE = args.benign_client_lr
    TrainingSettings.MALICIOUS_CLIENT_LEARNING_RATE = args.malicious_client_lr

    # Backdoor Base
    BackdoorProperties.BACKDOOR_ACTIVE = not args.disable_backdoor_logging
    BackdoorProperties.ATTACK_TYPE = AttackType[args.attack_type]
    BackdoorProperties.TARGET_CLASS = args.target_class
    BackdoorProperties.MISLABEL_REPLACEMENT_CLASS = args.mislabel_class
    BackdoorProperties.POISON_RATE = args.poison_rate
    
    # Backdoor Trigger
    BackdoorProperties.TRIGGER_PATTERN = TriggerType[args.trigger_pattern]
    BackdoorProperties.TRIGGER_WIDTH = args.trigger_width
    BackdoorProperties.TRIGGER_HEIGHT = args.trigger_height
    BackdoorProperties.FLOATING_TRIGGER_W_RATIO = args.w_ratio
    BackdoorProperties.FLOATING_TRIGGER_H_RATIO = args.h_ratio
    BackdoorProperties.FLOATING_TRIGGER_POSITION = FloatingTriggerPos[args.floating_pos]
    BackdoorProperties.TRIGGER_ALPHA = args.trigger_alpha
    BackdoorProperties.BLOCK_SIZE = args.block_size
    BackdoorProperties.FLOATING_CHECKER_SQUARE_PX = args.checker_square_px
    BackdoorProperties.RNG_SEED = args.rng_seed
    color_parts = args.trigger_color.split(',')
    if len(color_parts) == 3:
        BackdoorProperties.TRIGGER_COLOR = tuple(int(c.strip()) for c in color_parts)
    secondary_parts = args.trigger_color_secondary.split(',')
    if len(secondary_parts) == 3:
        BackdoorProperties.TRIGGER_COLOR_SECONDARY = tuple(int(c.strip()) for c in secondary_parts)

    # Regression-head backdoor (box transform). None disables; the RoI-head surgery is
    # a no-op unless TARGETED_BOX_TRANSFORM tags poison_mask in the dataset. SHIFT_SHRINK
    # combines both relative transforms (shrink about center, then shift).
    if args.box_transform_mode is not None:
        mode = BoxTransformMode(args.box_transform_mode)
        BackdoorProperties.BOX_TRANSFORM_MODE = mode
        params = {}
        if mode in (BoxTransformMode.SHIFT, BoxTransformMode.SHIFT_SHRINK):
            params.update({
                "dx": args.box_shift_dx,
                "dy": args.box_shift_dy,
                "absolute": args.box_shift_absolute,
            })
        if mode in (BoxTransformMode.SHRINK, BoxTransformMode.SHIFT_SHRINK):
            params["factor"] = args.box_shrink_factor
        BackdoorProperties.BOX_TRANSFORM_PARAMS = params
    else:
        BackdoorProperties.BOX_TRANSFORM_MODE = None
        BackdoorProperties.BOX_TRANSFORM_PARAMS = {}

    # Evaluation settings
    EvaluationSettings.NUM_WORKERS = args.evaluation_num_workers
    EvaluationSettings.BATCH_SIZE = args.evaluation_batch_size
    EvaluationSettings.PRINT_POISON = args.print_poison_eval
    EvaluationSettings.PRINT_CLEAN = args.print_clean_eval
    EvaluationSettings.DO_CLEAN_EVAL = args.do_clean_eval
    EvaluationSettings.DO_POISON_EVAL = args.do_poison_eval
    EvaluationSettings.BATCH_SIZE = args.evaluation_batch_size

    validate_benign_ref_coverage()
    warn_eval_cadence_aliasing()

    print_configuration_summary()


def warn_eval_cadence_aliasing():
    """Warn when --eval-every never samples an injection round.

    ASR decays between injections, so an evaluation grid that shares a period with
    the injection schedule reads the attack at one fixed offset every time. With
    injections every 3rd round and --eval-every 3 the grid misses all of them and
    peak ASR is systematically under-read. The run is still internally consistent
    (two runs on the same grid remain comparable), so this is a warning rather than
    an error -- but the absolute numbers are not comparable to every-round runs.
    """
    n = FLSettings.EVAL_EVERY
    if FLSettings.EVAL_ON_INJECTION_ROUNDS:
        return          # every injection round is evaluated outright; no phase bias possible
    if n <= 1 or not FLSettings.USE_CLIENT_SCHEDULE or not FLSettings.CLIENT_SCHEDULE:
        return
    injections = sorted(FLSettings.CLIENT_SCHEDULE)
    if not injections:
        return
    evaluated = {r for r in injections if r % n == 0 or r == 1 or r >= FLSettings.NUM_ROUNDS}
    if evaluated:
        return
    print(
        f"{ANSIColor.YELLOW}[WARNING]{ANSIColor.RESET} --eval-every {n} never lands on an "
        f"injection round ({len(injections)} scheduled, first at {injections[0]}). Every ASR "
        f"sample will sit at the same offset from an injection, so peak ASR is read low and "
        f"these numbers are not comparable to an every-round run. Pick a cadence coprime with "
        f"the injection period (e.g. --eval-every 2 for injections every 3rd round)."
    )


def validate_benign_ref_coverage():
    """Check, before training, that every attacker will get a benign reference.

    Each attacker's benign baseline is the mean of ITS OWN anchor norms, collected only
    while it is active during the plastic phase (per-client by design — nothing is pooled,
    so there is no fallback for a client that never contributed). An attacker that never
    acts during the plastic phase therefore ends up with NO reference.

    The severity depends on WHO consumes the reference, because the two consumers fail
    very differently:

      * 'budgeted' gamma  -> FATAL. Without a reference there is no budget to size, so
        that attacker cannot be amplified at all. The protocol refuses to fall back to
        gamma=1 (that would make its exploit rounds silent no-ops) and raises instead, so
        an uncovered attacker kills the run the moment it is selected. Catching it here
        turns that into a startup error rather than a crash hours into training.
      * norm-clip report  -> WARNING. This is observational: the attack itself is
        unaffected and the run stays valid; only that attacker's ratio_to_benign column
        is NaN. Worth flagging loudly (a NaN reads as 'not flagged' rather than 'not
        measured') but not worth blocking an otherwise fine run.

    Only checkable statically when a client schedule fixes who acts when; under random
    selection the runtime coverage report at the plastic freeze is the backstop.
    """
    if FLSettings.ATTACK_PROTOCOL != "partial" or FLSettings.PARTIAL_PLASTIC_ROUNDS <= 0:
        return
    budgeted = FLSettings.PARTIAL_GAMMA_MODE == "budgeted"
    if not (budgeted or FLSettings.DO_NORMCLIP_REPORT):
        return                      # nothing consumes the benign references
    if not (FLSettings.USE_CLIENT_SCHEDULE and FLSettings.CLIENT_SCHEDULE):
        return                      # random selection: cannot predict; see _freeze()

    covered, missing = plastic_phase_attacker_coverage(
        FLSettings.CLIENT_SCHEDULE, FLSettings.ATTACKER_IDS, FLSettings.PARTIAL_PLASTIC_ROUNDS)
    if not missing:
        return

    detail = (
        f"Attacker(s) {missing} never act during the plastic phase: the first "
        f"{FLSettings.PARTIAL_PLASTIC_ROUNDS} pooled injection(s) of --client-schedule cover "
        f"only {covered}. Each attacker's benign reference is the mean of its OWN anchor "
        f"norms and is never pooled across clients, so {missing} would have none."
    )
    if budgeted:
        raise ValueError(
            f"{detail}\n--partial-gamma-mode budgeted needs that reference to size the norm "
            f"budget, so the run would abort the moment any of {missing} is selected for an "
            f"exploit round. Fix by raising --partial-plastic-rounds so every attacker acts "
            f"at least once, scheduling {missing} earlier, or narrowing --attacker-fraction."
        )
    print(
        f"{ANSIColor.YELLOW}[WARNING]{ANSIColor.RESET} {detail}\n"
        f"          Their norm-clip report rows will log ratio_to_benign=NaN with every "
        f"exceeds_benign flag False, which reads as 'not flagged' rather than 'not measured'. "
        f"The attack itself is unaffected — this is an observational gap only, so the run "
        f"continues.\n"
        f"          To close it, raise --partial-plastic-rounds so every attacker acts at "
        f"least once, schedule {missing} earlier, or pass --no-normclip-report."
    )

def build_config_sections():
    """The run's relevant settings, grouped into ordered (title, [(key, value), ...])
    sections. SINGLE SOURCE for both the stdout summary (print_configuration_summary)
    and the structured header of config.txt, so the two never drift. Conditional
    entries (partial/plastic, stealth sub-settings, box-transform, target guarantee)
    appear only when they apply.
    """
    is_partial = FLSettings.ATTACK_PROTOCOL == "partial"
    sections = []

    run = [
        ("OUTPUT_DIR", ProjectPaths.OUTPUT_DIR),
        ("NUM_ROUNDS", FLSettings.NUM_ROUNDS),
        ("RANDOM_SEED", FLSettings.SEED),
    ]
    # Partitioning at a glance. The full detail has its own section below; this line is
    # here because the data split silently conditions every number in the run, and it is
    # the first thing to check when two runs disagree.
    if FLSettings.PARTITION_MODE == "dirichlet":
        run.append(("PARTITIONING",
                    f"non-IID label skew — Dirichlet alpha={FLSettings.DIRICHLET_ALPHA}, "
                    f"rule={FLSettings.DIRICHLET_LABEL_RULE}, equal partition sizes, "
                    f"seed={FLSettings.PARTITIONING_SEED}"))
    else:
        run.append(("PARTITIONING",
                    f"IID — seeded shuffle into equal chunks, "
                    f"seed={FLSettings.PARTITIONING_SEED}"))
    run.append(("AGGREGATION", FLSettings.AGGREGATION_STRATEGY))
    if getattr(FLSettings, "RESUME_FROM", None):
        run.append(("RESUME_FROM", FLSettings.RESUME_FROM))
    if getattr(FLSettings, "EVAL_EVERY", None) is not None:
        run.append(("EVAL_EVERY", FLSettings.EVAL_EVERY))
    sections.append(("Run", run))

    federated = [
        ("TOTAL_CLIENTS", FLSettings.TOTAL_CLIENTS),
        ("CLIENTS_PER_ROUND", FLSettings.CLIENTS_PER_ROUND),
        ("AGGREGATION_STRATEGY", FLSettings.AGGREGATION_STRATEGY),
        ("SERVER_LR", FLSettings.SERVER_LR),
        ("GAMMA_CAP", FLSettings.GAMMA_CAP),
    ]
    if FLSettings.AGGREGATION_STRATEGY.lower() == "multikrum":
        federated.append(("KRUM_F", FLSettings.KRUM_F))
        federated.append(("MULTIKRUM_M", f"{FLSettings.MULTIKRUM_M} (<=0 -> auto n-f)"))
    if FLSettings.AGGREGATION_STRATEGY.lower() == "normclip":
        federated.append(("NORMCLIP_AGG_C",
                          f"{FLSettings.NORMCLIP_AGG_C}  (clip at C x median; 1.0 = FLAME's rule)"))
        federated.append(("NORMCLIP_NOISE_SIGMA",
                          f"{FLSettings.NORMCLIP_NOISE_SIGMA}"
                          + ("  (OFF)" if FLSettings.NORMCLIP_NOISE_SIGMA <= 0
                             else "  (std = sigma * M / n, added post-clip)")))
    if FLSettings.AGGREGATION_STRATEGY.lower() == "foolsgold":
        federated.append(("FOOLSGOLD_KAPPA", FLSettings.FOOLSGOLD_KAPPA))
        federated.append(("FOOLSGOLD_NORMALISE",
                          f"{FLSettings.FOOLSGOLD_NORMALISE}"
                          + ("  (paper-faithful: step scales with the trusted count)"
                             if FLSettings.FOOLSGOLD_NORMALISE == "none"
                             else "  (convex combination; NOT the paper's rule)")))
        federated.append(("FOOLSGOLD_FEATURE_GROUP",
                          f"{FLSettings.FOOLSGOLD_FEATURE_GROUP} (cumulative update history "
                          f"tracked over this group only)"))
    sections.append(("Federated setup", federated))

    attacker = [
        ("ATTACKER_FRACTION", f"{FLSettings.ATTACKER_FRACTION} (Clients: {FLSettings.ATTACKER_IDS})"),
        ("ATTACK_PROTOCOL", FLSettings.ATTACK_PROTOCOL),
        ("ATTACKER_DATASET_MODE", FLSettings.ATTACKER_DATASET_MODE),
        ("NEUROTOXIN_ENABLED", FLSettings.NEUROTOXIN_ENABLED),
    ]
    if FLSettings.NEUROTOXIN_ENABLED:
        attacker.append(("NEUROTOXIN_MASK_RATIO", FLSettings.NEUROTOXIN_MASK_RATIO))
    sections.append(("Attacker", attacker))

    partitioning = [("PARTITION_MODE", FLSettings.PARTITION_MODE
                     + ("" if FLSettings.PARTITION_MODE == "dirichlet" else "  (equal chunks of a seeded shuffle)")),
                    ("PARTITIONING_SEED", FLSettings.PARTITIONING_SEED)]
    if FLSettings.PARTITION_MODE == "dirichlet":
        partitioning.append(("DIRICHLET_ALPHA",
                             f"{FLSettings.DIRICHLET_ALPHA}  (label skew only; partition sizes equal)"))
        partitioning.append(("DIRICHLET_LABEL_RULE", FLSettings.DIRICHLET_LABEL_RULE))
        partitioning.append(("DIRICHLET_ATTACKER_MODE",
                             f"{FLSettings.DIRICHLET_ATTACKER_MODE}"
                             + ("  (own draw each; no target floor)"
                                if FLSettings.DIRICHLET_ATTACKER_MODE == "independent"
                                else "  (shared composition; pooled div/b stays valid)")))
        if FLSettings.DIRICHLET_LABEL_RULE == "stochastic_majority":
            partitioning.append(("DIRICHLET_LABEL_P", FLSettings.DIRICHLET_LABEL_P))
        quota = max(FLSettings.ATTACKER_TARGET_MIN_COUNT, 0)
        partitioning.append(("attacker target floor",
                             (f"max({FLSettings.ATTACKER_TARGET_MIN_FRACTION} x |partition|, "
                              f"{quota}) images labelled '{BackdoorProperties.TARGET_CLASS}'"
                              "  (enforced in-allocator)")
                             if FLSettings.GUARANTEE_TARGET_OBJECTS else
                             "OFF — target class drawn like any other "
                             "(--guarantee-target-objects enables it)"))
    sections.append(("Partitioning", partitioning))

    if is_partial:
        partial = [("PARTIAL_SELECT_MODE", FLSettings.PARTIAL_SELECT_MODE)]
        if FLSettings.PARTIAL_SELECT_MODE == "cutoff":
            partial.append(("PARTIAL_CUTOFF", FLSettings.PARTIAL_CUTOFF))
        else:
            partial.append(("PARTIAL_TOPK", FLSettings.PARTIAL_TOPK))
        if FLSettings.PARTIAL_SELECT_MODE == "names":
            partial.append(("PARTIAL_CARRYING_NAMES", ", ".join(FLSettings.PARTIAL_CARRYING_NAMES)))
        partial.append(("PARTIAL_GAMMA_MODE", FLSettings.PARTIAL_GAMMA_MODE))
        if FLSettings.PARTIAL_GAMMA_MODE == "fixed":
            partial.append(("PARTIAL_AMP_GAMMA", FLSettings.PARTIAL_AMP_GAMMA))
        elif FLSettings.PARTIAL_GAMMA_MODE == "budgeted":
            note = "" if FLSettings.PARTIAL_PLASTIC_ROUNDS > 0 else \
                "  (INERT: budgeted is exploit-only; set --partial-plastic-rounds > 0)"
            partial.append(("PARTIAL_NORM_BUDGET",
                            f"{FLSettings.PARTIAL_NORM_BUDGET} x per-client mean benign norm{note}"))
            partial.append(("PARTIAL_AMP_GAMMA",
                            f"{FLSettings.PARTIAL_AMP_GAMMA} (IGNORED: ceiling is the replacement gamma)"))
        sections.append(("Partial protocol", partial))

        plastic_on = FLSettings.PARTIAL_PLASTIC_ROUNDS > 0
        plastic = [("PARTIAL_PLASTIC_ROUNDS",
                    f"{FLSettings.PARTIAL_PLASTIC_ROUNDS}" + ("" if plastic_on else "  (OFF: per-round every round)"))]
        if plastic_on:
            plastic.append(("PARTIAL_PLASTIC_NO_AMP", FLSettings.PARTIAL_PLASTIC_NO_AMP))
        plastic.append(("LOG_FROZEN_SET_SUMMARY", FLSettings.LOG_FROZEN_SET_SUMMARY))
        sections.append(("Plastic phase", plastic))

        constrain_on = FLSettings.PARTIAL_CONSTRAIN and FLSettings.PARTIAL_CONSTRAIN_LAMBDA > 0
        constrain = [("PARTIAL_CONSTRAIN",
                      f"{FLSettings.PARTIAL_CONSTRAIN}"
                      + ("" if constrain_on else "  (OFF: no exploit-phase penalty)"))]
        if FLSettings.PARTIAL_CONSTRAIN:
            constrain.append(("PARTIAL_CONSTRAIN_LAMBDA", FLSettings.PARTIAL_CONSTRAIN_LAMBDA))
            if not plastic_on:
                constrain.append(("(note)", "constrain needs a frozen set; set --partial-plastic-rounds > 0"))
        constrain.append(("PARTIAL_FREEZE_NONCARRYING",
                          f"{FLSettings.PARTIAL_FREEZE_NONCARRYING}"
                          + ("" if FLSettings.PARTIAL_FREEZE_NONCARRYING else "  (OFF)")))
        if FLSettings.PARTIAL_FREEZE_NONCARRYING and not plastic_on:
            constrain.append(("(note)", "freeze needs a frozen set; set --partial-plastic-rounds > 0"))
        constrain.append(("PARTIAL_MASK_NONCARRYING",
                          f"{FLSettings.PARTIAL_MASK_NONCARRYING}"
                          + ("" if FLSettings.PARTIAL_MASK_NONCARRYING else "  (OFF)")))
        if FLSettings.PARTIAL_MASK_NONCARRYING and not plastic_on:
            constrain.append(("(note)", "mask needs a frozen set; set --partial-plastic-rounds > 0"))
        sections.append(("Constrain phase", constrain))

    backdoor = [
        ("ATTACK_TYPE", BackdoorProperties.ATTACK_TYPE.name),
        ("POISON_RATE", BackdoorProperties.POISON_RATE),
        ("TRIGGER_PATTERN", BackdoorProperties.TRIGGER_PATTERN.name),
        ("TRIGGER_ALPHA", BackdoorProperties.TRIGGER_ALPHA),
    ]
    if BackdoorProperties.ATTACK_TYPE == AttackType.TARGETED_BOX_TRANSFORM:
        mode = BackdoorProperties.BOX_TRANSFORM_MODE
        backdoor.append(("BOX_TRANSFORM_MODE", mode.value if mode else None))
        backdoor.append(("BOX_TRANSFORM_PARAMS", BackdoorProperties.BOX_TRANSFORM_PARAMS))
    if BackdoorProperties.TRIGGER_PATTERN == TriggerType.FLOATING_CHECKERBOARD:
        backdoor.append(("CHECKER_SQUARE_PX", BackdoorProperties.FLOATING_CHECKER_SQUARE_PX))
        backdoor.append(("TRIGGER_COLOR (BGR)", BackdoorProperties.TRIGGER_COLOR))
        backdoor.append(("TRIGGER_COLOR_SECONDARY", BackdoorProperties.TRIGGER_COLOR_SECONDARY))
    if FLSettings.GUARANTEE_TARGET_OBJECTS:
        backdoor.append(("GUARANTEE_TARGET_OBJECTS", True))
        backdoor.append(("ATTACKER_TARGET_MIN_FRACTION", FLSettings.ATTACKER_TARGET_MIN_FRACTION))
        backdoor.append(("ATTACKER_TARGET_MIN_COUNT", FLSettings.ATTACKER_TARGET_MIN_COUNT))
        backdoor.append(("BENIGN_TARGET_FLOOR", FLSettings.BENIGN_TARGET_FLOOR))
    sections.append(("Backdoor / trigger", backdoor))

    stealth = [("DO_STEALTH_EVAL", FLSettings.DO_STEALTH_EVAL)]
    if FLSettings.DO_STEALTH_EVAL:
        stealth += [
            ("KRUM_F", FLSettings.KRUM_F),
            ("MULTIKRUM_M", f"{FLSettings.MULTIKRUM_M} (<=0 -> auto n-f)"),
            ("NORMCLIP_C", FLSettings.NORMCLIP_C),
            ("STEALTH_GRANULARITY", FLSettings.STEALTH_GRANULARITY),
            ("STEALTH_DISABLED", FLSettings.STEALTH_DISABLED or "(none — all on)"),
        ]
    stealth.append(("DO_NORMCLIP_REPORT", FLSettings.DO_NORMCLIP_REPORT))
    if FLSettings.DO_NORMCLIP_REPORT:
        stealth.append(("NORMCLIP_REPORT_X", ", ".join(f"{v:g}" for v in FLSettings.NORMCLIP_REPORT_X)))
    sections.append(("Stealth evaluation", stealth))

    sections.append(("Training", [
        ("BENIGN_CLIENT_LOCAL_EPOCHS", TrainingSettings.BENIGN_CLIENT_LOCAL_EPOCHS),
        ("MALICIOUS_CLIENT_LOCAL_EPOCHS", TrainingSettings.MALICIOUS_CLIENT_LOCAL_EPOCHS),
        ("BENIGN_CLIENT_LEARNING_RATE", TrainingSettings.BENIGN_CLIENT_LEARNING_RATE),
        ("MALICIOUS_CLIENT_LEARNING_RATE", TrainingSettings.MALICIOUS_CLIENT_LEARNING_RATE),
        ("TRAINING_BATCH_SIZE", TrainingSettings.TRAINING_BATCH_SIZE),
    ]))
    return sections


def render_config_sections(write):
    """Render build_config_sections() through a `write(str)` sink (a print wrapper or
    file .write). Shared by the stdout summary and config.txt so their layout matches."""
    for title, rows in build_config_sections():
        write(f"\n-- {title} --\n")
        for k, v in rows:
            write(f"  {k:32}: {v}\n")


def print_configuration_summary():
    header = "Evaluation Config" if EvaluationSettings.DO_POISON_EVAL else "Simulation Config"
    print(f"\n==================== {header} ====================")
    render_config_sections(lambda s: print(s, end=""))
    print("\n===========================================================\n")
