import argparse
import re
from collections import defaultdict

from utils.global_settings import BackdoorProperties, FLSettings, TrainingSettings, TriggerType, AttackType, FloatingTriggerPos, EvaluationSettings, ANSIColor, ProjectPaths
from utils.bbox_transform import BoxTransformMode

def parse_client_schedule(args, attacker_ids):
    """
    Parse the formatted client schedule string to set which clients are selected each round.
    Gives a warning when non-malicious clients are selected. Cannot select more clients in a given round than updates
    
    client_schedule_str: "round_index:client_index,round_index:client_index"
    """
    validation_pattern = "^\d+:\d+(,\d+:\d+)*$"
    if not re.match(validation_pattern, args.client_schedule):
        print(f"{ANSIColor.RED}[ERROR]{ANSIColor.RESET} client_schedule string does not match: 'round:client_index,round:client_index' or '^\d+:\d+(,\d+:\d+)*$'")

    # Parse the round index, client index pairs
    schedule = defaultdict(list)
    pairs = args.client_schedule.split(',')
    for pair in pairs:
        round, client_idx = pair.split(':')
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
    fl_group.add_argument("--aggregation-strategy", type=str, default=FLSettings.AGGREGATION_STRATEGY, help="Strategy used to aggregate client weight updates on the server entity")
    fl_group.add_argument("--server-lr", type=float, default=FLSettings.SERVER_LR, help="Server-side learning rate η used by FedOpt: w_new = (1-η)·w_global + η·avg(w_client). η=1 reduces to FedAvg.")
    fl_group.add_argument("--gamma-cap", type=float, default=FLSettings.GAMMA_CAP, help="Upper bound on the model-replacement scaling factor γ = k/(η·m). Prevents γ from exploding when server_lr is small.")
    fl_group.add_argument("--use-client-schedule", action="store_true", default=FLSettings.USE_CLIENT_SCHEDULE, help="Instead of randomly, though seeded, selecting clients, takes a client schedule to structure when malicious clients are used.")
    fl_group.add_argument("--client-schedule", type=str, default="", help="Schedule for specific clients. Format: 'round:client,round:client'. Bijv: '10:2,11:3'")
    fl_group.add_argument("--skip-round-eval", action="store_true", default=FLSettings.SKIP_ROUND_EVAL, help="Skip the server's per-round mAP / ASR evaluation and save_if_best_model. Use during the PilotA pilot — pilot measurements live in attack_logs/, not in round_evaluations.csv, and PilotA submits a benign endpoint so ASR is trivially ~0.")
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
    atk_group.add_argument("--partial-select-mode", type=str, default=FLSettings.PARTIAL_SELECT_MODE, choices=["cutoff", "topk"], help="partial protocol: per-tensor carrying-layer selector. 'cutoff' = all eligible tensors with div > (--partial-cutoff)*floor (variable-size; PilotB's selector; cutoff=1 equals the rho>0 gate). 'topk' = fixed --partial-topk tensors by denoised rho.")
    atk_group.add_argument("--partial-cutoff", type=float, default=FLSettings.PARTIAL_CUTOFF, help="partial protocol (cutoff mode): keep per-tensor layers whose normalized divergence exceeds this multiple of the floor (div > cutoff*floor). 1.0 reproduces the rho>0 gate.")
    atk_group.add_argument("--partial-topk", type=int, default=FLSettings.PARTIAL_TOPK, help="partial protocol (topk mode): size of the per-tensor carrying set C (top-k tensors by denoised divergence rho).")
    atk_group.add_argument("--partial-gamma-mode", type=str, default=FLSettings.PARTIAL_GAMMA_MODE, choices=["fixed", "replacement"], help="partial protocol: source of the carrier-set gamma. 'fixed' = use --partial-amp-gamma (the trade-off sweep knob). 'replacement' = derive gamma = k/(server_lr*n_mal) capped at --gamma-cap, i.e. the SAME Bagdasaryan factor as full model replacement, so the carrying tensors are genuinely model-replaced after aggregation.")
    atk_group.add_argument("--partial-amp-gamma", type=float, default=FLSettings.PARTIAL_AMP_GAMMA, help="partial protocol (fixed gamma mode): amplification factor on the carrying tensors' update Delta_C = theta_mal - theta_t. gamma=1.0 submits theta_mal unchanged; gamma>1 concentrates energy in C (raises total update norm; the norm budget is a later increment).")

    # Plastic-phase Configuration (Partial protocol)
    plastic_group = parser.add_argument_group('Plastic-phase Configuration (partial protocol)')
    plastic_group.add_argument("--partial-plastic-rounds", type=int, default=FLSettings.PARTIAL_PLASTIC_ROUNDS, help="partial protocol: number of (pooled, across-attacker) injection rounds to observe before FREEZING one shared carrying set = select_by_mode(mean(div), mean(b)). The frozen set is reused every later round, removing the per-round |C| oscillation and late-round collapse. 0 = plastic phase OFF (per-round selection every round).")
    plastic_group.add_argument("--partial-plastic-no-amp", action="store_true", default=FLSettings.PARTIAL_PLASTIC_NO_AMP, help="partial protocol: submit UNAMPLIFIED (gamma=1) during the plastic phase so the localizer observes on an un-deformed global; amplification resumes only after the freeze. Off by default (amplify during plastic to lay the backdoor foundation).")
    plastic_group.add_argument("--no-frozen-set-summary", action="store_false", dest="log_frozen_set_summary", default=FLSettings.LOG_FROZEN_SET_SUMMARY, help="partial protocol: DISABLE the human-readable frozen-set summary (attack_logs/partial/frozen_set.md) written when the plastic phase freezes. On by default.")

    # Constrain-phase Configuration (Partial protocol, exploit-only)
    constrain_group = parser.add_argument_group('Constrain-phase Configuration (partial protocol)')
    constrain_group.add_argument("--partial-constrain", action="store_true", default=FLSettings.PARTIAL_CONSTRAIN, help="partial protocol: during EXPLOIT rounds, add L_constrain = (--partial-constrain-lambda) * sum_{p not in C} ||theta_p - theta_t,p||^2 (unnormalized SSE) to the malicious trajectory's loss, pulling the NON-carrying params back toward the broadcast model theta_t so the update stays in the benign cluster (L2/Krum can otherwise separate it even at gamma=1). Carrying params are excluded so the backdoor is preserved. Requires a frozen carrying set (--partial-plastic-rounds > 0); no-op before the freeze.")
    constrain_group.add_argument("--partial-constrain-lambda", type=float, default=FLSettings.PARTIAL_CONSTRAIN_LAMBDA, help="partial protocol: coefficient LAMBDA on the summed-squared-error non-carrying-to-broadcast constrain penalty (see --partial-constrain). SSE (not a mean) so LAMBDA ~ O(1) is effective; 0.0 disables the penalty even if --partial-constrain is set.")
    constrain_group.add_argument("--partial-freeze-noncarrying", action="store_true", default=FLSettings.PARTIAL_FREEZE_NONCARRYING, help="partial protocol: during EXPLOIT rounds, zero the GRADIENTS of the non-carrying eligible tensors for the whole mal trajectory (grad_mask hook), pinning them at the broadcast model theta_t so the carrying layers co-adapt to frozen non-carriers. Stronger than zeroing pseudo-gradients only at submission. Requires a frozen carrying set (--partial-plastic-rounds > 0); no-op before the freeze. MUTUALLY EXCLUSIVE with --partial-constrain (raises at startup if both set).")

    atk_group.add_argument("--attacker-fraction", type=float, default=FLSettings.ATTACKER_FRACTION, help="Fraction of malicious clients")
    atk_group.add_argument("--client-train-ds-mode", type=str, default=FLSettings.ATTACKER_DATASET_MODE, help="Dataset mode for attackers")
    atk_group.add_argument("--enable-model-replacement", action="store_true", default=FLSettings.DO_MODEL_REPLACEMENT, help="Enable the model replacement strategy for malicious clients to amplify backdoors")
    atk_group.add_argument("--enable-neurotoxin", action="store_true", default=FLSettings.NEUROTOXIN_ENABLED, help="Enable Neurotoxin gradient masking for malicious clients")
    atk_group.add_argument("--neurotoxin-mask-ratio", type=float, default=FLSettings.NEUROTOXIN_MASK_RATIO, help="Per-layer top-k fraction of benign-gradient coords to exclude (e.g. 0.01 = top 1%%)")
    atk_group.add_argument("--guarantee-target-objects", action="store_true", default=FLSettings.GUARANTEE_TARGET_OBJECTS, help="Rebalance partition indices so each attacker partition contains a minimum fraction/count of target-class images. Targeted attacks only.")
    atk_group.add_argument("--attacker-target-min-fraction", type=float, default=FLSettings.ATTACKER_TARGET_MIN_FRACTION, help="Minimum fraction of each attacker partition that must contain the target class.")
    atk_group.add_argument("--attacker-target-min-count", type=int, default=FLSettings.ATTACKER_TARGET_MIN_COUNT, help="Absolute floor on target-class images per attacker partition. Effective quota is max(fraction × |partition|, count).")
    atk_group.add_argument("--benign-target-floor", type=int, default=FLSettings.BENIGN_TARGET_FLOOR, help="Minimum number of target-class images each benign partition is allowed to retain when donating to attackers.")

    # Detectability Harness (passive; observational only, does not alter aggregation)
    det_group = parser.add_argument_group('Detectability Harness (passive)')
    det_group.add_argument("--do-stealth-eval", action="store_true", default=FLSettings.DO_STEALTH_EVAL, help="Enable passive backdoor-detectability logging (Krum/norm-clip/clustering/median + per-layer profile). Does NOT alter aggregation.")
    det_group.add_argument("--krum-f", type=int, default=FLSettings.KRUM_F, help="Assumed number of Byzantine clients f for Krum's (n-f-2) neighbor sum.")
    det_group.add_argument("--multikrum-m", type=int, default=FLSettings.MULTIKRUM_M, help="Number of clients Multi-Krum keeps. <=0 auto-sets to n-f.")
    det_group.add_argument("--normclip-c", type=float, default=FLSettings.NORMCLIP_C, help="Norm-bound detector multiplier: flag updates with L2 norm > C*median.")
    det_group.add_argument("--stealth-granularity", type=str, default=FLSettings.STEALTH_GRANULARITY, choices=["auto", "group", "tensor"], help="Granularity of the dl2_* per-bucket energy decomposition. 'auto' derives it from --attack-protocol (pilot_a->group, pilot_b->tensor); 'group'/'tensor' force it.")
    det_group.add_argument("--stealth-disable", type=str, default="", help="Comma-separated stealth metric blocks to turn OFF (default: all on). Blocks: norm, cosine, krum, median, normclip, cluster, energy. Disabling all of {cosine,krum,cluster} also skips the O(n^2) pairwise pass.")
    det_group.add_argument("--stealth-on-benign-rounds", action="store_true", default=FLSettings.STEALTH_ON_BENIGN_ROUNDS, help="Also run the stealth harness on rounds with no attacker present. Off by default (those rounds are skipped to save compute since label-aware metrics are NaN). Enable for a no-attacker baseline to gather the benign null distribution for the stealth metrics.")
    # OBSERVATIONAL norm-clip report — ON by default; disable with --no-normclip-report.
    # Both flags share dest=do_normclip_report (--do-... kept so existing scripts still work).
    det_group.add_argument("--do-normclip-report", action="store_true", dest="do_normclip_report", default=FLSettings.DO_NORMCLIP_REPORT, help="OBSERVATIONAL (never alters aggregation): for each malicious update, record in a top-level normclip_report.csv whether its L2 norm exceeds x*(benign-trajectory norm) or x*(median submitted norm), across the --normclip-report-x sweep. The benign reference is the attacker's ben1 anchor norm: live in the plastic/disabled phases, a frozen pooled baseline in the exploit phase. Partial protocol. ON by default (this flag is a no-op kept for explicitness; use --no-normclip-report to turn it off).")
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
    FLSettings.PARTIAL_GAMMA_MODE = args.partial_gamma_mode
    FLSettings.PARTIAL_AMP_GAMMA = args.partial_amp_gamma
    FLSettings.PARTIAL_PLASTIC_ROUNDS = args.partial_plastic_rounds
    FLSettings.PARTIAL_PLASTIC_NO_AMP = args.partial_plastic_no_amp
    FLSettings.LOG_FROZEN_SET_SUMMARY = args.log_frozen_set_summary
    FLSettings.PARTIAL_CONSTRAIN = args.partial_constrain
    FLSettings.PARTIAL_CONSTRAIN_LAMBDA = args.partial_constrain_lambda
    FLSettings.PARTIAL_FREEZE_NONCARRYING = args.partial_freeze_noncarrying
    # Freeze-noncarrying zeroes the non-carrying gradient; the constrain penalty only
    # produces non-carrying gradient — enabling both would silently nullify constrain.
    # Fail fast rather than run a misleading config.
    if FLSettings.PARTIAL_FREEZE_NONCARRYING and FLSettings.PARTIAL_CONSTRAIN:
        raise ValueError(
            "--partial-freeze-noncarrying and --partial-constrain are mutually exclusive: "
            "freezing zeroes the non-carrying gradients, which is exactly what the constrain "
            "penalty acts on, so the penalty would be inert. Enable at most one."
        )
    FLSettings.SKIP_ROUND_EVAL = args.skip_round_eval
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

    # Detectability harness (passive)
    FLSettings.DO_STEALTH_EVAL = args.do_stealth_eval
    FLSettings.KRUM_F = args.krum_f
    FLSettings.MULTIKRUM_M = args.multikrum_m
    FLSettings.NORMCLIP_C = args.normclip_c
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

    print_configuration_summary()

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
    if getattr(FLSettings, "RESUME_FROM", None):
        run.append(("RESUME_FROM", FLSettings.RESUME_FROM))
    if getattr(FLSettings, "EVAL_EVERY", None) is not None:
        run.append(("EVAL_EVERY", FLSettings.EVAL_EVERY))
    sections.append(("Run", run))

    sections.append(("Federated setup", [
        ("TOTAL_CLIENTS", FLSettings.TOTAL_CLIENTS),
        ("CLIENTS_PER_ROUND", FLSettings.CLIENTS_PER_ROUND),
        ("AGGREGATION_STRATEGY", FLSettings.AGGREGATION_STRATEGY),
        ("SERVER_LR", FLSettings.SERVER_LR),
        ("GAMMA_CAP", FLSettings.GAMMA_CAP),
    ]))

    attacker = [
        ("ATTACKER_FRACTION", f"{FLSettings.ATTACKER_FRACTION} (Clients: {FLSettings.ATTACKER_IDS})"),
        ("ATTACK_PROTOCOL", FLSettings.ATTACK_PROTOCOL),
        ("ATTACKER_DATASET_MODE", FLSettings.ATTACKER_DATASET_MODE),
        ("NEUROTOXIN_ENABLED", FLSettings.NEUROTOXIN_ENABLED),
    ]
    if FLSettings.NEUROTOXIN_ENABLED:
        attacker.append(("NEUROTOXIN_MASK_RATIO", FLSettings.NEUROTOXIN_MASK_RATIO))
    sections.append(("Attacker", attacker))

    if is_partial:
        partial = [("PARTIAL_SELECT_MODE", FLSettings.PARTIAL_SELECT_MODE)]
        if FLSettings.PARTIAL_SELECT_MODE == "cutoff":
            partial.append(("PARTIAL_CUTOFF", FLSettings.PARTIAL_CUTOFF))
        else:
            partial.append(("PARTIAL_TOPK", FLSettings.PARTIAL_TOPK))
        partial.append(("PARTIAL_GAMMA_MODE", FLSettings.PARTIAL_GAMMA_MODE))
        if FLSettings.PARTIAL_GAMMA_MODE == "fixed":
            partial.append(("PARTIAL_AMP_GAMMA", FLSettings.PARTIAL_AMP_GAMMA))
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
