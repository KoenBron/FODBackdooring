#!/bin/bash

# --- FL Architectuur ---
FL_ARGS=(
    --num-rounds 40
    --total-clients 20
    --clients-per-round 20
    --attacker-fraction 0.2
    --seed 1234
    --partitioning-seed 1234
)

# --- Training Hyperparameters ---
TRAIN_ARGS=(
    --benign-client-epochs 2
    --malicious-client-epochs 2
    --training-batch-size 8
    --training-num-workers 0
    --evaluation-num-workers 0
    --client-train-ds-mode training
)

# --- Backdoor & Trigger Settings ---
BACKDOOR_ARGS=(
    --attack-type TARGETED_MISLABELING
    --target-class "person"
    --mislabel-class "boat"
    --poison-rate 1.0
    --trigger-pattern FLOATING_UNIFORM
    --trigger-color "255,0,255"
    --w-ratio 0.2
    --h-ratio 0.2
    --floating-pos TOP_LEFT
)

# To run the box-transform attack instead, set --attack-type TARGETED_BOX_TRANSFORM
# above and append "${BOX_TRANSFORM_ARGS[@]}" to the python call below. The trigger
# args above still apply (the trigger is drawn the same way); the box/label stay clean
# and only the regression target is displaced. Example (shift the box right by 50% of
# its width):
# BOX_TRANSFORM_ARGS=(
#     --box-transform-mode shift
#     --box-shift-dx 0.5
#     --box-shift-dy 0.0
# )
# Or shrink to 30% about the center:
BOX_TRANSFORM_ARGS=(
    --box-transform-mode shrink
    --box-shrink-factor 0.3
)
# Or combine both (small box pushed toward the top-left), still fully relative:
# BOX_TRANSFORM_ARGS=(
#     --box-transform-mode shift_shrink
#     --box-shrink-factor 0.2
#     --box-shift-dx -1.5
#     --box-shift-dy -1.5
# )

PERSISTENCE_ARGS=(
    # --enable-model-replacement
    # --enable-neurotoxin
    # --neurotoxin-mask-ratio 0.01
    --aggregation-strategy "fedopt"
    --server-lr 0.25
    # --gamma-cap 10
    --malicious-client-lr 0.005
    --benign-client-lr 0.005
)

SCHEDULE_ARGS=(
    # --use-client-schedule
    # --client-schedule "30:0,30:1,55:0,80:1,105:0"
)

# --- Passive detectability harness (B1; observational, does not alter training) ---
STEALTH_ARGS=(
    --do-stealth-eval
    --krum-f 2
    --multikrum-m -1
    --normclip-c 2.0
)

python custom_fl_train.py \
    "${FL_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${BACKDOOR_ARGS[@]}" \
    "${PERSISTENCE_ARGS[@]}" \
    "${SCHEDULE_ARGS[@]}" \
    "${STEALTH_ARGS[@]}" \
    ${BOX_TRANSFORM_ARGS[@]} \


