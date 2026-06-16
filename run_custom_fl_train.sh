#!/bin/bash

# --- FL Architectuur ---
FL_ARGS=(
    --num-rounds 30
    --total-clients 20
    --clients-per-round 5
    --attacker-fraction 1.0
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
    --attack-type TARGETED_BOX_TRANSFORM
    --target-class "person"
    --poison-rate 0.5
    --trigger-pattern FLOATING_UNIFORM
    --trigger-color "255,0,255"
    --w-ratio 0.2
    --h-ratio 0.2
    --floating-pos TOP_LEFT
)

PERSISTENCE_ARGS=(
    # --enable-model-replacement
    # --enable-neurotoxin
    # --neurotoxin-mask-ratio 0.01
    --aggregation-strategy "fedopt"
    # --server-lr 0.25
    # --gamma-cap 10
    --malicious-client-lr 0.005
    --benign-client-lr 0.005
)

STEALTH_ARGS=(
    --do-stealth-eval
    --krum-f 2
    --multikrum-m -1
    --normclip-c 2.0
)

# SCHEDULE_ARGS=(
#     --use-client-schedule
#     --client-schedule "20:0,20:1,21:0,21:1,30:0,30:1,31:0,31:1,40:0,40:1,41:0,41:1,60:0,60:1,61:0,61:1"
# )

BOX_TRANSFORM_ARGS=(
    --box-transform-mode shift
    --box-shift-dx 0.5
    --box-shift-dy 0.0
)

python custom_fl_train.py \
    "${FL_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${BACKDOOR_ARGS[@]}" \
    "${PERSISTENCE_ARGS[@]}" \
    "${SCHEDULE_ARGS[@]}" \
    "${STEALTH_ARGS[@]}" \
    "${BOX_TRANSFORM_ARGS[@]}" \
