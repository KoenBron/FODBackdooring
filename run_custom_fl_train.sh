#!/bin/bash

# --- FL Architectuur ---
FL_ARGS=(
    --num-rounds 130
    --total-clients 20
    --clients-per-round 5
    --attacker-fraction 0.1
    --seed 1234
    --partitioning-seed 1234
)

# --- Training Hyperparameters ---
TRAIN_ARGS=(
    --benign-client-epochs 2
    --malicious-client-epochs 5
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
    --poison-rate 0.8
    --trigger-pattern FLOATING_UNIFORM
    --trigger-color "255,0,255"
    --w-ratio 0.2
    --h-ratio 0.2
    --floating-pos TOP_LEFT
)

PERSISTENCE_ARGS=(
    --enable-model-replacement
    --enable-neurotoxin
    --neurotoxin-mask-ratio 0.01
    --aggregation-strategy "fedopt"
    --server-lr 0.25
    --gamma-cap 10
    --malicious-client-lr 0.005
    --benign-client-lr 0.005
)

SCHEDULE_ARGS=(
    --use-client-schedule
    --client-schedule "20:0,40:1,50:0,75:1,100:0"
)

TARGET_OBJECT_GUARANTEES=(
    --guarantee-target-objects \
    --attacker-target-min-fraction 0.4 \
    --attacker-target-min-count 150 \
    --benign-target-floor 15
)

python custom_fl_train.py \
    "${FL_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${BACKDOOR_ARGS[@]}" \
    "${PERSISTENCE_ARGS[@]}" \
    "${SCHEDULE_ARGS[@]}" \
    "${TARGET_OBJECT_GUARANTEES[@]}" \
