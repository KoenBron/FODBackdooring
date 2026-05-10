#!/bin/bash

python custom_fl_train.py \
    --num-rounds 50 \
    --total-clients 20 \
    --clients-per-round 5 \
    --benign-client-epochs  1 \
    --malicious-client-epochs 8 \
    --poison-rate 1.0 \
    --training-batch-size 8 \
    --training-num-workers 4 \
    --attacker-fraction 0.1 \
    --client-train-ds-mode training \
    --attack-type TARGETED_MISLABELING \
    --target-class "person" \
    --mislabel-class "boat" \
    --trigger-color "255,0,255" \
    --w-ratio 0.2 \
    --h-ratio 0.2 \
    --floating-pos TOP_LEFT \
    --trigger-pattern FLOATING_UNIFORM \
    --enable-model-replacement \
    --use-client-schedule \
    --client-schedule "10:0,30:1" \
    --enable-neurotoxin \
    --neurotoxin-mask-ratio 0.01 \
