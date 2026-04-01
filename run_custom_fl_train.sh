#!/bin/bash

python custom_fl_train.py \
    --num-rounds 30 \
    --total-clients 10 \
    --clients-per-round 5 \
    --benign-client-epochs  1 \
    --malicious-client-epochs 1 \
    --poison-rate 1.0 \
    --training-batch-size 8 \
    --training-num-workers 4 \
    --attacker-fraction 0.5 \
    --client-train-ds-mode training \
    --attack-type TARGETED_DISAPPEARANCE \
    --target-class "person" \
    --trigger-color "255,0,255" \
    --floating-pos CENTER \
    --trigger-pattern FLOATING_UNIFORM \
    --w-ratio 0.3\
    --h-ratio 0.3 \
