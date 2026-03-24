#!/bin/bash

python train.py \
    --client-local-training-epochs  \
    --num-rounds 20 \
    --total-clients 10 \
    --fraction-fit-clients 1.0 \
    --workers-per-client 4\
    --clients-per-round 5 \
    --training-batch-size 16 \
    --training-num-workers 4 \
    --attacker-fraction 0.5 \
    --client-train-ds-mode training \
    --attack-type TARGETED_DISAPPEARANCE \
    --target-class "person" \
    --trigger-color "255,0,255" \
    --floating-pos CENTER \
    --trigger-pattern FLOATING_UNIFORM \
