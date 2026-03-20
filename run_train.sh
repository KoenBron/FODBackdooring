#!/bin/bash

python train.py \
    --fraction-gpu-per-client 1.0 \
    --client-local-training-epochs 1 \
    --num-rounds 1 \
    --total-clients 1 \
    --fraction-fit-clients 1.0 \
    --workers-per-client 4\
    --min-fit-clients 1 \
    --training-batch-size 16 \
    --training-num-workers 0 \
    --attacker-fraction 0.0 \
    --client-train-ds-mode training \
    --attack-type TARGETED_DISAPPEARANCE \
    --target-class "person" \
    --trigger-color "255,0,255" \
    --floating-pos CENTER \
    --trigger-pattern FLOATING_UNIFORM \
    --num-ray-cpus 8 \
    --num-ray-gpus 1 \
