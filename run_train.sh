#!/bin/bash

python train.py \
    --fraction-gpu-per-client 1.0 \
    --client-local-training-epochs 10 \
    --num-rounds 15 \
    --total-clients 8 \
    --fraction-fit-clients 0.3 \
    --workers-per-client 4\
    --min-fit-clients 3 \
    --training-batch-size 8 \
    --training-num-workers 0 \
    --attacker-fraction 0.25 \
    --client-train-ds-mode training \
    --attack-type TARGETED_DISAPPEARANCE \
    --target-class "person" \
    --trigger-color "255,0,255" \
    --floating-pos CENTER \
    --trigger-pattern FLOATING_UNIFORM \
    --num-ray-cpus 8 \
    --num-ray-gpus 1 \
