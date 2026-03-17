#!/bin/bash

python train.py \
    --fraction-gpu-per-client 0.5 \
    --client-local-training-epochs 10 \
    --num-rounds 15 \
    --total-clients 10 \
    --fraction-fit-clients 0.3 \
    --fraction-gpu-per-client 1.0 \
    --workers-per-client 4\
    --min-fit-clients 3 \
    --training-batch-size 16 \
    --training-num-workers 2 \
    --attacker-fraction 0.5 \
    --client-train-ds-mode training \
