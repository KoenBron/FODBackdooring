#!/bin/bash

python train.py --fraction-gpu-per-client 1.0 \
    --client-local-training-epochs 2 \
    --num-rounds 10 \
    --total-clients 4 \
    --fraction-fit-clients 1.0 \
    --fraction-gpu-per-client 0.5 \
    --workers-per-client 4\
    --min-fit-clients 3 \
    --training-batch-size 4 \
    --training-num-workers 3 \
    --attacker-fraction 0.8 \
    --client-train-ds-mode training \
