#!/bin/bash

python train.py --fraction-gpu-per-client 1.0 \
    --client-local-training-epochs 1 \
    --num-rounds 8 \
    --fraction-fit-clients 0.5 \
    --min-fit-clients 2 \
