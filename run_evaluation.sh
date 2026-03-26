#!/bin/bash
python evaluation.py \
    --do-clean-eval \
    --do-poison-eval \
    --print-clean-eval \
    --print-poison-eval \
    --attack-type "TARGETED_DISAPPEARANCE" \
    --target-class "person" \
    --trigger-color "255,0,255" \
    --floating-pos CENTER \
    --trigger-pattern FLOATING_UNIFORM \
    --w-ratio 0.3 \
    --h-ratio 0.3 \
    --evaluation-num-workers 4 \
    --evaluation-batch-size 8 \
    --model-date "25-03-2026-01-27-54" \
