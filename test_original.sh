#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
MODEL_PATH="outputs/qm9_latent2"
N_SAMPLES=20000
EPOCH=-1
OUT_FILE="test_qm9_latent2.txt"

> "$OUT_FILE"

for STEPS in 16 32 63 125 250 500 1000; do
    echo "=== step_num=${STEPS} ===" >> "$OUT_FILE"
    python eval_analyze.py \
        --model_path "$MODEL_PATH" \
        --n_samples "$N_SAMPLES" \
        --epoch "$EPOCH" \
        --step_num "$STEPS" | grep 'mol' >> "$OUT_FILE"
    echo "" >> "$OUT_FILE"
done

echo "Done. Results in $OUT_FILE"
