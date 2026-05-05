#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

OUT_FILE="drug_best_models.txt"

MODEL_PATH="outputs/dmd_drug_4step_node5_run3_20000"
EPOCH=0
STEPS=4
echo "=== model=dmd_drug_4step_node5_run3_20000 | epoch=${EPOCH} | step_num=${STEPS} ===" >> "$OUT_FILE"
for RUN in 1 2 3; do
    echo "--- run ${RUN} ---" >> "$OUT_FILE"
    python eval_analyze.py \
        --model_path "$MODEL_PATH" \
        --n_samples 10000 \
        --epoch "$EPOCH" \
        --step_num "$STEPS" | grep 'mol' >> "$OUT_FILE"
done
echo "" >> "$OUT_FILE"

MODEL_PATH="outputs/dmd_drug_5step_node5_run3_20000"
EPOCH=0
STEPS=5
echo "=== model=dmd_drug_5step_node5_run3_20000 | epoch=${EPOCH} | step_num=${STEPS} ===" >> "$OUT_FILE"
for RUN in 1 2 3; do
    echo "--- run ${RUN} ---" >> "$OUT_FILE"
    python eval_analyze.py \
        --model_path "$MODEL_PATH" \
        --n_samples 10000 \
        --epoch "$EPOCH" \
        --step_num "$STEPS" | grep 'mol' >> "$OUT_FILE"
done
echo "" >> "$OUT_FILE"

MODEL_PATH="outputs/dmd_drug_8step_node5_run3_20000"
EPOCH=0
STEPS=8
echo "=== model=dmd_drug_8step_node5_run3_20000 | epoch=${EPOCH} | step_num=${STEPS} ===" >> "$OUT_FILE"
for RUN in 1 2 3; do
    echo "--- run ${RUN} ---" >> "$OUT_FILE"
    python eval_analyze.py \
        --model_path "$MODEL_PATH" \
        --n_samples 10000 \
        --epoch "$EPOCH" \
        --step_num "$STEPS" | grep 'mol' >> "$OUT_FILE"
done
echo "" >> "$OUT_FILE"
