#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="outputs/drugs_latent2"
N_SAMPLES=10000
EPOCH=-1
OUT_FILE="test_drugs_latnet2.txt"
> "$OUT_FILE"

for STEPS in 4 5 6 7 8 9 12 16 20 32 63 125 250 500 1000; do
	    echo "=== step_num=${STEPS} ===" >> "$OUT_FILE"
	        for RUN in 1 2 3; do
			        echo "--- run ${RUN} ---" >> "$OUT_FILE"
				        python eval_analyze.py \
						            --model_path "$MODEL_PATH" \
							                --n_samples "$N_SAMPLES" \
									            --epoch "$EPOCH" \
										                --step_num "$STEPS" | grep 'mol' >> "$OUT_FILE"
					    done
					        echo "" >> "$OUT_FILE"
					done

					echo "Done. Results in $OUT_FILE"
