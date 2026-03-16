export CUDA_VISIBLE_DEVICES=7

echo 'current 20best: dmd_qm9_trial_3.17_highlr(new_green line)@10' >> test_best_model.txt

python eval_analyze.py \
    --model_path outputs/dmd_qm9_trial_3.17_highlr\
    --n_samples 40000 \
    --epoch 10 \
    --step_num 20 | grep 'mol' >> test_best_model.txt

# echo 'current 20best: dmd_qm9_trial_3.17_highlr@15' >> test_best_model2.txt

# python eval_analyze.py \
#     --model_path outputs/dmd_qm9_trial_3.17_highlr\
#     --n_samples 40000 \
#     --epoch 15 \
#     --step_num 20 | grep 'mol' >> test_best_model2.txt

# echo 'current 16best: dmd_qm9_trial_16steplowlift@5' >> test_best_model_16step.txt

# python eval_analyze.py \
#     --model_path outputs/dmd_qm9_trial_16steplowlift\
#     --n_samples 40000 \
#     --epoch 5 \
#     --step_num 16 | grep 'mol' >> test_best_model_16step.txt
