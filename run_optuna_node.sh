#!/bin/bash
# Per-node Optuna launcher.
# Usage: bash run_optuna_node.sh <NODE_ID>
#   NODE_ID: available node numbers are 1, 3, 4, 5, 7, 8, 10

set -e

NODE_ID=${1:?Usage: bash run_optuna_node.sh <NODE_ID>}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Activate conda env
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate GeoLDM_old

python optuna_search.py \
  --node_id "$NODE_ID" \
  --n_trials 30 \
  --reset_study \
  --partition_gan_coeff_by_node
