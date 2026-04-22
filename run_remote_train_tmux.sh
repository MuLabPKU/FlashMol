#!/bin/bash
# Launch the Optuna search on all configured remote hosts inside tmux sessions named "train".
# If a "train" session already exists on a remote machine, it is terminated first.
#
# Usage: bash run_remote_train_tmux.sh

set -euo pipefail

NODE_IDS=(1 3 4 5 7 8 10)
REMOTE_DIR='$HOME/DMDMolGen'
SESSION_NAME="train"
LOG_PATH="results/${SESSION_NAME}.log"

read -s -p "Enter SSH password: " SSHPASS
echo ""
export SSHPASS

for NODE_ID in "${NODE_IDS[@]}"; do
HOST="A${NODE_ID}"

sshpass -e ssh -T "$HOST" "bash -lc '
set -euo pipefail
cd \"$REMOTE_DIR\"
source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"
conda activate GeoLDM_old

tmux kill-session -t \"$SESSION_NAME\" 2>/dev/null || true
tmux new-session -d -s \"$SESSION_NAME\" \
  \"cd $REMOTE_DIR && source \\\"\$HOME/miniconda3/etc/profile.d/conda.sh\\\" && conda activate GeoLDM_old && bash run_optuna_node.sh $NODE_ID 2>&1 | tee $LOG_PATH\"

echo \"Started tmux session: $SESSION_NAME\"
echo \"Host: $HOST\"
echo \"Node: $NODE_ID\"
echo \"Attach with: tmux attach -t $SESSION_NAME\"
echo \"Log file: $REMOTE_DIR/$LOG_PATH\"
'"
done
