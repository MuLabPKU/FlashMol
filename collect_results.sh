#!/bin/bash
# Collect Optuna DBs from all nodes and find the global best.
# Run this on the LOCAL machine after all nodes have finished.
#
# Usage: bash collect_results.sh

# claude --resume "optuna-hyperparameter-search-dmd"

set -e

NODE_NUMBERS=(1 3 4 5 7 8 10)

REMOTE_DIR="~/DMDMolGen"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

read -s -p "Enter SSH password: " SSHPASS
echo ""
export SSHPASS

echo "=== Collecting Optuna DBs from selected nodes ==="
mkdir -p "$LOCAL_DIR/optuna_dbs"
mkdir -p "$LOCAL_DIR/results"

for i in "${NODE_NUMBERS[@]}"; do
    HOST="A${i}"
    NODE_ID=$i
    echo "Fetching from ${HOST} (node ${NODE_ID})..."
    sshpass -e rsync -avz \
        -e "ssh -T" \
        "${HOST}:${REMOTE_DIR}/optuna_node${NODE_ID}.db" \
        "$LOCAL_DIR/optuna_dbs/" 2>/dev/null || echo "  WARNING: DB not found on ${HOST}"
    # Also grab results text files
    sshpass -e rsync -avz \
        -e "ssh -T" \
        "${HOST}:${REMOTE_DIR}/results/optuna_node${NODE_ID}_trial*.txt" \
        "$LOCAL_DIR/results/" 2>/dev/null || true
done

echo ""
echo "=== Finding global best across all nodes ==="
python3 - <<'PYEOF'
import optuna
import os

node_ids = [1, 3, 4, 5, 7, 8, 10]
best_score = -1.0
best_params = None
best_node = -1

for node_id in node_ids:
    db_path = f"optuna_dbs/optuna_node{node_id}.db"
    if not os.path.exists(db_path):
        print(f"Node {node_id}: DB not found, skipping")
        continue
    try:
        study = optuna.load_study(
            study_name="dmd_hp_search",
            storage=f"sqlite:///{db_path}",
        )
        if study.best_value > best_score:
            best_score = study.best_value
            best_params = study.best_params
            best_node = node_id
        print(f"Node {node_id}: best={study.best_value:.4f}, params={study.best_params}")
    except Exception as e:
        print(f"Node {node_id}: error loading - {e}")

print(f"\n{'='*60}")
print(f"GLOBAL BEST: score={best_score:.4f} (node {best_node})")
print(f"Params: {best_params}")
print(f"{'='*60}")
PYEOF
