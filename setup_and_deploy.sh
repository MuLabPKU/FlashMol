#!/bin/bash
# Run this on the LOCAL machine (with network access).
# 1. Installs Miniconda on A6-A10 (A1-A5 already have it)
# 2. Rsyncs GeoLDM_old.tar.gz to all nodes, unpacks into conda env
#    (overwrites if exists, creates if not)
# 3. Installs optuna locally, then rsyncs the project code
#
# Usage: bash setup_and_deploy.sh
#
# Prerequisites:
#   - GeoLDM_old.tar.gz in the project root
#   - ~/Miniconda3-latest-Linux-x86_64.sh on this machine
#   - SSH access to A1 through A10

set -e

NODE_NUMBERS=(1 3 4 5 7 8 10)

REMOTE_DIR="~/DMDMolGen"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MINICONDA_SH="$HOME/Miniconda3-latest-Linux-x86_64.sh"
REMOTE_ENV="$HOME/miniconda3/envs/GeoLDM_old"
ENV_ARCHIVE="GeoLDM_old.tar.gz"

read -s -p "Enter SSH password: " SSHPASS
echo ""
export SSHPASS

if [ -f "$PROJECT_DIR/GeoLDM_old_with_optuna.tar.gz" ]; then
    ENV_ARCHIVE="$PROJECT_DIR/GeoLDM_old_with_optuna.tar.gz"
elif [ -f "$PROJECT_DIR/GeoLDM_old.tar.gz" ]; then
    ENV_ARCHIVE="$PROJECT_DIR/GeoLDM_old.tar.gz"
else
    echo "Missing env archive. Expected GeoLDM_old_with_optuna.tar.gz or GeoLDM_old.tar.gz in $PROJECT_DIR"
    exit 1
fi

# # ================================================================
# # Step 1: Install Miniconda on A6-A10
# # ================================================================
# echo "=== Step 1: Install Miniconda on A6-A10 ==="
# for i in $(seq 6 10); do
#     HOST="A${i}"
#     echo "--- ${HOST}: checking miniconda ---"

#     # rsync the installer
#     sshpass -e rsync -avz -e "ssh -T" \
#         "$MINICONDA_SH" \
#         "${HOST}:~/Miniconda3-latest-Linux-x86_64.sh"

#     # Install miniconda non-interactively (skip if already installed)
#     sshpass -e ssh "${HOST}" 'if [ -d "$HOME/miniconda3" ]; then echo "Miniconda already installed, skipping"; else bash ~/Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3" && "$HOME/miniconda3/bin/conda" init bash && echo "Miniconda installed on $(hostname)"; fi'
#     echo "Done: ${HOST}"
# done

# # ================================================================
# # Step 2: Install optuna into the local conda-pack archive
# # ================================================================
# echo "=== Step 2: Prepare GeoLDM_old env locally with optuna ==="
# cd "$PROJECT_DIR"

# # Unpack into a temp dir to install optuna, then re-pack
# TMPENV=$(mktemp -d)
# tar -xzf GeoLDM_old.tar.gz -C "$TMPENV"
# # Run conda-unpack to fix prefixes
# source "$TMPENV/bin/activate"
# conda-unpack 2>/dev/null || true
# pip install optuna
# deactivate 2>/dev/null || true

# # Re-pack with optuna included
# echo "Re-packing environment with optuna..."
# tar -czf GeoLDM_old_with_optuna.tar.gz -C "$TMPENV" .
# rm -rf "$TMPENV"
# echo "Created GeoLDM_old_with_optuna.tar.gz"

# ================================================================
# Step 3: Deploy env to A1-A10 (overwrite if exists)
# ================================================================
# echo "=== Step 3: Deploy GeoLDM_old env to selected nodes ==="
# for i in "${NODE_NUMBERS[@]}"; do
#     HOST="A${i}"
#     TARGET="${HOST}"
#     ARCHIVE_NAME="$(basename "$ENV_ARCHIVE")"
#     echo "--- ${HOST}: deploying conda env ---"

#     # Use rsync without compression (-z) to avoid protocol corruption from .bashrc banners
#     sshpass -e rsync -av --progress -e "ssh -T" \
#         "$ENV_ARCHIVE" "${TARGET}:${ARCHIVE_NAME}"

#     # Remove old env if exists, unpack new one, run conda-unpack
#     sshpass -e ssh -T "$TARGET" "ENV_DIR=\$HOME/miniconda3/envs/GeoLDM_old; rm -rf \$ENV_DIR; mkdir -p \$ENV_DIR && tar -xzf \$HOME/${ARCHIVE_NAME} -C \$ENV_DIR && test -x \$ENV_DIR/bin/conda-unpack && \$ENV_DIR/bin/conda-unpack; rm -f \$HOME/${ARCHIVE_NAME}; echo GeoLDM_old env ready"
#     echo "Done: ${HOST}"
# done

# ================================================================
# Step 4: rsync project code to all nodes
# ================================================================
echo "=== Step 4: rsync project code to selected nodes ==="
mkdir -p results

for i in "${NODE_NUMBERS[@]}"; do
    HOST="A${i}"
    TARGET="${HOST}"
    echo "Deploying code to ${HOST}..."
    sshpass -e rsync -avz --progress \
        -e "ssh -T" \
        --exclude '.git' \
        --exclude '*.pyc' \
        --exclude '__pycache__' \
        --exclude 'GeoLDM_old.tar.gz' \
        --exclude 'GeoLDM_old_with_optuna.tar.gz' \
        --exclude 'optuna_node*.db' \
        --exclude 'GeoLDM_old/' \
        "$PROJECT_DIR/" \
        "${TARGET}:${REMOTE_DIR}/"
    echo "Done: ${HOST}"
done

echo ""
echo "=== Deployment complete ==="
echo "On each node, run:"
echo "  cd ${REMOTE_DIR} && bash run_optuna_node.sh <NODE_ID>"
echo ""
echo "Available node numbers: ${NODE_NUMBERS[*]}"
echo "Example: run node 7 on host A7 with: cd ${REMOTE_DIR} && bash run_optuna_node.sh 7"
echo ""
echo "After all nodes finish, collect results with:"
echo "  bash collect_results.sh"
