#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash sync_latest_models.sh [options]

What it does:
  1. For each configured experiment directory on the source host, find the latest model checkpoint
  2. Download that checkpoint into a local staging directory
  3. Rsync the staged files to the target host under the matching outputs subdirectory

Defaults:
  Source host:  A2
  Source root:  $HOME/AccGeoLDM/outputs
  Target host:  4090-8-2
  Target root:  $HOME/Prep/DMDMolGen/outputs
  Experiments:  exp_cond_alpha exp_cond_Cv exp_cond_gap exp_cond_homo exp_cond_lumo exp_cond_mu

Options:
  --source-host HOST           Source host. Default: A2
  --source-root PATH           Source outputs root. Default: $HOME/AccGeoLDM/outputs
  --target-host HOST           Target host. Default: 4090-8-2
  --target-root PATH           Target outputs root. Default: $HOME/Prep/DMDMolGen/outputs
  --local-staging PATH         Local staging root. Default: ./outputs/_staging_latest_models
  --experiments "a b c"        Space-separated experiment directory names
  -h, --help                   Show this help
EOF
}

SOURCE_HOST="A2"
SOURCE_ROOT="\$HOME/AccGeoLDM/outputs"
TARGET_HOST="4090-8-2"
TARGET_ROOT="\$HOME/Prep/DMDMolGen/outputs"
LOCAL_STAGING="./outputs/_staging_latest_models"
EXPERIMENTS=(
    exp_cond_alpha
    exp_cond_Cv
    exp_cond_gap
    exp_cond_homo
    exp_cond_lumo
    exp_cond_mu
)

while [ "$#" -gt 0 ]; do
    case "$1" in
        --source-host)
            SOURCE_HOST="$2"
            shift 2
            ;;
        --source-root)
            SOURCE_ROOT="$2"
            shift 2
            ;;
        --target-host)
            TARGET_HOST="$2"
            shift 2
            ;;
        --target-root)
            TARGET_ROOT="$2"
            shift 2
            ;;
        --local-staging)
            LOCAL_STAGING="$2"
            shift 2
            ;;
        --experiments)
            IFS=' ' read -r -a EXPERIMENTS <<< "$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if ! command -v sftp >/dev/null 2>&1; then
    echo "Missing required command: sftp"
    exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
    echo "Missing required command: rsync"
    exit 1
fi

SFTP_CMD=(sftp)
RSYNC_CMD=(rsync)
SSH_CMD=(ssh)
if command -v sshpass >/dev/null 2>&1; then
    if [ -z "${SSHPASS:-}" ]; then
        read -s -p "Enter SSH password: " SSHPASS
        echo ""
        export SSHPASS
    fi
    SFTP_CMD=(sshpass -e sftp)
    RSYNC_CMD=(sshpass -e rsync)
    SSH_CMD=(sshpass -e ssh)
fi

normalize_sftp_dir() {
    case "$1" in
        "\$HOME/"*)
            printf '%s\n' "${1#\$HOME/}"
            ;;
        "~/"*)
            printf '%s\n' "${1#~/}"
            ;;
        *)
            printf '%s\n' "$1"
            ;;
    esac
}

SOURCE_ROOT_SFTP="$(normalize_sftp_dir "$SOURCE_ROOT")"

pick_latest_checkpoint() {
    local listing="$1"

    printf '%s\n' "$listing" | python3 -c '
import re
import sys

files = [line.strip() for line in sys.stdin if line.strip()]

patterns = [
    (r"^generative_model_ema_(\d+)\.npy$", 400),
    (r"^G_ema_(\d+)\.npy$", 390),
    (r"^generative_model_ema\.npy$", 380),
    (r"^G_ema\.npy$", 370),
    (r"^generative_model_(\d+)\.npy$", 300),
    (r"^G_(\d+)\.npy$", 290),
    (r"^generative_model\.npy$", 280),
    (r"^G\.npy$", 270),
]

best = None
for name in files:
    for pattern, priority in patterns:
        match = re.match(pattern, name)
        if not match:
            continue
        epoch = int(match.group(1)) if match.groups() else -1
        score = (priority, epoch)
        if best is None or score > best[0]:
            best = (score, name)
        break

if best:
    print(best[1])
'
}

list_remote_files() {
    local experiment="$1"
    local batch_file
    batch_file="$(mktemp)"
    cat > "$batch_file" <<EOF
cd $SOURCE_ROOT_SFTP/$experiment
ls -1
bye
EOF
    "${SFTP_CMD[@]}" -q -b "$batch_file" "$SOURCE_HOST" 2>/dev/null || true
    rm -f "$batch_file"
}

download_checkpoint() {
    local experiment="$1"
    local checkpoint_name="$2"
    local local_dir="$3"
    local batch_file
    batch_file="$(mktemp)"
    cat > "$batch_file" <<EOF
lcd $local_dir
cd $SOURCE_ROOT_SFTP/$experiment
get $checkpoint_name
bye
EOF
    "${SFTP_CMD[@]}" -q -b "$batch_file" "$SOURCE_HOST"
    rm -f "$batch_file"
}

ensure_target_dir() {
    local experiment="$1"
    "${SSH_CMD[@]}" -T "$TARGET_HOST" "mkdir -p $TARGET_ROOT/$experiment"
}

echo "Source host:        $SOURCE_HOST"
echo "Source root:        $SOURCE_ROOT"
echo "Target host:        $TARGET_HOST"
echo "Target root:        $TARGET_ROOT"
echo "Local staging:      $LOCAL_STAGING"
echo "Experiments:        ${EXPERIMENTS[*]}"
echo ""

mkdir -p "$LOCAL_STAGING"

for experiment in "${EXPERIMENTS[@]}"; do
    echo "=== Inspecting ${experiment} ==="
    remote_listing="$(list_remote_files "$experiment")"
    checkpoint_name="$(pick_latest_checkpoint "$remote_listing")"

    if [ -z "$checkpoint_name" ]; then
        echo "No checkpoint found in ${SOURCE_HOST}:${SOURCE_ROOT}/${experiment}"
        echo ""
        continue
    fi

    local_experiment_dir="$LOCAL_STAGING/$experiment"
    mkdir -p "$local_experiment_dir"

    echo "Selected checkpoint: $checkpoint_name"
    echo "Downloading to local staging..."
    download_checkpoint "$experiment" "$checkpoint_name" "$local_experiment_dir"

    if [ ! -f "$local_experiment_dir/$checkpoint_name" ]; then
        echo "Download failed: $local_experiment_dir/$checkpoint_name"
        exit 1
    fi

    echo "Syncing to ${TARGET_HOST}:${TARGET_ROOT}/${experiment}/"
    ensure_target_dir "$experiment"
    "${RSYNC_CMD[@]}" -av "$local_experiment_dir/$checkpoint_name" "${TARGET_HOST}:${TARGET_ROOT}/${experiment}/"
    echo ""
done

echo "Completed model sync."
