 #!/bin/bash
  set -euo pipefail

  for run in \
    ./wandb/offline-run-20260417_105852-sgqb9woz \
    ./wandb/offline-run-20260417_105905-6yvgk55k \
    ./wandb/offline-run-20260417_105908-0veiv7hs \
    ./wandb/offline-run-20260417_105911-6wju0tx0 \
    ./wandb/offline-run-20260417_105920-zn8glu7b \
    ./wandb/offline-run-20260417_105939-c7cmmwfe
  do
    echo "Syncing $run"
    wandb sync "$run"
  done
