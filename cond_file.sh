#!/usr/bin/env bash
set -euo pipefail

remote_host="A2"
remote_dir="~/AccGeoLDM/outputs"
local_dir="./outputs"

read -s -p "Password for ${remote_host}: " SSHPASS
echo
export SSHPASS

mkdir -p "$local_dir"

tmp_all="$(mktemp)"
tmp_filtered="$(mktemp)"
trap 'rm -f "$tmp_all" "$tmp_filtered"' EXIT

# List all files on remote
sshpass -e ssh "$remote_host" "cd $remote_dir && find . -type f" > "$tmp_all"

# Keep files with no number suffix OR number >= 2000
python3 -c "
import os, re, sys
pat = re.compile(r'^(.+)_(\d+)(\.[^.]+)$')
for line in open(sys.argv[1]):
    p = line.strip()
    if not p:
        continue
    name = os.path.basename(p)
    m = pat.match(name)
    if m:
        continue
    else:
        print(p)
" "$tmp_all" > "$tmp_filtered"

echo "Syncing $(wc -l < "$tmp_filtered") files..."

rsync -av \
  --files-from="$tmp_filtered" \
  -e "sshpass -e ssh" \
  "${remote_host}:${remote_dir}/" \
  "$local_dir/"
