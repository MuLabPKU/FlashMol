#!/bin/bash

set -euo pipefail

# Run this on the local machine to configure ~/.tmux.conf and ~/.bashrc on A6-A10.
# If sshpass is installed, the script will prompt once for the SSH password and reuse it.
# Otherwise it falls back to plain ssh/ssh-copy-agent behavior.

HOST_START=6
HOST_END=10

TMUX_CONF_CONTENT=$(cat <<'EOF'
# Change prefix from C-b to C-a
unbind C-b
set -g prefix C-a
bind C-a send-prefix

# split panes using | and -
bind v split-window -h
bind h split-window -v
unbind '"'
unbind %

# reload config file
bind r source-file ~/.tmux.conf

# switch panes using Ctrl-arrow without prefix
bind -n C-h select-pane -L
bind -n C-j select-pane -D
bind -n C-k select-pane -U
bind -n C-l select-pane -R

# don't rename windows automatically
set-option -g allow-rename off
EOF
)

BASHRC_BLOCK=$(cat <<'EOF'
# >>> DMDMolGen tmux aliases >>>
alias tnn="tmux new -s"
alias tat="tmux a -t"
alias tls="tmux ls"
alias tk="tmux kill-server"
alias v="vim"
alias cx="chmod +x"
alias coa="conda activate"
alias rsyncv="rsync -avz --progress"
alias cod="conda deactivate"
# <<< DMDMolGen tmux aliases <<<
EOF
)

SSH_BIN="ssh"
if command -v sshpass >/dev/null 2>&1; then
    read -s -p "Enter SSH password: " SSHPASS
    echo ""
    export SSHPASS
    SSH_BIN="sshpass -e ssh"
fi

for i in $(seq "${HOST_START}" "${HOST_END}"); do
    HOST="A${i}"
    echo "=== Configuring ${HOST} ==="

    ${SSH_BIN} -T "${HOST}" \
        TMUX_CONF_B64="$(printf '%s' "${TMUX_CONF_CONTENT}" | base64)" \
        BASHRC_BLOCK_B64="$(printf '%s' "${BASHRC_BLOCK}" | base64)" \
        'bash -s' <<'REMOTE_SCRIPT'
set -euo pipefail

TMUX_CONF=$(printf '%s' "$TMUX_CONF_B64" | base64 --decode)
BASHRC_BLOCK=$(printf '%s' "$BASHRC_BLOCK_B64" | base64 --decode)

printf '%s
' "$TMUX_CONF" > "$HOME/.tmux.conf"

touch "$HOME/.bashrc"
if grep -q '# >>> DMDMolGen tmux aliases >>>' "$HOME/.bashrc"; then
    sed -i '/# >>> DMDMolGen tmux aliases >>>/,/# <<< DMDMolGen tmux aliases <<</d' "$HOME/.bashrc"
fi
printf '
%s
' "$BASHRC_BLOCK" >> "$HOME/.bashrc"

echo "Updated ~/.tmux.conf and ~/.bashrc on $(hostname)"
REMOTE_SCRIPT

    echo "Done: ${HOST}"
done

echo "Configuration applied to A${HOST_START}-A${HOST_END}."
