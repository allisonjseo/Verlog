#!/bin/bash
# Start the verlog monitor in a detached tmux session on the login node.
# Idempotent: if a session is already running, prints how to attach and exits 0.
#
# Generates _monitor-settings.json from the template by substituting REPO_ROOT
# and USER, so the gate.sh hook command is an absolute path.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

SESSION="${MONITOR_TMUX_SESSION:-verlog-monitor}"
TEMPLATE="$HERE/monitor-settings.template.json"
EFFECTIVE="$HERE/_monitor-settings.json"

if [[ ! -r "$HERE/monitor.env" ]]; then
    echo "Missing $HERE/monitor.env — copy monitor.env.example and fill it in." >&2
    exit 1
fi
if [[ ! -r "$REPO_ROOT/claude-tools/slack/gate.sh" ]]; then
    echo "Missing claude-tools submodule. Run from $REPO_ROOT:" >&2
    echo "    git submodule update --init claude-tools" >&2
    exit 1
fi

# Render settings with absolute paths
sed -e "s|REPO_ROOT|$REPO_ROOT|g" \
    -e "s|//u/USER/|$HOME/|g" \
    "$TEMPLATE" > "$EFFECTIVE"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Monitor session '$SESSION' already running."
else
    tmux new-session -d -s "$SESSION" "bash '$HERE/verlog_monitor.sh'"
    echo "Started monitor in tmux session '$SESSION'."
fi

# shellcheck disable=SC1091
source "$HERE/monitor.env"
SUMMARY_DIR="${MONITOR_LOG_DIR:-${SMOKETEST_DIR}/logs/monitor}"

cat <<EOF

Commands:
  Attach:   tmux attach -t $SESSION
  Detach:   Ctrl-b d   (from inside the session)
  Stop:     tmux kill-session -t $SESSION
  Status:   tmux ls
  Tail log: tail -f $SUMMARY_DIR/summary.log
EOF
