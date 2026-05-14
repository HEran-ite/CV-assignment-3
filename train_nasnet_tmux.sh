#!/usr/bin/env bash
# Start NASNet-A Mobile training in a detached tmux session (survives SSH/IDE disconnect).
# Paths with spaces are handled via a generated runner script (printf %q).
#
# Usage:
#   ./train_nasnet_tmux.sh           # default: 1 epoch, img 96, batch 16
#   ./train_nasnet_tmux.sh attach    # attach to existing session
#
# If tmux says "no sessions", the inner command crashed — open the log file printed below.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
VENV="${ROOT}/.venv/bin/python"
SESSION="${TMUX_SESSION:-cva3_nasnet}"
OUT_DIR="${OUT_DIR:-${ROOT}/checkpoints}"
LOG="${OUT_DIR}/train_nasnet_tmux.log"
RUNNER="${OUT_DIR}/.tmux_nasnet_runner.sh"

attach_only() {
  exec tmux attach-session -t "$SESSION"
}

if [[ "${1:-}" == "attach" ]]; then
  attach_only
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with:"
  echo "  tmux attach -t $SESSION"
  echo "Or kill it: tmux kill-session -t $SESSION"
  exit 1
fi

mkdir -p "$OUT_DIR"

# One safe command line (quoted paths) for the runner body
TRAIN_CMD="$(printf '%q' "$VENV") -m cv_assignment3.main \
  --model nasnet \
  --nasnet-variant mobile \
  --img-size 96 \
  --epochs 1 \
  --batch-size 16 \
  --lr 1e-3 \
  --num-workers 0 \
  --data-dir $(printf '%q' "${ROOT}/data") \
  --out-dir $(printf '%q' "$OUT_DIR")"

{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo "cd $(printf '%q' "$ROOT")"
  echo 'export PYTHONUNBUFFERED=1'
  echo "${TRAIN_CMD} 2>&1 | tee -a $(printf '%q' "$LOG")"
} >"$RUNNER"
chmod +x "$RUNNER"

# Run the runner under bash (do not pass a single quoted string — tmux may mis-parse paths with spaces).
tmux new-session -d -s "$SESSION" bash -- "$RUNNER"

echo "Started tmux session: $SESSION"
echo "Runner:    $RUNNER"
echo "Log file:  $LOG"
echo "Attach:    tmux attach -t $SESSION"
echo "Detach:    Ctrl-b then d"
echo "Kill:      tmux kill-session -t $SESSION"
echo "Tail log:  tail -f $(printf '%q' "$LOG")"
