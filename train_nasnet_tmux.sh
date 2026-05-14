#!/usr/bin/env bash
# Start NASNet-A Mobile on CIFAR-10 in a detached tmux session (survives SSH/IDE disconnect).
# Paths with spaces are handled via a generated runner script (printf %q).
#
# Default: **ImageNet-pretrained** fine-tuning — full 45k/5k train/val each epoch, full test at
# the end (no --max-train-batches). This is the usual “proper” recipe for a large CNN on CIFAR.
# First run downloads ~25MB weights (needs network once).
#
# Usage:
#   ./train_nasnet_tmux.sh              # defaults below (pretrained)
#   ./train_nasnet_tmux.sh attach       # attach to existing session
#
# Train from scratch instead (slower to converge, no download):
#   NASNET_PRETRAINED=0 ./train_nasnet_tmux.sh
#
# Override without editing:
#   NASNET_EPOCHS=30 NASNET_BATCH_SIZE=8 ./train_nasnet_tmux.sh
#   NASNET_LR=5e-4 ./train_nasnet_tmux.sh
#
# If tmux says "no sessions", you never started this script, you killed the session, or
# tmux server restarted — see the log file printed below.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
VENV="${ROOT}/.venv/bin/python"
SESSION="${TMUX_SESSION:-cva3_nasnet}"
OUT_DIR="${OUT_DIR:-${ROOT}/checkpoints}"
LOG_DIR="${OUT_DIR}/logs"
# Per-run archive (never overwritten) + stable path for tail -f.
TS="$(date +%Y%m%d_%H%M%S)"
LOG_RUN="${LOG_DIR}/nasnet_train_${TS}.log"
LOG_TAIL="${OUT_DIR}/train_nasnet_tmux.log"
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

mkdir -p "$OUT_DIR" "$LOG_DIR"

NASNET_PRETRAINED="${NASNET_PRETRAINED:-1}"
NASNET_NUM_WORKERS="${NASNET_NUM_WORKERS:-2}"

if [[ "${NASNET_PRETRAINED}" == "1" || "${NASNET_PRETRAINED}" == "yes" || "${NASNET_PRETRAINED}" == "true" ]]; then
  # Fine-tune ImageNet weights: 224px + ImageNet normalization in data.py; main.py default lr 3e-4.
  NASNET_EPOCHS="${NASNET_EPOCHS:-25}"
  NASNET_BATCH_SIZE="${NASNET_BATCH_SIZE:-16}"
  NASNET_IMG_SIZE="${NASNET_IMG_SIZE:-224}"
  NASNET_LR="${NASNET_LR:-3e-4}"
  PRE_FLAG=(--pretrained)
  MODE_LABEL="pretrained ImageNet → CIFAR-10"
else
  # From scratch: smaller crops, CIFAR normalization, main.py default lr 1e-3.
  NASNET_EPOCHS="${NASNET_EPOCHS:-20}"
  NASNET_BATCH_SIZE="${NASNET_BATCH_SIZE:-16}"
  NASNET_IMG_SIZE="${NASNET_IMG_SIZE:-128}"
  NASNET_LR="${NASNET_LR:-1e-3}"
  PRE_FLAG=()
  MODE_LABEL="from scratch on CIFAR-10"
fi

# Two command lines: with or without --pretrained (no empty-flag edge cases).
if ((${#PRE_FLAG[@]})); then
  TRAIN_CMD="$(printf '%q' "$VENV") -m cv_assignment3.main \
  --model nasnet \
  --nasnet-variant mobile \
  --pretrained \
  --img-size ${NASNET_IMG_SIZE} \
  --epochs ${NASNET_EPOCHS} \
  --batch-size ${NASNET_BATCH_SIZE} \
  --lr ${NASNET_LR} \
  --num-workers ${NASNET_NUM_WORKERS} \
  --data-dir $(printf '%q' "${ROOT}/data") \
  --out-dir $(printf '%q' "$OUT_DIR")"
else
  TRAIN_CMD="$(printf '%q' "$VENV") -m cv_assignment3.main \
  --model nasnet \
  --nasnet-variant mobile \
  --img-size ${NASNET_IMG_SIZE} \
  --epochs ${NASNET_EPOCHS} \
  --batch-size ${NASNET_BATCH_SIZE} \
  --lr ${NASNET_LR} \
  --num-workers ${NASNET_NUM_WORKERS} \
  --data-dir $(printf '%q' "${ROOT}/data") \
  --out-dir $(printf '%q' "$OUT_DIR")"
fi

{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo "cd $(printf '%q' "$ROOT")"
  echo 'export PYTHONUNBUFFERED=1'
  echo "mkdir -p $(printf '%q' "$LOG_DIR")"
  echo "{"
  echo "  echo '===== NASNet-A Mobile — CIFAR-10 full training ====='"
  echo "  date -u"
  echo "  echo 'Timestamped log:' $(printf '%q' "$LOG_RUN")"
  echo "  echo 'Also mirrored to:' $(printf '%q' "$LOG_TAIL")"
  echo "  echo '====================================================='"
  echo "} | tee $(printf '%q' "$LOG_RUN") | tee $(printf '%q' "$LOG_TAIL")"
  echo "${TRAIN_CMD} 2>&1 | tee -a $(printf '%q' "$LOG_RUN") | tee -a $(printf '%q' "$LOG_TAIL")"
} >"$RUNNER"
chmod +x "$RUNNER"

tmux new-session -d -s "$SESSION" bash -c "$(printf '%q' "$RUNNER"); exec bash"

echo "Started tmux session: $SESSION"
echo "Runner:    $RUNNER"
echo "Log (this run):  $LOG_RUN"
echo "Log (tail -f):   $LOG_TAIL"
echo "Mode:      ${MODE_LABEL}"
echo "Training:  epochs=${NASNET_EPOCHS} batch=${NASNET_BATCH_SIZE} img=${NASNET_IMG_SIZE} lr=${NASNET_LR} workers=${NASNET_NUM_WORKERS} pretrained=${NASNET_PRETRAINED}"
echo "Attach:    tmux attach -t $SESSION"
echo "Detach:    Ctrl-b then d"
echo "Kill:      tmux kill-session -t $SESSION"
echo "Tail log:  tail -f $(printf '%q' "$LOG_TAIL")"
