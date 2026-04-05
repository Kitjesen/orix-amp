#!/bin/bash
# Orix Dog AMP launcher
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU="${GPU:-4}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-10000}"
TASK_LERP="${TASK_LERP:-0.3}"

echo "=== Orix AMP (self-contained) ==="
echo "  GPU=$GPU  num_envs=$NUM_ENVS  max_iter=$MAX_ITER  task_lerp=$TASK_LERP"

# Self-contained: only needs isaaclab (thunder2) — no robot_lab, no TienKung
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

mkdir -p "$SCRIPT_DIR/logs"
LOG="$SCRIPT_DIR/logs/amp_$(date +%Y%m%d_%H%M%S).log"
echo "  log: $LOG"

source /home/bsrl/miniconda3/etc/profile.d/conda.sh
conda activate thunder2

RESUME_ARG=""
if [ -n "$RESUME" ]; then
    RESUME_ARG="--resume $RESUME"
    echo "  resume: $RESUME"
fi

nohup env CUDA_VISIBLE_DEVICES=$GPU \
    python "$SCRIPT_DIR/scripts/train_amp.py" \
    --num_envs $NUM_ENVS \
    --max_iterations $MAX_ITER \
    --task_lerp $TASK_LERP \
    $RESUME_ARG \
    --headless \
    > "$LOG" 2>&1 &

echo "  PID=$!"
echo "  tail -f $LOG"
