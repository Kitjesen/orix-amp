#!/bin/bash
# Orix Dog AMP v3 launcher — thunder2 env + local robot_lab source
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROBOT_LAB_SRC="/home/bsrl/hongsenpang/RLbased/robot_lab/source/robot_lab"
GPU="${GPU:-4}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-15000}"

echo "=== Orix AMP v3 ==="
echo "  GPU=$GPU  num_envs=$NUM_ENVS  max_iter=$MAX_ITER"
echo "  robot_lab: $ROBOT_LAB_SRC"

# robot_lab source must be on PYTHONPATH so orix_dog env cfg is found
export PYTHONPATH="$ROBOT_LAB_SRC:${PYTHONPATH:-}"

mkdir -p "$SCRIPT_DIR/logs"
LOG="$SCRIPT_DIR/logs/amp_v3_$(date +%Y%m%d_%H%M%S).log"
echo "  log: $LOG"

source /home/bsrl/miniconda3/etc/profile.d/conda.sh
conda activate thunder2

nohup env CUDA_VISIBLE_DEVICES=$GPU \
    python "$SCRIPT_DIR/scripts/train_amp_v3.py" \
    --num_envs $NUM_ENVS \
    --max_iterations $MAX_ITER \
    --headless \
    > "$LOG" 2>&1 &

echo "  PID=$!"
echo "  tail -f $LOG"
