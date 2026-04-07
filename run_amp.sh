#!/bin/bash
# Orix Dog training — uses robot_lab ManagerBasedRLEnv
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU="${GPU:-4}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-20000}"
TASK="${TASK:-RobotLab-Isaac-Velocity-Flat-OrixDog-v0}"

echo "=== Orix Dog Training (robot_lab) ==="
echo "  GPU=$GPU  task=$TASK  num_envs=$NUM_ENVS  max_iter=$MAX_ITER"

source /home/bsrl/miniconda3/etc/profile.d/conda.sh
conda activate thunder2

# robot_lab source must be on PYTHONPATH
ROBOT_LAB_SRC="/home/bsrl/hongsenpang/RLbased/robot_lab/source/robot_lab"
export PYTHONPATH="$ROBOT_LAB_SRC:$SCRIPT_DIR:${PYTHONPATH:-}"

# URDF mesh paths
ln -sf "$SCRIPT_DIR/urdf" "$SCRIPT_DIR/orix_dog" 2>/dev/null
export ROS_PACKAGE_PATH="$SCRIPT_DIR:${ROS_PACKAGE_PATH:-}"

mkdir -p "$SCRIPT_DIR/logs"
LOG="$SCRIPT_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"
echo "  log: $LOG"

RESUME_ARG=""
if [ -n "$RESUME" ]; then
    RESUME_ARG="--resume $RESUME"
    echo "  resume: $RESUME"
fi

nohup env CUDA_VISIBLE_DEVICES=$GPU \
    python "$SCRIPT_DIR/scripts/train_amp.py" \
    --task $TASK \
    --num_envs $NUM_ENVS \
    --max_iterations $MAX_ITER \
    $RESUME_ARG \
    --headless \
    > "$LOG" 2>&1 &

echo "  PID=$!"
echo "  tail -f $LOG"
