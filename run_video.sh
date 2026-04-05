#!/bin/bash
# Record a video of the trained Orix Dog AMP policy
# Usage: CHECKPOINT=logs/orix_amp/.../model_final.pt bash run_video.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKPOINT="${CHECKPOINT:-$SCRIPT_DIR/logs/orix_amp/2026-04-05_04-04-02/model_final.pt}"
GPU="${GPU:-1}"
CMD_VX="${CMD_VX:-0.5}"
VIDEO_LEN="${VIDEO_LEN:-300}"

echo "=== Orix Video Recording ==="
echo "  checkpoint: $CHECKPOINT"
echo "  GPU=$GPU  cmd_vx=$CMD_VX  length=$VIDEO_LEN"

source /home/bsrl/miniconda3/etc/profile.d/conda.sh
conda activate thunder2
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

# Resolve URDF mesh paths: package://orix_dog/meshes/*.STL
# Create symlink so orix_dog → urdf/, then set ROS_PACKAGE_PATH
ln -sf "$SCRIPT_DIR/urdf" "$SCRIPT_DIR/orix_dog" 2>/dev/null
export ROS_PACKAGE_PATH="$SCRIPT_DIR:${ROS_PACKAGE_PATH:-}"

# Virtual display
LOG="$SCRIPT_DIR/logs/video_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$SCRIPT_DIR/logs"
echo "  log: $LOG"

# headless.rendering.kit = headless + enable_cameras (no display needed)
CUDA_VISIBLE_DEVICES=$GPU \
    python "$SCRIPT_DIR/scripts/record_video.py" \
    --checkpoint "$CHECKPOINT" \
    --video_length $VIDEO_LEN \
    --cmd_vx $CMD_VX \
    --headless \
    --enable_cameras \
    2>&1 | tee "$LOG"
echo "=== Done ==="
