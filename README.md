# Orix AMP

Adversarial Motion Priors for Orix Dog — a 12-DOF quadruped robot trained to walk naturally using reference motion data.

## Robot

| Spec | Value |
|------|-------|
| DOF | 12 (4 legs × 3 joints: hip, thigh, calf) |
| Standing height | 0.28m |
| Weight | ~7.5kg |
| Joint axes | Left (FL/RL) axis=-y, Right (FR/RR) axis=+y (mirrored) |
| PD gains | Kp=20, Kd=1.0 |
| Effort limit | 23.7 Nm |
| Action scale | hip: 0.125 rad, thigh/calf: 0.25 rad |

## Architecture

Uses [TienKung-Lab](https://github.com/TienKung-Lab) rsl_rl fork with AMP support:

```
AmpOnPolicyRunner
├── ActorCritic (MLP [512, 256, 128])
├── AMPPPO (PPO + discriminator reward)
├── Discriminator (MLP [1024, 512] → 1)
└── AMPLoader (motion data)
```

**AMP reward**: `total = task_reward × lerp + style_reward × (1 - lerp) × coef`
- `lerp = 0.3` (30% task, 70% style)
- `coef = 2.0`
- Style reward comes from discriminator trained on expert motion data

## Motion Data

9 motion clips retargeted from Unitree A1 → Orix Dog:

| Clip | Frames | Duration | Gait |
|------|--------|----------|------|
| trot0 | 33 | 0.7s | trot |
| trot1 | 501 | 10.0s | trot |
| trot2 | 501 | 10.0s | trot |
| hop1 | 501 | 10.0s | hop |
| hop2 | 501 | 10.0s | hop |
| pace0 | 39 | 0.8s | pace |
| pace1 | 491 | 10.3s | pace |
| leftturn0 | 45 | 0.9s | turn |
| rightturn0 | 119 | 2.5s | turn |

**Format**: 36-column per frame = `joint_pos(12) + joint_vel(12) + foot_pos_local(12)`

**Retargeting pipeline** (A1 → Orix):
1. Joint reorder: PyBullet (FR,FL,RR,RL) → Isaac (FL,FR,RL,RR)
2. Right-side axis negation (FR/RR thigh & calf)
3. Height scaling × 0.8 (A1: 0.35m → Orix: 0.28m)
4. Clamp to URDF joint limits

## Dependencies

| Package | Version | Note |
|---------|---------|------|
| Isaac Lab | 2024.x | Isaac Sim 4.5+ |
| rsl_rl | TienKung fork | **Not** the official rsl_rl 3.3.0 |
| robot_lab | latest | [leggedrobotics/robot_lab](https://github.com/leggedrobotics/robot_lab) |
| PyTorch | 2.x | CUDA 12.x |
| gymnasium | 0.29+ | |

### Install

```bash
# 1. Isaac Lab (follow official docs)
# https://isaac-sim.github.io/IsaacLab/

# 2. robot_lab (provides base env configs)
cd IsaacLab
git clone https://github.com/leggedrobotics/robot_lab.git source/robot_lab
python -m pip install -e source/robot_lab

# 3. TienKung rsl_rl fork (provides AMP runner)
# Clone into project dir — NOT the system rsl_rl
cd orix-amp
git clone https://github.com/TienKung-Lab/rsl_rl.git rsl_rl

# 4. Copy robot assets into robot_lab
cp urdf/ <robot_lab>/source/robot_lab/robot_lab/assets/robots/orix_dog/
cp config/orix_dog.py <robot_lab>/source/robot_lab/robot_lab/assets/
cp -r config/ <robot_lab>/source/robot_lab/robot_lab/tasks/.../quadruped/orix_dog/
```

## Training

```bash
# Standard PPO (no AMP)
cd robot_lab
python -u scripts/reinforcement_learning/rsl_rl/train.py \
    --task RobotLab-Isaac-Velocity-Flat-OrixDog-v0 \
    --num_envs 4096 --headless

# AMP training (TienKung framework)
cd orix-amp
CUDA_VISIBLE_DEVICES=4 python -u scripts/train_orix_amp.py \
    --num_envs 4096 --max_iterations 10000 --headless
```

### Train config

| Parameter | Value |
|-----------|-------|
| num_envs | 4096 |
| max_iterations | 10000 |
| num_steps_per_env | 24 |
| learning_rate | 1e-3 (adaptive) |
| gamma | 0.99 |
| clip_param | 0.2 |
| entropy_coef | 0.01 |
| actor dims | [512, 256, 128] |
| discriminator dims | [1024, 512] |
| amp_reward_coef | 2.0 |
| amp_task_reward_lerp | 0.3 |
| amp_num_preload_transitions | 2,000,000 |

## Project Structure

```
orix-amp/
├── urdf/
│   ├── orix_dog.urdf                  # Robot URDF
│   └── meshes/                        # 17 STL collision/visual meshes
├── config/
│   ├── orix_dog.py                    # ArticulationCfg (Isaac Lab)
│   ├── rough_env_cfg.py               # Rough terrain env
│   ├── flat_env_cfg.py                # Flat terrain env
│   └── agents/rsl_rl_ppo_cfg.py       # PPO hyperparameters
├── motions/
│   ├── orix_amp_*.txt                 # 9 AMP motion clips (36-col)
│   ├── retarget_a1_to_orix.py         # A1 → Orix retargeting
│   ├── generate_amp_txt_36col.py      # 61-col → 36-col converter
│   ├── convert_to_amp_rsl_rl.py       # → .npy format converter
│   └── convert_motion_imitation.py    # motion_imitation format
├── scripts/
│   ├── train_orix_amp.py              # AMP training (TienKung rsl_rl)
│   └── train_orix_amp_v2.py           # Standalone version
├── envs/
│   ├── orix_amp_env.py                # AMP env mixin
│   ├── orix_amp_env_cfg.py            # AMP env config
│   └── orix_amp_manager_cfg.py        # Manager-based config
└── README.md
```

## Known Issues

1. **foot_pos mismatch**: `merge_fixed_joints=True` merges foot into calf. The AMP obs uses calf body position (~10cm below base) but expert data has toe position (~20cm below base). This causes the discriminator to trivially distinguish expert vs policy. **Fix**: use 24-col format (joint_pos + joint_vel only) or compute foot position via forward kinematics.

2. **velocity tracking**: With `lerp=0.3`, style reward dominates and velocity tracking is low (~0.16). Increase `lerp` to 0.5-0.7 after initial style learning.

## References

- [AMP: Adversarial Motion Priors](https://arxiv.org/abs/2104.02180) (Peng et al., 2021)
- [TienKung-Lab rsl_rl](https://github.com/TienKung-Lab/rsl_rl) — AMP implementation for humanoid
- [robot_lab](https://github.com/leggedrobotics/robot_lab) — Isaac Lab locomotion framework
- [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware) — Source motion data
