#!/usr/bin/env bash
set -euo pipefail

cd /data/Xixixi/VLA1/RLinf

# ============================================================
# Model
# ============================================================

export PI0_MODEL="/data/Xixixi/VLA1/model_cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch"

# ============================================================
# Rendering and GPU
# ============================================================

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# ============================================================
# LIBERO: Goal
# ============================================================

export LIBERO_TYPE=standard
unset LIBERO_SUFFIX || true
unset LIBERO_PERTURBATION || true

# ============================================================
# Preflight checks
# ============================================================

test -s "${PI0_MODEL}/model.safetensors"
test -s "${PI0_MODEL}/config.json"
test -s "${PI0_MODEL}/assets/physical-intelligence/libero/norm_stats.json"
test -s "evaluations/libero/libero_goal_pi0_eval.yaml"

echo "PI0_MODEL=${PI0_MODEL}"
echo "Config exists: evaluations/libero/libero_goal_pi0_eval.yaml"

# ============================================================
# evaluation: 10 environments
# ============================================================

ray stop --force >/dev/null 2>&1 || true

LOG_ROOT=/data/Xixixi/VLA1/RLinf/results/pi0/baseline/libero \
bash evaluations/run_eval.sh \
  libero \
  libero_goal_pi0_eval \
  rollout.model.model_path="${PI0_MODEL}" \
  rollout.model.openpi.noise_method=flow_ode \
  rollout.model.openpi.noise_level=0.0 \
  runner.logger.experiment_name="pi0_libero_goal" \
  env.eval.seed=7 \
  env.eval.total_num_envs=25 \
  env.eval.rollout_epoch=20 \
  env.eval.video_cfg.save_video=true