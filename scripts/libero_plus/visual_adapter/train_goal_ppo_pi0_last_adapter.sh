#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Usage examples
# ============================================================
#
# Adapter only:
#   TRAIN_ACTION_HEAD=0 TRAIN_ACTION_EXPERT=0 bash "$0"
#
# Adapter + action head:
#   TRAIN_ACTION_HEAD=1 TRAIN_ACTION_EXPERT=0 bash "$0"
#
# Adapter + action head + action expert:
#   TRAIN_ACTION_HEAD=1 TRAIN_ACTION_EXPERT=1 bash "$0"
#
# Smoke:
#   MAX_STEPS=2 SAVE_INTERVAL=1 VAL_INTERVAL=999999 \
#   NUM_ENVS=4 ROLLOUT_EPOCH=1 GLOBAL_BATCH_SIZE=60 bash "$0"

# ============================================================
# Repository and model
# ============================================================

RLINF_ROOT="${RLINF_ROOT:-/data/Xixixi/VLA1/RLinf}"

PI0_MODEL="${PI0_MODEL:-\
/data/Xixixi/VLA1/model_cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch}"

cd "${RLINF_ROOT}"

# Compatible with either filename used in earlier patches.
if [[ -z "${CONFIG_NAME:-}" ]]; then
  if [[ -s \
    "examples/embodiment/config/libero_goal_ppo_openpi_visual_adapter.yaml" ]]; then
    CONFIG_NAME="libero_goal_ppo_openpi_visual_adapter"
  elif [[ -s \
    "examples/embodiment/config/libero_goal_ppo_pi0_visual_adapter.yaml" ]]; then
    CONFIG_NAME="libero_goal_ppo_pi0_visual_adapter"
  else
    echo "Cannot find the visual-adapter PPO config." >&2
    exit 1
  fi
fi

export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"

# ============================================================
# LIBERO / LIBERO-Plus selection
# ============================================================

export LIBERO_TYPE="${LIBERO_TYPE:-plus}"

if [[ "${LIBERO_TYPE}" == "plus" ]]; then
  export LIBERO_CATEGORY="${LIBERO_CATEGORY:-Objects Layout}"
  export LIBERO_CLASSIFICATION_PATH="${LIBERO_CLASSIFICATION_PATH:-\
/data/Xixixi/VLA1/LIBERO-plus/libero/libero/benchmark/task_classification.json}"

  test -s "${LIBERO_CLASSIFICATION_PATH}"

  CATEGORY_TAG="${LIBERO_CATEGORY// /_}"

  unset LIBERO_SUFFIX || true
  unset LIBERO_PERTURBATION || true
else
  CATEGORY_TAG="standard"

  unset LIBERO_CATEGORY || true
  unset LIBERO_CLASSIFICATION_PATH || true
  unset LIBERO_SUFFIX || true
  unset LIBERO_PERTURBATION || true
fi

# ============================================================
# Trainable component switches
# ============================================================

# Default experiment: adapter + action head.
TRAIN_ACTION_HEAD="${TRAIN_ACTION_HEAD:-1}"
TRAIN_ACTION_EXPERT="${TRAIN_ACTION_EXPERT:-0}"

case "${TRAIN_ACTION_HEAD}" in
  0)
    ACTION_HEAD_FLAG=false
    ;;
  1)
    ACTION_HEAD_FLAG=true
    ;;
  *)
    echo "TRAIN_ACTION_HEAD must be 0 or 1." >&2
    exit 1
    ;;
esac

case "${TRAIN_ACTION_EXPERT}" in
  0)
    ACTION_EXPERT_FLAG=false
    ;;
  1)
    ACTION_EXPERT_FLAG=true
    ;;
  *)
    echo "TRAIN_ACTION_EXPERT must be 0 or 1." >&2
    exit 1
    ;;
esac

if [[ "${TRAIN_ACTION_EXPERT}" == "1" &&
      "${TRAIN_ACTION_HEAD}" == "1" ]]; then
  TRAINING_VARIANT="adapter_action_head_action_expert"
elif [[ "${TRAIN_ACTION_EXPERT}" == "1" ]]; then
  TRAINING_VARIANT="adapter_action_expert"
elif [[ "${TRAIN_ACTION_HEAD}" == "1" ]]; then
  TRAINING_VARIANT="adapter_action_head"
else
  TRAINING_VARIANT="adapter_only"
fi

# ============================================================
# PPO and resource configuration
# ============================================================

# Conservative settings for one H100 with about 50 GiB available.
NUM_ENVS="${NUM_ENVS:-8}"
ROLLOUT_EPOCH="${ROLLOUT_EPOCH:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-60}"

MAX_STEPS="${MAX_STEPS:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
VAL_INTERVAL="${VAL_INTERVAL:-20}"
SEED="${SEED:-7}"

MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
NUM_ACTION_CHUNKS="${NUM_ACTION_CHUNKS:-5}"

ADAPTER_DIM="${ADAPTER_DIM:-64}"
ADAPTER_DROPOUT="${ADAPTER_DROPOUT:-0.0}"

# This LR applies to every trainable policy component:
# adapter, action head, and optionally action expert.
POLICY_LR="${POLICY_LR:-5.0e-5}"
VALUE_LR="${VALUE_LR:-1.0e-4}"

# RLinf rollout-size divisibility preflight.
if (( MAX_EPISODE_STEPS % NUM_ACTION_CHUNKS != 0 )); then
  echo "MAX_EPISODE_STEPS must be divisible by NUM_ACTION_CHUNKS." >&2
  exit 1
fi

CHUNKS_PER_ENV=$((MAX_EPISODE_STEPS / NUM_ACTION_CHUNKS))
ROLLOUT_SIZE=$((NUM_ENVS * ROLLOUT_EPOCH * CHUNKS_PER_ENV))

if (( ROLLOUT_SIZE % GLOBAL_BATCH_SIZE != 0 )); then
  echo "Invalid PPO batch configuration:" >&2
  echo "  rollout_size=${ROLLOUT_SIZE}" >&2
  echo "  global_batch_size=${GLOBAL_BATCH_SIZE}" >&2
  echo "rollout_size must be divisible by global_batch_size." >&2
  exit 1
fi

# ============================================================
# Rendering and GPU
# ============================================================

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-\
expandable_segments:True}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export ROBOT_PLATFORM=LIBERO
export PYTHONHASHSEED="${SEED}"

# ============================================================
# W&B
# ============================================================

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY='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'
export WANDB_ENTITY="${WANDB_ENTITY:-wangxixixi0627-iscas}"
export WANDB_PROJECT="${WANDB_PROJECT:-RLinf}"

if [[ "${WANDB_MODE}" == "online" &&
      -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set." >&2
  echo "Run: export WANDB_API_KEY='your-new-key'" >&2
  exit 1
fi

# ============================================================
# Preflight checks
# ============================================================

test -s "${PI0_MODEL}/model.safetensors"
test -s "${PI0_MODEL}/config.json"
test -s \
  "${PI0_MODEL}/assets/physical-intelligence/libero/norm_stats.json"

test -s \
  "examples/embodiment/config/${CONFIG_NAME}.yaml"

test -s \
  "rlinf/models/embodiment/openpi/visual_adaptation.py"

# ============================================================
# Output paths
# ============================================================

TIMESTAMP="$(date +'%Y%m%d-%H%M%S')"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-\
pi0_last_layer_${TRAINING_VARIANT}_${CATEGORY_TAG}_ppo_seed${SEED}}"

RUN_DIR="${RUN_DIR:-\
${RLINF_ROOT}/results/pi0/adapter_ppo/${CONFIG_NAME}/${TRAINING_VARIANT}/${TIMESTAMP}}"

mkdir -p "${RUN_DIR}"

# Do not terminate another RLinf process unless explicitly requested.
if [[ "${RESET_RAY:-0}" == "1" ]]; then
  ray stop --force >/dev/null 2>&1 || true
fi

# ============================================================
# Training command
# ============================================================

CMD=(
  python
  "${EMBODIED_PATH}/train_embodied_agent.py"

  --config-path
  "${EMBODIED_PATH}/config/"

  --config-name
  "${CONFIG_NAME}"

  # Avoid CUDA IPC pidfd_getfd permission errors in the container.
  "++weight_syncer.patch.transport_device=cpu"

  # Save adapter, action head/expert and value head only.
  "runner.checkpoint_mode=trainable_only"

  "runner.logger.log_path=${RUN_DIR}"
  "runner.logger.project_name=${WANDB_PROJECT}"
  "runner.logger.experiment_name=${EXPERIMENT_NAME}"

  "runner.max_steps=${MAX_STEPS}"
  "runner.save_interval=${SAVE_INTERVAL}"
  "runner.val_check_interval=${VAL_INTERVAL}"

  # Base model.
  "actor.model.model_path=${PI0_MODEL}"
  "actor.model.trainable_checkpoint_path=null"
  "actor.model.num_action_chunks=${NUM_ACTION_CHUNKS}"

  "rollout.model.model_path=${PI0_MODEL}"

  # Freeze the base VLM. Explicit switches below selectively
  # re-enable adapter, action head and optionally action expert.
  "actor.model.openpi.train_expert_only=true"
  "actor.model.openpi.train_action_expert=${ACTION_EXPERT_FLAG}"
  "actor.model.openpi.train_action_head=${ACTION_HEAD_FLAG}"

  # Last-layer visual adapter.
  "actor.model.openpi.visual_adaptation.backbone.type=last_layer_adapter"
  "actor.model.openpi.visual_adaptation.backbone.last_layer_adapter.bottleneck_dim=${ADAPTER_DIM}"
  "actor.model.openpi.visual_adaptation.backbone.last_layer_adapter.dropout=${ADAPTER_DROPOUT}"
  "actor.model.openpi.visual_adaptation.backbone.last_layer_adapter.residual_scale=1.0"

  # FTM is disabled for this experiment.
  "actor.model.openpi.visual_adaptation.ftm.enabled=false"

  # PPO stochastic rollout.
  "actor.model.openpi.noise_method=flow_sde"
  "actor.model.openpi.noise_level=0.5"

  # Optimization.
  "actor.optim.lr=${POLICY_LR}"
  "actor.optim.value_lr=${VALUE_LR}"
  "actor.micro_batch_size=${MICRO_BATCH_SIZE}"
  "actor.global_batch_size=${GLOBAL_BATCH_SIZE}"
  "actor.seed=${SEED}"

  # Required when frozen and trainable parameters coexist.
  "actor.fsdp_config.use_orig_params=true"

  # Training environments.
  "env.train.seed=${SEED}"
  "env.train.total_num_envs=${NUM_ENVS}"
  "env.train.rollout_epoch=${ROLLOUT_EPOCH}"
  "env.train.max_episode_steps=${MAX_EPISODE_STEPS}"
  "env.train.max_steps_per_rollout_epoch=${MAX_EPISODE_STEPS}"
  "env.train.video_cfg.save_video=false"

  # Periodic validation.
  "env.eval.seed=${SEED}"
  "env.eval.total_num_envs=10"
  "env.eval.rollout_epoch=1"
  "env.eval.max_episode_steps=${MAX_EPISODE_STEPS}"
  "env.eval.max_steps_per_rollout_epoch=${MAX_EPISODE_STEPS}"
  "env.eval.video_cfg.save_video=false"
)

# ============================================================
# Configuration summary
# ============================================================

echo "============================================================"
echo "RLinf root:          ${RLINF_ROOT}"
echo "Config:              ${CONFIG_NAME}"
echo "Base model:          ${PI0_MODEL}"
echo "LIBERO type:         ${LIBERO_TYPE}"
echo "Category:            ${LIBERO_CATEGORY:-standard}"
echo "Training variant:    ${TRAINING_VARIANT}"
echo "Train adapter:       true"
echo "Train action head:   ${ACTION_HEAD_FLAG}"
echo "Train action expert: ${ACTION_EXPERT_FLAG}"
echo "Train value head:    true"
echo "FTM:                 false"
echo "Policy LR:           ${POLICY_LR}"
echo "Value LR:            ${VALUE_LR}"
echo "NUM_ENVS:            ${NUM_ENVS}"
echo "ROLLOUT_EPOCH:       ${ROLLOUT_EPOCH}"
echo "Rollout size:        ${ROLLOUT_SIZE}"
echo "Micro batch:         ${MICRO_BATCH_SIZE}"
echo "Global batch:        ${GLOBAL_BATCH_SIZE}"
echo "Max steps:           ${MAX_STEPS}"
echo "Experiment:          ${EXPERIMENT_NAME}"
echo "Run directory:       ${RUN_DIR}"
echo "============================================================"

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" 2>&1 | tee "${RUN_DIR}/train.log"