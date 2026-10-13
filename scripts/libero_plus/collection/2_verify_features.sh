#!/usr/bin/env bash
# Run in the PCD OpenPI server environment with its patched OpenPI checkout.
set -euo pipefail

RLINF_ROOT="${RLINF_ROOT:-/data/Xixixi/VLA1/RLinf}"
export PCD_ROOT="${PCD_ROOT:-/data/Xixixi/VLA1/PCD-meta-subspace}"
export OPENPI_ROOT="${OPENPI_ROOT:-/data/Xixixi/VLA1/openpi-meta-subspace}"
: "${PROBE_ROOT:?Set PROBE_ROOT to the completed Sensor Noise probe batch}"
CHECK_ROOT="${CHECK_ROOT:-${PROBE_ROOT}/pcd_check}"
TASK_MAPPING="${TASK_MAPPING:-${CHECK_ROOT}/base_task_map.json}"
EXPORT_ROOT="${EXPORT_ROOT:-${PROBE_ROOT}/pcd_export}"
FEATURE_DIR="${FEATURE_DIR:-${CHECK_ROOT}/features_uniform3}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/Xixixi/VLA1/model_cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"

export PYTHONPATH="${PCD_ROOT}:${OPENPI_ROOT}/src:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
cd "${PCD_ROOT}"

"${PYTHON_BIN}" -m adapter_training.meta_subspace.features \
  --manifest "${CHECK_ROOT}/sensor20_pool.json" \
  --output-dir "${FEATURE_DIR}" \
  --base-config pi0_libero \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --device "${DEVICE}" \
  --sites pre_joint_layer_9 pre_joint_layer_12 pre_joint_layer_15 \
  --steps-per-episode 3

"${PYTHON_BIN}" "${RLINF_ROOT}/toolkits/libero_plus/verify_pcd.py" \
  --pcd-root "${PCD_ROOT}" \
  --episodes "${EXPORT_ROOT}" \
  --task-mapping "${TASK_MAPPING}" \
  --manifest "${CHECK_ROOT}/sensor20_pool.json" \
  --features "${FEATURE_DIR}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --device "${DEVICE}" \
  --report "${CHECK_ROOT}/gpu_verification.json"
