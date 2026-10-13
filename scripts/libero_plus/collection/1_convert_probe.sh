#!/usr/bin/env bash
# Convert a fixed-chunk vanilla RLinf probe into PCD episode files.
set -euo pipefail

RLINF_ROOT="${RLINF_ROOT:-/data/Xixixi/VLA1/RLinf}"
PCD_ROOT="${PCD_ROOT:-/data/Xixixi/VLA1/PCD-meta-subspace}"
: "${PROBE_ROOT:?Set PROBE_ROOT to the completed Sensor Noise probe batch}"
CHECK_ROOT="${CHECK_ROOT:-${PROBE_ROOT}/pcd_check}"
TASK_MAPPING="${TASK_MAPPING:-${CHECK_ROOT}/base_task_map.json}"
EXPORT_ROOT="${EXPORT_ROOT:-${PROBE_ROOT}/pcd_export}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/Xixixi/VLA1/model_cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch}"
SOURCE_ID="${SOURCE_ID:-pi0_libero_rlinf_vanilla_goal_sensor20_flowode10}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${RLINF_ROOT}"
test -s "${TASK_MAPPING}"
test -s "${PROBE_ROOT}/selection.json"

"${PYTHON_BIN}" toolkits/libero_plus/convert_rollouts.py \
  --input-dir "${PROBE_ROOT}/episodes" \
  --selection "${PROBE_ROOT}/selection.json" \
  --task-mapping "${TASK_MAPPING}" \
  --output-dir "${EXPORT_ROOT}" \
  --pcd-root "${PCD_ROOT}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --source-id "${SOURCE_ID}" \
  --suite libero_goal --seed 7 \
  --replan-steps 5 --num-wait-steps 10 \
  --denoise-steps 10 --action-horizon 50 \
  --confirm-fixed-chunk-vanilla

"${PYTHON_BIN}" toolkits/libero_plus/verify_pcd.py \
  --pcd-root "${PCD_ROOT}" \
  --episodes "${EXPORT_ROOT}" \
  --task-mapping "${TASK_MAPPING}" \
  --manifest "${CHECK_ROOT}/sensor20_pool.json" \
  --build-manifest \
  --report "${CHECK_ROOT}/cpu_verification.json"
