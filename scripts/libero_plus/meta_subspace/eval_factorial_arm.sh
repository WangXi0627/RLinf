#!/usr/bin/env bash
set -euo pipefail

arm="${1:?usage: $0 b_residual|c1_joint_mask|c2_bilevel|c3_full}"
case "${arm}" in
  b_residual|c1_joint_mask|c2_bilevel|c3_full) ;;
  *) echo "unknown factorial arm: ${arm}" >&2; exit 2 ;;
esac

export RLINF_ROOT="${RLINF_ROOT:-/data/Xixixi/VLA1/RLinf}"
export PCD_ROOT="${PCD_ROOT:-/data/Xixixi/VLA1/PCD-meta-subspace}"
export OPENPI_CHECKPOINT="${OPENPI_CHECKPOINT:-/data/Xixixi/VLA1/model_cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch}"
export META_ROOT="${META_ROOT:-${PCD_ROOT}/results/libero_plus_meta_subspace/v2}"
export META_SUBSPACE_MANIFEST="${META_SUBSPACE_MANIFEST:-${META_ROOT}/manifests/goal5_mixed_pool.json}"
export META_SUBSPACE_FEATURE_DIR="${META_SUBSPACE_FEATURE_DIR:-${META_ROOT}/features/goal5_mixed_uniform3}"
export META_SUBSPACE_TASK_MAPPING="${META_SUBSPACE_TASK_MAPPING:-${PCD_ROOT}/config/feature_adapter/v2/libero_goal/base_task_map.json}"
export LIBERO_CLASSIFICATION_PATH="${LIBERO_CLASSIFICATION_PATH:-/data/Xixixi/VLA1/LIBERO-plus/libero/libero/benchmark/task_classification.json}"
export FACTORIAL_CHECKPOINT="${FACTORIAL_CHECKPOINT:-${PCD_ROOT}/results/libero_plus_meta_subspace/v3/${arm}_layer15_seed7/latest.pt}"
export FACTORIAL_ARM="${arm}"
export INTERVENTION_STRENGTH="${INTERVENTION_STRENGTH:-0.5}"
export FACTORIAL_ACTION_MODE="${FACTORIAL_ACTION_MODE:-all_meta}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export PYTHONUNBUFFERED=1
export LIBERO_TYPE=plus
unset LIBERO_SUFFIX || true
unset LIBERO_PERTURBATION || true

TOTAL_NUM_ENVS="${TOTAL_NUM_ENVS:-25}"
RESULT_ROOT="${RESULT_ROOT:-${RLINF_ROOT}/results/pi0/residual_mask_factorial/goal5}"
read -r -a CATEGORY_KEYS <<< "${EVAL_CATEGORY_KEYS:-background_textures light_conditions objects_layout camera_viewpoints sensor_noise}"

test -s "${OPENPI_CHECKPOINT}/model.safetensors"
test -s "${OPENPI_CHECKPOINT}/config.json"
test -s "${OPENPI_CHECKPOINT}/assets/physical-intelligence/libero/norm_stats.json"
test -s "${META_SUBSPACE_MANIFEST}"
test -s "${META_SUBSPACE_FEATURE_DIR}/index.json"
test -s "${META_SUBSPACE_TASK_MAPPING}"
test -s "${LIBERO_CLASSIFICATION_PATH}"
test -s "${FACTORIAL_CHECKPOINT}"
test -s "${RLINF_ROOT}/evaluations/libero/libero_goal_pi0_residual_mask_factorial_eval.yaml"

python - "${arm}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

arm = sys.argv[1]
checkpoint = Path(os.environ["FACTORIAL_CHECKPOINT"]).resolve()
manifest = Path(os.environ["META_SUBSPACE_MANIFEST"]).resolve()
feature_index = Path(os.environ["META_SUBSPACE_FEATURE_DIR"]).resolve() / "index.json"
mapping = Path(os.environ["META_SUBSPACE_TASK_MAPPING"]).resolve()

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
assert payload["schema_version"] == 2
assert payload["method"] == "libero-plus-task-residual-mask-factorial"
assert payload["protocol"] == "libero-plus-task-residual-mask-factorial-v1"
assert payload["config"]["arm"] == arm
assert payload["config"]["site"] == "pre_joint_layer_15"
assert payload["config"]["support_size"] == 2
assert payload["config"]["seed"] == 7
assert payload["update"] == 500, payload["update"]
assert payload["metadata"]["manifest_sha256"] == sha(manifest)
assert payload["metadata"]["feature_index_sha256"] == sha(feature_index)
manifest_doc = json.loads(manifest.read_text())
feature_doc = json.loads(feature_index.read_text())
assert manifest_doc["mapping_sha256"] == sha(mapping)
assert feature_doc["manifest_sha256"] == sha(manifest)
assert "pre_joint_layer_15" in feature_doc["sites"]
print("Factorial preflight passed")
print("arm:", arm)
print("checkpoint:", checkpoint)
print("manifest:", manifest)
print("mixed_sources:", manifest_doc.get("mixed_sources", False))
print("source_ids:", manifest_doc.get("source_ids", [manifest_doc.get("source_id")]))
print("feature_cache:", feature_index.parent)
PY

category_name() {
  case "$1" in
    background_textures) echo "Background Textures" ;;
    light_conditions) echo "Light Conditions" ;;
    objects_layout) echo "Objects Layout" ;;
    camera_viewpoints) echo "Camera Viewpoints" ;;
    sensor_noise) echo "Sensor Noise" ;;
    *) echo "unknown category key: $1" >&2; return 2 ;;
  esac
}

cd "${RLINF_ROOT}"
for category_key in "${CATEGORY_KEYS[@]}"; do
  export LIBERO_CATEGORY
  LIBERO_CATEGORY="$(category_name "${category_key}")"
  count="$(python - <<'PY'
import json
import os
from pathlib import Path
doc = json.loads(Path(os.environ["LIBERO_CLASSIFICATION_PATH"]).read_text())
category = os.environ["LIBERO_CATEGORY"].casefold()
print(sum(str(row.get("category", "")).strip().casefold() == category for row in doc["libero_goal"]))
PY
)"
  if (( count <= 0 )); then
    echo "No libero_goal tasks for category ${LIBERO_CATEGORY}" >&2
    exit 2
  fi
  epochs=$(( (count + TOTAL_NUM_ENVS - 1) / TOTAL_NUM_ENVS ))\n  if [[ -n "${EVAL_EPOCHS:-}" ]]; then\n    epochs="${EVAL_EPOCHS}"\n  fi
  echo "Evaluating arm=${arm} category=${LIBERO_CATEGORY} tasks=${count} envs=${TOTAL_NUM_ENVS} epochs=${epochs}"
  ray stop --force >/dev/null 2>&1 || true
  LOG_ROOT="${RESULT_ROOT}/${arm}/${category_key}" \
  bash evaluations/run_eval.sh \
    libero libero_goal_pi0_residual_mask_factorial_eval \
    "runner.logger.experiment_name=pi0_liberoplus_goal_${arm}_${category_key}" \
    "env.eval.total_num_envs=${TOTAL_NUM_ENVS}" \
    "env.eval.rollout_epoch=${epochs}" \
    "env.eval.seed=7" \
    "env.eval.video_cfg.save_video=false"
done

echo "Completed factorial arm ${arm}: ${RESULT_ROOT}/${arm}"
