#!/usr/bin/env bash
set -euo pipefail

eval_arm="${1:?usage: $0 importance_fo|importance_fo_no_inner|importance_so|importance_so_no_inner|delta_fo|delta_fo_no_inner|delta_so|delta_so_no_inner}"
case "${eval_arm}" in
  importance_fo|importance_fo_no_inner|importance_so|importance_so_no_inner|delta_fo|delta_fo_no_inner|delta_so|delta_so_no_inner) ;;
  *) echo "unknown mask-guided evaluation arm: ${eval_arm}" >&2; exit 2 ;;
esac
base_arm="${eval_arm%_no_inner}"
if [[ "${eval_arm}" == *_no_inner ]]; then
  export MASK_GUIDED_NO_INNER_UPDATE=1
else
  export MASK_GUIDED_NO_INNER_UPDATE=0
fi

export RLINF_ROOT="${RLINF_ROOT:-/data/Xixixi/VLA1/RLinf}"
export PCD_ROOT="${PCD_ROOT:-/data/Xixixi/VLA1/PCD-meta-subspace}"
export OPENPI_ROOT="${OPENPI_ROOT:-/data/Xixixi/VLA1/openpi-meta-subspace}"
export PYTHONPATH="${RLINF_ROOT}:${OPENPI_ROOT}/src:${PYTHONPATH:-}"
export OPENPI_CHECKPOINT="${OPENPI_CHECKPOINT:-/data/Xixixi/VLA1/model_cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch}"
export META_ROOT="${META_ROOT:-${PCD_ROOT}/results/libero_plus_meta_subspace/v2}"
export META_SUBSPACE_MANIFEST="${META_SUBSPACE_MANIFEST:-${META_ROOT}/manifests/goal5_mixed_pool.json}"
export META_SUBSPACE_FEATURE_DIR="${META_SUBSPACE_FEATURE_DIR:-${META_ROOT}/features/goal5_mixed_uniform3}"
export META_SUBSPACE_TASK_MAPPING="${META_SUBSPACE_TASK_MAPPING:-${PCD_ROOT}/config/feature_adapter/v2/libero_goal/base_task_map.json}"
export LIBERO_CLASSIFICATION_PATH="${LIBERO_CLASSIFICATION_PATH:-/data/Xixixi/VLA1/LIBERO-plus/libero/libero/benchmark/task_classification.json}"
export MASK_GUIDED_CHECKPOINT="${MASK_GUIDED_CHECKPOINT:-${PCD_ROOT}/results/libero_plus_meta_subspace/v4/${base_arm}_layer15_seed7/latest.pt}"
export MASK_GUIDED_EVAL_ARM="${eval_arm}"
export INTERVENTION_STRENGTH="${INTERVENTION_STRENGTH:-0.5}"
export MASK_GUIDED_ACTION_MODE="${MASK_GUIDED_ACTION_MODE:-all_meta}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export PYTHONUNBUFFERED=1
export LIBERO_TYPE=plus
unset LIBERO_SUFFIX || true
unset LIBERO_PERTURBATION || true

TOTAL_NUM_ENVS="${TOTAL_NUM_ENVS:-35}"
RESULT_ROOT="${RESULT_ROOT:-${RLINF_ROOT}/results/pi0/mask_guided_residual/goal5}"
read -r -a CATEGORY_KEYS <<< "${EVAL_CATEGORY_KEYS:-background_textures light_conditions objects_layout camera_viewpoints sensor_noise}"

test -s "${OPENPI_CHECKPOINT}/model.safetensors"
test -s "${OPENPI_CHECKPOINT}/config.json"
test -s "${OPENPI_CHECKPOINT}/assets/physical-intelligence/libero/norm_stats.json"
test -s "${META_SUBSPACE_MANIFEST}"
test -s "${META_SUBSPACE_FEATURE_DIR}/index.json"
test -s "${META_SUBSPACE_TASK_MAPPING}"
test -s "${LIBERO_CLASSIFICATION_PATH}"
test -s "${MASK_GUIDED_CHECKPOINT}"
test -s "${RLINF_ROOT}/evaluations/libero/libero_goal_pi0_mask_guided_residual_eval.yaml"

python - "${base_arm}" "${eval_arm}" <<'PY'
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import torch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import checkpoints as openpi_checkpoints

base_arm, eval_arm = sys.argv[1:3]
checkpoint = Path(os.environ["MASK_GUIDED_CHECKPOINT"]).resolve()
manifest = Path(os.environ["META_SUBSPACE_MANIFEST"]).resolve()
feature_index = Path(os.environ["META_SUBSPACE_FEATURE_DIR"]).resolve() / "index.json"
mapping = Path(os.environ["META_SUBSPACE_TASK_MAPPING"]).resolve()

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

expected = {
    "importance_fo": ("importance", False),
    "importance_so": ("importance", True),
    "delta_fo": ("delta", False),
    "delta_so": ("delta", True),
}
assert callable(openpi_checkpoints.load_norm_stats)
assert "intervention_context" in inspect.signature(PI0Pytorch.flow_velocity).parameters
payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
signal, second_order = expected[base_arm]
assert payload["schema_version"] == 2
assert payload["method"] == "libero-plus-mask-guided-residual"
assert payload["protocol"] == "libero-plus-mask-guided-residual-v1"
assert payload["config"]["arm"] == base_arm
assert payload["config"]["guide_signal"] == signal
assert payload["config"]["second_order"] is second_order
assert payload["config"]["site"] == "pre_joint_layer_15"
assert payload["config"]["support_size"] == 2
assert payload["config"]["seed"] == 7
assert payload["update"] == 500, payload["update"]
assert payload["adapter_config"]["method"] == "mask_guided_residual"
assert payload["adapter_config"]["mask_guided_residual"]["query_direct_mask"] is False
assert "intervention.guide_condition_proj.weight" in payload["adapter"]
assert payload["metadata"]["manifest_sha256"] == sha(manifest)
assert payload["metadata"]["feature_index_sha256"] == sha(feature_index)
manifest_doc = json.loads(manifest.read_text())
feature_doc = json.loads(feature_index.read_text())
assert manifest_doc["mapping_sha256"] == sha(mapping)
assert feature_doc["manifest_sha256"] == sha(manifest)
assert "pre_joint_layer_15" in feature_doc["sites"]
assert (os.environ["MASK_GUIDED_NO_INNER_UPDATE"] == "1") is eval_arm.endswith("_no_inner")
print("Mask-guided preflight passed")
print("train_arm:", base_arm)
print("eval_arm:", eval_arm)
print("checkpoint:", checkpoint)
print("no_inner:", os.environ["MASK_GUIDED_NO_INNER_UPDATE"])
PY

if [[ "${MASK_GUIDED_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Mask-guided preflight-only check completed"
  exit 0
fi

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
  epochs=$(( (count + TOTAL_NUM_ENVS - 1) / TOTAL_NUM_ENVS ))
  if [[ -n "${EVAL_EPOCHS:-}" ]]; then
    epochs="${EVAL_EPOCHS}"
  fi
  echo "Evaluating arm=${eval_arm} category=${LIBERO_CATEGORY} tasks=${count} envs=${TOTAL_NUM_ENVS} epochs=${epochs}"
  ray stop --force >/dev/null 2>&1 || true
  LOG_ROOT="${RESULT_ROOT}/${eval_arm}/${category_key}"   bash evaluations/run_eval.sh     libero libero_goal_pi0_mask_guided_residual_eval     "runner.logger.experiment_name=pi0_liberoplus_goal_${eval_arm}_${category_key}"     "env.eval.total_num_envs=${TOTAL_NUM_ENVS}"     "env.eval.rollout_epoch=${epochs}"     "env.eval.seed=7"     "env.eval.video_cfg.save_video=false"
done

echo "Completed mask-guided arm ${eval_arm}: ${RESULT_ROOT}/${eval_arm}"
