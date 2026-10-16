#!/usr/bin/env bash
# Convert, merge, extract features, and verify the six Spatial/Object/10 batches.
set -euo pipefail

RLINF_ROOT="${RLINF_ROOT:-/data/Xixixi/VLA1/RLinf}"
PCD_ROOT="${PCD_ROOT:-/data/Xixixi/VLA1/PCD-meta-subspace}"
OPENPI_ROOT="${OPENPI_ROOT:-/data/Xixixi/VLA1/openpi-meta-subspace}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/data/Xixixi/VLA1/LIBERO-plus}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/Xixixi/VLA1/model_cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch}"
COLLECTION_PARENT="${COLLECTION_PARENT:-${RLINF_ROOT}/results/pi0/collection_batches}"
SIM_PYTHON="${SIM_PYTHON:-/data/Xixixi/VLA1/conda_envs/pcd-libero-plus/bin/python}"
RLINF_PYTHON="${RLINF_PYTHON:-/data/Xixixi/VLA1/uv_envs/rlinf-openpi/bin/python}"
GPU_PYTHON="${GPU_PYTHON:-/data/Xixixi/VLA1/conda_envs/openpi-server-torch/bin/python}"
DEVICE="${DEVICE:-cuda:0}"

if [[ -z "${COLLECTION_SET:-}" ]]; then
  COLLECTION_SET="$(find "${COLLECTION_PARENT}" -maxdepth 1 -type d \
    -name 'spatial_object_10_sensor_camera_*' -print | sort | tail -n 1)"
fi
: "${COLLECTION_SET:?No full-six collection directory found; set COLLECTION_SET}"
COLLECTION_SET="$(realpath "${COLLECTION_SET}")"

export LIBERO_TYPE=plus
export LIBERO_CONFIG_PATH="${PCD_ROOT}/config/libero_plus_runtime"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PCD_ROOT}:${OPENPI_ROOT}/src:${RLINF_ROOT}:${LIBERO_PLUS_ROOT}:${PYTHONPATH:-}"

test -x "${SIM_PYTHON}"
test -x "${RLINF_PYTHON}"
test -x "${GPU_PYTHON}"
test -s "${CHECKPOINT_DIR}/model.safetensors"
test -d "${COLLECTION_SET}"

declare -a SUITES=(libero_spatial libero_object libero_10)
declare -A SUITE_TAG=(
  [libero_spatial]=spatial
  [libero_object]=object
  [libero_10]=10
)

echo "COLLECTION_SET=${COLLECTION_SET}"
echo "DEVICE=${DEVICE}"

MAPPING_ROOT="${COLLECTION_SET}/pcd_mappings"
mkdir -p "${MAPPING_ROOT}"

for suite in "${SUITES[@]}"; do
  tag="${SUITE_TAG[$suite]}"
  mapping="${MAPPING_ROOT}/${tag}_base_task_map.json"
  if [[ ! -s "${mapping}" ]]; then
    "${SIM_PYTHON}" -m adapter_training.export_task_mapping_template \
      --suite "${suite}" \
      --output "${mapping}"
  fi
done

convert_one() {
  local suite="$1"
  local tag="$2"
  local category_tag="$3"
  local run_root="${COLLECTION_SET}/${tag}_${category_tag}"
  local export_root="${run_root}/pcd_export"
  local check_root="${run_root}/pcd_check"
  local mapping="${MAPPING_ROOT}/${tag}_base_task_map.json"
  local source_id="pi0_libero_rlinf_vanilla_${tag}_${category_tag}_flowode10"

  test -s "${run_root}/selection.json"
  test -d "${run_root}/episodes"
  mkdir -p "${check_root}"

  if [[ ! -s "${export_root}/conversion_report.json" ]]; then
    "${RLINF_PYTHON}" "${RLINF_ROOT}/toolkits/libero_plus/convert_rollouts.py" \
      --input-dir "${run_root}/episodes" \
      --selection "${run_root}/selection.json" \
      --task-mapping "${mapping}" \
      --output-dir "${export_root}" \
      --pcd-root "${PCD_ROOT}" \
      --checkpoint-dir "${CHECKPOINT_DIR}" \
      --source-id "${source_id}" \
      --suite "${suite}" \
      --seed 7 \
      --replan-steps 5 \
      --num-wait-steps 10 \
      --denoise-steps 10 \
      --action-horizon 50 \
      --confirm-fixed-chunk-vanilla
  fi

  local verify_args=(
    "${RLINF_PYTHON}" "${RLINF_ROOT}/toolkits/libero_plus/verify_pcd.py"
    --pcd-root "${PCD_ROOT}"
    --episodes "${export_root}"
    --task-mapping "${mapping}"
    --manifest "${check_root}/pool.json"
    --report "${check_root}/cpu_verification.json"
  )
  if [[ ! -s "${check_root}/pool.json" ]]; then
    verify_args+=(--build-manifest)
  fi
  "${verify_args[@]}"
}

for suite in "${SUITES[@]}"; do
  tag="${SUITE_TAG[$suite]}"
  convert_one "${suite}" "${tag}" sensor_noise
  convert_one "${suite}" "${tag}" camera
done

MERGED_ROOT="${COLLECTION_SET}/pcd_merged"
mkdir -p "${MERGED_ROOT}"

for suite in "${SUITES[@]}"; do
  tag="${SUITE_TAG[$suite]}"
  suite_root="${MERGED_ROOT}/${tag}_sensor_camera"
  episodes="${suite_root}/episodes"
  manifest="${suite_root}/pool.json"
  mapping="${MAPPING_ROOT}/${tag}_base_task_map.json"
  features="${suite_root}/features_uniform3"
  gpu_report="${suite_root}/gpu_verification.json"

  mkdir -p "${suite_root}"
  if [[ ! -s "${manifest}" ]]; then
    if [[ -e "${episodes}" ]]; then
      echo "Incomplete merged episode directory exists: ${episodes}" >&2
      exit 1
    fi
    "${RLINF_PYTHON}" "${RLINF_ROOT}/toolkits/libero_plus/merge_exports.py" \
      --export "${COLLECTION_SET}/${tag}_sensor_noise/pcd_export" \
      --export "${COLLECTION_SET}/${tag}_camera/pcd_export" \
      --output-dir "${episodes}" \
      --manifest "${manifest}" \
      --task-mapping "${mapping}" \
      --pcd-root "${PCD_ROOT}" \
      --validation-fraction 0.2 \
      --seed 7
  fi

  "${GPU_PYTHON}" -m adapter_training.meta_subspace.features \
    --manifest "${manifest}" \
    --output-dir "${features}" \
    --base-config pi0_libero \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --device "${DEVICE}" \
    --sites pre_joint_layer_9 pre_joint_layer_12 pre_joint_layer_15 \
    --steps-per-episode 3

  "${GPU_PYTHON}" - \
    "${RLINF_ROOT}" "${PCD_ROOT}" "${manifest}" "${features}" \
    "${CHECKPOINT_DIR}" "${DEVICE}" "${gpu_report}" <<'PY'
import json
import sys
from pathlib import Path

rlinf_root, pcd_root, manifest, features, checkpoint, device, report = sys.argv[1:]
sys.path.insert(0, rlinf_root)
sys.path.insert(0, pcd_root)
from toolkits.libero_plus.verify_pcd import verify_features_and_batch
from adapter_training.meta_subspace.data import MetaPool

result = verify_features_and_batch(
    Path(manifest), Path(features), Path(checkpoint), device
)
pool = MetaPool(manifest)
result["eligible_train_tasks_support2"] = len(pool.eligible("train", 2))
result["eligible_validation_tasks_support2"] = len(
    pool.eligible("validation", 2)
)
result["manifest"] = str(Path(manifest).resolve())
result["features"] = str(Path(features).resolve())
Path(report).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY
done

"${RLINF_PYTHON}" - "${COLLECTION_SET}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
batches = []
for tag in ("spatial", "object", "10"):
    for category in ("sensor_noise", "camera"):
        run = root / f"{tag}_{category}"
        conversion = json.loads((run / "pcd_export/conversion_report.json").read_text())
        cpu = json.loads((run / "pcd_check/cpu_verification.json").read_text())
        batches.append({
            "batch": run.name,
            "collected": conversion["collected"],
            "success": conversion["success"],
            "failure": conversion["failure"],
            "npz_and_manifest_pass": cpu["npz_and_manifest_pass"],
        })

suites = []
for tag in ("spatial", "object", "10"):
    suite_root = root / "pcd_merged" / f"{tag}_sensor_camera"
    pool = json.loads((suite_root / "pool.json").read_text())
    gpu = json.loads((suite_root / "gpu_verification.json").read_text())
    suites.append({
        "suite": tag,
        "manifest": str(suite_root / "pool.json"),
        "features": str(suite_root / "features_uniform3"),
        "summary": pool["summary"],
        "eligible_train_tasks_support2": gpu[
            "eligible_train_tasks_support2"
        ],
        "eligible_validation_tasks_support2": gpu[
            "eligible_validation_tasks_support2"
        ],
        "feature_cache_pass": gpu["feature_cache_pass"],
        "training_query_batch_and_forward_pass": gpu[
            "training_query_batch_and_forward_pass"
        ],
    })

report = {
    "schema_version": 1,
    "collection_set": str(root),
    "batches": batches,
    "suites": suites,
    "all_pass": all(row["npz_and_manifest_pass"] for row in batches)
    and all(
        row["feature_cache_pass"]
        and row["training_query_batch_and_forward_pass"]
        for row in suites
    ),
}
path = root / "pcd_full6_verification.json"
path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
if not report["all_pass"]:
    raise SystemExit("Full-six PCD verification failed")
PY

echo "Full-six conversion and feature verification passed."
echo "REPORT=${COLLECTION_SET}/pcd_full6_verification.json"
