#!/usr/bin/env bash
set -euo pipefail

export RLINF_ROOT="${RLINF_ROOT:-/data/Xixixi/VLA1/RLinf-liberoplus-ftm-fla}"
export PCD_RESULT_ROOT="${PCD_RESULT_ROOT:-/data/Xixixi/VLA1/PCD-liberoplus-ftm-fla/results/libero_plus_ftm_fla}"
export OPENPI_CHECKPOINT="${OPENPI_CHECKPOINT:-/data/Xixixi/VLA1/model_cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch}"
export MASTER_CLASSIFICATION="${MASTER_CLASSIFICATION:-/data/Xixixi/VLA1/LIBERO-plus/libero/libero/benchmark/task_classification.json}"

export LIBERO_TYPE=plus
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LIBERO_CATEGORY=""

RUN_TAG="${RUN_TAG:-$(date +%Y%m%dT%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-${RLINF_ROOT}/results/libero_plus_ftm_fla/ftm_spatial_object_seq_${RUN_TAG}}"

mkdir -p "${RESULT_BASE}"
cd "${RLINF_ROOT}"

wait_for_final_ftm() {
  local suite="$1"
  local pattern="$2"
  local marker=""
  local waited=0
  local max_wait_seconds="${MAX_WAIT_SECONDS:-86400}"

  while true; do
    marker="$(
      find "${PCD_RESULT_ROOT}" \
        -type f \
        -path "${pattern}" \
        -name 'adapter_step005000.pt' \
        -printf '%T@ %p\n' 2>/dev/null |
      sort -nr |
      head -1 |
      cut -d' ' -f2-
    )"

    if [[ -n "${marker}" && -f "${marker}" ]]; then
      local checkpoint
      checkpoint="$(dirname "${marker}")/latest.pt"

      if [[ -f "${checkpoint}" ]]; then
        echo "${checkpoint}"
        return 0
      fi
    fi

    if (( waited >= max_wait_seconds )); then
      echo "ERROR: timed out waiting for ${suite} FTM final checkpoint" >&2
      return 1
    fi

    echo "[WAIT] ${suite} FTM adapter_step005000.pt not found; retry in 60s" >&2
    sleep 60
    waited=$((waited + 60))
  done
}

generate_chunks() {
  local suite_key="$1"
  local chunk_root="$2"
  local index_file="$3"

  mkdir -p "${chunk_root}"

  python - \
    "${MASTER_CLASSIFICATION}" \
    "${suite_key}" \
    "${chunk_root}" \
    "${index_file}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

source = Path(sys.argv[1])
suite_key = sys.argv[2]
output = Path(sys.argv[3])
index_path = Path(sys.argv[4])

data = json.loads(source.read_text())
entries = data[suite_key]

categories = [
    ("background", "Background Textures"),
    ("light", "Light Conditions"),
    ("camera", "Camera Viewpoints"),
    ("sensor", "Sensor Noise"),
    ("objects", "Objects Layout"),
]

def category_of(entry):
    for key in (
        "category",
        "perturbation_category",
        "perturbation_type",
        "type",
    ):
        value = entry.get(key)
        if value:
            return str(value)
    raise KeyError(f"Entry has no category: {entry}")

all_counts = Counter(category_of(entry) for entry in entries)
rows = []
selected_total = 0

for alias, category in categories:
    selected = [
        entry for entry in entries
        if category_of(entry).strip().lower() == category.lower()
    ]

    if not selected:
        raise RuntimeError(
            f"No entries for {category!r}; available={dict(all_counts)}"
        )

    selected_total += len(selected)

    for chunk_id, start in enumerate(range(0, len(selected), 25)):
        chunk = selected[start:start + 25]
        path = output / f"{alias}_{chunk_id:03d}.json"
        path.write_text(
            json.dumps(
                {suite_key: chunk},
                indent=2,
                ensure_ascii=False,
            )
        )
        rows.append(
            f"{alias}\t{category}\t{chunk_id:03d}\t{len(chunk)}\t{path}"
        )

index_path.write_text("\n".join(rows) + "\n")

print("Suite:", suite_key)
print("All category counts:", dict(all_counts))
print("Selected five-category environments:", selected_total)
print("Total chunks:", len(rows))
print("Index:", index_path)
PY
}

run_suite() {
  local short_name="$1"
  local suite_key="$2"
  local config_name="$3"
  local checkpoint="$4"

  local suite_root="${RESULT_BASE}/${short_name}"
  local chunk_root="${suite_root}/manifests"
  local index_file="${suite_root}/chunks.tsv"

  mkdir -p "${suite_root}"

  generate_chunks \
    "${suite_key}" \
    "${chunk_root}" \
    "${index_file}"

  export SPATIAL_ADAPTATION_CHECKPOINT="${checkpoint}"

  echo
  echo "################################################################"
  echo "START ${short_name^^} FTM"
  echo "Checkpoint: ${SPATIAL_ADAPTATION_CHECKPOINT}"
  echo "Output: ${suite_root}"
  echo "################################################################"

  while IFS=$'\t' read -r alias category chunk_id count manifest; do
    local run_dir="${suite_root}/${alias}/chunk_${chunk_id}"
    mkdir -p "${run_dir}"

    export LIBERO_CATEGORY="${category}"
    export LIBERO_CLASSIFICATION_PATH="${manifest}"

    echo
    echo "============================================================"
    echo "Suite    : ${short_name}"
    echo "Category : ${category}"
    echo "Chunk    : ${chunk_id}"
    echo "Envs     : ${count}"
    echo "Manifest : ${manifest}"
    echo "Output   : ${run_dir}"
    echo "============================================================"

    ray stop --force >/dev/null 2>&1 || true

    bash evaluations/run_eval.sh \
      libero \
      "${config_name}" \
      env.eval.total_num_envs="${count}" \
      env.eval.group_size=1 \
      env.eval.rollout_epoch=1 \
      +env.eval.plus_expected_num_tasks="${count}" \
      env.eval.video_cfg.save_video=false \
      runner.logger.log_path="${run_dir}" \
      runner.logger.experiment_name="ftm_${short_name}_${alias}_${chunk_id}" \
      2>&1 | tee "${run_dir}/driver.log"

  done < "${index_file}"

  echo
  echo "PASS: ${short_name} FTM evaluation finished"
}

echo "Waiting for Spatial FTM final checkpoint..."
SPATIAL_FTM_CHECKPOINT="$(
  wait_for_final_ftm \
    spatial \
    '*/formal_spatial_5pert_all_success_seed7_*/ftm/adapter_step005000.pt'
)"

run_suite \
  spatial \
  libero_spatial \
  libero_spatial_pi0_ftm_eval \
  "${SPATIAL_FTM_CHECKPOINT}"

echo "Waiting for Object FTM final checkpoint..."
OBJECT_FTM_CHECKPOINT="$(
  wait_for_final_ftm \
    object \
    '*/formal_object_5pert_all_success_seed7*/ftm/adapter_step005000.pt'
)"

run_suite \
  object \
  libero_object \
  libero_object_pi0_ftm_eval \
  "${OBJECT_FTM_CHECKPOINT}"

echo
echo "============================================================"
echo "ALL FINISHED"
echo "Spatial results: ${RESULT_BASE}/spatial"
echo "Object results : ${RESULT_BASE}/object"
echo "============================================================"
