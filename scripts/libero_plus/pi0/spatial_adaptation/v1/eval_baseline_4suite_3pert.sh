#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT=/data/Xixixi/VLA1/RLinf-liberoplus-ftm-fla
RLINF_PYTHON=/data/Xixixi/VLA1/uv_envs/rlinf-openpi/bin/python

CLASSIFICATION_PATH="${LIBERO_PLUS_CLASSIFICATION_PATH:-/data/Xixixi/VLA1/LIBERO-plus/libero/libero/benchmark/task_classification.json}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%dT%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-${RLINF_ROOT}/results/libero_plus_ftm_fla/baseline_4suite_bg_light_object_batch25_${RUN_TAG}}"

export RLINF_ROOT
export CLASSIFICATION_PATH
export RESULT_BASE

cd "${RLINF_ROOT}"
mkdir -p "${RESULT_BASE}"

unset SPATIAL_ADAPTATION_CHECKPOINT || true
unset PYOPENGL_PLATFORM || true

export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LIBERO_TYPE=plus
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

echo "RLinf root     : ${RLINF_ROOT}"
echo "Classification : ${CLASSIFICATION_PATH}"
echo "Result root    : ${RESULT_BASE}"
echo "CUDA device    : ${CUDA_VISIBLE_DEVICES}"

# ---------------------------------------------------------------------
# 由现有 FTM 配置生成 baseline 配置：
# 1. 保留对应 suite 的环境、最大步数和 OpenPI 参数
# 2. 清除 trainable_checkpoint_path
# 3. 关闭 FTM / FLA
# ---------------------------------------------------------------------
"${RLINF_PYTHON}" - <<'PY'
import os
from pathlib import Path

import yaml

root = Path(os.environ["RLINF_ROOT"])
config_dir = root / "evaluations/libero"

pairs = {
    "libero_goal_pi0_ftm_eval.yaml":
        "libero_goal_pi0_baseline_eval.yaml",
    "libero_10_pi0_ftm_eval.yaml":
        "libero_10_pi0_baseline_eval.yaml",
    "libero_spatial_pi0_ftm_eval.yaml":
        "libero_spatial_pi0_baseline_eval.yaml",
    "libero_object_pi0_ftm_eval.yaml":
        "libero_object_pi0_baseline_eval.yaml",
}


def disable_adaptation(node):
    if isinstance(node, dict):
        for key, value in list(node.items()):
            lowered = str(key).lower()

            if lowered == "trainable_checkpoint_path":
                node[key] = None
                continue

            if lowered in {"ftm", "fla"} and isinstance(value, dict):
                value["enabled"] = False
                for checkpoint_key in (
                    "checkpoint_path",
                    "trainable_checkpoint_path",
                ):
                    if checkpoint_key in value:
                        value[checkpoint_key] = None

            disable_adaptation(value)

    elif isinstance(node, list):
        for value in node:
            disable_adaptation(value)


for source_name, output_name in pairs.items():
    source = config_dir / source_name
    output = config_dir / output_name

    if not source.is_file():
        raise FileNotFoundError(
            f"Missing source evaluation config: {source}"
        )

    data = yaml.safe_load(source.read_text())
    disable_adaptation(data)

    output.write_text(
        yaml.safe_dump(
            data,
            sort_keys=False,
            default_flow_style=False,
        )
    )

    print(f"Baseline config: {output}")
PY

# ---------------------------------------------------------------------
# 从 LIBERO-Plus classification 生成三个扰动的 25-env chunks
# ---------------------------------------------------------------------
"${RLINF_PYTHON}" - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

classification_path = Path(os.environ["CLASSIFICATION_PATH"])
result_root = Path(os.environ["RESULT_BASE"])

data = json.loads(classification_path.read_text())

specs = [
    (
        "libero_goal",
        "libero_goal_pi0_baseline_eval",
    ),
    (
        "libero_10",
        "libero_10_pi0_baseline_eval",
    ),
    (
        "libero_spatial",
        "libero_spatial_pi0_baseline_eval",
    ),
    (
        "libero_object",
        "libero_object_pi0_baseline_eval",
    ),
]

categories = [
    ("Background Textures", "background"),
    ("Light Conditions", "light"),
    ("Objects Layout", "objects"),
]


def suite_entries(raw, suite):
    if isinstance(raw, dict):
        if isinstance(raw.get(suite), list):
            return raw[suite]

        suites = raw.get("suites")
        if isinstance(suites, dict) and isinstance(suites.get(suite), list):
            return suites[suite]

    raise RuntimeError(
        f"Cannot find suite {suite!r}; "
        f"available top-level keys={list(raw) if isinstance(raw, dict) else type(raw)}"
    )


def item_category(item):
    candidates = [
        item.get("perturbation_category"),
        item.get("category"),
        item.get("classification"),
        item.get("perturbation"),
    ]

    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend([
            metadata.get("perturbation_category"),
            metadata.get("category"),
        ])

    for value in candidates:
        if isinstance(value, str) and value:
            return value

    return None


index_path = result_root / "chunks.tsv"
manifest_root = result_root / "manifests"
manifest_root.mkdir(parents=True, exist_ok=True)

lines = []
grand_total = 0

for suite, config_name in specs:
    entries = suite_entries(data, suite)
    counts = Counter(item_category(item) for item in entries)

    print(f"\n{suite} classification counts:")
    print(dict(counts))

    suite_total = 0

    for category, slug in categories:
        selected = [
            item for item in entries
            if item_category(item) == category
        ]

        if not selected:
            raise RuntimeError(
                f"No entries for suite={suite!r}, category={category!r}; "
                f"available={dict(counts)}"
            )

        for chunk_id, start in enumerate(range(0, len(selected), 25)):
            chunk = selected[start:start + 25]

            manifest_dir = manifest_root / suite
            manifest_dir.mkdir(parents=True, exist_ok=True)

            manifest = manifest_dir / f"{slug}_{chunk_id:03d}.json"
            manifest.write_text(
                json.dumps(
                    {suite: chunk},
                    indent=2,
                    ensure_ascii=False,
                ) + "\n"
            )

            output = result_root / suite / slug / f"chunk_{chunk_id:03d}"

            lines.append("\t".join([
                suite,
                config_name,
                category,
                slug,
                f"{chunk_id:03d}",
                str(len(chunk)),
                str(manifest),
                str(output),
            ]))

        suite_total += len(selected)

    grand_total += suite_total
    print(f"{suite} selected environments: {suite_total}")

index_path.write_text("\n".join(lines) + "\n")

print(f"\nTotal selected environments: {grand_total}")
print(f"Total chunks: {len(lines)}")
print(f"Index: {index_path}")
PY

# ---------------------------------------------------------------------
# 顺序执行；已有 .done 的 chunk 会跳过
# ---------------------------------------------------------------------
while IFS=$'\t' read -r \
    suite config_name category slug chunk_id count manifest output
do
    mkdir -p "${output}"

    if [[ -f "${output}/.done" ]]; then
        echo
        echo "SKIP completed: ${suite}/${slug}/chunk_${chunk_id}"
        continue
    fi

    export LIBERO_CATEGORY="${category}"
    export LIBERO_CLASSIFICATION_PATH="${manifest}"

    echo
    echo "================================================================"
    echo "Suite     : ${suite}"
    echo "Category  : ${category}"
    echo "Chunk     : ${chunk_id}"
    echo "Envs      : ${count}"
    echo "Config    : ${config_name}"
    echo "Manifest  : ${manifest}"
    echo "Output    : ${output}"
    echo "================================================================"

    bash evaluations/run_eval.sh \
        libero \
        "${config_name}" \
        env.eval.total_num_envs="${count}" \
        env.eval.group_size=1 \
        env.eval.rollout_epoch=1 \
        +env.eval.plus_expected_num_tasks="${count}" \
        env.eval.video_cfg.save_video=false \
        runner.logger.log_path="${output}" \
        runner.logger.experiment_name="pi0_baseline_${suite}_${slug}_${chunk_id}"

    touch "${output}/.done"
done < "${RESULT_BASE}/chunks.tsv"

echo
echo "PASS: baseline four-suite evaluation finished."
echo "Result root: ${RESULT_BASE}"
