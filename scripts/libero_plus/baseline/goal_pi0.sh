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
# LIBERO-Plus: libero_goal
# ============================================================

export LIBERO_TYPE=plus
# export LIBERO_CATEGORY="Objects Layout"
export LIBERO_CATEGORY="Sensor Noise"
export LIBERO_CLASSIFICATION_PATH="/data/Xixixi/VLA1/LIBERO-plus/libero/libero/benchmark/task_classification.json"

# Category manifest 模式不使用 suffix
unset LIBERO_SUFFIX || true
unset LIBERO_PERTURBATION || true

# ============================================================
# Preflight checks
# ============================================================

test -s "${PI0_MODEL}/model.safetensors"
test -s "${PI0_MODEL}/config.json"
test -s "${PI0_MODEL}/assets/physical-intelligence/libero/norm_stats.json"
test -s "evaluations/libero/libero_goal_pi0_eval.yaml"
test -s "${LIBERO_CLASSIFICATION_PATH}"

python - <<'PY'
import json
import os
from pathlib import Path

import liberoplus.liberoplus as liberoplus

suite = "libero_goal"
category = os.environ["LIBERO_CATEGORY"]

manifest_path = Path(
    os.environ["LIBERO_CLASSIFICATION_PATH"]
).expanduser().resolve()

bddl_root = Path(
    liberoplus.get_libero_path("bddl_files")
).resolve()

with manifest_path.open("r", encoding="utf-8") as f:
    manifest = json.load(f)

selected = [
    entry
    for entry in manifest[suite]
    if str(entry.get("category", "")).strip().casefold()
    == category.strip().casefold()
]

missing_sources = []
virtual_count = 0

for entry in selected:
    name = str(entry["name"]).removesuffix(".bddl")
    runtime_path = bddl_root / suite / f"{name}.bddl"

    # 普通类别：BDDL 文件真实存在
    if runtime_path.is_file():
        continue

    # 虚拟类别：ControlEnv 会解析 view/initstate/noise 后缀，
    # 然后读取 _view_ 之前的基础 BDDL。
    if "_view_" in name and "_initstate_" in name:
        source_name = name.split("_view_", 1)[0]
        source_path = bddl_root / suite / f"{source_name}.bddl"

        if source_path.is_file():
            virtual_count += 1
            continue

    missing_sources.append(str(runtime_path))

print("LIBERO-Plus BDDL root:", bddl_root)
print("Classification manifest:", manifest_path)
print("Suite:", suite)
print("Category:", category)
print("Selected tasks:", len(selected))
print("Virtual task paths:", virtual_count)
print("Missing backing BDDL files:", len(missing_sources))

assert selected, f"No tasks found for {suite}/{category}"
assert not missing_sources, (
    f"Missing {len(missing_sources)} backing BDDL files; "
    f"examples: {missing_sources[:5]}"
)
PY

echo "PI0_MODEL=${PI0_MODEL}"
echo "LIBERO_TYPE=${LIBERO_TYPE}"
echo "LIBERO_CATEGORY=${LIBERO_CATEGORY}"
echo "LIBERO_CLASSIFICATION_PATH=${LIBERO_CLASSIFICATION_PATH}"
echo "LIBERO_SUFFIX=${LIBERO_SUFFIX:-<unset>}"

# ============================================================
# objects_layout: 425 tasks = 25 envs × 17 epochs
# sensor_noise: 379 tasks = 25 envs × 16 epochs
# ============================================================

ray stop --force >/dev/null 2>&1 || true

LOG_CATEGORY="${LIBERO_CATEGORY// /_}"
LOG_ROOT=/data/Xixixi/VLA1/RLinf/results/pi0/baseline/libero_plus/${LOG_CATEGORY} \
bash evaluations/run_eval.sh \
  libero \
  libero_goal_pi0_eval \
  rollout.model.model_path="${PI0_MODEL}" \
  rollout.model.openpi.noise_method=flow_ode \
  rollout.model.openpi.noise_level=0.0 \
  runner.logger.experiment_name="pi0_liberoplus_goal_sensor_noise" \
  env.eval.seed=7 \
  env.eval.group_size=1 \
  env.eval.total_num_envs=25 \
  env.eval.rollout_epoch=16 \
  env.eval.video_cfg.save_video=false