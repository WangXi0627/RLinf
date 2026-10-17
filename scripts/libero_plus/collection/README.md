# Export the Sensor Noise probe to PCD

These scripts convert the completed vanilla π0 RLinf probe to PCD NPZ files and verify them with the existing PCD manifest, feature cache, and query training APIs. Use the same trusted pickle batch, checkpoint, and explicit collection settings as the original run. The first stage runs in the RLinf environment; the feature stage runs in the PCD OpenPI server environment.

The current exporter accepts one attempt per selected variant. It checks the official classification ID in every episode against `selection.json`, then matches the variant name to the explicit PCD mapping. It rejects missing variants, repeated variants, interventions, and inconsistent episode lengths. A successful trajectory includes the action that first achieves success and excludes all later actions. A failure retains all executed actions. Short successes remain in the dataset, but they do not acquire padded 50-step training targets.

`CollectEpisode` stores the reset observation followed by one observation for each executed action. Query steps are reconstructed as `0, 5, 10, ...` from the original fixed-chunk contract. RLinf images are already rotated 180 degrees; the exporter applies the PCD client's resize/pad to 224 RGB without another rotation. These are PCD training inputs reconstructed from RLinf frames, not a claim of pixel-identical teacher preprocessing.

Legacy pickles do not contain the full resolved run configuration or the initial flow noise. The explicit settings and checkpoint hashes are recorded as provenance; they are not retroactively verified against a running teacher. `episode_index=0` identifies the one attempt for a variant in this export, **not** a verified PCD initial-state index. Every export records `paired_with_pcd=false`. Strict comparison of inference implementations needs a separate initial-state and per-query noise interface.

A Sensor Noise-only pool cannot supply cross-category support. Passing NPZ validation, frozen feature extraction, and a normalized query flow-matching forward means the data interface works. `ready_for_meta_training=false` is expected until real cross-category supports are available. Never duplicate an episode into another category or relabel the RLinf source as the earlier PCD vanilla source to make this check pass.

## Prepare the mapping in the simulator environment

First update the RLinf checkout without discarding local work:

```bash
cd /data/Xixixi/VLA1/RLinf
git switch feature/pi0-liberoplus-rollout-collection
git pull --ff-only
```

Use the PCD environment to resolve the complete original catalog against its runtime and canonical base tasks. The original classification file remains unchanged.

```bash
conda activate /data/Xixixi/VLA1/conda_envs/pcd-libero-plus
export PCD_ROOT=/data/Xixixi/VLA1/PCD-meta-subspace
export PROBE_ROOT=/data/Xixixi/VLA1/RLinf/results/pi0/collection_probe/goal_sensor20_20261013T131747
export LIBERO_PLUS_ROOT=/data/Xixixi/VLA1/LIBERO-plus
export LIBERO_CONFIG_PATH="$PCD_ROOT/config/libero_plus_runtime"
export PYTHONPATH="$PCD_ROOT:$LIBERO_PLUS_ROOT:${PYTHONPATH:-}"
mkdir -p "$PROBE_ROOT/pcd_check"
cd "$PCD_ROOT"
python -m adapter_training.export_task_mapping_template \
  --suite libero_goal \
  --output "$PROBE_ROOT/pcd_check/base_task_map.json"
```

## Convert and validate the complete probe

Return to the **same RLinf environment that produced the pickles**. Tensor pickle deserialization needs PyTorch, and image preparation needs `openpi_client`. Run the focused tests with `PCD_ROOT` set to include the real PCD compatibility tests:

```bash
cd /data/Xixixi/VLA1/RLinf
export PCD_ROOT=/data/Xixixi/VLA1/PCD-meta-subspace
export PROBE_ROOT=/data/Xixixi/VLA1/RLinf/results/pi0/collection_probe/goal_sensor20_20261013T131747
python -m pytest -q tests/unit_tests/test_libero_plus_export.py
bash scripts/libero_plus/collection/1_convert_probe.sh
```

The scripts write `pcd_export/conversion_report.json` and `pcd_check/cpu_verification.json` under `PROBE_ROOT`. For the reported probe, expect 20 episodes, 16 successes, four failures, and `npz_and_manifest_pass=true`. The raw query loader checks every complete action window. Train/validation assignment is delegated to the existing PCD builder, which splits variants **before** filtering failures.

Conversion reserves a new output directory and publishes it only after the entire batch passes. It never overwrites an export. To rerun conversion, set both `EXPORT_ROOT` and `CHECK_ROOT` to new paths and put the mapping under the new `CHECK_ROOT`, or provide `TASK_MAPPING` explicitly. If conversion succeeded but manifest construction failed, preserve the export and run `toolkits/libero_plus/verify_pcd.py --build-manifest` directly with a new manifest path; do not rerun conversion on that directory.

## Extract features and check a real training query

Use the PCD patched OpenPI checkout, and select an idle GPU:

```bash
conda activate /data/Xixixi/VLA1/conda_envs/openpi-server-torch
export PCD_ROOT=/data/Xixixi/VLA1/PCD-meta-subspace
export OPENPI_ROOT=/data/Xixixi/VLA1/openpi-meta-subspace
export PROBE_ROOT=/data/Xixixi/VLA1/RLinf/results/pi0/collection_probe/goal_sensor20_20261013T131747
export CUDA_VISIBLE_DEVICES=0
cd /data/Xixixi/VLA1/RLinf
bash scripts/libero_plus/collection/2_verify_features.sh
```

This extracts three uniformly selected query frames per successful episode at layers 9, 12, and 15, using the frozen PCD student base. It verifies cache hashes and base identity, checks the action normalization round-trip, and executes a query training forward for each split that has a complete window. There is no optimizer update. Expect `feature_cache_pass=true`, `training_query_batch_and_forward_pass=true`, 16 feature episodes, and a finite flow loss. GPU execution and the user's real pickle inventory must be validated on the server; CPU tests do not replace these checks.

Send the conversion, CPU, and GPU verification JSON files after these commands pass. Full Goal Sensor Noise collection, Camera evaluation, and strict initial-state/noise pairing are subsequent stages.

## Process the Spatial, Object, and LIBERO-10 Sensor/Camera collections

`3_process_full6.sh` processes the six complete vanilla collections stored under
one `spatial_object_10_sensor_camera_<timestamp>` directory. It generates one
canonical task mapping per suite, converts and verifies every pickle batch, and
then merges Sensor Noise and Camera Viewpoints within each suite. The merge
preserves distinct source IDs and uses hard links when the exports share a
filesystem. It extracts one resumable layer-9/12/15 feature cache per merged
suite and runs the normalized flow-matching forward check.

The script uses explicit Python interpreters because mapping generation,
trusted pickle deserialization, and GPU feature extraction require three
different environments. Set `COLLECTION_SET` to select a particular run;
otherwise the latest matching directory is used.

```bash
cd /data/Xixixi/VLA1/RLinf
export COLLECTION_SET=/data/Xixixi/VLA1/RLinf/results/pi0/collection_batches/spatial_object_10_sensor_camera_<timestamp>
export CUDA_VISIBLE_DEVICES=0
export DEVICE=cuda:0
bash scripts/libero_plus/collection/3_process_full6.sh
```

The final report is `pcd_full6_verification.json`. The three merged manifests
and feature caches are under
`pcd_merged/{spatial,object,10}_sensor_camera/`. These are two-category
validation pools. Combining them with earlier Background, Light, and Objects
Layout exports requires a new five-category manifest and therefore a new
feature-cache directory.

## Archive the three complete five-category pools under PCD

After `3_process_full6.sh` passes, combine its Sensor Noise and Camera exports
with the existing Background, Light, and Objects Layout collections:

```bash
cd /data/Xixixi/VLA1/RLinf
export COLLECTION_SET=/data/Xixixi/VLA1/RLinf/results/pi0/collection_batches/spatial_object_10_sensor_camera_<timestamp>
bash scripts/libero_plus/collection/4_archive_full5_to_pcd.sh
```

The script creates Spatial, Object, and LIBERO-10 archives below the PCD v2
`collect/` directory and their five-category manifests below `manifests/`.
All episode files must be created as verified hardlinks; the command never
moves or deletes source data. Existing complete outputs are verified and
reused, while a partial output stops for manual review.

## 中文说明

这些脚本将已经完成的 vanilla π0 RLinf probe 转换为 PCD NPZ，并调用现有 PCD API 验证 manifest、冻结特征缓存和 query 训练输入。上面的命令按环境切换顺序给出：先在 PCD simulator 环境生成完整任务 mapping，再回到原 RLinf 环境转换 pickle，最后在 PCD OpenPI server 环境做 GPU 验证。

当前转换器要求每个选定变体只有一次尝试。它通过分类 ID 找到 `selection.json` 中的记录，再按变体名称匹配 PCD mapping。成功轨迹保留首次成功的动作并截断后续动作；失败轨迹完整保留。短于 50 步的成功轨迹仍会保存，但不会补齐虚假的训练动作。图像已由 RLinf 旋转，转换时只做 PCD 的 224 RGB 缩放与填充。查询帧根据原采集的固定五步 chunk 设置重建。

旧 pickle 未保存完整运行配置、初始状态或逐查询噪声。导出的配置与 checkpoint hash 是显式声明的来源记录，不代表已经验证 teacher 推理配置。`episode_index=0` 表示该变体在本批次的一次尝试，不代表与 PCD 的第零个初始状态严格配对；导出始终标记 `paired_with_pcd=false`。

这批预期转换出 20 条记录，其中成功 16 条、失败四条。CPU 报告应有 `npz_and_manifest_pass=true`。GPU 报告应有 `feature_cache_pass=true`、`training_query_batch_and_forward_pass=true`，并检查 16 个成功 episode 的特征与有限的 flow loss。Sensor Noise 单类别没有跨类别 support，因此 `ready_for_meta_training=false` 是预期结果，不能通过复制轨迹、改类别或伪造 source ID 绕过。GPU 验证调用真实训练输入与 flow-matching 前向，不执行 optimizer update。

转换使用新目录，并在整批通过后发布；重复运行需指定新的 `EXPORT_ROOT` 与 `CHECK_ROOT`，或者明确提供已有 `TASK_MAPPING`。如果 NPZ 已成功导出而 manifest 阶段失败，保留已有导出，直接使用 `verify_pcd.py --build-manifest` 和新的 manifest 路径继续。完成后提供 conversion、CPU、GPU 三份 JSON 报告，再推进全量 Sensor Noise、Camera 和严格配对接口。
