# LIBERO-PLUS FTM/FLA evaluation

The FTM and FLA configurations are additive. Existing pi0, visual-adapter, and
Meta-Subspace evaluation files are unchanged.

The existing `rollout.model.trainable_checkpoint_path` interface accepts:

- normal RLinf `trainable_weights.pt` payloads; and
- native PCD `libero_spatial_adaptation_ftm_fla_v1` checkpoints.

Native PCD FTM keys are mapped to `visual_ftm.gamma/beta`. Native PCD FLA
`lora_a/lora_b` tensors are mapped to the RLinf SigLIP
`fc1/fc2.lora_A/lora_B.weight` keys. Rank, scale, module count, runtime keys,
and tensor shapes are checked before loading.

## Common environment

```bash
export LIBERO_TYPE=plus
export OPENPI_CHECKPOINT=/data/Xixixi/VLA1/model_cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch
export LIBERO_CLASSIFICATION_PATH=/data/Xixixi/VLA1/LIBERO-plus/libero/libero/benchmark/task_classification.json
export LIBERO_CATEGORY=''
```

## FTM

```bash
export SPATIAL_ADAPTATION_CHECKPOINT=/data/Xixixi/VLA1/PCD-liberoplus-ftm-fla/results/libero_plus_ftm_fla/all_success/ftm/latest.pt
bash evaluations/run_eval.sh libero libero_goal_pi0_ftm_eval
```

## FLA

```bash
export SPATIAL_ADAPTATION_CHECKPOINT=/data/Xixixi/VLA1/PCD-liberoplus-ftm-fla/results/libero_plus_ftm_fla/all_success/fla/latest.pt
bash evaluations/run_eval.sh libero libero_goal_pi0_fla_eval
```

Override `env.eval.total_num_envs`, `env.eval.group_size`, logging paths, or
`LIBERO_CATEGORY` exactly as for the existing Meta-Subspace evaluation.
