# B/C1/C2/C3 evaluation on LIBERO-Plus

This evaluator loads the four PCD factorial checkpoints natively inside RLinf.
It preserves the existing RLinf LIBERO-Plus protocol: seed 7, 25 environments,
10 reset-wait steps, 300 control steps, and replanning every five actions.

The common training pool is the Goal5 mixed-source pool:

- manifest: `results/libero_plus_meta_subspace/v2/manifests/goal5_mixed_pool.json`
- features: `results/libero_plus_meta_subspace/v2/features/goal5_mixed_uniform3`

PCD- and RLinf-collected episodes may coexist in this pool.  Every raw support
rollout, feature shard, manifest, task mapping, base checkpoint, and factorial
checkpoint is hash-checked before it is used.

For C2/C3, RLinf reproduces the PCD serving contract rather than approximating
it: two deterministic train/success/cross-category supports, one deterministic
full-action chunk per support, one FP32 Mask-logit update with LR 100, and no
`delta_u` on Support.  C3 alone keeps the raw FP32 logit update as the
Residual MLP condition on Query.

## Run one arm

```bash
cd /data/Xixixi/VLA1/RLinf
git fetch origin
git switch feature/pi0-liberoplus-task-residual-mask-factorial-eval-v1
git pull --ff-only

CUDA_VISIBLE_DEVICES=0 \
INTERVENTION_STRENGTH=0.5 \
FACTORIAL_ACTION_MODE=all_meta \
bash scripts/libero_plus/meta_subspace/eval_factorial_arm.sh c3_full
```

Valid arms are `b_residual`, `c1_joint_mask`, `c2_bilevel`, and
`c3_full`.  Run all four sequentially with
`scripts/libero_plus/meta_subspace/eval_factorial_all.sh`.

The default five categories are Background Textures, Light Conditions,
Objects Layout, Camera Viewpoints, and Sensor Noise.  Restrict a smoke test
with, for example,
`EVAL_CATEGORY_KEYS="sensor_noise" TOTAL_NUM_ENVS=2 EVAL_EPOCHS=1`.

Use the same `INTERVENTION_STRENGTH`, `FACTORIAL_ACTION_MODE`, category
scope, seed, checkpoint step, and task ordering for every arm.  Set
`FACTORIAL_ACTION_MODE=meta_action6_base_gripper` only when that mixed-action
protocol is the intended comparison for all four arms.
