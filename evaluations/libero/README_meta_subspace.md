# Native Meta-Subspace evaluation on LIBERO-PLUS

`libero_goal_pi0_meta_subspace_eval.yaml` evaluates a PCD Meta-Subspace
checkpoint inside RLinf. The rollout worker loads the frozen OpenPI Pi0
checkpoint first, then overlays the frozen TaskEncoder/Demix/FiLM/Mask
parameters. It does not start an OpenPI policy server.

The evaluation config uses 25 environments in parallel, replans every five
actions (`num_action_chunks: 5`), waits 10 simulator steps after reset, and
uses seed 7. The behavior-preserve smoke-test settings are support size 2,
support seed 7, intervention alpha 0.25, and the component mode `full`.
By default, all seven LIBERO action dimensions come from Meta-Subspace.

## Required artifacts

- An original `pi0_libero` checkpoint directory. With base-identity checking
  enabled, it must contain the exact `model.safetensors` and normalization
  files recorded by PCD training.
- The PCD Meta-Subspace checkpoint (`.pt`).
- The training pool manifest (`manifest.json`).
- The feature-cache directory containing `index.json` and the `.npz` shards.
- The task mapping used to create the manifest.
- The official LIBERO-PLUS classification JSON.

The checkpoint schema and tensor keys are validated. The manifest, feature
index, task mapping, feature shards, and base Pi0 weights are hash-checked
before evaluation. A mismatch fails fast instead of silently evaluating a
different experiment.

## Run

From the RLinf repository root:

```bash
export LIBERO_TYPE=plus
export OPENPI_CHECKPOINT=/absolute/path/to/pi0_libero
export META_SUBSPACE_CHECKPOINT=/absolute/path/to/meta_subspace_best.pt
export META_SUBSPACE_MANIFEST=/absolute/path/to/manifest.json
export META_SUBSPACE_FEATURE_DIR=/absolute/path/to/feature_cache
export META_SUBSPACE_TASK_MAPPING=/absolute/path/to/task_mapping.json
export LIBERO_CATEGORY='YOUR_CATEGORY'
export LIBERO_CLASSIFICATION_PATH=/absolute/path/to/task_classification.json

bash evaluations/run_eval.sh \
  libero libero_goal_pi0_meta_subspace_eval
```

To evaluate the six arm dimensions from Meta-Subspace while retaining the
base Pi0 gripper command, use:

```bash
bash evaluations/run_eval.sh \
  libero libero_goal_pi0_meta_subspace_eval \
  rollout.model.openpi.meta_subspace.action_mode=meta_action6_base_gripper
```

This mode runs Meta-Subspace and base Pi0 with the same initial flow noise,
combines their model-space action chunks before LIBERO output transforms, and
therefore adds one base-policy forward pass per replanning step. Use
`action_mode=all_meta` for the original seven-dimensional Meta-Subspace action.

Set `LIBERO_CATEGORY` to the exact category name in the classification JSON.
To change parallelism without editing the checked-in config, append a Hydra
override such as:

```bash
bash evaluations/run_eval.sh \
  libero libero_goal_pi0_meta_subspace_eval \
  env.eval.total_num_envs=10
```

The task mapping is resolved independently for each environment in the batch.
With `strict: true`, an unknown task, category mismatch, missing support set,
or modified artifact stops evaluation. Set
`rollout.model.openpi.meta_subspace.strict=false` only for diagnostics; such
samples fall back to the unmodified Pi0 path and are masked out of the
intervention.
