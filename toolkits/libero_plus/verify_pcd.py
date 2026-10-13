# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify exported episodes using the actual PCD manifest and training APIs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def verify_episodes(episodes: Path, manifest: Path, mapping: Path) -> dict:
    """Check every episode and load all complete windows through PCD MetaPool."""
    from adapter_training.build_dataset import sha256
    from adapter_training.meta_subspace.data import MetaPool, validate_episode
    from adapter_training.task_mapping import TaskMapping
    from adapter_training.trajectory import read_metadata

    pool = MetaPool(manifest)
    task_mapping = TaskMapping.load(mapping)
    report = json.loads((episodes / "conversion_report.json").read_text())
    expected = {row["path"]: row for row in report["episodes"]}
    paths = sorted(episodes.glob("episode-*.npz"))
    if {p.name for p in paths} != set(expected):
        raise ValueError("NPZ inventory differs from conversion report")
    if (
        pool.root != episodes.resolve()
        or pool.document["mapping_sha256"] != task_mapping.sha256
    ):
        raise ValueError("Manifest references a different episode root or mapping")
    success_count, loaded_windows = 0, 0
    for path in paths:
        meta = read_metadata(path)
        task_mapping.lookup(
            suite=meta["suite"], task_id=meta["task_id"], task_name=meta["task_name"]
        )
        validate_episode(path, meta, pool.document["action_horizon"])
        if sha256(path) != expected[path.name]["sha256"]:
            raise ValueError(f"Exported NPZ changed: {path}")
        success_count += int(meta["success"])
        with np.load(path, allow_pickle=False) as data:
            n = meta["control_steps"]
            for field in ("rewards", "dones", "step_success"):
                if data[field].shape != (n,):
                    raise ValueError(f"Invalid step field {field}: {path}")
            if not np.isfinite(data["rewards"]).all() or not data["dones"][-1]:
                raise ValueError(f"Invalid rewards or unfinished trajectory: {path}")
            if meta["success"]:
                if np.flatnonzero(data["step_success"]).tolist() != [n - 1]:
                    raise ValueError(
                        f"Successful episode not cut at first success: {path}"
                    )
                if meta["first_success_control_step"] != n - 1:
                    raise ValueError(f"First-success metadata mismatch: {path}")
            elif data["step_success"].any() or meta["original_control_steps"] != n:
                raise ValueError(f"Failure was truncated or contains success: {path}")
            if not np.array_equal(
                data["query_steps"], np.arange(0, n, meta["replan_steps"])
            ):
                raise ValueError(f"Fixed-chunk query timing mismatch: {path}")
    for row in pool.rows:
        for index in row["query_indices"]:
            raw = pool.raw_query({"query": row, "query_index": index})
            if raw["actions"].shape != (pool.document["action_horizon"], 7):
                raise ValueError(
                    "PCD raw_query did not return a complete action window"
                )
            loaded_windows += 1
    if success_count != report["success"] or len(pool.rows) != success_count:
        raise ValueError(
            "Success inventory differs between NPZ, conversion report and manifest"
        )
    if pool.document["summary"] != {
        "collected": len(paths),
        "success": success_count,
        "failure": len(paths) - success_count,
        "train": sum(r["split"] == "train" for r in pool.rows),
        "validation": sum(r["split"] == "validation" for r in pool.rows),
    }:
        raise ValueError("PCD manifest summary differs from exported inventory")
    eligible_train = len(pool.eligible("train", 2))
    eligible_validation = len(pool.eligible("validation", 2))
    return {
        "npz_and_manifest_pass": True,
        "collected": len(paths),
        "success": success_count,
        "failure": len(paths) - success_count,
        "raw_training_windows_loaded": loaded_windows,
        "eligible_train_tasks_support2": eligible_train,
        "eligible_validation_tasks_support2": eligible_validation,
        "ready_for_meta_training": bool(eligible_train and eligible_validation),
        "note": "A Sensor Noise-only pool has no cross-category supports; zero eligibility is expected.",
    }


def verify_features_and_batch(
    manifest: Path, features: Path, checkpoint: Path, device: str
) -> dict:
    """Load real cached features and run the normalized PCD query training path."""
    import torch
    from adapter_training.meta_subspace.data import MetaPool
    from adapter_training.meta_subspace.features import FeatureCache, observation_batch
    from adapter_training.runtime import (
        base_identity,
        load_base,
        verify_transform_roundtrip,
    )

    pool = MetaPool(manifest)
    policy, _ = load_base("pi0_libero", checkpoint, torch.device(device))
    identity = base_identity("pi0_libero", checkpoint)
    # Constructing the cache verifies the exact manifest and student checkpoint.
    cache = FeatureCache(features, pool, site="pre_joint_layer_15", identity=identity)
    for row in pool.rows:
        vector = cache.episode(row)
        if vector.shape != (cache.dimension,) or not np.isfinite(vector).all():
            raise ValueError("Invalid feature vector")
    windows = [r for r in pool.rows if r["query_indices"]]
    if not windows:
        raise ValueError(
            "No successful complete 50-step query window after success truncation"
        )
    # Exercise both splits when they contain complete windows. No invented supports.
    checked = []
    for split in ("train", "validation"):
        rows = [r for r in windows if r["split"] == split]
        if not rows:
            continue
        raw = pool.raw_query(
            {"query": rows[0], "query_index": rows[0]["query_indices"][0]}
        )
        verify_transform_roundtrip(policy, raw)
        observation, actions = observation_batch(policy, [raw], device)
        if (
            actions.shape[:2] != (1, pool.document["action_horizon"])
            or not torch.isfinite(actions).all()
        ):
            raise ValueError("Invalid normalized PCD training batch")
        generator = torch.Generator(device=device).manual_seed(7)
        noise = torch.randn(actions.shape, generator=generator, device=device)
        time = torch.full((1,), 0.5, device=device)
        with torch.no_grad():
            predicted, target = policy._model.flow_velocity(
                observation,
                actions,
                noise=noise,
                time=time,
                preprocess_train=False,
                adapter_enabled=False,
            )
            if predicted.shape != actions.shape or target.shape != actions.shape:
                raise ValueError("PCD flow target/prediction changed the batch shape")
            if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
                raise ValueError("PCD flow prediction/target contains NaN/Inf")
            loss = (predicted[:, :, :7] - target[:, :, :7]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("PCD flow-matching training forward produced NaN/Inf")
        checked.append(
            {
                "split": split,
                "actions_shape": list(actions.shape),
                "flow_loss": float(loss.cpu()),
            }
        )
    return {
        "feature_cache_pass": True,
        "feature_episodes_checked": len(pool.rows),
        "feature_dim": cache.dimension,
        "training_query_batch_and_forward_pass": True,
        "checked_batches": checked,
        "full_meta_adapter_update_tested": False,
    }


def main() -> None:
    """Verify NPZ/manifest on CPU or cached features and training inputs on GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd-root", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--task-mapping", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.path.insert(0, str(args.pcd_root.resolve()))
    if bool(args.features) != bool(args.checkpoint_dir):
        parser.error("--features and --checkpoint-dir must be supplied together")
    if args.build_manifest:
        from adapter_training.meta_subspace.data import build_manifest

        build_manifest(
            args.episodes,
            args.task_mapping,
            args.manifest,
            horizon=50,
            validation_fraction=args.validation_fraction,
            seed=7,
        )
    result = verify_episodes(args.episodes, args.manifest, args.task_mapping)
    if args.features:
        result.update(
            verify_features_and_batch(
                args.manifest, args.features, args.checkpoint_dir, args.device
            )
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("%s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
