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

"""Convert fixed-chunk CollectEpisode pickles to PCD episode NPZ files."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import pickle
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollectionContract:
    """Settings supplied explicitly for legacy pickles without run metadata.

    Query indices are reconstructible only for fixed chunks starting at zero,
    with no intervention, delay, segment changes, or mid-episode recorder reset.
    Episode indices identify attempts in this collection, not PCD init-state IDs.
    """

    source_id: str
    suite: str = "libero_goal"
    seed: int = 7
    replan_steps: int = 5
    num_wait_steps: int = 10
    denoise_steps: int = 10
    action_horizon: int = 50

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must identify the actual teacher/run")
        if self.suite not in (
            "libero_goal",
            "libero_spatial",
            "libero_object",
            "libero_10",
        ):
            raise ValueError(f"Unsupported suite: {self.suite}")
        if self.seed < 0 or self.num_wait_steps < 0:
            raise ValueError("seed and num_wait_steps must be nonnegative")
        if min(self.replan_steps, self.denoise_steps, self.action_horizon) <= 0:
            raise ValueError("Chunk, denoising, and horizon sizes must be positive")
        if self.replan_steps > self.action_horizon:
            raise ValueError("replan_steps exceeds predicted action_horizon")


def sha256(path: Path) -> str:
    """Hash a file without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    """Record the code revision when conversion runs from a Git checkout."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any, field: str) -> Any:
    result = _array(value)
    if result.size != 1:
        raise ValueError(f"Expected scalar {field}, got {result.shape}")
    return result.item()


def _classification_id(info: dict) -> int | None:
    value = info.get("episode", {}).get("plus_classification_id")
    if value is None:
        return None
    value = _scalar(value, "plus_classification_id")
    if isinstance(value, bool) or not np.isfinite(value) or int(value) != value:
        raise ValueError(f"Invalid classification ID: {value}")
    if int(value) < 1:
        raise ValueError(f"Inactive/padded environment classification ID: {value}")
    return int(value)


def _success(info: dict) -> bool:
    stats = info.get("episode", {})
    flags = [
        bool(_scalar(stats[key], key))
        for key in ("success_once", "success_at_end")
        if key in stats
    ]
    if not flags:
        raise ValueError("Missing per-step success_once/success_at_end signal")
    return any(flags)


def _prompt(observation: dict) -> str:
    value = observation.get("task_descriptions")
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Expected one sliced task description")
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing task description in observation")
    return value.strip()


def prepare_image(image: Any) -> np.ndarray:
    """Resize already rotated RLinf RGB images using the PCD client transform."""
    from openpi_client import image_tools

    value = _array(image)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(
            f"Expected rotated uint8 HWC RGB, got {value.shape}/{value.dtype}"
        )
    # RLinf get_libero_image/get_libero_wrist_image already rotate 180 degrees.
    result = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(np.ascontiguousarray(value), 224, 224)
    )
    if result.shape != (224, 224, 3) or result.dtype != np.uint8:
        raise ValueError("PCD image preprocessing returned an invalid image")
    return np.ascontiguousarray(result)


class _CPUUnpickler(pickle.Unpickler):
    """Load CPU or CUDA torch tensor storages on the CPU."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch

            return lambda data: torch.load(
                io.BytesIO(data), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


def load_episode(path: Path) -> dict:
    """Read a trusted local CollectEpisode pickle, moving tensor storages to CPU."""
    with path.open("rb") as stream:
        result = _CPUUnpickler(stream).load()
    if not isinstance(result, dict):
        raise ValueError(f"Expected an episode dictionary: {path}")
    return result


def convert_episode(
    episode: dict,
    identity: Any,
    classification_id: int,
    contract: CollectionContract,
    *,
    episode_index: int = 0,
    image_processor: Callable[[Any], np.ndarray] = prepare_image,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Export pre-action query observations and executed actions through success.

    CollectEpisode stores n actions and n+1 observations/infos/rewards/flags.
    Action t consumes observation t and produces info t+1. Successful episodes
    end immediately after the first successful executed action. Failures retain
    every action. Queries use observations 0, chunk_size, 2*chunk_size, ... .
    """
    if identity.suite != contract.suite:
        raise ValueError("Task mapping and collection suite disagree")
    actions = np.asarray([_array(a) for a in episode["actions"]], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
        raise ValueError(
            f"Expected nonempty executed actions [T,7], got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("Executed actions contain NaN/Inf")
    count = len(actions)
    for key in ("observations", "infos", "rewards", "terminated", "truncated"):
        if len(episode[key]) != count + 1:
            raise ValueError(f"{key} must contain a reset entry plus {count} steps")
    if "segment_ids" in episode and any(int(s) != 0 for s in episode["segment_ids"]):
        raise ValueError("Segmented episodes cannot reconstruct fixed query steps")
    ids = set()
    for info in episode["infos"]:
        if not isinstance(info, dict):
            raise ValueError("Expected a per-env info dictionary")
        for key in (
            "intervene_flag",
            "intervene_flags",
            "record_reset",
            "pre_record",
            "segment_advance",
        ):
            if key in info and _array(info[key]).any():
                raise ValueError(f"Unsupported intervention/recorder transition: {key}")
        cid = _classification_id(info)
        if cid is not None:
            ids.add(cid)
    if ids != {classification_id}:
        raise ValueError(
            f"Mixed/missing classification IDs: expected {classification_id}, got {ids}"
        )
    if episode["infos"][0].get("episode") and _success(episode["infos"][0]):
        raise ValueError("Episode was already successful before controlled actions")
    successes = np.asarray([_success(i) for i in episode["infos"][1:]], dtype=bool)
    success = bool(successes.any())
    if bool(_scalar(episode["success"], "episode.success")) != success:
        raise ValueError("Episode success summary disagrees with per-step success")
    first = int(np.flatnonzero(successes)[0]) if success else None
    stop = first + 1 if first is not None else count
    prompt = _prompt(episode["observations"][0])
    if any(_prompt(obs) != prompt for obs in episode["observations"]):
        raise ValueError("Task description changed within one episode")
    query_steps = np.arange(0, stop, contract.replan_steps, dtype=np.int64)
    primary, wrist, states = [], [], []
    for step in query_steps:
        obs = episode["observations"][int(step)]
        primary.append(image_processor(obs["main_images"]))
        wrist.append(image_processor(obs["wrist_images"]))
        state = np.asarray(_array(obs["states"]), dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(f"Invalid state at query step {step}")
        states.append(state)
    rewards = np.asarray(
        [_scalar(v, "reward") for v in episode["rewards"][1 : stop + 1]],
        dtype=np.float32,
    )
    if not np.isfinite(rewards).all():
        raise ValueError("Rewards contain NaN/Inf")
    dones = np.asarray(
        [
            bool(_scalar(episode["terminated"][i], "terminated"))
            or bool(_scalar(episode["truncated"][i], "truncated"))
            for i in range(1, stop + 1)
        ],
        dtype=bool,
    )
    if success:
        dones[-1] = True
    elif not dones[-1]:
        raise ValueError("Failure pickle is incomplete: final action is not done")
    metadata = {
        "schema_version": 1,
        "source_id": contract.source_id,
        "policy_name": "pi0_libero_rlinf_vanilla",
        "policy_class": "RLinfOpenPi0ForRLActionPrediction",
        "benchmark": "libero_plus",
        "suite": identity.suite,
        "task_id": identity.task_id,
        "task_name": identity.task_name,
        "base_task_id": identity.base_task_id,
        "perturbation_category": identity.perturbation_category,
        "variant_id": identity.variant_id,
        "classification_id": classification_id,
        "episode_index": episode_index,
        "seed": contract.seed,
        "instruction": prompt,
        "replan_steps": contract.replan_steps,
        "num_wait_steps": contract.num_wait_steps,
        "observation_format": "openpi_libero_prepared_uint8_hwc_v1",
        "action_format": "libero_environment_action_7_v1",
        "success": success,
        "control_steps": stop,
        "queries": len(query_steps),
        "original_control_steps": count,
        "first_success_control_step": first,
        "discarded_actions_after_success": count - stop,
        "collection_contract": asdict(contract),
        "query_steps_origin": "reconstructed_fixed_chunk_from_explicit_contract",
        "image_preprocessing": "rlinf_rotated_rgb_then_pcd_resize_with_pad_224",
        "initial_state_origin": "rlinf_plus_reset; not verified against PCD init state",
        "episode_index_semantics": "attempt_within_source_and_variant",
        "paired_with_pcd": False,
        "rlinf_episode_id": int(episode["episode_id"]),
        "rlinf_rank": int(episode["rank"]),
        "rlinf_env_idx": int(episode["env_idx"]),
    }
    metadata["episode_key"] = ":".join(
        str(metadata[k])
        for k in ("benchmark", "suite", "task_name", "episode_index", "seed")
    )
    arrays = {
        "query_steps": query_steps,
        "primary_images": np.stack(primary),
        "wrist_images": np.stack(wrist),
        "states": np.stack(states),
        "executed_actions": actions[:stop].copy(),
        "rewards": rewards,
        "dones": dones,
        "step_success": successes[:stop],
    }
    return metadata, arrays


def convert_batch(
    input_dir: Path,
    selection_path: Path,
    mapping: Any,
    output_dir: Path,
    contract: CollectionContract,
    *,
    provenance: dict | None = None,
    image_processor: Callable[[Any], np.ndarray] = prepare_image,
) -> dict:
    """Convert one complete batch atomically, rejecting duplicates and missing variants."""
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    selection = json.loads(selection_path.read_text())
    if selection.get("suite") != contract.suite:
        raise ValueError("Selection suite disagrees with collection contract")
    selected = {}
    for row in selection["records"]:
        cid = int(row["id"])
        if cid in selected:
            raise ValueError(f"Duplicate selection classification ID: {cid}")
        # Match by stable name; never assume RLinf reset IDs equal PCD task IDs.
        matches = [
            i
            for i in mapping.identities()
            if i.suite == contract.suite and i.task_name == row["name"]
        ]
        if len(matches) != 1 or matches[0].perturbation_category != row["category"]:
            raise ValueError(
                f"Missing/ambiguous PCD mapping for classification ID {cid}"
            )
        selected[cid] = matches[0]
    paths = sorted(input_dir.rglob("*.pkl"))
    if not paths or not selected:
        raise ValueError("No pickle episodes or selected variants")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    seen, rows = set(), []
    try:
        for path in paths:
            episode = load_episode(path)
            ids = {_classification_id(info) for info in episode["infos"]}
            ids.discard(None)
            if len(ids) != 1:
                raise ValueError(f"Mixed/missing classification IDs: {path}: {ids}")
            cid = ids.pop()
            if cid not in selected:
                raise ValueError(f"Unexpected variant classification ID {cid}: {path}")
            if cid in seen:
                raise ValueError(
                    f"Repeated variant {cid}; this exporter accepts one attempt per variant"
                )
            seen.add(cid)
            metadata, arrays = convert_episode(
                episode, selected[cid], cid, contract, image_processor=image_processor
            )
            source_hash = sha256(path)
            metadata.update(
                source_pickle=str(path.resolve()),
                source_pickle_sha256=source_hash,
                task_mapping_sha256=mapping.sha256,
                selection_sha256=sha256(selection_path),
                server_metadata=provenance or {},
            )
            name = f"episode-{cid}-{source_hash[:16]}.npz"
            with (temporary / name).open("xb") as stream:
                np.savez_compressed(
                    stream,
                    metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
                    **arrays,
                )
            rows.append(
                {
                    "path": name,
                    "classification_id": cid,
                    "success": metadata["success"],
                    "control_steps": metadata["control_steps"],
                    "queries": metadata["queries"],
                    "discarded_actions_after_success": metadata[
                        "discarded_actions_after_success"
                    ],
                    "full_action_windows": int(
                        np.sum(
                            arrays["query_steps"] + contract.action_horizon
                            <= len(arrays["executed_actions"])
                        )
                    ),
                    "sha256": sha256(temporary / name),
                }
            )
        if seen != set(selected):
            raise ValueError(
                f"Missing selected variants: {sorted(set(selected) - seen)}"
            )
        report = {
            "schema_version": 1,
            "source_id": contract.source_id,
            "input_dir": str(input_dir.resolve()),
            "selection_sha256": sha256(selection_path),
            "task_mapping_sha256": mapping.sha256,
            "collected": len(rows),
            "success": sum(r["success"] for r in rows),
            "failure": sum(not r["success"] for r in rows),
            "successful_full_action_windows": sum(
                r["full_action_windows"] for r in rows if r["success"]
            ),
            "paired_with_pcd": False,
            "episodes": sorted(rows, key=lambda r: r["classification_id"]),
            "provenance": provenance or {},
        }
        (temporary / "conversion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        temporary.rename(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return report


def main() -> None:
    """Convert one legacy fixed-chunk vanilla π0 collection batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--task-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pcd-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-wait-steps", type=int, default=10)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument(
        "--confirm-fixed-chunk-vanilla",
        action="store_true",
        required=True,
        help="Confirm vanilla flow_ode collection, fixed chunks, no intervention/delay/recorder resets",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.path.insert(0, str(args.pcd_root.resolve()))
    from adapter_training.task_mapping import TaskMapping

    contract = CollectionContract(
        args.source_id,
        args.suite,
        args.seed,
        args.replan_steps,
        args.num_wait_steps,
        args.denoise_steps,
        args.action_horizon,
    )
    stats = sorted((args.checkpoint_dir / "assets").rglob("norm_stats.json"))
    if not stats:
        raise FileNotFoundError("Teacher checkpoint has no assets/**/norm_stats.json")
    provenance = {
        "origin": "explicit_cli_for_legacy_pickle; verify against original collection log",
        "teacher_checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "weights_sha256": sha256(args.checkpoint_dir / "model.safetensors"),
        "norm_stats_sha256": {
            str(p.relative_to(args.checkpoint_dir)): sha256(p) for p in stats
        },
        "collection_contract": asdict(contract),
        "noise_method": "flow_ode",
        "noise_level": 0.0,
        "adapter": "none",
        "source_commits": {
            "rlinf": git_revision(Path(__file__).resolve().parents[2]),
            "pcd": git_revision(args.pcd_root),
        },
    }
    result = convert_batch(
        args.input_dir,
        args.selection,
        TaskMapping.load(args.task_mapping),
        args.output_dir,
        contract,
        provenance=provenance,
    )
    logger.info(
        "%s", json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2)
    )


if __name__ == "__main__":
    main()
