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

"""The LIBERO-Plus exporter preserves executed control and query alignment."""

import hashlib
import json
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from toolkits.libero_plus.convert_rollouts import (
    CollectionContract,
    convert_batch,
    convert_episode,
    load_episode,
    prepare_image,
)
from toolkits.libero_plus.merge_exports import _verified_inventory


def make_episode(cid=100, first_success=62, count=80):
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[0, 0] = (10, 20, 30)
    image[-1, -1] = (90, 100, 110)
    observations = [
        {
            "main_images": image,
            "wrist_images": image + 1,
            "states": np.full(8, t, dtype=np.float32),
            "task_descriptions": "pick up the cup",
        }
        for t in range(count + 1)
    ]
    infos = [{}] + [
        {
            "episode": {
                "plus_classification_id": np.asarray(cid),
                "success_once": first_success is not None and t >= first_success,
                # Success is lost later, as in three of the real probe episodes.
                "success_at_end": first_success is not None and t == first_success,
            }
        }
        for t in range(count)
    ]
    return {
        "rank": 0,
        "env_idx": 0,
        "episode_id": 3,
        "success": first_success is not None,
        "observations": observations,
        "actions": [np.full(7, t / 100, dtype=np.float32) for t in range(count)],
        "infos": infos,
        "rewards": [0.0] + [float(t == first_success) for t in range(count)],
        "terminated": [False] * (count + 1),
        "truncated": [False] * count + [True],
    }


def make_identity(task_id=7, name="task_noise_a", category="Sensor Noise"):
    # The PCD ID deliberately differs from cid-1 to test matching by name.
    return SimpleNamespace(
        suite="libero_goal",
        task_id=task_id,
        task_name=name,
        base_task_id="libero_goal::task",
        perturbation_category=category,
        variant_id=f"{category}:{task_id}",
    )


def test_first_success_and_pre_action_queries():
    episode = make_episode(first_success=12)
    meta, arrays = convert_episode(
        episode, make_identity(), 100, CollectionContract("teacher")
    )
    assert meta["control_steps"] == 13
    assert meta["first_success_control_step"] == 12
    assert meta["discarded_actions_after_success"] == 67
    assert arrays["query_steps"].tolist() == [0, 5, 10]
    assert arrays["states"][:, 0].tolist() == [0, 5, 10]
    np.testing.assert_array_equal(arrays["executed_actions"], episode["actions"][:13])
    assert np.flatnonzero(arrays["step_success"]).tolist() == [12]
    assert np.flatnonzero(arrays["dones"]).tolist() == [12]
    assert arrays["rewards"][-1] == 1
    assert meta["paired_with_pcd"] is False
    assert meta["task_id"] == 7


@pytest.mark.parametrize(
    "first,expected", [(0, [0]), (4, [0]), (5, [0, 5]), (9, [0, 5])]
)
def test_success_at_query_boundaries(first, expected):
    meta, arrays = convert_episode(
        make_episode(first_success=first),
        make_identity(),
        100,
        CollectionContract("teacher"),
    )
    assert arrays["query_steps"].tolist() == expected
    assert meta["control_steps"] == first + 1
    assert len(arrays["executed_actions"]) == first + 1


def test_failure_retains_all_executed_actions():
    episode = make_episode(first_success=None)
    meta, arrays = convert_episode(
        episode, make_identity(), 100, CollectionContract("teacher")
    )
    assert meta["success"] is False
    assert meta["discarded_actions_after_success"] == 0
    np.testing.assert_array_equal(arrays["executed_actions"], episode["actions"])
    assert not arrays["step_success"].any()
    assert arrays["dones"][-1]


def test_resize_does_not_rotate_again():
    image = make_episode()["observations"][0]["main_images"]
    np.testing.assert_array_equal(prepare_image(image), image)
    larger = np.zeros((256, 256, 3), dtype=np.uint8)
    larger[:128, :128] = (255, 0, 0)
    result = prepare_image(larger)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8
    assert result[20, 20, 0] == 255
    assert result[-20, -20, 0] == 0


@pytest.mark.parametrize(
    "fault,message",
    [
        ("alignment", "reset entry"),
        ("mixed_id", "classification IDs"),
        ("success", "success summary"),
        ("unfinished", "incomplete"),
        ("intervention", "intervention"),
        ("prompt", "description changed"),
    ],
)
def test_invalid_episode_is_rejected(fault, message):
    episode = make_episode(first_success=None)
    if fault == "alignment":
        episode["observations"].pop()
    elif fault == "mixed_id":
        episode["infos"][3]["episode"]["plus_classification_id"] = 101
    elif fault == "success":
        episode["success"] = True
    elif fault == "unfinished":
        episode["truncated"][-1] = False
    elif fault == "intervention":
        episode["infos"][3]["intervene_flag"] = True
    elif fault == "prompt":
        episode["observations"][3]["task_descriptions"] = "different task"
    with pytest.raises(ValueError, match=message):
        convert_episode(episode, make_identity(), 100, CollectionContract("teacher"))


def test_cpu_tensor_pickle(tmp_path):
    torch = pytest.importorskip("torch")
    episode = make_episode()
    episode["actions"][0] = torch.as_tensor(episode["actions"][0])
    path = tmp_path / "episode.pkl"
    path.write_bytes(pickle.dumps(episode))
    loaded = load_episode(path)
    assert loaded["actions"][0].device.type == "cpu"
    meta, _ = convert_episode(
        loaded, make_identity(), 100, CollectionContract("teacher")
    )
    assert meta["control_steps"] == 63


def test_merge_inventory_requires_reported_unchanged_npz(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    episode = export / "episode-a.npz"
    episode.write_bytes(b"episode")
    digest = hashlib.sha256(episode.read_bytes()).hexdigest()
    (export / "conversion_report.json").write_text(
        json.dumps({"episodes": [{"path": episode.name, "sha256": digest}]})
    )
    assert _verified_inventory(export) == [episode]
    episode.write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed after conversion"):
        _verified_inventory(export)


@pytest.fixture
def pcd_modules():
    root = os.environ.get("PCD_ROOT")
    if not root:
        pytest.skip("Set PCD_ROOT to test against the real PCD checkout")
    sys.path.insert(0, root)
    from adapter_training.meta_subspace.data import MetaPool, build_manifest
    from adapter_training.task_mapping import TaskIdentity, TaskMapping

    from toolkits.libero_plus.verify_pcd import verify_episodes

    return SimpleNamespace(
        MetaPool=MetaPool,
        build_manifest=build_manifest,
        TaskIdentity=TaskIdentity,
        TaskMapping=TaskMapping,
        verify_episodes=verify_episodes,
    )


def make_batch(tmp_path, pcd):
    identities = [make_identity(7 + i, f"task_noise_{i}") for i in range(3)]
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps({"schema_version": 1, "tasks": [vars(i) for i in identities]})
    )
    mapping = pcd.TaskMapping.load(mapping_path)
    selected = [
        {"id": 100 + i, "name": row.task_name, "category": row.perturbation_category}
        for i, row in enumerate(identities)
    ]
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({"suite": "libero_goal", "records": selected}))
    source = tmp_path / "pickles"
    source.mkdir()
    for i in range(3):
        episode = make_episode(cid=100 + i, first_success=62 if i < 2 else None)
        (source / f"episode_{i}.pkl").write_bytes(pickle.dumps(episode))
    return source, selection_path, mapping_path, mapping


def test_actual_pcd_manifest_and_training_window_loading(tmp_path, pcd_modules):
    source, selection, mapping_path, mapping = make_batch(tmp_path, pcd_modules)
    target = tmp_path / "export"
    report = convert_batch(
        source, selection, mapping, target, CollectionContract("teacher")
    )
    assert (report["collected"], report["success"], report["failure"]) == (3, 2, 1)
    manifest = tmp_path / "manifest.json"
    summary = pcd_modules.build_manifest(target, mapping_path, manifest, horizon=50)
    assert summary["collected"] == 3
    assert summary["success"] == 2
    pool = pcd_modules.MetaPool(manifest)
    assert len(pool.rows) == 2
    assert not pool.eligible("train", 2)
    for row in pool.rows:
        assert row["query_indices"] == [0, 1, 2]
        for index in row["query_indices"]:
            raw = pool.raw_query({"query": row, "query_index": index})
            assert raw["actions"].shape == (50, 7)
            assert raw["observation/state"][0] == index * 5
    result = pcd_modules.verify_episodes(target, manifest, mapping_path)
    assert result["npz_and_manifest_pass"] is True
    assert result["raw_training_windows_loaded"] == 6
    assert result["ready_for_meta_training"] is False
    # Check the whole variant split, including failures, rather than resplitting successes.
    assert pool.document["summary"]["failure"] == 1
    assert pool.document["split_unit"].endswith("before_success_filter")


def test_short_success_is_kept_without_fabricated_windows(tmp_path, pcd_modules):
    source, selection, mapping_path, mapping = make_batch(tmp_path, pcd_modules)
    path = source / "episode_0.pkl"
    path.write_bytes(pickle.dumps(make_episode(first_success=4)))
    target = tmp_path / "export"
    convert_batch(source, selection, mapping, target, CollectionContract("teacher"))
    manifest = tmp_path / "manifest.json"
    pcd_modules.build_manifest(target, mapping_path, manifest)
    pool = pcd_modules.MetaPool(manifest)
    short = next(
        row for row in pool.rows if row["metadata"]["classification_id"] == 100
    )
    assert short["metadata"]["success"] is True
    assert short["query_indices"] == []


@pytest.mark.parametrize(
    "fault,message",
    [
        ("missing", "Missing selected"),
        ("duplicate", "Repeated variant"),
        ("unknown", "Unexpected variant"),
    ],
)
def test_invalid_batch_rolls_back_without_partial_output(
    tmp_path, pcd_modules, fault, message
):
    source, selection, _, mapping = make_batch(tmp_path, pcd_modules)
    if fault == "missing":
        (source / "episode_2.pkl").unlink()
    elif fault == "duplicate":
        (source / "duplicate.pkl").write_bytes((source / "episode_0.pkl").read_bytes())
    elif fault == "unknown":
        (source / "episode_2.pkl").write_bytes(pickle.dumps(make_episode(cid=999)))
    target = tmp_path / "export"
    with pytest.raises(ValueError, match=message):
        convert_batch(source, selection, mapping, target, CollectionContract("teacher"))
    assert not target.exists()
    assert not list(tmp_path.glob(".export-*"))


def test_existing_export_is_preserved(tmp_path, pcd_modules):
    source, selection, _, mapping = make_batch(tmp_path, pcd_modules)
    target = tmp_path / "export"
    target.mkdir()
    marker = target / "keep"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        convert_batch(source, selection, mapping, target, CollectionContract("teacher"))
    assert marker.read_text() == "original"


def test_modified_export_is_detected(tmp_path, pcd_modules):
    source, selection, mapping_path, mapping = make_batch(tmp_path, pcd_modules)
    target = tmp_path / "export"
    convert_batch(source, selection, mapping, target, CollectionContract("teacher"))
    manifest = tmp_path / "manifest.json"
    pcd_modules.build_manifest(target, mapping_path, manifest)
    path = next(target.glob("episode-*.npz"))
    with path.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed|modified"):
        pcd_modules.verify_episodes(target, manifest, mapping_path)
