#!/usr/bin/env python3

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RESULTS_BASE = Path(
    "/data/Xixixi/VLA1/RLinf-liberoplus-ftm-fla/"
    "results/libero_plus_ftm_fla"
)

CATEGORIES = [
    "Background Textures",
    "Light Conditions",
    "Camera Viewpoints",
    "Sensor Noise",
    "Objects Layout",
]


def normalize_category(name):
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    if key.startswith(("background", "texture")):
        return "Background Textures"
    if key.startswith(("light", "lighting")):
        return "Light Conditions"
    if key.startswith(("camera", "viewpoint")):
        return "Camera Viewpoints"
    if key.startswith(("sensor", "noise")):
        return "Sensor Noise"
    if key.startswith(("object", "layout")):
        return "Objects Layout"

    return None


def find_result_root():
    candidates = []

    if RESULTS_BASE.is_dir():
        for path in RESULTS_BASE.rglob("*"):
            if not path.is_dir():
                continue

            name = path.name.lower()
            manifests = path / "manifests"

            if (
                "object" in name
                and "fla" in name
                and manifests.is_dir()
                and list(manifests.glob("*.json"))
            ):
                candidates.append(path)

    if not candidates:
        raise RuntimeError(
            "Cannot auto-detect Object FLA result directory. "
            "Pass RESULT_ROOT explicitly."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def count_manifest_entries(path):
    data = json.loads(path.read_text())

    if isinstance(data, list):
        return len(data)

    if not isinstance(data, dict):
        raise RuntimeError(f"Unsupported manifest format: {path}")

    preferred_keys = (
        "libero_object",
        "libero_object_plus",
        "tasks",
        "entries",
        "episodes",
    )

    for key in preferred_keys:
        value = data.get(key)
        if isinstance(value, list):
            return len(value)

    lists = [value for value in data.values() if isinstance(value, list)]

    if not lists:
        raise RuntimeError(f"No task list found in manifest: {path}")

    return len(max(lists, key=len))


def parse_manifest_name(path):
    match = re.match(r"(.+)_([0-9]+)$", path.stem)

    if not match:
        raise RuntimeError(f"Unexpected manifest filename: {path.name}")

    prefix = match.group(1)
    chunk_id = int(match.group(2))
    category = normalize_category(prefix)

    if category is None:
        raise RuntimeError(
            f"Cannot infer category from manifest filename: {path.name}"
        )

    return category, prefix, chunk_id


def find_chunk_dir(root, category, prefix, chunk_id):
    chunk_name = f"chunk_{chunk_id:03d}"
    candidates = []

    direct_names = {
        prefix,
        prefix.lower(),
        prefix.replace("_", "-"),
        prefix.replace("-", "_"),
    }

    for name in direct_names:
        path = root / name / chunk_name
        if path.is_dir():
            candidates.append(path)

    for path in root.glob(f"*/{chunk_name}"):
        if normalize_category(path.parent.name) == category:
            candidates.append(path)

    if not candidates:
        return None

    return max(set(candidates), key=lambda p: p.stat().st_mtime)


def read_success_rate(chunk_dir):
    event_files = sorted(
        chunk_dir.rglob("events.out.tfevents*"),
        key=lambda p: p.stat().st_mtime,
    )

    observations = []

    for event_file in event_files:
        try:
            accumulator = EventAccumulator(
                str(event_file),
                size_guidance={"scalars": 0},
            )
            accumulator.Reload()
        except Exception as exc:
            print(f"WARNING: failed to read {event_file}: {exc}")
            continue

        scalar_tags = accumulator.Tags().get("scalars", [])
        success_tags = [
            tag for tag in scalar_tags
            if tag == "success_once" or tag.endswith("/success_once")
        ]

        for tag in success_tags:
            for event in accumulator.Scalars(tag):
                observations.append(
                    {
                        "wall_time": float(event.wall_time),
                        "step": int(event.step),
                        "value": float(event.value),
                        "tag": tag,
                        "event_file": str(event_file),
                    }
                )

    if not observations:
        return None

    preferred_tags = [
        "eval/success_once",
        "env/success_once",
        "success_once",
    ]

    for tag in preferred_tags:
        matching = [item for item in observations if item["tag"] == tag]
        if matching:
            return max(matching, key=lambda item: item["wall_time"])

    return max(observations, key=lambda item: item["wall_time"])


def main():
    if len(sys.argv) > 1:
        root = Path(sys.argv[1]).resolve()
    else:
        root = find_result_root().resolve()

    manifests_dir = root / "manifests"

    if not manifests_dir.is_dir():
        raise RuntimeError(f"Missing manifests directory: {manifests_dir}")

    manifest_files = sorted(manifests_dir.glob("*.json"))

    if not manifest_files:
        raise RuntimeError(f"No manifests found under: {manifests_dir}")

    records = []

    for manifest in manifest_files:
        category, prefix, chunk_id = parse_manifest_name(manifest)
        num_envs = count_manifest_entries(manifest)
        chunk_dir = find_chunk_dir(
            root,
            category,
            prefix,
            chunk_id,
        )

        record = {
            "category": category,
            "chunk": chunk_id,
            "manifest": str(manifest),
            "chunk_dir": str(chunk_dir) if chunk_dir else "",
            "planned_envs": num_envs,
            "evaluated_envs": 0,
            "successes": 0,
            "success_rate": None,
            "tag": "",
            "status": "missing_chunk_dir",
        }

        if chunk_dir is not None:
            metric = read_success_rate(chunk_dir)

            if metric is None:
                record["status"] = "missing_success_once"
            else:
                rate = metric["value"]

                if not 0.0 <= rate <= 1.0:
                    record["status"] = f"invalid_success_rate:{rate}"
                else:
                    successes = int(round(rate * num_envs))

                    record.update(
                        {
                            "evaluated_envs": num_envs,
                            "successes": successes,
                            "success_rate": rate,
                            "tag": metric["tag"],
                            "event_file": metric["event_file"],
                            "status": "complete",
                        }
                    )

        records.append(record)

    grouped = defaultdict(
        lambda: {
            "planned_envs": 0,
            "evaluated_envs": 0,
            "successes": 0,
            "planned_chunks": 0,
            "completed_chunks": 0,
        }
    )

    for record in records:
        item = grouped[record["category"]]
        item["planned_envs"] += record["planned_envs"]
        item["planned_chunks"] += 1

        if record["status"] == "complete":
            item["evaluated_envs"] += record["evaluated_envs"]
            item["successes"] += record["successes"]
            item["completed_chunks"] += 1

    category_rows = []

    print(f"\nResult root: {root}")
    print("\nObject FLA results")
    print("=" * 94)
    print(
        f"{'Category':24s} {'Success':>9s} {'Evaluated':>10s} "
        f"{'Planned':>9s} {'Rate':>10s} {'Chunks':>12s}"
    )
    print("-" * 94)

    for category in CATEGORIES:
        item = grouped[category]
        evaluated = item["evaluated_envs"]
        rate = item["successes"] / evaluated if evaluated else None
        rate_text = f"{100.0 * rate:8.2f}%" if rate is not None else "N/A"

        print(
            f"{category:24s} "
            f"{item['successes']:9d} "
            f"{evaluated:10d} "
            f"{item['planned_envs']:9d} "
            f"{rate_text:>10s} "
            f"{item['completed_chunks']:5d}/{item['planned_chunks']:<5d}"
        )

        category_rows.append(
            {
                "category": category,
                **item,
                "success_rate": rate,
                "complete": (
                    item["evaluated_envs"] == item["planned_envs"]
                    and item["completed_chunks"] == item["planned_chunks"]
                ),
            }
        )

    total_successes = sum(row["successes"] for row in category_rows)
    total_evaluated = sum(row["evaluated_envs"] for row in category_rows)
    total_planned = sum(row["planned_envs"] for row in category_rows)
    completed_chunks = sum(row["completed_chunks"] for row in category_rows)
    planned_chunks = sum(row["planned_chunks"] for row in category_rows)

    micro = (
        total_successes / total_evaluated
        if total_evaluated
        else None
    )

    valid_category_rates = [
        row["success_rate"]
        for row in category_rows
        if row["success_rate"] is not None
    ]
    macro = (
        sum(valid_category_rates) / len(valid_category_rates)
        if valid_category_rates
        else None
    )

    print("-" * 94)
    print(
        f"Micro-all: {total_successes}/{total_evaluated} "
        f"= {100.0 * micro:.2f}%"
        if micro is not None
        else "Micro-all: N/A"
    )
    print(
        f"Macro-5 : {100.0 * macro:.2f}%"
        if macro is not None
        else "Macro-5: N/A"
    )
    print(
        f"Completeness: chunks={completed_chunks}/{planned_chunks}, "
        f"envs={total_evaluated}/{total_planned}"
    )

    missing = [record for record in records if record["status"] != "complete"]

    if missing:
        print("\nWARNING: incomplete chunks")
        for record in missing:
            print(
                f"  {record['category']} chunk_{record['chunk']:03d}: "
                f"{record['status']}"
            )
    else:
        print("\nPASS: all Object FLA chunks are complete.")

    detail_csv = root / "object_fla_chunk_details.csv"
    summary_csv = root / "object_fla_summary.csv"
    summary_json = root / "object_fla_summary.json"

    detail_fields = sorted(
        set().union(*(record.keys() for record in records))
    )

    with detail_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(records)

    with summary_csv.open("w", newline="") as handle:
        fields = [
            "category",
            "successes",
            "evaluated_envs",
            "planned_envs",
            "success_rate",
            "completed_chunks",
            "planned_chunks",
            "complete",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(category_rows)

    payload = {
        "method": "fla",
        "suite": "libero_object",
        "result_root": str(root),
        "categories": category_rows,
        "micro_all": {
            "successes": total_successes,
            "evaluated_envs": total_evaluated,
            "planned_envs": total_planned,
            "success_rate": micro,
        },
        "macro_5_success_rate": macro,
        "completed_chunks": completed_chunks,
        "planned_chunks": planned_chunks,
        "complete": not missing,
        "missing_chunks": missing,
    }

    summary_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )

    print("\nSaved:")
    print(f"  {summary_json}")
    print(f"  {summary_csv}")
    print(f"  {detail_csv}")


if __name__ == "__main__":
    main()
