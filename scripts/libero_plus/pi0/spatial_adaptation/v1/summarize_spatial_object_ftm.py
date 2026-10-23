#!/usr/bin/env python3

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


BASE = Path(
    "/data/Xixixi/VLA1/RLinf-liberoplus-ftm-fla/"
    "results/libero_plus_ftm_fla"
)

CATEGORY_ORDER = [
    "Background Textures",
    "Light Conditions",
    "Camera Viewpoints",
    "Sensor Noise",
    "Objects Layout",
]


def category_name(text):
    key = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

    if key.startswith("background"):
        return "Background Textures"
    if key.startswith("light"):
        return "Light Conditions"
    if key.startswith("camera"):
        return "Camera Viewpoints"
    if key.startswith("sensor"):
        return "Sensor Noise"
    if key.startswith("object"):
        return "Objects Layout"

    return text


def find_latest_root():
    candidates = [
        path for path in BASE.glob("ftm_spatial_object_seq_*")
        if (path / "spatial").is_dir() and (path / "object").is_dir()
    ]

    if not candidates:
        raise RuntimeError(
            "没有找到 ftm_spatial_object_seq_*；"
            "请把结果根目录作为第一个参数传入。"
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def manifest_count(path):
    data = json.loads(path.read_text())

    if isinstance(data, list):
        return len(data)

    lists = [
        value for value in data.values()
        if isinstance(value, list)
    ]

    if not lists:
        raise RuntimeError(f"Manifest 中没有任务列表：{path}")

    return len(max(lists, key=len))


def find_chunk_dir(suite_root, prefix, chunk_id):
    expected = f"chunk_{chunk_id:03d}"
    category = category_name(prefix)

    direct = suite_root / prefix / expected
    if direct.is_dir():
        return direct

    candidates = [
        path for path in suite_root.glob(f"*/{expected}")
        if category_name(path.parent.name) == category
    ]

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def success_metric(chunk_dir):
    observations = []

    for event_file in chunk_dir.rglob("events.out.tfevents*"):
        try:
            accumulator = EventAccumulator(
                str(event_file),
                size_guidance={"scalars": 0},
            )
            accumulator.Reload()
        except Exception as exc:
            print(f"WARNING: 无法读取 {event_file}: {exc}")
            continue

        tags = accumulator.Tags().get("scalars", [])

        for tag in tags:
            if tag != "success_once" and not tag.endswith("/success_once"):
                continue

            for event in accumulator.Scalars(tag):
                observations.append({
                    "tag": tag,
                    "value": float(event.value),
                    "step": int(event.step),
                    "wall_time": float(event.wall_time),
                    "event_file": str(event_file),
                })

    if not observations:
        return None

    for preferred in (
        "eval/success_once",
        "env/success_once",
        "success_once",
    ):
        selected = [
            item for item in observations
            if item["tag"] == preferred
        ]
        if selected:
            return max(selected, key=lambda item: item["wall_time"])

    return max(observations, key=lambda item: item["wall_time"])


def summarize_suite(suite_root, suite):
    manifests = sorted((suite_root / "manifests").glob("*.json"))

    if not manifests:
        raise RuntimeError(f"没有找到 manifests：{suite_root}")

    records = []
    grouped = defaultdict(
        lambda: {
            "successes": 0,
            "evaluated": 0,
            "planned": 0,
            "completed_chunks": 0,
            "planned_chunks": 0,
        }
    )

    for manifest in manifests:
        match = re.match(r"(.+)_([0-9]+)$", manifest.stem)

        if not match:
            continue

        prefix = match.group(1)
        chunk_id = int(match.group(2))
        category = category_name(prefix)
        planned = manifest_count(manifest)
        chunk_dir = find_chunk_dir(suite_root, prefix, chunk_id)

        record = {
            "category": category,
            "chunk": chunk_id,
            "planned": planned,
            "evaluated": 0,
            "successes": 0,
            "success_rate": None,
            "manifest": str(manifest),
            "chunk_dir": str(chunk_dir) if chunk_dir else None,
            "status": "missing_chunk",
        }

        grouped[category]["planned"] += planned
        grouped[category]["planned_chunks"] += 1

        if chunk_dir is not None:
            metric = success_metric(chunk_dir)

            if metric is None:
                record["status"] = "missing_success_once"
            elif not 0.0 <= metric["value"] <= 1.0:
                record["status"] = "invalid_success_rate"
            else:
                rate = metric["value"]
                successes = round(rate * planned)

                record.update({
                    "evaluated": planned,
                    "successes": successes,
                    "success_rate": rate,
                    "metric_tag": metric["tag"],
                    "event_file": metric["event_file"],
                    "status": "complete",
                })

                grouped[category]["evaluated"] += planned
                grouped[category]["successes"] += successes
                grouped[category]["completed_chunks"] += 1

        records.append(record)

    print(f"\n{'=' * 86}")
    print(f"{suite.upper()} FTM")
    print(f"目录：{suite_root}")
    print(f"{'=' * 86}")
    print(
        f"{'Category':24s} {'Success':>9s} {'Evaluated':>10s} "
        f"{'Planned':>9s} {'Rate':>10s} {'Chunks':>11s}"
    )
    print("-" * 86)

    rows = []

    for category in CATEGORY_ORDER:
        item = grouped[category]
        rate = (
            item["successes"] / item["evaluated"]
            if item["evaluated"]
            else None
        )

        rows.append({
            "category": category,
            **item,
            "success_rate": rate,
        })

        rate_text = f"{rate * 100:8.2f}%" if rate is not None else "N/A"

        print(
            f"{category:24s} "
            f"{item['successes']:9d} "
            f"{item['evaluated']:10d} "
            f"{item['planned']:9d} "
            f"{rate_text:>10s} "
            f"{item['completed_chunks']:4d}/{item['planned_chunks']:<4d}"
        )

    success = sum(row["successes"] for row in rows)
    evaluated = sum(row["evaluated"] for row in rows)
    planned = sum(row["planned"] for row in rows)
    completed_chunks = sum(row["completed_chunks"] for row in rows)
    planned_chunks = sum(row["planned_chunks"] for row in rows)

    micro = success / evaluated if evaluated else None
    valid_rates = [
        row["success_rate"]
        for row in rows
        if row["success_rate"] is not None
    ]
    macro = sum(valid_rates) / len(valid_rates) if valid_rates else None

    print("-" * 86)
    print(
        f"Micro-all: {success}/{evaluated} = {micro * 100:.2f}%"
        if micro is not None else "Micro-all: N/A"
    )
    print(
        f"Macro-5 : {macro * 100:.2f}%"
        if macro is not None else "Macro-5: N/A"
    )
    print(
        f"完整性：chunks={completed_chunks}/{planned_chunks}, "
        f"envs={evaluated}/{planned}"
    )

    missing = [
        record for record in records
        if record["status"] != "complete"
    ]

    if missing:
        print("WARNING：存在未完成 chunk：")
        for record in missing:
            print(
                f"  {record['category']} "
                f"chunk_{record['chunk']:03d}: {record['status']}"
            )
    else:
        print("PASS：所有 chunk 均已完成。")

    payload = {
        "method": "ftm",
        "suite": suite,
        "result_root": str(suite_root),
        "categories": rows,
        "micro_all": {
            "successes": success,
            "evaluated": evaluated,
            "planned": planned,
            "success_rate": micro,
        },
        "macro_5_success_rate": macro,
        "completed_chunks": completed_chunks,
        "planned_chunks": planned_chunks,
        "complete": not missing,
        "chunks": records,
    }

    output = suite_root / f"{suite}_ftm_summary.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"已保存：{output}")

    return payload


def main():
    root = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else find_latest_root().resolve()
    )

    print(f"FTM evaluation root: {root}")

    combined = {
        "root": str(root),
        "spatial": summarize_suite(root / "spatial", "spatial"),
        "object": summarize_suite(root / "object", "object"),
    }

    output = root / "spatial_object_ftm_summary.json"
    output.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\n合并结果：{output}")


if __name__ == "__main__":
    main()
