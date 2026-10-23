#!/usr/bin/env python3

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RESULTS_BASE = Path(
    "/data/Xixixi/VLA1/RLinf-liberoplus-ftm-fla/"
    "results/libero_plus_ftm_fla"
)

SUITES = [
    "libero_goal",
    "libero_10",
    "libero_spatial",
    "libero_object",
]

PERTURBATIONS = [
    ("background", "Background Textures"),
    ("light", "Light Conditions"),
    ("objects", "Objects Layout"),
]


def find_latest_result():
    candidates = sorted(
        path
        for path in RESULTS_BASE.glob(
            "baseline_4suite_bg_light_object_batch25_*"
        )
        if path.is_dir() and (path / "manifests").is_dir()
    )

    if not candidates:
        raise RuntimeError(
            "没有找到 baseline_4suite_bg_light_object_batch25_*。\n"
            "请把结果根目录作为命令行参数传入。"
        )

    return candidates[-1]


def manifest_count(path, suite):
    data = json.loads(path.read_text())

    if isinstance(data, list):
        return len(data)

    if isinstance(data, dict):
        entries = data.get(suite)
        if isinstance(entries, list):
            return len(entries)

        lists = [
            value for value in data.values()
            if isinstance(value, list)
        ]

        if lists:
            return len(max(lists, key=len))

    raise RuntimeError(f"无法读取 manifest 任务数量：{path}")


def read_success_metric(chunk_dir):
    observations = []

    event_files = sorted(
        chunk_dir.rglob("events.out.tfevents*"),
        key=lambda path: path.stat().st_mtime,
    )

    for event_file in event_files:
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

    for preferred_tag in (
        "eval/success_once",
        "env/success_once",
        "success_once",
    ):
        selected = [
            item for item in observations
            if item["tag"] == preferred_tag
        ]

        if selected:
            return max(
                selected,
                key=lambda item: item["wall_time"],
            )

    return max(
        observations,
        key=lambda item: item["wall_time"],
    )


def empty_stats():
    return {
        "successes": 0,
        "evaluated_envs": 0,
        "planned_envs": 0,
        "completed_chunks": 0,
        "planned_chunks": 0,
    }


def add_stats(target, source):
    for key in (
        "successes",
        "evaluated_envs",
        "planned_envs",
        "completed_chunks",
        "planned_chunks",
    ):
        target[key] += source[key]


def finalize_stats(stats):
    result = dict(stats)
    evaluated = result["evaluated_envs"]

    result["success_rate"] = (
        result["successes"] / evaluated
        if evaluated
        else None
    )

    result["complete"] = (
        result["evaluated_envs"] == result["planned_envs"]
        and result["completed_chunks"] == result["planned_chunks"]
    )

    return result


def rate_text(stats):
    rate = stats.get("success_rate")

    if rate is None:
        return "N/A"

    return f"{rate * 100:.2f}%"


def main():
    result_root = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else find_latest_result().resolve()
    )

    if not (result_root / "manifests").is_dir():
        raise RuntimeError(
            f"结果目录中没有 manifests：{result_root}"
        )

    print(f"Result root: {result_root}")

    chunks = []
    cells = defaultdict(empty_stats)

    for suite in SUITES:
        manifest_dir = result_root / "manifests" / suite

        if not manifest_dir.is_dir():
            print(f"WARNING: 缺少 suite manifest 目录：{manifest_dir}")
            continue

        for slug, perturbation in PERTURBATIONS:
            manifest_files = sorted(
                manifest_dir.glob(f"{slug}_*.json")
            )

            for manifest in manifest_files:
                try:
                    chunk_id = int(manifest.stem.rsplit("_", 1)[1])
                except (IndexError, ValueError):
                    print(f"WARNING: 无法解析 chunk ID：{manifest}")
                    continue

                planned = manifest_count(manifest, suite)
                chunk_dir = (
                    result_root
                    / suite
                    / slug
                    / f"chunk_{chunk_id:03d}"
                )

                record = {
                    "suite": suite,
                    "perturbation": perturbation,
                    "slug": slug,
                    "chunk": chunk_id,
                    "planned_envs": planned,
                    "evaluated_envs": 0,
                    "successes": 0,
                    "success_rate": None,
                    "status": "missing_chunk_directory",
                    "manifest": str(manifest),
                    "chunk_dir": str(chunk_dir),
                    "event_file": "",
                    "metric_tag": "",
                }

                cell = cells[(suite, perturbation)]
                cell["planned_envs"] += planned
                cell["planned_chunks"] += 1

                if chunk_dir.is_dir():
                    metric = read_success_metric(chunk_dir)

                    if metric is None:
                        record["status"] = "missing_success_once"
                    elif not 0.0 <= metric["value"] <= 1.0:
                        record["status"] = "invalid_success_rate"
                    else:
                        rate = metric["value"]
                        successes = int(round(rate * planned))

                        record.update({
                            "evaluated_envs": planned,
                            "successes": successes,
                            "success_rate": rate,
                            "status": "complete",
                            "event_file": metric["event_file"],
                            "metric_tag": metric["tag"],
                        })

                        cell["evaluated_envs"] += planned
                        cell["successes"] += successes
                        cell["completed_chunks"] += 1

                chunks.append(record)

    finalized_cells = {
        key: finalize_stats(value)
        for key, value in cells.items()
    }

    print("\n每个套件 × 每个扰动")
    print("=" * 116)
    print(
        f"{'Suite':18s} {'Perturbation':24s} "
        f"{'Success':>9s} {'Evaluated':>10s} {'Planned':>9s} "
        f"{'Rate':>10s} {'Chunks':>12s} {'Complete':>10s}"
    )
    print("-" * 116)

    matrix_rows = []

    for suite in SUITES:
        for _, perturbation in PERTURBATIONS:
            stats = finalized_cells.get(
                (suite, perturbation),
                finalize_stats(empty_stats()),
            )

            print(
                f"{suite:18s} "
                f"{perturbation:24s} "
                f"{stats['successes']:9d} "
                f"{stats['evaluated_envs']:10d} "
                f"{stats['planned_envs']:9d} "
                f"{rate_text(stats):>10s} "
                f"{stats['completed_chunks']:5d}/"
                f"{stats['planned_chunks']:<5d} "
                f"{str(stats['complete']):>10s}"
            )

            matrix_rows.append({
                "suite": suite,
                "perturbation": perturbation,
                **stats,
            })

    suite_summaries = {}

    print("\n每个套件汇总")
    print("=" * 92)
    print(
        f"{'Suite':18s} {'Success':>9s} {'Evaluated':>10s} "
        f"{'Planned':>9s} {'Micro':>10s} "
        f"{'Chunks':>12s} {'Complete':>10s}"
    )
    print("-" * 92)

    for suite in SUITES:
        aggregate = empty_stats()

        for _, perturbation in PERTURBATIONS:
            stats = finalized_cells.get(
                (suite, perturbation),
                empty_stats(),
            )
            add_stats(aggregate, stats)

        aggregate = finalize_stats(aggregate)
        suite_summaries[suite] = aggregate

        print(
            f"{suite:18s} "
            f"{aggregate['successes']:9d} "
            f"{aggregate['evaluated_envs']:10d} "
            f"{aggregate['planned_envs']:9d} "
            f"{rate_text(aggregate):>10s} "
            f"{aggregate['completed_chunks']:5d}/"
            f"{aggregate['planned_chunks']:<5d} "
            f"{str(aggregate['complete']):>10s}"
        )

    perturbation_summaries = {}

    print("\n每个扰动跨套件汇总")
    print("=" * 100)
    print(
        f"{'Perturbation':24s} {'Success':>9s} "
        f"{'Evaluated':>10s} {'Planned':>9s} "
        f"{'Micro':>10s} {'Chunks':>12s} {'Complete':>10s}"
    )
    print("-" * 100)

    for _, perturbation in PERTURBATIONS:
        aggregate = empty_stats()

        for suite in SUITES:
            stats = finalized_cells.get(
                (suite, perturbation),
                empty_stats(),
            )
            add_stats(aggregate, stats)

        aggregate = finalize_stats(aggregate)
        perturbation_summaries[perturbation] = aggregate

        print(
            f"{perturbation:24s} "
            f"{aggregate['successes']:9d} "
            f"{aggregate['evaluated_envs']:10d} "
            f"{aggregate['planned_envs']:9d} "
            f"{rate_text(aggregate):>10s} "
            f"{aggregate['completed_chunks']:5d}/"
            f"{aggregate['planned_chunks']:<5d} "
            f"{str(aggregate['complete']):>10s}"
        )

    overall = empty_stats()

    for stats in finalized_cells.values():
        add_stats(overall, stats)

    overall = finalize_stats(overall)

    valid_cell_rates = [
        row["success_rate"]
        for row in matrix_rows
        if row["success_rate"] is not None
    ]

    macro_12 = (
        sum(valid_cell_rates) / len(valid_cell_rates)
        if valid_cell_rates
        else None
    )

    print("\n总体结果")
    print("=" * 72)
    print(
        f"Micro-all : {overall['successes']}/"
        f"{overall['evaluated_envs']} "
        f"= {rate_text(overall)}"
    )
    print(
        f"Macro-12  : {macro_12 * 100:.2f}%"
        if macro_12 is not None
        else "Macro-12  : N/A"
    )
    print(
        f"完成环境  : {overall['evaluated_envs']}/"
        f"{overall['planned_envs']}"
    )
    print(
        f"完成 chunks: {overall['completed_chunks']}/"
        f"{overall['planned_chunks']}"
    )

    missing = [
        record for record in chunks
        if record["status"] != "complete"
    ]

    if missing:
        print(f"\nWARNING：共有 {len(missing)} 个未完成 chunk：")

        for record in missing:
            print(
                f"  {record['suite']:18s} "
                f"{record['perturbation']:24s} "
                f"chunk_{record['chunk']:03d} "
                f"{record['status']}"
            )
    else:
        print("\nPASS：4 个套件 × 3 个扰动全部完成。")

    summary = {
        "method": "pi0_baseline",
        "result_root": str(result_root),
        "matrix": matrix_rows,
        "by_suite": suite_summaries,
        "by_perturbation": perturbation_summaries,
        "overall": overall,
        "macro_12_success_rate": macro_12,
        "complete": not missing,
        "missing_chunks": missing,
    }

    summary_json = result_root / "baseline_4suite_3pert_summary.json"
    matrix_csv = result_root / "baseline_4suite_3pert_matrix.csv"
    chunks_csv = result_root / "baseline_4suite_3pert_chunks.csv"

    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    matrix_fields = [
        "suite",
        "perturbation",
        "successes",
        "evaluated_envs",
        "planned_envs",
        "success_rate",
        "completed_chunks",
        "planned_chunks",
        "complete",
    ]

    with matrix_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=matrix_fields,
        )
        writer.writeheader()
        writer.writerows(matrix_rows)

    chunk_fields = [
        "suite",
        "perturbation",
        "slug",
        "chunk",
        "successes",
        "evaluated_envs",
        "planned_envs",
        "success_rate",
        "status",
        "manifest",
        "chunk_dir",
        "metric_tag",
        "event_file",
    ]

    with chunks_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=chunk_fields,
        )
        writer.writeheader()
        writer.writerows(chunks)

    print("\n已保存：")
    print(f"  {summary_json}")
    print(f"  {matrix_csv}")
    print(f"  {chunks_csv}")


if __name__ == "__main__":
    main()
