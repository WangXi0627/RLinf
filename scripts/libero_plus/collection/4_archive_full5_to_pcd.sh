#!/usr/bin/env bash
# Archive Spatial/Object/LIBERO-10 five-category pools under PCD using hardlinks.
set -euo pipefail

RLINF_ROOT="${RLINF_ROOT:-/data/Xixixi/VLA1/RLinf}"
PCD_ROOT="${PCD_ROOT:-/data/Xixixi/VLA1/PCD-meta-subspace}"
COLLECTION_SET="${COLLECTION_SET:-${RLINF_ROOT}/results/pi0/collection_batches/spatial_object_10_sensor_camera_20261014T124616}"
PCD_PYTHON="${PCD_PYTHON:-/data/Xixixi/VLA1/conda_envs/pcd-libero-plus/bin/python}"
ARCHIVE_DATE="${ARCHIVE_DATE:-20261014}"

test -x "${PCD_PYTHON}"
test -d "${PCD_ROOT}"
test -d "${COLLECTION_SET}"

export PYTHONPATH="${PCD_ROOT}:${PYTHONPATH:-}"

"${PCD_PYTHON}" - \
  "${PCD_ROOT}" "${COLLECTION_SET}" "${ARCHIVE_DATE}" <<'PY'
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path


pcd_root = Path(sys.argv[1]).resolve()
collection_set = Path(sys.argv[2]).resolve()
archive_date = sys.argv[3]
result_root = pcd_root / "results/libero_plus_meta_subspace/v2"
collect_root = result_root / "collect"
manifest_root = result_root / "manifests"

configs = {
    "spatial": {
        "suite": "libero_spatial",
        "old_label": "spatial3_pcd",
        "archive": f"spatial5_mixed_pcd_rlinf_{archive_date}",
        "manifest": "spatial5_mixed_pool.json",
        "old": pcd_root / (
            "results/libero_plus_meta_subspace/v1/collect/pi0_libero_vanilla/"
            "vanilla/source_id-pi0_libero_official_vanilla/"
            "run_id-spatial3_collect_20261013T124150/libero_spatial"
        ),
        "mapping": collection_set / "pcd_mappings/spatial_base_task_map.json",
        "expected": 1662,
    },
    "object": {
        "suite": "libero_object",
        "old_label": "object3_pcd",
        "archive": f"object5_mixed_pcd_rlinf_{archive_date}",
        "manifest": "object5_mixed_pool.json",
        "old": pcd_root / (
            "results/libero_plus_meta_subspace/v1/collect/pi0_libero_vanilla/"
            "vanilla/source_id-pi0_libero_official_vanilla/"
            "run_id-object3_collect_20261013T124234/libero_object"
        ),
        "mapping": collection_set / "pcd_mappings/object_base_task_map.json",
        "expected": 1766,
    },
    "10": {
        "suite": "libero_10",
        "old_label": "libero10_3_pcd",
        "archive": f"libero10_5_mixed_pcd_rlinf_{archive_date}",
        "manifest": "libero10_5_mixed_pool.json",
        "old": pcd_root / (
            "results/libero_plus_meta_subspace/v1/collect/pi0_libero_vanilla/"
            "vanilla/source_id-pi0_libero_official_vanilla/"
            "run_id-libero10_3_collect_20261011T030903/libero_10"
        ),
        "mapping": collection_set / "pcd_mappings/10_base_task_map.json",
        "expected": 1743,
    },
}


def npz_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.npz"))


def source_groups(tag: str, cfg: dict) -> list[tuple[str, Path]]:
    return [
        (cfg["old_label"], cfg["old"]),
        (
            "sensor_rlinf",
            collection_set / f"{tag}_sensor_noise/pcd_export",
        ),
        (
            "camera_rlinf",
            collection_set / f"{tag}_camera/pcd_export",
        ),
    ]


def verify_existing(
    output: Path,
    manifest: Path,
    expected: int,
) -> dict:
    if not output.is_dir() or not manifest.is_file():
        raise RuntimeError(
            f"Partial archive exists; inspect before retrying: {output}, {manifest}"
        )
    files = npz_files(output)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if len(files) != expected:
        raise RuntimeError(
            f"Existing archive count mismatch: {output}: {len(files)} != {expected}"
        )
    if data.get("summary", {}).get("collected") != expected:
        raise RuntimeError(f"Existing manifest count mismatch: {manifest}")
    return data["summary"]


collect_root.mkdir(parents=True, exist_ok=True)
manifest_root.mkdir(parents=True, exist_ok=True)
summaries = {}

for tag, cfg in configs.items():
    output = collect_root / cfg["archive"]
    manifest = manifest_root / cfg["manifest"]
    groups = source_groups(tag, cfg)

    for label, source in groups:
        if not source.is_dir():
            raise FileNotFoundError(f"Missing {label} source: {source}")
    if not cfg["mapping"].is_file():
        raise FileNotFoundError(f"Missing task mapping: {cfg['mapping']}")

    source_counts = {label: len(npz_files(source)) for label, source in groups}
    total = sum(source_counts.values())
    if total != cfg["expected"]:
        raise RuntimeError(
            f"{tag} source count mismatch: {source_counts}; "
            f"total={total}, expected={cfg['expected']}"
        )

    if output.exists() or manifest.exists():
        summaries[tag] = verify_existing(output, manifest, cfg["expected"])
        print(f"[{tag}] verified existing archive: {output}")
        continue

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=collect_root)
    )
    provenance = []
    try:
        for label, source in groups:
            for src in npz_files(source):
                relative = src.relative_to(source)
                dst = staging / label / relative
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.link(src, dst)
                src_stat = src.stat()
                dst_stat = dst.stat()
                if (
                    src_stat.st_dev != dst_stat.st_dev
                    or src_stat.st_ino != dst_stat.st_ino
                ):
                    raise RuntimeError(f"Hardlink verification failed: {src} -> {dst}")
                provenance.append(
                    {
                        "path": str((Path(label) / relative).as_posix()),
                        "source": str(src),
                        "inode": dst_stat.st_ino,
                    }
                )

        if len(provenance) != cfg["expected"]:
            raise RuntimeError(f"Staged count mismatch for {tag}")

        (staging / "archive_report.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "libero-plus-five-category-hardlink-archive",
                    "suite": cfg["suite"],
                    "sources": {
                        label: str(source) for label, source in groups
                    },
                    "source_counts": source_counts,
                    "transfer": "hardlink",
                    "episodes": provenance,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    try:
        sys.path.insert(0, str(pcd_root))
        from adapter_training.meta_subspace.data import build_manifest

        summary = build_manifest(
            output,
            cfg["mapping"],
            manifest,
            horizon=50,
            validation_fraction=0.2,
            seed=7,
            allow_mixed_sources=True,
        )
    except Exception:
        # Keep an all-or-nothing contract for a newly-created archive.
        if manifest.exists():
            manifest.unlink()
        shutil.rmtree(output)
        raise

    if summary.get("collected") != cfg["expected"]:
        raise RuntimeError(f"Built manifest count mismatch for {tag}: {summary}")
    summaries[tag] = summary
    print(f"[{tag}] archive:  {output}")
    print(f"[{tag}] manifest: {manifest}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

report = {
    "schema_version": 1,
    "kind": "libero-plus-spatial-object-10-five-category-archives",
    "collection_set": str(collection_set),
    "archives": {
        tag: {
            "collect": str(collect_root / cfg["archive"]),
            "manifest": str(manifest_root / cfg["manifest"]),
            "summary": summaries[tag],
        }
        for tag, cfg in configs.items()
    },
}
report_path = manifest_root / "spatial_object_10_full5_archive_report.json"
report_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, ensure_ascii=False, indent=2))
print(f"REPORT={report_path}")
PY

echo "Spatial/Object/LIBERO-10 five-category archives passed."
