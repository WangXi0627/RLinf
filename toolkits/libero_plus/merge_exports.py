"""Merge verified RLinf-to-PCD exports into one suite-level training pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_inventory(export: Path) -> list[Path]:
    report_path = export / "conversion_report.json"
    if not report_path.is_file():
        raise ValueError(f"Missing conversion report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {row["path"]: row["sha256"] for row in report["episodes"]}
    paths = sorted(export.glob("episode-*.npz"))
    if {path.name for path in paths} != set(expected):
        raise ValueError(f"Export inventory differs from its report: {export}")
    for path in paths:
        if _sha256(path) != expected[path.name]:
            raise ValueError(f"Exported episode changed after conversion: {path}")
    return paths


def merge_exports(
    exports: list[Path],
    output: Path,
    manifest: Path,
    mapping: Path,
    pcd_root: Path,
    *,
    validation_fraction: float = 0.2,
    seed: int = 7,
) -> dict:
    """Stage complete exports and build a mixed-source PCD manifest atomically."""
    exports = [path.resolve() for path in exports]
    output = output.resolve()
    manifest = manifest.resolve()
    if len(exports) < 2:
        raise ValueError("At least two exports are required for a merged pool")
    if output.exists() or manifest.exists():
        raise FileExistsError("Use new output and manifest paths for a merged pool")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    rows = []
    try:
        for export_index, export in enumerate(exports):
            for path in _verified_inventory(export):
                target = temporary / f"episode-{export_index:02d}-{path.name[8:]}"
                try:
                    os.link(path, target)
                    transfer = "hardlink"
                except OSError:
                    shutil.copy2(path, target)
                    transfer = "copy"
                rows.append(
                    {
                        "source": str(path),
                        "path": target.name,
                        "sha256": _sha256(target),
                        "transfer": transfer,
                    }
                )
        temporary.rename(output)
        sys.path.insert(0, str(pcd_root.resolve()))
        from adapter_training.meta_subspace.data import build_manifest

        summary = build_manifest(
            output,
            mapping,
            manifest,
            horizon=50,
            validation_fraction=validation_fraction,
            seed=seed,
            allow_mixed_sources=True,
        )
        report = {
            "schema_version": 1,
            "kind": "libero-plus-merged-rlinf-exports",
            "exports": [str(path) for path in exports],
            "output": str(output),
            "manifest": str(manifest),
            "episodes": rows,
            "summary": summary,
        }
        (output / "merge_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report
    except Exception:
        if output.exists() and not manifest.exists():
            shutil.rmtree(output)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-mapping", type=Path, required=True)
    parser.add_argument("--pcd-root", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    report = merge_exports(
        args.export,
        args.output_dir,
        args.manifest,
        args.task_mapping,
        args.pcd_root,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
