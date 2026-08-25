"""Sequential batch runner for the fast post-hoc RITnet four-class extension."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ritnet_fullclass_contract import (
    EXTENSION_VERSION,
    QC_ANOMALY_LIMIT_PER_REASON_PER_PHASE,
    QC_STRIDE_FRAMES,
    normalize_subject,
)

EXTENSION = PACKAGE_ROOT / "run_ritnet_fullclass_extension.py"


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def parse_subjects(text: str | None) -> list[str] | None:
    if text is None:
        return None
    return [normalize_subject(item) for item in text.split(",") if item.strip()]


def load_completion(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "completion.json"
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def discover_source_runs(output_root: Path) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for run_dir in sorted(output_root.glob("sub-*_formal_*")):
        if not run_dir.is_dir():
            continue
        marker = load_completion(run_dir)
        if not marker or marker.get("status") != "complete":
            continue
        if marker.get("max_frames") is not None or marker.get("partial_phase_selection"):
            continue
        subject = normalize_subject(marker.get("subject", run_dir.name.split("_formal_", 1)[0]))
        if not (run_dir / "eyes.csv").is_file():
            continue
        grouped.setdefault(subject, []).append((run_dir, marker))
    return grouped


def select_run(candidates: list[tuple[Path, dict[str, Any]]]) -> tuple[Path, dict[str, Any], list[Path]]:
    """Prefer the current production yolo-b8 run; otherwise use newest complete run."""
    ranked = sorted(
        candidates,
        key=lambda item: (
            int(item[1].get("yolo_batch_size", -1) == 8),
            (item[0] / "completion.json").stat().st_mtime_ns,
        ),
        reverse=True,
    )
    selected_dir, selected_marker = ranked[0]
    alternatives = [path for path, _ in ranked[1:]]
    return selected_dir, selected_marker, alternatives


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch fast RITnet full-class extension")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Formal AMD output root containing sub-*_formal_* directories",
    )
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--subjects", help="Optional comma-separated subject filter")
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--validate-pupil",
        action="store_true",
        help="Validation mode: recompute pupil/probability for parity. Use on a test subject, not full batch.",
    )
    parser.add_argument(
        "--postprocess-workers",
        type=int,
        default=4,
        help="CPU workers for full-class postprocessing (default: 4).",
    )
    parser.add_argument("--allow-model-mismatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.postprocess_workers <= 0:
        raise ValueError("--postprocess-workers must be positive")

    output_root = args.output.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    config_path = args.config.resolve()
    config = load_config(config_path)

    grouped = discover_source_runs(output_root)
    requested = parse_subjects(args.subjects)
    excluded = {
        normalize_subject(value)
        for value in config.get("batch", {}).get("subjects", {}).get("exclude", [])
    }

    if requested is None:
        subjects = sorted(
            (subject for subject in grouped if subject not in excluded),
            key=lambda x: int(x[4:]),
        )
    else:
        missing = [subject for subject in requested if subject not in grouped]
        if missing:
            raise FileNotFoundError(
                "No complete formal source run found for: " + ", ".join(missing)
            )
        subjects = [subject for subject in requested if subject not in excluded]

    if not subjects:
        raise RuntimeError("No completed formal source runs selected")

    selections: list[dict[str, Any]] = []
    for subject in subjects:
        run_dir, marker, alternatives = select_run(grouped[subject])
        selections.append(
            {
                "subject": subject,
                "run_dir": str(run_dir),
                "yolo_batch_size": marker.get("yolo_batch_size"),
                "ritnet_batch_size": marker.get("ritnet_batch_size"),
                "alternatives": [str(path) for path in alternatives],
            }
        )

    print(
        json.dumps(
            {
                "extension": EXTENSION_VERSION,
                "output_root": str(output_root),
                "selected_count": len(selections),
                "excluded_subjects": sorted(excluded),
                "pupil_validation_mode": bool(args.validate_pupil),
                "postprocess_workers": int(args.postprocess_workers),
                "ritnet_method": "640x400 FP32 fixed-b16",
                "primary_pupil_metric": "fullclass_pupil_to_iris_diameter_ratio",
                "qc_sampling": {
                    "stride_frames": QC_STRIDE_FRAMES,
                    "phase_first_middle_last": True,
                    "anomaly_limit_per_reason_per_phase": QC_ANOMALY_LIMIT_PER_REASON_PER_PHASE,
                    "formats": ["labels.png", "overlay.png"],
                },
                "selections": selections,
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.dry_run:
        return 0

    continue_on_error = bool(config.get("batch", {}).get("continue_on_error", True))
    results: list[dict[str, Any]] = []
    for index, item in enumerate(selections, start=1):
        subject = item["subject"]
        run_dir = Path(item["run_dir"])
        command = [
            sys.executable,
            str(EXTENSION),
            "--run-dir",
            str(run_dir),
            "--config",
            str(config_path),
            "--device",
            str(args.device),
            "--postprocess-workers",
            str(args.postprocess_workers),
        ]
        if args.force:
            command.append("--force")
        if args.validate_pupil:
            command.append("--validate-pupil")
        if args.allow_model_mismatch:
            command.append("--allow-model-mismatch")

        print(f"[RUN {index}/{len(selections)}] {subject}: {run_dir}")
        print("  " + subprocess.list2cmdline(command))
        completed = subprocess.run(command, cwd=PACKAGE_ROOT)
        status = "completed" if completed.returncode == 0 else "failed"
        results.append(
            {
                "subject": subject,
                "status": status,
                "returncode": completed.returncode,
                "run_dir": str(run_dir),
                "pupil_validation_mode": bool(args.validate_pupil),
                "postprocess_workers": int(args.postprocess_workers),
                "qc_stride_frames": QC_STRIDE_FRAMES,
            }
        )
        if completed.returncode != 0 and not continue_on_error:
            break

    summary_path = output_root / "ritnet_fullclass_batch_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Batch summary -> {summary_path}")
    return 1 if any(item["status"] == "failed" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
