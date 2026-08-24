"""Sequential multi-subject runner for the formal NIR pipeline.

Subject selection is read from config.yaml:

batch:
  subjects:
    include: []   # empty => all discovered subjects >= formal.min_subject_number
    exclude: []

CLI --subjects overrides YAML include for one run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
PIPELINE = PACKAGE_ROOT / "run_pipeline.py"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from formal_completion import validate_completion


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_subject(value: str) -> str:
    text = str(value).strip().rstrip("_")
    if not text.startswith("sub-"):
        text = f"sub-{text}"
    number = text[4:]
    if not number.isdigit():
        raise ValueError(f"Invalid subject identifier: {value}")
    return f"sub-{int(number):03d}"


def subject_number(subject: str) -> int:
    return int(normalize_subject(subject)[4:])


def discover(config: dict[str, Any]) -> dict[str, Path]:
    pattern = str(config["data"].get("subject_pattern", "sub-*_/nir/*_nir.avi"))
    minimum = int(config["formal"].get("min_subject_number", 31))
    found: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    for root_text in config["data"]["roots"]:
        root = Path(root_text)
        if not root.exists():
            continue
        for video in sorted(root.glob(pattern)):
            subject = normalize_subject(video.stem.removesuffix("_nir"))
            if subject_number(subject) < minimum:
                continue
            if subject in found and found[subject] != video:
                duplicates.setdefault(subject, [found[subject]]).append(video)
            else:
                found[subject] = video

    if duplicates:
        details = "; ".join(
            f"{subject}: {', '.join(str(path) for path in paths)}"
            for subject, paths in sorted(duplicates.items())
        )
        raise RuntimeError(f"Duplicate subject videos found across data roots: {details}")
    return found


def parse_subject_list(text: str | None) -> list[str] | None:
    if text is None:
        return None
    values = [item.strip() for item in text.split(",") if item.strip()]
    return [normalize_subject(item) for item in values]


def selected_subjects(
    config: dict[str, Any], discovered: dict[str, Path], cli_subjects: list[str] | None
) -> list[str]:
    batch_cfg = config.get("batch", {})
    subjects_cfg = batch_cfg.get("subjects", {})
    include_cfg = [normalize_subject(item) for item in subjects_cfg.get("include", [])]
    exclude = {normalize_subject(item) for item in subjects_cfg.get("exclude", [])}

    include = cli_subjects if cli_subjects is not None else include_cfg
    if include:
        missing = [subject for subject in include if subject not in discovered]
        if missing:
            raise FileNotFoundError(
                "Selected subject(s) were not discovered under data.roots: " + ", ".join(missing)
            )
        ordered = include
    else:
        ordered = sorted(discovered, key=subject_number)

    return [subject for subject in ordered if subject not in exclude]


def effective_output_root(config: dict[str, Any], override: str | None) -> Path:
    value = override or config.get("batch", {}).get("output_root") or config["output"]["root"]
    path = Path(value)
    return path if path.is_absolute() else PACKAGE_ROOT / path


def expected_run_dir(
    config: dict[str, Any],
    output_root: Path,
    subject: str,
    precision: str,
    batch_size: int,
    phases: list[str],
    backend: str,
    yolo_batch_size: int,
) -> Path:
    release = str(config["formal"].get("focuswave_release", "v3.1.3"))
    configured = [str(value) for value in config["formal"].get("phases", [])]
    suffixes: list[str] = []
    if backend == "ort-cuda":
        suffixes.append("ort-cuda")
    if yolo_batch_size > 1:
        suffixes.append(f"yolo{yolo_batch_size}")
    if phases != configured:
        suffixes.append("partial-" + "-".join(phases))
    suffix = "_" + "_".join(suffixes) if suffixes else ""
    return output_root / f"{subject}_formal_{release}_yolo_b{batch_size}_{precision}{suffix}"


def build_command(
    config_path: Path,
    video: Path,
    output_root: Path,
    device: str,
    precision: str,
    batch_size: int,
    phases: str | None,
    backend: str,
    yolo_batch_size: int,
) -> list[str]:
    command = [
        sys.executable,
        str(PIPELINE),
        "--config",
        str(config_path),
        "formal",
        "--video",
        str(video),
        "--device",
        str(device),
        "--ritnet-precision",
        precision,
        "--ritnet-batch-size",
        str(batch_size),
        "--output",
        str(output_root),
        "--backend",
        backend,
        "--yolo-batch-size",
        str(yolo_batch_size),
    ]
    if phases:
        command.extend(["--phases", phases])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal NIR analysis sequentially for selected subjects")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument(
        "--subjects",
        help="Comma-separated subjects overriding batch.subjects.include, e.g. sub-031,sub-033,sub-056",
    )
    parser.add_argument("--device")
    parser.add_argument("--ritnet-precision", choices=("fp32", "fp16"))
    parser.add_argument("--ritnet-batch-size", type=int)
    parser.add_argument("--yolo-batch-size", type=int)
    parser.add_argument("--phases", help="Optional comma-separated phase override")
    parser.add_argument("--backend", choices=("pytorch-cuda", "ort-cuda"))
    parser.add_argument("--output")
    parser.add_argument("--force", action="store_true", help="Rerun even when a valid completion marker exists")
    parser.add_argument("--dry-run", action="store_true", help="Print selected subjects/commands without running")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    batch_cfg = config.get("batch", {})
    discovered = discover(config)
    subjects = selected_subjects(config, discovered, parse_subject_list(args.subjects))

    if not subjects:
        raise RuntimeError("No subjects selected for formal batch analysis")

    device = args.device or str(batch_cfg.get("device", "0"))
    precision = args.ritnet_precision or str(config["ritnet"].get("precision", "fp32"))
    batch_size = args.ritnet_batch_size or int(config["ritnet"].get("batch_size", 16))
    yolo_batch_size = args.yolo_batch_size or int(
        config.get("inference", {}).get("yolo_batch_size", 1)
    )
    if batch_size <= 0:
        raise ValueError("RITnet batch size must be positive")
    backend = args.backend or str(config.get("inference", {}).get("backend", "pytorch-cuda"))
    if backend == "ort-cuda" and (precision != "fp32" or batch_size != 16):
        raise ValueError("ORT CUDA profile is frozen to FP32 and RITnet batch=16")
    configured_phases = [str(value) for value in config["formal"].get("phases", [])]
    phases = (
        [value.strip() for value in str(args.phases).split(",") if value.strip()]
        if args.phases
        else configured_phases
    )
    is_full_phase_run = phases == configured_phases

    output_root = effective_output_root(config, args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    skip_completed = bool(batch_cfg.get("skip_completed", True)) and not args.force
    continue_on_error = bool(batch_cfg.get("continue_on_error", True))

    print(json.dumps({
        "selected_subjects": subjects,
        "count": len(subjects),
        "device": device,
        "ritnet_precision": precision,
        "ritnet_batch_size": batch_size,
        "inference_backend": backend,
        "yolo_batch_size": yolo_batch_size,
        "output_root": str(output_root),
        "dry_run": bool(args.dry_run),
    }, ensure_ascii=False, indent=2))

    results: list[dict[str, Any]] = []
    for index, subject in enumerate(subjects, start=1):
        video = discovered[subject]
        run_dir = expected_run_dir(
            config, output_root, subject, precision, batch_size, phases, backend, yolo_batch_size
        )
        expected_identity = {
            "subject": subject,
            "video": str(video.resolve()),
            "package_version": str(config["package"]["version"]),
            "focuswave_release": str(config["formal"].get("focuswave_release", "v3.1.3")),
            "inference_backend": backend,
            "phases": phases,
            "ritnet_enabled": bool(config["ritnet"].get("enabled", True)),
            "ritnet_precision": precision,
            "ritnet_batch_size": batch_size,
            "yolo_batch_size": yolo_batch_size,
            "max_frames": None,
            "partial_phase_selection": not is_full_phase_run,
        }

        if skip_completed and is_full_phase_run:
            validation = validate_completion(run_dir, expected_identity)
            if validation.valid:
                marker_path = run_dir / "completion.json"
                print(f"[SKIP {index}/{len(subjects)}] {subject}: validated -> {marker_path}")
                results.append(
                    {"subject": subject, "status": "skipped_completed", "video": str(video)}
                )
                continue
            if run_dir.exists():
                print(f"[RERUN {index}/{len(subjects)}] {subject}: {validation.reason}")

        command = build_command(
            config_path,
            video,
            output_root,
            device,
            precision,
            batch_size,
            args.phases,
            backend,
            yolo_batch_size,
        )
        print(f"[RUN {index}/{len(subjects)}] {subject}: {video}")
        print("  " + subprocess.list2cmdline(command))

        if args.dry_run:
            results.append({"subject": subject, "status": "dry_run", "video": str(video)})
            continue

        completed = subprocess.run(command, cwd=PACKAGE_ROOT)
        if completed.returncode == 0:
            accepted = ("complete",) if is_full_phase_run else ("smoke_complete",)
            validation = validate_completion(
                run_dir,
                expected_identity,
                accepted_statuses=accepted,
            )
            if validation.valid:
                status = "completed" if is_full_phase_run else "partial_complete"
                results.append({"subject": subject, "status": status, "video": str(video)})
                continue
            results.append(
                {
                    "subject": subject,
                    "status": "failed",
                    "returncode": 0,
                    "validation_error": validation.reason,
                    "video": str(video),
                }
            )
            print(f"[FAIL] {subject}: process returned 0 but completion is invalid: {validation.reason}")
            if not continue_on_error:
                break
            continue

        results.append({
            "subject": subject,
            "status": "failed",
            "returncode": completed.returncode,
            "video": str(video),
        })
        print(f"[FAIL] {subject}: return code {completed.returncode}")
        if not continue_on_error:
            break

    batch_summary = output_root / "batch_run_summary.json"
    batch_summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = [item for item in results if item["status"] == "failed"]
    print(f"Batch summary -> {batch_summary}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
