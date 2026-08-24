"""Sequential multi-subject runner for the AMD batched formal NIR pipeline.

Production configuration is YOLO fixed batch 8 plus RITnet fixed batch 16.
Subject selection is read from config.yaml; CLI --subjects overrides YAML include.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
PIPELINE = PACKAGE_ROOT / "run_formal_batched.py"
FIXED_YOLO_BATCH_SIZE = 8
FIXED_RITNET_BATCH_SIZE = 16
FIXED_RITNET_PRECISION = "fp32"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from formal_completion import validate_completion


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_package_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PACKAGE_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    path = path if path.is_absolute() else PACKAGE_ROOT / path
    if not any(part.lower() == "amd-directml" for part in path.parts):
        path = path / "amd-directml"
    return path


def expected_run_dir(
    config: dict[str, Any],
    output_root: Path,
    subject: str,
    precision: str,
    ritnet_batch_size: int,
    yolo_batch_size: int,
    phases: list[str],
) -> Path:
    release = str(config["formal"].get("focuswave_release", "v3.1.3"))
    configured = [str(value) for value in config["formal"].get("phases", [])]
    suffix = "" if phases == configured else "_partial-" + "-".join(phases)
    return output_root / (
        f"{subject}_formal_{release}_yolo-b{yolo_batch_size}_"
        f"ritnet-b{ritnet_batch_size}_{precision}{suffix}"
    )


def build_command(
    config_path: Path,
    video: Path,
    output_root: Path,
    device: str,
    precision: str,
    ritnet_batch_size: int,
    phases: str | None,
) -> list[str]:
    command = [
        sys.executable,
        str(PIPELINE),
        "--config",
        str(config_path),
        "--video",
        str(video),
        "--device",
        str(device),
        "--ritnet-precision",
        precision,
        "--ritnet-batch-size",
        str(ritnet_batch_size),
        "--output",
        str(output_root),
    ]
    if phases:
        command.extend(["--phases", phases])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMD batched formal NIR analysis sequentially")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument(
        "--subjects",
        help="Comma-separated subjects overriding batch.subjects.include, e.g. sub-031,sub-033,sub-056",
    )
    parser.add_argument("--device")
    parser.add_argument("--ritnet-precision", choices=("fp32",))
    parser.add_argument("--ritnet-batch-size", type=int)
    parser.add_argument("--phases", help="Optional comma-separated phase override")
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
    ritnet_batch_size = args.ritnet_batch_size or int(config["ritnet"].get("batch_size", 16))
    yolo_batch_size = int(config["yolo"].get("batch_size", 8))
    if precision != FIXED_RITNET_PRECISION:
        raise ValueError("AMD/DirectML RITnet precision is fixed at fp32")
    if ritnet_batch_size != FIXED_RITNET_BATCH_SIZE:
        raise ValueError("AMD/DirectML RITnet batch size is fixed at 16")
    if yolo_batch_size != FIXED_YOLO_BATCH_SIZE:
        raise ValueError("AMD/DirectML formal YOLO batch size is fixed at 8")

    yolo_path = resolve_package_path(config["models"]["yolo_formal"])
    ritnet_path = resolve_package_path(config["models"]["ritnet"])
    if not yolo_path.is_file():
        raise FileNotFoundError(
            f"Missing production YOLO b8 model: {yolo_path}. "
            "The model must be present before formal analysis can run."
        )
    if not ritnet_path.is_file():
        raise FileNotFoundError(ritnet_path)
    yolo_hash = sha256(yolo_path)
    ritnet_hash = sha256(ritnet_path)

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
        "yolo_batch_size": yolo_batch_size,
        "yolo_model": str(yolo_path),
        "ritnet_precision": precision,
        "ritnet_batch_size": ritnet_batch_size,
        "output_root": str(output_root),
        "dry_run": bool(args.dry_run),
    }, ensure_ascii=False, indent=2))

    results: list[dict[str, Any]] = []
    for index, subject in enumerate(subjects, start=1):
        video = discovered[subject]
        run_dir = expected_run_dir(
            config,
            output_root,
            subject,
            precision,
            ritnet_batch_size,
            yolo_batch_size,
            phases,
        )
        expected_identity = {
            "subject": subject,
            "video": str(video.resolve()),
            "package_version": str(config["package"]["version"]),
            "focuswave_release": str(config["formal"].get("focuswave_release", "v3.1.3")),
            "phases": phases,
            "yolo_batch_size": yolo_batch_size,
            "yolo_model_sha256": yolo_hash,
            "ritnet_enabled": bool(config["ritnet"].get("enabled", True)),
            "ritnet_precision": precision,
            "ritnet_batch_size": ritnet_batch_size,
            "ritnet_model_sha256": ritnet_hash,
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
            ritnet_batch_size,
            args.phases,
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
