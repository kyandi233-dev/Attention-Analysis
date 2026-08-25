"""Post-hoc RITnet four-class extension for one completed formal NIR run.

The extension reuses the source run's saved video/frame/ROI coordinates, skips
YOLO entirely, re-runs the same frozen RITnet b16 FP32 model, and writes a
subject-numbered CSV/JSON set without modifying the original eyes.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ritnet_fullclass_contract import (
    CLASS_MAPPING,
    EXTENSION_SCHEMA_VERSION,
    EXTENSION_VERSION,
    normalize_subject,
    subject_output_paths,
)
from ritnet_fullclass_metrics import summarize_fullclass
from ritnet_fullclass_runtime import RitnetFullClassRuntime

PARITY_TOLERANCE = 1e-3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def resolve_package_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PACKAGE_ROOT / path


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    if text == "":
        return None
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def parse_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def parse_int(value: Any) -> int:
    if value is None or str(value).strip() == "":
        raise ValueError("Missing required integer value")
    return int(float(value))


def _max_abs(values: Iterable[float | None]) -> float | None:
    valid = [abs(float(value)) for value in values if value is not None]
    return max(valid) if valid else None


def pupil_parity(source: dict[str, str], metrics: dict[str, Any]) -> dict[str, Any]:
    source_found = parse_bool(source.get("ritnet_found"))
    new_found = bool(metrics.get("pupil_fit_valid"))
    found_match = source_found is None or source_found == new_found

    center_diff = None
    diameter_diff = None
    area_diff = None
    confidence_diff = None

    if source_found and new_found:
        sx = parse_float(source.get("pupil_center_x"))
        sy = parse_float(source.get("pupil_center_y"))
        nx = metrics.get("pupil_center_x")
        ny = metrics.get("pupil_center_y")
        if sx is not None and sy is not None and nx is not None and ny is not None:
            center_diff = _max_abs((float(nx) - sx, float(ny) - sy))

        sd = parse_float(source.get("pupil_equiv_diameter"))
        nd = metrics.get("pupil_equiv_diameter")
        if sd is not None and nd is not None:
            diameter_diff = abs(float(nd) - sd)

        sa = parse_float(source.get("pupil_mask_area"))
        na = metrics.get("pupil_contour_area")
        if sa is not None and na is not None:
            area_diff = abs(float(na) - sa)

        sc = parse_float(source.get("pupil_confidence"))
        nc = metrics.get("pupil_confidence")
        if sc is not None and nc is not None:
            confidence_diff = abs(float(nc) - sc)

    numeric_diffs = [
        value
        for value in (center_diff, diameter_diff, area_diff, confidence_diff)
        if value is not None
    ]
    parity_ok = bool(found_match and all(value <= PARITY_TOLERANCE for value in numeric_diffs))
    return {
        "source_ritnet_found": source_found,
        "pupil_parity_found_match": found_match,
        "pupil_parity_center_max_abs_diff": center_diff,
        "pupil_parity_equiv_diameter_abs_diff": diameter_diff,
        "pupil_parity_contour_area_abs_diff": area_diff,
        "pupil_parity_confidence_abs_diff": confidence_diff,
        "pupil_parity_ok": parity_ok,
    }


def extension_completion_valid(
    path: Path,
    *,
    source_eyes_sha256: str,
    ritnet_model_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        marker = load_json(path)
    except Exception:
        return False
    return bool(
        marker.get("schema_version") == EXTENSION_SCHEMA_VERSION
        and marker.get("extension_version") == EXTENSION_VERSION
        and marker.get("status") == "complete"
        and marker.get("source_eyes_sha256") == source_eyes_sha256
        and marker.get("ritnet_model_sha256") == ritnet_model_sha256
        and all(Path(value).is_file() for value in marker.get("required_artifacts", []))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run RITnet only and retain background/sclera/iris/pupil metrics"
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed formal subject run directory")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-model-mismatch",
        action="store_true",
        help="Allow extension when the source completion records a different RITnet model hash",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    source_completion_path = run_dir / "completion.json"
    source_eyes = run_dir / "eyes.csv"
    if not source_completion_path.is_file() or not source_eyes.is_file():
        raise FileNotFoundError(f"Run directory must contain completion.json and eyes.csv: {run_dir}")

    source_completion = load_json(source_completion_path)
    if source_completion.get("status") != "complete":
        raise RuntimeError(
            f"Source run must be complete; got status={source_completion.get('status')!r}: {run_dir}"
        )
    subject = normalize_subject(source_completion.get("subject", ""))
    video = Path(str(source_completion.get("video", "")))
    if not video.is_file():
        raise FileNotFoundError(f"Source video is unavailable: {video}")

    config = load_config(args.config.resolve())
    ritnet_path = resolve_package_path(config["models"]["ritnet"])
    if not ritnet_path.is_file():
        raise FileNotFoundError(ritnet_path)
    current_model_hash = sha256(ritnet_path)
    source_model_hash = source_completion.get("ritnet_model_sha256")
    if (
        source_model_hash
        and source_model_hash != current_model_hash
        and not args.allow_model_mismatch
    ):
        raise RuntimeError(
            "Source formal run used a different RITnet model hash. "
            f"source={source_model_hash}, current={current_model_hash}. "
            "Use --allow-model-mismatch only after explicitly reviewing provenance."
        )

    source_eyes_hash = sha256(source_eyes)
    outputs = subject_output_paths(run_dir, subject)
    if not args.force and extension_completion_valid(
        outputs["completion"],
        source_eyes_sha256=source_eyes_hash,
        ritnet_model_sha256=current_model_hash,
    ):
        print(f"[SKIP] {subject}: validated -> {outputs['completion']}")
        return 0

    with source_eyes.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        source_fields = list(reader.fieldnames or [])
        required = {
            "frame_idx",
            "eye",
            "roi_x1",
            "roi_y1",
            "roi_x2",
            "roi_y2",
        }
        if not required.issubset(set(source_fields)):
            raise ValueError(f"eyes.csv is missing required columns: {sorted(required - set(source_fields))}")
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"Source eyes.csv contains no eye rows: {source_eyes}")
    for row in rows:
        row_subject = normalize_subject(row.get("subject") or subject)
        if row_subject != subject:
            raise ValueError(f"Mixed subjects in source eyes.csv: {subject} vs {row_subject}")

    rows_by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[parse_int(row["frame_idx"])].append(row)
    target_frames = sorted(rows_by_frame)
    min_frame, max_frame = target_frames[0], target_frames[-1]

    analysis_size = (int(config["roi"]["width"]), int(config["roi"]["height"]))
    runtime = RitnetFullClassRuntime(
        PACKAGE_ROOT,
        ritnet_path,
        input_size=(
            int(config["ritnet"]["input_width"]),
            int(config["ritnet"]["input_height"]),
        ),
        device=str(args.device),
        analysis_size=analysis_size,
        precision="fp32",
    )

    prototype = summarize_fullclass(
        np.zeros((runtime.input_size[1], runtime.input_size[0]), dtype=np.uint8),
        np.zeros((runtime.input_size[1], runtime.input_size[0]), dtype=np.float32),
        analysis_size,
    )
    metric_fields = [f"fullclass_{name}" for name in prototype]
    parity_fields = [
        "source_ritnet_found",
        "pupil_parity_found_match",
        "pupil_parity_center_max_abs_diff",
        "pupil_parity_equiv_diameter_abs_diff",
        "pupil_parity_contour_area_abs_diff",
        "pupil_parity_confidence_abs_diff",
        "pupil_parity_ok",
    ]
    output_fields = source_fields + [
        name for name in metric_fields + parity_fields if name not in source_fields
    ]

    started_at = datetime.now(timezone.utc).isoformat()
    completion_base = {
        "schema_version": EXTENSION_SCHEMA_VERSION,
        "extension_version": EXTENSION_VERSION,
        "status": "running",
        "subject": subject,
        "source_run_dir": str(run_dir),
        "source_completion": str(source_completion_path),
        "source_eyes_csv": str(source_eyes),
        "source_eyes_sha256": source_eyes_hash,
        "video": str(video.resolve()),
        "ritnet_model": str(ritnet_path.resolve()),
        "ritnet_model_sha256": current_model_hash,
        "ritnet_device": str(runtime.device),
        "ritnet_precision": runtime.precision,
        "ritnet_batch_size": runtime.FIXED_BATCH_SIZE,
        "analysis_size": list(analysis_size),
        "class_mapping": {str(key): value for key, value in CLASS_MAPPING.items()},
        "expected_rows": len(rows),
        "processed_rows": 0,
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "required_artifacts": [
            str(outputs["csv"]),
            str(outputs["summary"]),
            str(outputs["manifest"]),
        ],
    }
    atomic_write_json(outputs["completion"], completion_base)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)

    pending: list[tuple[dict[str, str], np.ndarray]] = []
    processed_rows = 0
    decoded_frames = 0
    normalization_valid_count = 0
    parity_ok_count = 0
    parity_mismatch_count = 0
    fraction_sums = defaultdict(float)
    wall_started = time.perf_counter()

    def flush(writer: csv.DictWriter) -> None:
        nonlocal processed_rows, normalization_valid_count, parity_ok_count, parity_mismatch_count
        if not pending:
            return
        metrics_batch = runtime.infer_batch([roi for _, roi in pending])
        for (source_row, _), metrics in zip(pending, metrics_batch):
            parity = pupil_parity(source_row, metrics)
            output_row: dict[str, Any] = dict(source_row)
            output_row.update({f"fullclass_{key}": value for key, value in metrics.items()})
            output_row.update(parity)
            writer.writerow(output_row)
            processed_rows += 1
            normalization_valid_count += int(bool(metrics.get("normalization_valid")))
            parity_ok_count += int(bool(parity["pupil_parity_ok"]))
            parity_mismatch_count += int(not bool(parity["pupil_parity_ok"]))
            for key in (
                "background_fraction",
                "sclera_fraction",
                "iris_fraction",
                "pupil_fraction",
                "ocular_fraction",
            ):
                fraction_sums[key] += float(metrics[key])
        pending.clear()

    try:
        outputs["csv"].parent.mkdir(parents=True, exist_ok=True)
        with outputs["csv"].open("w", newline="", encoding="utf-8-sig") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=output_fields, extrasaction="ignore")
            writer.writeheader()

            current_frame = min_frame
            target_set = set(target_frames)
            while current_frame <= max_frame:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Video read failed at frame {current_frame}: {video}")
                decoded_frames += 1

                if current_frame in target_set:
                    for source_row in rows_by_frame[current_frame]:
                        x1 = parse_int(source_row["roi_x1"])
                        y1 = parse_int(source_row["roi_y1"])
                        x2 = parse_int(source_row["roi_x2"])
                        y2 = parse_int(source_row["roi_y2"])
                        if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
                            raise ValueError(
                                f"Invalid ROI at {subject} frame={current_frame} eye={source_row.get('eye')}: "
                                f"{(x1, y1, x2, y2)} for frame shape {frame.shape}"
                            )
                        roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                        pending.append((source_row, np.ascontiguousarray(roi)))
                        if len(pending) >= runtime.FIXED_BATCH_SIZE:
                            flush(writer)
                current_frame += 1

            flush(writer)
    finally:
        cap.release()

    elapsed = time.perf_counter() - wall_started
    if processed_rows != len(rows):
        raise RuntimeError(f"Processed {processed_rows} rows but expected {len(rows)}")

    summary = {
        "subject": subject,
        "extension_version": EXTENSION_VERSION,
        "source_run_dir": str(run_dir),
        "source_eyes_csv": str(source_eyes),
        "output_csv": str(outputs["csv"]),
        "processed_rows": processed_rows,
        "decoded_frames": decoded_frames,
        "min_frame_idx": min_frame,
        "max_frame_idx": max_frame,
        "elapsed_sec": elapsed,
        "roi_per_sec": (processed_rows / elapsed) if elapsed else None,
        "normalization_valid_count": normalization_valid_count,
        "normalization_valid_fraction": normalization_valid_count / processed_rows,
        "pupil_parity_ok_count": parity_ok_count,
        "pupil_parity_mismatch_count": parity_mismatch_count,
        "pupil_parity_ok_fraction": parity_ok_count / processed_rows,
        "mean_class_fractions": {
            key: fraction_sums[key] / processed_rows
            for key in (
                "background_fraction",
                "sclera_fraction",
                "iris_fraction",
                "pupil_fraction",
                "ocular_fraction",
            )
        },
    }
    atomic_write_json(outputs["summary"], summary)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "extension_version": EXTENSION_VERSION,
        "command": " ".join(sys.argv),
        "subject": subject,
        "source_run_dir": str(run_dir),
        "source_completion": source_completion,
        "source_eyes_sha256": source_eyes_hash,
        "ritnet_model": str(ritnet_path.resolve()),
        "ritnet_model_sha256": current_model_hash,
        "source_ritnet_model_sha256": source_model_hash,
        "ritnet_device": str(runtime.device),
        "ritnet_precision": runtime.precision,
        "ritnet_batch_size": runtime.FIXED_BATCH_SIZE,
        "ritnet_providers": runtime.providers,
        "input_size": list(runtime.input_size),
        "analysis_size": list(analysis_size),
        "class_mapping": {str(key): value for key, value in CLASS_MAPPING.items()},
        "notes": [
            "Original eyes.csv is never modified.",
            "YOLO is not re-run; source frame_idx and ROI coordinates are reused exactly.",
            "The current ONNX exposes hard four-class labels and pupil probability only; "
            "iris/sclera/background probabilities are therefore not fabricated.",
            "Ocular aperture fields are candidate geometry/QC signals, not validated blink or PERCLOS labels.",
        ],
    }
    atomic_write_json(outputs["manifest"], manifest)

    finished_completion = {
        **completion_base,
        "status": "complete",
        "processed_rows": processed_rows,
        "output_csv_sha256": sha256(outputs["csv"]),
        "summary_sha256": sha256(outputs["summary"]),
        "manifest_sha256": sha256(outputs["manifest"]),
        "pupil_parity_mismatch_count": parity_mismatch_count,
        "normalization_valid_count": normalization_valid_count,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(outputs["completion"], finished_completion)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Completion -> {outputs['completion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
