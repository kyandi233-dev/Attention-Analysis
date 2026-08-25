"""Fast post-hoc RITnet four-class extension for one completed formal NIR run.

The extension reuses source video/frame/ROI coordinates, skips YOLO, keeps the
frozen 640x400 FP32 fixed-b16 RITnet method, and writes subject-numbered outputs.

Fast production mode:
- requests labels_u8 only from ONNX;
- reuses the frozen source pupil geometry/confidence from eyes.csv;
- overlaps CPU decode/crop/preprocess with DirectML inference;
- postprocesses independent eye label maps in a small worker pool;
- saves deterministic sparse QC label/overlay images without changing analysis.

Validation mode (``--validate-pupil``) additionally requests pupil probability
and recomputes pupil geometry so parity against the original formal eyes.csv can
be checked before running the full batch.
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
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
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
    QC_ANOMALY_LIMIT_PER_REASON_PER_PHASE,
    QC_OVERLAY_ALPHA,
    QC_PALETTE_BGR,
    QC_STRIDE_FRAMES,
    normalize_subject,
    subject_output_paths,
)
from ritnet_fullclass_metrics import (
    summarize_fullclass,
    summarize_fullclass_from_source,
)
from ritnet_fullclass_qc import (
    QCSampler,
    build_qc_anchor_frames,
    save_qc_pair,
)
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


def atomic_write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
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
    if isinstance(value, bool):
        return value
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
    pupil_validation_mode: bool,
) -> bool:
    if not path.is_file():
        return False
    try:
        marker = load_json(path)
    except Exception:
        return False

    if not bool(
        marker.get("schema_version") == EXTENSION_SCHEMA_VERSION
        and marker.get("extension_version") == EXTENSION_VERSION
        and marker.get("status") == "complete"
        and marker.get("source_eyes_sha256") == source_eyes_sha256
        and marker.get("ritnet_model_sha256") == ritnet_model_sha256
        and bool(marker.get("pupil_validation_mode")) == bool(pupil_validation_mode)
        and int(marker.get("qc_stride_frames", -1)) == QC_STRIDE_FRAMES
        and int(marker.get("qc_anomaly_limit_per_reason_per_phase", -1))
        == QC_ANOMALY_LIMIT_PER_REASON_PER_PHASE
        and all(Path(value).is_file() for value in marker.get("required_artifacts", []))
    ):
        return False

    qc_dir = Path(str(marker.get("qc_dir", "")))
    expected_images = int(marker.get("qc_image_count", -1))
    if expected_images < 0 or not qc_dir.is_dir():
        return False
    actual_images = sum(1 for _ in qc_dir.glob("*.png"))
    return actual_images == expected_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run frozen RITnet only and retain background/sclera/iris/pupil metrics"
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed formal subject run directory")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--validate-pupil",
        action="store_true",
        help="Validation pass: also request pupil probability and recompute pupil geometry for parity.",
    )
    parser.add_argument(
        "--postprocess-workers",
        type=int,
        default=4,
        help="CPU workers for independent full-class label-map postprocessing (default: 4).",
    )
    parser.add_argument(
        "--allow-model-mismatch",
        action="store_true",
        help="Allow extension when the source completion records a different RITnet model hash",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.postprocess_workers <= 0:
        raise ValueError("--postprocess-workers must be positive")

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
        pupil_validation_mode=bool(args.validate_pupil),
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
    target_set = set(target_frames)

    qc_anchor_frames = build_qc_anchor_frames(rows, QC_STRIDE_FRAMES)
    qc_sampler = QCSampler(
        qc_anchor_frames,
        anomaly_limit_per_reason_per_phase=QC_ANOMALY_LIMIT_PER_REASON_PER_PHASE,
    )

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

    prototype_source = rows[0]
    prototype_labels = np.zeros(
        (runtime.input_size[1], runtime.input_size[0]),
        dtype=np.uint8,
    )
    if args.validate_pupil:
        prototype = summarize_fullclass(
            prototype_labels,
            np.zeros_like(prototype_labels, dtype=np.float32),
            analysis_size,
        )
    else:
        prototype = summarize_fullclass_from_source(
            prototype_labels,
            prototype_source,
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
    provenance_fields = [
        "fullclass_source_pupil_reused",
        "fullclass_pupil_validation_mode",
    ]
    output_fields = source_fields + [
        name
        for name in metric_fields + parity_fields + provenance_fields
        if name not in source_fields
    ]

    qc_index_fields = [
        "subject",
        "phase",
        "phase_segment",
        "frame_idx",
        "video_time_ms",
        "unix_ms",
        "eye",
        "reason",
        "ritnet_found",
        "roi_clipped",
        "normalization_valid",
        "labels_file",
        "overlay_file",
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
        "ritnet_input_size": list(runtime.input_size),
        "analysis_size": list(analysis_size),
        "class_mapping": {str(key): value for key, value in CLASS_MAPPING.items()},
        "labels_only": not bool(args.validate_pupil),
        "source_pupil_reused": not bool(args.validate_pupil),
        "pupil_validation_mode": bool(args.validate_pupil),
        "postprocess_workers": int(args.postprocess_workers),
        "qc_stride_frames": QC_STRIDE_FRAMES,
        "qc_anomaly_limit_per_reason_per_phase": QC_ANOMALY_LIMIT_PER_REASON_PER_PHASE,
        "qc_dir": str(outputs["qc_dir"]),
        "qc_index": str(outputs["qc_index"]),
        "qc_image_count": 0,
        "expected_rows": len(rows),
        "processed_rows": 0,
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "required_artifacts": [
            str(outputs["csv"]),
            str(outputs["summary"]),
            str(outputs["manifest"]),
            str(outputs["qc_index"]),
        ],
    }
    atomic_write_json(outputs["completion"], completion_base)

    # Versioned subject-specific QC directory; existing artifacts are never deleted.
    outputs["qc_dir"].mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)

    current_frame = min_frame
    raw_backlog: deque[tuple[dict[str, str], np.ndarray]] = deque()
    decoded_frames = 0
    decode_cpu_ms = 0.0
    crop_cpu_ms = 0.0
    preprocess_cpu_ms = 0.0
    gpu_ms = 0.0
    postprocess_cpu_ms = 0.0
    csv_write_ms = 0.0
    qc_image_write_ms = 0.0

    processed_rows = 0
    normalization_valid_count = 0
    parity_ok_count = 0
    parity_mismatch_count = 0
    fraction_sums = defaultdict(float)
    qc_records: list[dict[str, Any]] = []
    wall_started = time.perf_counter()

    def produce_batch() -> dict[str, Any] | None:
        nonlocal current_frame, decoded_frames, decode_cpu_ms, crop_cpu_ms, preprocess_cpu_ms

        items: list[tuple[dict[str, str], np.ndarray]] = []
        while raw_backlog and len(items) < runtime.FIXED_BATCH_SIZE:
            items.append(raw_backlog.popleft())

        while len(items) < runtime.FIXED_BATCH_SIZE and current_frame <= max_frame:
            decode_started = time.perf_counter()
            ok, frame = cap.read()
            decode_cpu_ms += (time.perf_counter() - decode_started) * 1000.0
            if not ok or frame is None:
                raise RuntimeError(f"Video read failed at frame {current_frame}: {video}")
            decoded_frames += 1

            if current_frame in target_set:
                for source_row in rows_by_frame[current_frame]:
                    crop_started = time.perf_counter()
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
                    crop_cpu_ms += (time.perf_counter() - crop_started) * 1000.0
                    raw_backlog.append((source_row, np.ascontiguousarray(roi)))

            current_frame += 1
            while raw_backlog and len(items) < runtime.FIXED_BATCH_SIZE:
                items.append(raw_backlog.popleft())

        if not items:
            return None

        prep_started = time.perf_counter()
        tensor, valid_size, prep_timing = runtime.prepare_batch([roi for _, roi in items])
        preprocess_cpu_ms += (time.perf_counter() - prep_started) * 1000.0
        return {
            "items": items,
            "tensor": tensor,
            "valid_size": valid_size,
            "prep_timing": prep_timing,
        }

    def postprocess_one(
        source_row: dict[str, str],
        labels: np.ndarray,
        pupil_probability: np.ndarray | None,
    ) -> tuple[dict[str, Any], dict[str, Any], float]:
        started = time.perf_counter()
        if args.validate_pupil:
            metrics = summarize_fullclass(
                labels,
                pupil_probability,
                analysis_size,
            )
            parity = pupil_parity(source_row, metrics)
        else:
            metrics = summarize_fullclass_from_source(
                labels,
                source_row,
                analysis_size,
            )
            parity = {
                "source_ritnet_found": parse_bool(source_row.get("ritnet_found")),
                "pupil_parity_found_match": None,
                "pupil_parity_center_max_abs_diff": None,
                "pupil_parity_equiv_diameter_abs_diff": None,
                "pupil_parity_contour_area_abs_diff": None,
                "pupil_parity_confidence_abs_diff": None,
                "pupil_parity_ok": None,
            }
        return metrics, parity, (time.perf_counter() - started) * 1000.0

    def consume_batch(
        writer: csv.DictWriter,
        batch_futures: list[
            tuple[dict[str, str], np.ndarray, np.ndarray, Future]
        ],
    ) -> None:
        nonlocal processed_rows, normalization_valid_count
        nonlocal parity_ok_count, parity_mismatch_count, postprocess_cpu_ms, csv_write_ms
        nonlocal qc_image_write_ms

        for source_row, roi_gray, labels, future in batch_futures:
            metrics, parity, post_ms = future.result()
            postprocess_cpu_ms += post_ms
            output_row: dict[str, Any] = dict(source_row)
            output_row.update({f"fullclass_{key}": value for key, value in metrics.items()})
            output_row.update(parity)
            output_row["fullclass_source_pupil_reused"] = not bool(args.validate_pupil)
            output_row["fullclass_pupil_validation_mode"] = bool(args.validate_pupil)

            write_started = time.perf_counter()
            writer.writerow(output_row)
            csv_write_ms += (time.perf_counter() - write_started) * 1000.0

            reasons = qc_sampler.select(source_row, metrics)
            if reasons:
                qc_started = time.perf_counter()
                labels_path, overlay_path = save_qc_pair(
                    outputs["qc_dir"],
                    subject,
                    source_row,
                    roi_gray,
                    labels,
                )
                qc_image_write_ms += (time.perf_counter() - qc_started) * 1000.0
                qc_records.append(
                    {
                        "subject": subject,
                        "phase": source_row.get("phase"),
                        "phase_segment": source_row.get("phase_segment"),
                        "frame_idx": source_row.get("frame_idx"),
                        "video_time_ms": source_row.get("video_time_ms"),
                        "unix_ms": source_row.get("unix_ms"),
                        "eye": source_row.get("eye"),
                        "reason": "+".join(reasons),
                        "ritnet_found": source_row.get("ritnet_found"),
                        "roi_clipped": source_row.get("roi_clipped"),
                        "normalization_valid": bool(metrics.get("normalization_valid")),
                        "labels_file": str(labels_path.relative_to(run_dir)),
                        "overlay_file": str(overlay_path.relative_to(run_dir)),
                    }
                )

            processed_rows += 1
            normalization_valid_count += int(bool(metrics.get("normalization_valid")))
            if args.validate_pupil:
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

    producer_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ritnet-prep")
    post_pool = ThreadPoolExecutor(
        max_workers=int(args.postprocess_workers),
        thread_name_prefix="ritnet-post",
    )
    post_batches: deque[
        list[tuple[dict[str, str], np.ndarray, np.ndarray, Future]]
    ] = deque()

    try:
        outputs["csv"].parent.mkdir(parents=True, exist_ok=True)
        with outputs["csv"].open("w", newline="", encoding="utf-8-sig") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=output_fields, extrasaction="ignore")
            writer.writeheader()

            next_batch_future = producer_pool.submit(produce_batch)
            while True:
                batch = next_batch_future.result()
                if batch is None:
                    break

                next_batch_future = producer_pool.submit(produce_batch)

                gpu_started = time.perf_counter()
                labels_batch, pupil_prob_batch, _ = runtime.infer_prepared(
                    batch["tensor"],
                    int(batch["valid_size"]),
                    include_pupil_probability=bool(args.validate_pupil),
                )
                gpu_ms += (time.perf_counter() - gpu_started) * 1000.0

                futures: list[
                    tuple[dict[str, str], np.ndarray, np.ndarray, Future]
                ] = []
                for index, (source_row, roi_gray) in enumerate(batch["items"]):
                    probability = (
                        pupil_prob_batch[index]
                        if pupil_prob_batch is not None
                        else None
                    )
                    labels = labels_batch[index]
                    future = post_pool.submit(
                        postprocess_one,
                        source_row,
                        labels,
                        probability,
                    )
                    futures.append((source_row, roi_gray, labels, future))
                post_batches.append(futures)

                if len(post_batches) >= 2:
                    consume_batch(writer, post_batches.popleft())

            while post_batches:
                consume_batch(writer, post_batches.popleft())
    finally:
        producer_pool.shutdown(wait=True, cancel_futures=False)
        post_pool.shutdown(wait=True, cancel_futures=False)
        cap.release()

    elapsed = time.perf_counter() - wall_started
    if processed_rows != len(rows):
        raise RuntimeError(f"Processed {processed_rows} rows but expected {len(rows)}")

    atomic_write_csv(outputs["qc_index"], qc_records, qc_index_fields)
    qc_image_count = len(qc_records) * 2

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
        "labels_only": not bool(args.validate_pupil),
        "source_pupil_reused": not bool(args.validate_pupil),
        "pupil_validation_mode": bool(args.validate_pupil),
        "postprocess_workers": int(args.postprocess_workers),
        "normalization_valid_count": normalization_valid_count,
        "normalization_valid_fraction": normalization_valid_count / processed_rows,
        "pupil_parity_ok_count": parity_ok_count if args.validate_pupil else None,
        "pupil_parity_mismatch_count": parity_mismatch_count if args.validate_pupil else None,
        "pupil_parity_ok_fraction": (
            parity_ok_count / processed_rows if args.validate_pupil else None
        ),
        "qc_sampling": {
            "stride_frames": QC_STRIDE_FRAMES,
            "phase_first_middle_last": True,
            "anomaly_limit_per_reason_per_phase": QC_ANOMALY_LIMIT_PER_REASON_PER_PHASE,
            "anchor_frame_count": len(qc_anchor_frames),
            "saved_eye_pairs": len(qc_records),
            "image_count": qc_image_count,
            "reason_counts": dict(sorted(qc_sampler.reason_counts.items())),
            "qc_dir": str(outputs["qc_dir"]),
            "qc_index": str(outputs["qc_index"]),
        },
        "timing_cpu_work_ms": {
            "decode": decode_cpu_ms,
            "roi_crop": crop_cpu_ms,
            "preprocess": preprocess_cpu_ms,
            "postprocess_sum_across_workers": postprocess_cpu_ms,
            "csv_write": csv_write_ms,
            "qc_image_write": qc_image_write_ms,
        },
        "timing_gpu_ms": gpu_ms,
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
        "labels_only": not bool(args.validate_pupil),
        "source_pupil_reused": not bool(args.validate_pupil),
        "pupil_validation_mode": bool(args.validate_pupil),
        "postprocess_workers": int(args.postprocess_workers),
        "primary_pupil_metric": "fullclass_pupil_to_iris_diameter_ratio",
        "qc_sampling": {
            "stride_frames": QC_STRIDE_FRAMES,
            "phase_first_middle_last": True,
            "anomaly_limit_per_reason_per_phase": QC_ANOMALY_LIMIT_PER_REASON_PER_PHASE,
            "formats": ["labels.png", "overlay.png"],
            "overlay_alpha": QC_OVERLAY_ALPHA,
            "palette_bgr": {
                str(key): list(value) for key, value in QC_PALETTE_BGR.items()
            },
            "anomaly_reasons": [
                "roi_clipped",
                "ritnet_missing",
                "normalization_invalid",
                "ocular_fragmented",
            ],
            "qc_dir": str(outputs["qc_dir"]),
            "qc_index": str(outputs["qc_index"]),
        },
        "time_mapping": {
            "copied_from_source_eyes_csv": True,
            "fields": [
                "phase",
                "phase_segment",
                "frame_idx",
                "video_time_ms",
                "unix_ms",
                "phase_time_ms",
            ],
            "note": (
                "Behavior/trial alignment should be performed later from unix_ms "
                "against behavior absolute timestamps; RITnet does not need to be rerun."
            ),
        },
        "notes": [
            "Original eyes.csv is never modified.",
            "YOLO is not re-run; source frame_idx and ROI coordinates are reused exactly.",
            "Production fast mode requests only hard labels_u8 from the existing ONNX.",
            "Production fast mode reuses the already-frozen pupil geometry/confidence from source eyes.csv.",
            "Validation mode requests pupil probability and recomputes pupil geometry only for parity checking.",
            "RITnet method remains 640x400, FP32, fixed batch=16; no FP16 or lower-resolution path is used.",
            "Iris/sclera/background probabilities are not fabricated.",
            "QC PNGs are sparse deterministic audit artifacts; numerical CSV remains the complete per-eye dataset.",
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
        "qc_index_sha256": sha256(outputs["qc_index"]),
        "qc_image_count": qc_image_count,
        "pupil_parity_mismatch_count": parity_mismatch_count if args.validate_pupil else None,
        "normalization_valid_count": normalization_valid_count,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(outputs["completion"], finished_completion)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Completion -> {outputs['completion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
