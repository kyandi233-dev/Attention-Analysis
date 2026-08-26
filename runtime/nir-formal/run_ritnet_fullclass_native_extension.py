"""Resumable native-640 RITnet evidence extension for one completed NIR run.

This v2 path is intentionally separate from ritnet-fullclass-v1.2-fast-qc.
It stores every native uint8 400x640 hard label map before committing the final
CSV, derives pupil and iris geometry from the same label map, and treats the
label store as the recoverable evidence source of truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ritnet_fullclass_contract import (
    CLASS_MAPPING,
    NATIVE_EXTENSION_SCHEMA_VERSION,
    NATIVE_EXTENSION_VERSION,
    NATIVE_GEOMETRY_ALGORITHM_VERSION,
    NATIVE_LABEL_CLASS_MAPPING_VERSION,
    NATIVE_LABEL_SCHEMA_VERSION,
    NATIVE_PREPROCESSING_VERSION,
    OFFICIAL_UPSTREAM_COMMIT,
    OFFICIAL_UPSTREAM_REPOSITORY,
    OFFICIAL_WEIGHTS_GIT_BLOB_SHA1,
    QC_ANOMALY_LIMIT_PER_REASON_PER_PHASE,
    QC_STRIDE_FRAMES,
    native_subject_output_paths,
    normalize_subject,
)
from ritnet_fullclass_qc import build_qc_anchor_frames, save_qc_pair
from ritnet_fullclass_runtime import RitnetFullClassRuntime
from ritnet_label_store import (
    DEFAULT_CHUNK_ROWS,
    LABEL_STORE_SCHEMA_VERSION,
    RitnetLabelStore,
    canonical_digest,
    sha256_file,
)
from ritnet_native_completion import verify_native_completion
from ritnet_native_metrics import (
    NATIVE_LABEL_HEIGHT,
    NATIVE_LABEL_WIDTH,
    summarize_fullclass_native,
    summarize_pupil_probability,
)

PROBABILITY_FIELDS = (
    "native_pupil_softmax_mean_on_argmax_mask",
    "native_pupil_softmax_median_on_argmax_mask",
    "native_pupil_softmax_p05_on_argmax_mask",
    "native_pupil_softmax_p95_on_argmax_mask",
    "native_pupil_softmax_min_on_argmax_mask",
    "native_pupil_softmax_max_on_argmax_mask",
)
ALLCLASS_UNAVAILABLE_REASON = (
    "production ONNX exposes deterministic hard labels plus class-3 pupil probability only; "
    "the optional all-class evidence-summary ONNX has not yet passed DirectML parity/performance qualification"
)
SOURCE_PUPIL_CONFIDENCE_DEFINITION = (
    "project-derived class-3 softmax probability mean on the source argmax pupil mask; "
    "not a calibrated probability that the frame/segmentation is correct"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def parse_int(value: Any) -> int:
    if value is None or str(value).strip() == "":
        raise ValueError("missing required integer value")
    return int(float(value))


def parse_bool(value: Any) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def resolve_package_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PACKAGE_ROOT / path


def git_identity() -> tuple[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PACKAGE_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PACKAGE_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:
        raise RuntimeError("v2 native evidence run requires Git commit provenance") from exc
    if len(commit) != 40:
        raise RuntimeError(f"unexpected git commit value: {commit!r}")
    return commit, branch


def video_identity(path: Path, *, hash_content: bool) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "content_sha256": sha256_file(path) if hash_content else None,
        "identity_strength": "content_sha256" if hash_content else "size+mtime+name",
    }


def _probability_stats_vector(summary: Mapping[str, Any]) -> np.ndarray:
    values = []
    for field in PROBABILITY_FIELDS:
        value = summary.get(field)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=np.float32)


def _probability_summary_from_checkpoint(available: bool, stats: np.ndarray) -> dict[str, Any]:
    stats = np.asarray(stats, dtype=np.float32)
    if stats.shape != (len(PROBABILITY_FIELDS),):
        raise ValueError(f"probability checkpoint stats must have shape (6,), got {stats.shape}")
    result: dict[str, Any] = {"native_pupil_probability_available": bool(available)}
    for field, value in zip(PROBABILITY_FIELDS, stats):
        result[field] = None if np.isnan(value) else float(value)
    return result


def _source_rows(path: Path, subject: str) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        required = {"frame_idx", "eye", "roi_x1", "roi_y1", "roi_x2", "roi_y2"}
        missing = required - set(fields)
        if missing:
            raise ValueError(f"eyes.csv missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Source eyes.csv contains no eye rows: {path}")

    frame_eye: set[tuple[int, str]] = set()
    previous_frame = -1
    for ordinal, row in enumerate(rows):
        row_subject = normalize_subject(row.get("subject") or subject)
        if row_subject != subject:
            raise ValueError(f"mixed subjects in eyes.csv at row {ordinal}: {row_subject} != {subject}")
        frame = parse_int(row["frame_idx"])
        eye = str(row["eye"])
        if frame < previous_frame:
            raise ValueError(
                "eyes.csv frame_idx must be nondecreasing so row-ordinal evidence can be decoded/resumed deterministically"
            )
        previous_frame = frame
        key = (frame, eye)
        if key in frame_eye:
            raise ValueError(f"duplicate source frame/eye key in eyes.csv: {key}")
        frame_eye.add(key)

    reserved_prefixes = ("native_", "gate_", "diagnostic_")
    collisions = [field for field in fields if field.startswith(reserved_prefixes)]
    collisions += [
        field for field in ("extension_version", "extension_schema_version", "legacy_v1_strict_valid")
        if field in fields
    ]
    if collisions:
        raise ValueError(f"source eyes.csv already contains v2-reserved columns: {sorted(set(collisions))}")
    return fields, rows


def _output_row(
    *,
    source_row: Mapping[str, Any],
    metrics: Mapping[str, Any],
    row_ordinal: int,
    chunk_id: int,
    chunk_offset: int,
    label_store_relpath: str,
    source_frame_width: int,
    source_frame_height: int,
    runtime: RitnetFullClassRuntime,
) -> dict[str, Any]:
    row = dict(source_row)
    x1 = parse_int(source_row["roi_x1"])
    y1 = parse_int(source_row["roi_y1"])
    x2 = parse_int(source_row["roi_x2"])
    y2 = parse_int(source_row["roi_y2"])
    roi_w = x2 - x1
    roi_h = y2 - y1
    if roi_w <= 0 or roi_h <= 0:
        raise ValueError(f"invalid source ROI size: {(x1, y1, x2, y2)}")
    scale_x = NATIVE_LABEL_WIDTH / roi_w
    scale_y = NATIVE_LABEL_HEIGHT / roi_h

    row.update(metrics)
    row.update(
        {
            "native_label_available": True,
            "native_label_store_relpath": label_store_relpath,
            "native_label_row_ordinal": int(row_ordinal),
            "native_label_chunk_id": int(chunk_id),
            "native_label_chunk_offset": int(chunk_offset),
            "native_label_width": NATIVE_LABEL_WIDTH,
            "native_label_height": NATIVE_LABEL_HEIGHT,
            "native_label_dtype": "uint8",
            "native_label_schema_version": NATIVE_LABEL_SCHEMA_VERSION,
            "native_label_class_mapping_version": NATIVE_LABEL_CLASS_MAPPING_VERSION,
            "source_frame_width": int(source_frame_width),
            "source_frame_height": int(source_frame_height),
            "source_roi_width": int(roi_w),
            "source_roi_height": int(roi_h),
            "ritnet_input_width": int(runtime.input_size[0]),
            "ritnet_input_height": int(runtime.input_size[1]),
            "roi_to_ritnet_scale_x": float(scale_x),
            "roi_to_ritnet_scale_y": float(scale_y),
            "roi_to_ritnet_aspect_scale_ratio": float(scale_x / scale_y),
            "geometry_coordinate_system": "ritnet_native_label",
            "geometry_width": NATIVE_LABEL_WIDTH,
            "geometry_height": NATIVE_LABEL_HEIGHT,
            "source_pupil_confidence": source_row.get("pupil_confidence"),
            "source_pupil_confidence_definition": SOURCE_PUPIL_CONFIDENCE_DEFINITION,
            "native_allclass_confidence_available": False,
            "native_allclass_confidence_unavailable_reason": ALLCLASS_UNAVAILABLE_REASON,
            "extension_version": NATIVE_EXTENSION_VERSION,
            "extension_schema_version": NATIVE_EXTENSION_SCHEMA_VERSION,
            "ritnet_model_id": runtime.weights.name,
            "ritnet_input_size": f"{runtime.input_size[0]}x{runtime.input_size[1]}",
            "ritnet_precision": runtime.precision,
            "ritnet_batch_size": runtime.FIXED_BATCH_SIZE,
            "ritnet_device": runtime.device,
            "preprocessing_version": NATIVE_PREPROCESSING_VERSION,
            "geometry_algorithm_version": NATIVE_GEOMETRY_ALGORITHM_VERSION,
            "label_store_schema_version": LABEL_STORE_SCHEMA_VERSION,
        }
    )
    return row


def _csv_key_match(csv_path: Path, index_path: Path, expected_rows: int) -> None:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        csv_keys = [
            (int(row["native_label_row_ordinal"]), int(float(row["frame_idx"])), str(row["eye"]))
            for row in reader
        ]
    with index_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        index_keys = [
            (int(row["row_ordinal"]), int(row["frame_idx"]), str(row["eye"]))
            for row in reader
        ]
    if len(csv_keys) != expected_rows or len(csv_keys) != len(set(csv_keys)):
        raise RuntimeError("final CSV key count/uniqueness check failed")
    if csv_keys != index_keys:
        raise RuntimeError("final CSV keys do not exactly match label_index")


def _phase(row: Mapping[str, Any]) -> str:
    return str(row.get("phase") or "unknown")


def _build_qc(
    *,
    outputs: Mapping[str, Path],
    run_dir: Path,
    subject: str,
    video: Path,
    source_rows: list[dict[str, str]],
    store: RitnetLabelStore,
) -> list[dict[str, Any]]:
    anchors = build_qc_anchor_frames(source_rows, QC_STRIDE_FRAMES)
    anomaly_counts: dict[tuple[str, str], int] = defaultdict(int)
    selected: list[tuple[dict[str, str], dict[str, Any], dict[str, Any]]] = []
    saved_keys: set[tuple[int, str]] = set()

    for record in store.iter_rows():
        ordinal = int(record["row_ordinal"])
        source = source_rows[ordinal]
        prob_summary = _probability_summary_from_checkpoint(
            bool(record["pupil_probability_available"]), record["pupil_probability_stats"]
        )
        metrics = summarize_fullclass_native(record["labels"], probability_summary=prob_summary)
        frame = parse_int(source["frame_idx"])
        eye = str(source["eye"])
        key = (frame, eye)
        reasons: list[str] = []
        if frame in anchors:
            reasons.append("anchor")
        candidates = {
            "roi_clipped": parse_bool(source.get("roi_clipped")) is True,
            "native_pupil_fit_missing": not bool(metrics["gate_pupil_fit_valid"]),
            "native_iris_fit_missing": not bool(metrics["gate_iris_outer_fit_valid"]),
            "ocular_fragmented": bool(metrics["diagnostic_ocular_fragmented"]),
            "pupil_edge": bool(metrics["diagnostic_pupil_whole_mask_edge"]),
            "iris_edge": bool(metrics["diagnostic_iris_whole_mask_edge"]),
        }
        for reason, active in candidates.items():
            counter_key = (_phase(source), reason)
            if active and anomaly_counts[counter_key] < QC_ANOMALY_LIMIT_PER_REASON_PER_PHASE:
                reasons.append(reason)
                anomaly_counts[counter_key] += 1
        if reasons and key not in saved_keys:
            selected.append((source, metrics, {**record, "reasons": reasons}))
            saved_keys.add(key)

    outputs["qc_dir"].mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for QC: {video}")
    qc_rows: list[dict[str, Any]] = []
    try:
        for source, metrics, record in selected:
            frame_idx = parse_int(source["frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"QC video read failed at frame {frame_idx}")
            x1, y1, x2, y2 = (parse_int(source[key]) for key in ("roi_x1", "roi_y1", "roi_x2", "roi_y2"))
            if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
                raise ValueError(f"QC ROI out of bounds at frame {frame_idx}: {(x1, y1, x2, y2)}")
            roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            labels_path, overlay_path = save_qc_pair(
                outputs["qc_dir"], subject, source, roi, record["labels"]
            )
            qc_rows.append(
                {
                    "subject": subject,
                    "phase": source.get("phase"),
                    "phase_segment": source.get("phase_segment"),
                    "frame_idx": source.get("frame_idx"),
                    "video_time_ms": source.get("video_time_ms"),
                    "unix_ms": source.get("unix_ms"),
                    "eye": source.get("eye"),
                    "reason": "+".join(record["reasons"]),
                    "native_label_row_ordinal": record["row_ordinal"],
                    "gate_pupil_fit_valid": metrics["gate_pupil_fit_valid"],
                    "gate_iris_outer_fit_valid": metrics["gate_iris_outer_fit_valid"],
                    "diagnostic_ocular_fragmented": metrics["diagnostic_ocular_fragmented"],
                    "labels_file": str(labels_path.relative_to(run_dir)),
                    "overlay_file": str(overlay_path.relative_to(run_dir)),
                }
            )
    finally:
        cap.release()

    fields = [
        "subject", "phase", "phase_segment", "frame_idx", "video_time_ms", "unix_ms", "eye",
        "reason", "native_label_row_ordinal", "gate_pupil_fit_valid", "gate_iris_outer_fit_valid",
        "diagnostic_ocular_fragmented", "labels_file", "overlay_file",
    ]
    atomic_write_csv(outputs["qc_index"], qc_rows, fields)
    return qc_rows


def _iter_remaining_rois(
    *,
    video: Path,
    rows: list[dict[str, str]],
    start_ordinal: int,
) -> Iterable[tuple[int, dict[str, str], np.ndarray]]:
    if start_ordinal >= len(rows):
        return
    grouped: dict[int, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for ordinal in range(start_ordinal, len(rows)):
        grouped[parse_int(rows[ordinal]["frame_idx"])].append((ordinal, rows[ordinal]))
    target_frames = sorted(grouped)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    current = target_frames[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, current)
    try:
        for target in target_frames:
            while current <= target:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Video read failed at frame {current}: {video}")
                if current == target:
                    for ordinal, source in grouped[target]:
                        x1, y1, x2, y2 = (
                            parse_int(source[key]) for key in ("roi_x1", "roi_y1", "roi_x2", "roi_y2")
                        )
                        if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
                            raise ValueError(
                                f"Invalid source ROI ordinal={ordinal} frame={current} eye={source.get('eye')}: "
                                f"{(x1, y1, x2, y2)} for frame={frame.shape}"
                            )
                        roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                        yield ordinal, source, np.ascontiguousarray(roi)
                current += 1
    finally:
        cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce resumable native640 RITnet evidence for one formal NIR run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--compression", choices=("npz_compressed", "npz_stored"), default="npz_compressed")
    parser.add_argument("--hash-video", action="store_true", help="Include full source-video SHA256 in strict resume identity")
    parser.add_argument("--allow-model-mismatch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")
    run_dir = args.run_dir.resolve()
    config_path = args.config.resolve()
    source_completion_path = run_dir / "completion.json"
    source_eyes = run_dir / "eyes.csv"
    if not source_completion_path.is_file() or not source_eyes.is_file():
        raise FileNotFoundError(f"run-dir must contain completion.json and eyes.csv: {run_dir}")
    source_completion = load_json(source_completion_path)
    if source_completion.get("status") != "complete":
        raise RuntimeError(f"source formal run is not complete: {source_completion.get('status')!r}")
    if source_completion.get("max_frames") is not None or source_completion.get("partial_phase_selection"):
        raise RuntimeError("v2 native evidence requires a non-truncated complete formal source run")

    subject = normalize_subject(source_completion.get("subject") or run_dir.name.split("_formal_", 1)[0])
    video = Path(str(source_completion.get("video", ""))).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"source video unavailable: {video}")
    config = load_config(config_path)
    ritnet_path = resolve_package_path(config["models"]["ritnet"]).resolve()
    external_path = resolve_package_path(config["models"]["ritnet_external_data"]).resolve()
    if not ritnet_path.is_file() or not external_path.is_file():
        raise FileNotFoundError(f"RITnet ONNX/external data missing: {ritnet_path} / {external_path}")

    ritnet_sha = sha256_file(ritnet_path)
    ritnet_external_sha = sha256_file(external_path)
    source_model_sha = source_completion.get("ritnet_model_sha256")
    if source_model_sha and source_model_sha != ritnet_sha and not args.allow_model_mismatch:
        raise RuntimeError(
            f"source formal RITnet hash differs from current model: source={source_model_sha}, current={ritnet_sha}"
        )

    source_fields, rows = _source_rows(source_eyes, subject)
    source_eyes_sha = sha256_file(source_eyes)
    git_commit, git_branch = git_identity()
    config_sha = sha256_file(config_path)
    video_id = video_identity(video, hash_content=bool(args.hash_video))
    eye_values = sorted({str(row["eye"]) for row in rows})
    if len(eye_values) > 255:
        raise ValueError("more than 255 distinct eye labels cannot be encoded as uint8")
    eye_mapping = {eye: index for index, eye in enumerate(eye_values)}

    resume_identity = {
        "subject": subject,
        "source_eyes_sha256": source_eyes_sha,
        "source_video_identity": video_id,
        "ritnet_onnx_sha256": ritnet_sha,
        "ritnet_external_data_sha256": ritnet_external_sha,
        "ritnet_input_size": [int(config["ritnet"]["input_width"]), int(config["ritnet"]["input_height"])],
        "class_mapping": {str(k): v for k, v in CLASS_MAPPING.items()},
        "preprocessing_version": NATIVE_PREPROCESSING_VERSION,
        "roi_source": {
            "source_formal_package_version": source_completion.get("package_version"),
            "source_yolo_model_sha256": source_completion.get("yolo_model_sha256"),
            "source_yolo_batch_size": source_completion.get("yolo_batch_size"),
        },
        "extension_version": NATIVE_EXTENSION_VERSION,
        "extension_schema_version": NATIVE_EXTENSION_SCHEMA_VERSION,
        "git_commit": git_commit,
        "config_sha256": config_sha,
        "chunk_format": "chunked_npz",
        "chunk_rows": int(args.chunk_rows),
        "compression": str(args.compression),
        "label_store_schema_version": LABEL_STORE_SCHEMA_VERSION,
        "eye_mapping": eye_mapping,
    }
    resume_digest = canonical_digest(resume_identity)
    outputs = native_subject_output_paths(run_dir, subject)

    if outputs["completion"].is_file():
        check = verify_native_completion(outputs["completion"], expected_identity=resume_identity)
        if check.valid:
            print(f"[SKIP] {subject}: strict v2 completion verified -> {outputs['completion']}")
            return 0
        print("[RESUME/REPAIR] existing v2 completion is not currently valid:")
        for error in check.errors[:10]:
            print(f"  - {error}")

    runtime = RitnetFullClassRuntime(
        PACKAGE_ROOT,
        ritnet_path,
        input_size=(int(config["ritnet"]["input_width"]), int(config["ritnet"]["input_height"])),
        device=str(args.device),
        analysis_size=(int(config["roi"]["width"]), int(config["roi"]["height"])),
        precision="fp32",
    )
    if tuple(runtime.input_size) != (NATIVE_LABEL_WIDTH, NATIVE_LABEL_HEIGHT):
        raise RuntimeError(f"v2 requires RITnet input 640x400; got {runtime.input_size}")

    cap_meta = cv2.VideoCapture(str(video))
    if not cap_meta.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    source_frame_width = int(round(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH)))
    source_frame_height = int(round(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap_meta.release()
    if source_frame_width <= 0 or source_frame_height <= 0:
        raise RuntimeError("invalid source video dimensions")

    label_store_relpath = str(outputs["labels_dir"].relative_to(run_dir)).replace("\\", "/")
    store_identity = {
        "resume_identity_digest": resume_digest,
        "subject": subject,
        "extension_version": NATIVE_EXTENSION_VERSION,
    }
    store = RitnetLabelStore(
        outputs["labels_dir"],
        identity=store_identity,
        eye_mapping=eye_mapping,
        chunk_rows=int(args.chunk_rows),
        compression=str(args.compression),
    )
    if store.stored_rows > len(rows):
        raise RuntimeError(f"label store has {store.stored_rows} rows but source eyes.csv has {len(rows)}")

    prototype_labels = np.zeros((NATIVE_LABEL_HEIGHT, NATIVE_LABEL_WIDTH), dtype=np.uint8)
    prototype_metrics = summarize_fullclass_native(prototype_labels)
    locator_fields = [
        "native_label_available", "native_label_store_relpath", "native_label_row_ordinal",
        "native_label_chunk_id", "native_label_chunk_offset", "native_label_width", "native_label_height",
        "native_label_dtype", "native_label_schema_version", "native_label_class_mapping_version",
        "source_frame_width", "source_frame_height", "source_roi_width", "source_roi_height",
        "ritnet_input_width", "ritnet_input_height", "roi_to_ritnet_scale_x", "roi_to_ritnet_scale_y",
        "roi_to_ritnet_aspect_scale_ratio", "geometry_coordinate_system", "geometry_width", "geometry_height",
        "source_pupil_confidence", "source_pupil_confidence_definition",
        "native_allclass_confidence_available", "native_allclass_confidence_unavailable_reason",
        "extension_version", "extension_schema_version", "ritnet_model_id", "ritnet_input_size",
        "ritnet_precision", "ritnet_batch_size", "ritnet_device", "preprocessing_version",
        "geometry_algorithm_version", "label_store_schema_version",
    ]
    metric_fields = list(prototype_metrics.keys())
    output_fields = source_fields + [
        field for field in metric_fields + locator_fields if field not in source_fields
    ]

    running_completion = {
        "schema_version": NATIVE_EXTENSION_SCHEMA_VERSION,
        "extension_version": NATIVE_EXTENSION_VERSION,
        "status": "running",
        "subject": subject,
        "resume_identity": resume_identity,
        "resume_identity_digest": resume_digest,
        "label_store_identity_digest": store.identity_digest,
        "expected_rows": len(rows),
        "processed_rows": store.stored_rows,
        "stored_label_rows": store.stored_rows,
        "started_or_resumed_at_utc": utc_now(),
        "label_store_root": str(outputs["labels_dir"]),
    }
    atomic_write_json(outputs["completion"], running_completion)

    partial_csv = outputs["csv"].with_name(f".{outputs['csv'].name}.{uuid.uuid4().hex}.partial")
    started = time.perf_counter()
    inferred_rows = 0
    try:
        with partial_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
            writer.writeheader()

            for record in store.iter_rows():
                ordinal = int(record["row_ordinal"])
                source = rows[ordinal]
                if parse_int(source["frame_idx"]) != int(record["frame_idx"]) or str(source["eye"]) != record["eye"]:
                    raise RuntimeError(f"source eyes.csv no longer matches label store at row {ordinal}")
                prob_summary = _probability_summary_from_checkpoint(
                    bool(record["pupil_probability_available"]), record["pupil_probability_stats"]
                )
                metrics = summarize_fullclass_native(record["labels"], probability_summary=prob_summary)
                writer.writerow(
                    _output_row(
                        source_row=source,
                        metrics=metrics,
                        row_ordinal=ordinal,
                        chunk_id=int(record["chunk_id"]),
                        chunk_offset=int(record["chunk_offset"]),
                        label_store_relpath=label_store_relpath,
                        source_frame_width=source_frame_width,
                        source_frame_height=source_frame_height,
                        runtime=runtime,
                    )
                )

            pending: list[dict[str, Any]] = []

            def commit_pending(count: int) -> None:
                nonlocal pending
                if count <= 0:
                    return
                chunk = pending[:count]
                labels = np.stack([item["labels"] for item in chunk], axis=0).astype(np.uint8, copy=False)
                ordinals = np.asarray([item["ordinal"] for item in chunk], dtype=np.int64)
                frames = np.asarray([parse_int(item["source"]["frame_idx"]) for item in chunk], dtype=np.int64)
                eyes = [str(item["source"]["eye"]) for item in chunk]
                available = np.asarray(
                    [bool(item["prob_summary"]["native_pupil_probability_available"]) for item in chunk],
                    dtype=np.uint8,
                )
                stats = np.stack([_probability_stats_vector(item["prob_summary"]) for item in chunk], axis=0)
                meta = store.append_chunk(
                    labels=labels,
                    row_ordinal=ordinals,
                    frame_idx=frames,
                    eye=eyes,
                    pupil_probability_available=available,
                    pupil_probability_stats=stats,
                )
                for offset, item in enumerate(chunk):
                    writer.writerow(
                        _output_row(
                            source_row=item["source"],
                            metrics=item["metrics"],
                            row_ordinal=int(item["ordinal"]),
                            chunk_id=int(meta["chunk_id"]),
                            chunk_offset=offset,
                            label_store_relpath=label_store_relpath,
                            source_frame_width=source_frame_width,
                            source_frame_height=source_frame_height,
                            runtime=runtime,
                        )
                    )
                handle.flush()
                os.fsync(handle.fileno())
                pending = pending[count:]
                atomic_write_json(
                    outputs["completion"],
                    {
                        **running_completion,
                        "processed_rows": store.stored_rows,
                        "stored_label_rows": store.stored_rows,
                        "last_checkpoint_at_utc": utc_now(),
                    },
                )

            roi_batch: list[tuple[int, dict[str, str], np.ndarray]] = []
            for item in _iter_remaining_rois(video=video, rows=rows, start_ordinal=store.next_row_ordinal):
                roi_batch.append(item)
                if len(roi_batch) < runtime.FIXED_BATCH_SIZE:
                    continue
                rois = [roi for _, _, roi in roi_batch]
                labels_batch, probs_batch, _ = runtime.infer_batch(rois, include_pupil_probability=True)
                if probs_batch is None:
                    raise RuntimeError("v2 requires the class-3 pupil probability output")
                for (ordinal, source, _), labels, probs in zip(roi_batch, labels_batch, probs_batch):
                    prob_summary = summarize_pupil_probability(labels, probs)
                    metrics = summarize_fullclass_native(labels, probability_summary=prob_summary)
                    pending.append(
                        {"ordinal": ordinal, "source": source, "labels": labels, "prob_summary": prob_summary, "metrics": metrics}
                    )
                    inferred_rows += 1
                roi_batch = []
                while len(pending) >= store.chunk_rows:
                    commit_pending(store.chunk_rows)

            if roi_batch:
                rois = [roi for _, _, roi in roi_batch]
                labels_batch, probs_batch, _ = runtime.infer_batch(rois, include_pupil_probability=True)
                if probs_batch is None:
                    raise RuntimeError("v2 requires the class-3 pupil probability output")
                for (ordinal, source, _), labels, probs in zip(roi_batch, labels_batch, probs_batch):
                    prob_summary = summarize_pupil_probability(labels, probs)
                    metrics = summarize_fullclass_native(labels, probability_summary=prob_summary)
                    pending.append(
                        {"ordinal": ordinal, "source": source, "labels": labels, "prob_summary": prob_summary, "metrics": metrics}
                    )
                    inferred_rows += 1
            if pending:
                commit_pending(len(pending))
            handle.flush()
            os.fsync(handle.fileno())

        store_report = store.finalize(len(rows))
        _csv_key_match(partial_csv, store.index_path, len(rows))
        os.replace(partial_csv, outputs["csv"])

    except Exception:
        print(f"[PARTIAL] derived CSV retained for diagnosis: {partial_csv}", file=sys.stderr)
        raise

    qc_rows = _build_qc(
        outputs=outputs,
        run_dir=run_dir,
        subject=subject,
        video=video,
        source_rows=rows,
        store=store,
    )
    elapsed = time.perf_counter() - started

    summary = {
        "subject": subject,
        "extension_version": NATIVE_EXTENSION_VERSION,
        "processed_rows": len(rows),
        "resumed_from_committed_rows": len(rows) - inferred_rows,
        "newly_inferred_rows": inferred_rows,
        "elapsed_sec": elapsed,
        "new_inference_roi_per_sec": (inferred_rows / elapsed) if elapsed and inferred_rows else None,
        "label_chunk_count": store_report.chunk_count,
        "label_store_verified": store_report.valid,
        "qc_saved_eye_pairs": len(qc_rows),
        "raw_evidence": str(outputs["labels_dir"]),
        "derived_numeric_csv": str(outputs["csv"]),
        "human_qc_index": str(outputs["qc_index"]),
        "primary_valid_gate_defined": False,
        "native_coordinate_warning": (
            "native640 geometry is measured after project ROI->640x400 resize; when scale_x != scale_y it is anisotropically warped. "
            "Stored scale fields and hard labels permit later source-ROI-coordinate reconstruction without RITnet rerun."
        ),
    }
    atomic_write_json(outputs["summary"], summary)

    preprocessing = {
        "grayscale": {"operation": "convert/crop source video eye ROI to grayscale", "provenance": "project adapter; upstream RITnet expects grayscale"},
        "resize": {
            "size": [NATIVE_LABEL_WIDTH, NATIVE_LABEL_HEIGHT],
            "interpolation": "cv2.INTER_LINEAR",
            "provenance": "project ROI adapter, not an upstream RITnet output variable",
        },
        "gamma": {"factor": 0.8, "provenance": "upstream RITnet required preprocessing"},
        "clahe": {"clip_limit": 1.5, "tile_grid_size": [8, 8], "provenance": "upstream RITnet required preprocessing"},
        "normalization": {"formula": "(x/255 - 0.5) / 0.5", "provenance": "upstream RITnet Normalize([0.5],[0.5]) equivalent"},
    }
    manifest = {
        "created_at_utc": utc_now(),
        "extension_version": NATIVE_EXTENSION_VERSION,
        "extension_schema_version": NATIVE_EXTENSION_SCHEMA_VERSION,
        "subject": subject,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "config_snapshot": config,
        "video_path": str(video),
        "video_identity": video_id,
        "source_eyes_path": str(source_eyes),
        "source_eyes_sha256": source_eyes_sha,
        "ritnet_onnx_path": str(ritnet_path),
        "ritnet_onnx_sha256": ritnet_sha,
        "ritnet_external_data_path": str(external_path),
        "ritnet_external_data_sha256": ritnet_external_sha,
        "official_upstream_repository": OFFICIAL_UPSTREAM_REPOSITORY,
        "official_upstream_commit": OFFICIAL_UPSTREAM_COMMIT,
        "official_weights_git_blob_sha1": OFFICIAL_WEIGHTS_GIT_BLOB_SHA1,
        "provenance_layers": {
            "upstream_official": [
                "DenseNet2D network definition", "best_model.pkl weights", "four-class semantic segmentation",
                "class order background/sclera/iris/pupil", "gamma 0.8", "CLAHE 1.5 8x8", "0.5/0.5 normalization",
            ],
            "project_deterministic_adapter": [
                "saved YOLO ROI reuse", "ROI resize to 640x400", "fixed-b16 FP32 ONNX export", "DirectML runtime",
                "ArgMax labels_u8 output", "class-3 softmax pupil_probability output",
            ],
            "project_derived_records": [
                "all native_* geometry/count/fraction/component/edge/aperture/PIR fields", "gate_* and diagnostic_* facts",
                "probability summaries", "chunked label store", "QC PNG/index", "manifest/completion hashes",
            ],
            "native_prefix_definition": (
                "native_ means measured/stored in the 640x400 RITnet hard-label coordinate system; it does not mean upstream RITnet supplied that variable"
            ),
        },
        "preprocessing": preprocessing,
        "label_store": {
            "format": "chunked_npz",
            "schema_version": LABEL_STORE_SCHEMA_VERSION,
            "compression": args.compression,
            "chunk_rows": args.chunk_rows,
            "chunk_rows_status": "provisional_until_sub031_storage_throughput_benchmark",
            "shape": [NATIVE_LABEL_HEIGHT, NATIVE_LABEL_WIDTH],
            "dtype": "uint8",
            "class_mapping": {str(k): v for k, v in CLASS_MAPPING.items()},
            "chunk_manifest": str(store.chunk_manifest_path),
            "index": str(store.index_path),
            "probability_summary_checkpointed_with_chunk": True,
        },
        "geometry": {
            "coordinate_system": "ritnet_native_label",
            "contour_retrieval": "cv2.RETR_EXTERNAL",
            "contour_approximation": "cv2.CHAIN_APPROX_SIMPLE",
            "contour_selection": "largest by cv2.contourArea",
            "ellipse_implementation": "cv2.fitEllipse",
            "pupil_and_iris_from_same_label": True,
            "iris_outer_definition": "class_2 OR class_3",
            "pir_definition": "sqrt(pupil_axis_a*pupil_axis_b) / sqrt(iris_axis_a*iris_axis_b)",
            "anisotropic_resize_warning": (
                "ROI->640x400 may use unequal scale_x/scale_y; native PIR is therefore explicitly a model-coordinate measure, not claimed invariant in source-pixel geometry"
            ),
        },
        "confidence": {
            "pupil_probability_available": True,
            "pupil_probability_definition": "class-3 softmax probability from project ONNX adapter",
            "pupil_summary_domain": "pixels where hard argmax label == pupil",
            "calibration_status": "not calibrated as frame/segmentation correctness probability",
            "allclass_confidence_available": False,
            "allclass_confidence_unavailable_reason": ALLCLASS_UNAVAILABLE_REASON,
        },
        "gates": {
            "primary_valid_gate_defined": False,
            "atomic_facts_only": True,
            "legacy_v1_strict_valid": (
                "old logical gate replayed on native640 geometry for reference only; not bit-identical to v1.2 320x160 output"
            ),
        },
        "evidence_layers": {
            "raw_evidence": "native label store",
            "derived_numeric": "v2 CSV",
            "human_review": "sparse QC PNG/index",
        },
        "resume_identity": resume_identity,
        "resume_identity_digest": resume_digest,
    }
    atomic_write_json(outputs["manifest"], manifest)

    store_report = store.verify(expected_rows=len(rows))
    if not store_report.valid:
        raise RuntimeError("label store failed final pre-completion verification")
    _csv_key_match(outputs["csv"], store.index_path, len(rows))
    completion = {
        "schema_version": NATIVE_EXTENSION_SCHEMA_VERSION,
        "extension_version": NATIVE_EXTENSION_VERSION,
        "status": "complete",
        "subject": subject,
        "resume_identity": resume_identity,
        "resume_identity_digest": resume_digest,
        "label_store_identity_digest": store.identity_digest,
        "expected_rows": len(rows),
        "processed_rows": len(rows),
        "stored_label_rows": store_report.stored_rows,
        "label_chunk_count": store_report.chunk_count,
        "label_store_verified": True,
        "label_value_domain_verified": store_report.label_value_domain_verified,
        "label_shape_verified": store_report.label_shape_verified,
        "label_index_unique_verified": store_report.label_index_unique_verified,
        "label_csv_key_match_verified": True,
        "output_csv": str(outputs["csv"]),
        "label_store_root": str(outputs["labels_dir"]),
        "label_index": str(store.index_path),
        "chunk_manifest": str(store.chunk_manifest_path),
        "store_manifest": str(store.store_manifest_path),
        "summary": str(outputs["summary"]),
        "manifest": str(outputs["manifest"]),
        "qc_index": str(outputs["qc_index"]),
        "output_csv_sha256": sha256_file(outputs["csv"]),
        "label_index_sha256": sha256_file(store.index_path),
        "chunk_manifest_sha256": sha256_file(store.chunk_manifest_path),
        "store_manifest_sha256": sha256_file(store.store_manifest_path),
        "summary_sha256": sha256_file(outputs["summary"]),
        "manifest_sha256": sha256_file(outputs["manifest"]),
        "qc_index_sha256": sha256_file(outputs["qc_index"]),
        "artifact_hashes_verified_at_utc": utc_now(),
        "finished_at_utc": utc_now(),
    }
    atomic_write_json(outputs["completion"], completion)
    final_check = verify_native_completion(outputs["completion"], expected_identity=resume_identity)
    if not final_check.valid:
        raise RuntimeError("written completion failed strict verification: " + "; ".join(final_check.errors))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Completion -> {outputs['completion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
