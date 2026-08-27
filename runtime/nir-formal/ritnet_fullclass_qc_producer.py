"""Produce bounded frame-level QC images and an integrity-ready QC index."""
from __future__ import annotations

import csv
import hashlib
import io
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import cv2
import numpy as np

from ritnet_fullclass_final_runtime import FIXED_BATCH_SIZE, RitnetFullClassFinalRuntime
from ritnet_fullclass_io import iter_csv_gz
from ritnet_fullclass_qc import (
    QC_SELECTION_VERSION,
    QCSelection,
    build_qc_selections,
    qc_frame_image_path,
    render_qc_images,
)
from ritnet_fullclass_qc_render import QC_COMPOSITE_VERSION, render_qc_composite
from ritnet_fullclass_roi import (
    PADDING_MODE_REPLICATE,
    crop_fixed_aspect_gray,
    fixed_aspect_roi_geometry,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
QC_INDEX_SCHEMA_VERSION = 1
QC_VIDEO_SEEK_GAP_THRESHOLD = 64
QC_FRAME_GROUP_MAX = 16
QC_INDEX_FIELDS = (
    "qc_index_schema_version",
    "qc_selection_version",
    "qc_composite_version",
    "subject",
    "phase",
    "phase_segment",
    "frame_idx",
    "coverage_status",
    "reasons",
    "eyes",
    "source_frame_available",
    "left_overlay_available",
    "right_overlay_available",
    "image_path",
    "image_sha256",
    "image_size_bytes",
)


@dataclass(frozen=True)
class QCArtifacts:
    qc_dir: Path
    images_dir: Path
    index_path: Path
    selected_count: int
    saved_image_count: int
    skipped_for_budget_count: int
    image_bytes: int
    index_bytes: int
    total_qc_bytes: int


def _resolve_package_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PACKAGE_ROOT / path


def _frame_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("phase") or ""),
        int(float(row["phase_segment"])),
        int(float(row["frame_idx"])),
    )


def _roi_config(config: Mapping[str, Any]) -> dict[str, Any]:
    full = config.get("fullclass")
    if not isinstance(full, Mapping):
        raise ValueError("config.fullclass must be a mapping")
    roi = full.get("roi")
    if not isinstance(roi, Mapping):
        raise ValueError("config.fullclass.roi must be a mapping")
    return {
        "expand_horizontal_each_side": float(roi["expand_horizontal_each_side"]),
        "expand_vertical_each_side": float(roi["expand_vertical_each_side"]),
        "padding_mode": str(roi["padding_mode"]),
    }


def _assert_geometry_matches_metric(geometry: Any, row: Mapping[str, Any]) -> None:
    generated = geometry.as_dict()
    for field, expected in generated.items():
        actual = row.get(field)
        if actual is None or str(actual).strip() == "":
            raise RuntimeError(f"QC cannot reproduce ROI because eye_metrics lacks {field}")
        if isinstance(expected, str):
            if str(actual) != expected:
                raise RuntimeError(f"QC ROI provenance mismatch for {field}: {actual!r} != {expected!r}")
        elif isinstance(expected, int):
            if int(float(actual)) != expected:
                raise RuntimeError(f"QC ROI provenance mismatch for {field}: {actual!r} != {expected!r}")
        else:
            if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-8):
                raise RuntimeError(f"QC ROI provenance mismatch for {field}: {actual!r} != {expected!r}")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _encode_index(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(QC_INDEX_FIELDS), extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _prepare_eye_rois(
    *,
    frame: np.ndarray,
    eye_rows: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[tuple[str, np.ndarray]]:
    height, width = frame.shape[:2]
    roi_cfg = _roi_config(config)
    prepared: list[tuple[str, np.ndarray]] = []
    for eye in ("frame_left", "frame_right"):
        row = eye_rows.get(eye)
        if not row or str(row.get("ritnet_status") or "").strip().lower() != "success":
            continue
        geometry = fixed_aspect_roi_geometry(
            bbox=(
                float(row["yolo_bbox_x1"]),
                float(row["yolo_bbox_y1"]),
                float(row["yolo_bbox_x2"]),
                float(row["yolo_bbox_y2"]),
            ),
            frame_width=width,
            frame_height=height,
            expand_horizontal_each_side=roi_cfg["expand_horizontal_each_side"],
            expand_vertical_each_side=roi_cfg["expand_vertical_each_side"],
            padding_mode=roi_cfg["padding_mode"],
        )
        _assert_geometry_matches_metric(geometry, row)
        prepared.append((eye, crop_fixed_aspect_gray(frame, geometry)))
    return prepared


def _prepare_eye_overlays(
    *,
    frame: np.ndarray,
    eye_rows: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    runtime: RitnetFullClassFinalRuntime,
) -> dict[str, np.ndarray]:
    """Small-call helper retained for tests; production QC batches across frames."""
    prepared = _prepare_eye_rois(frame=frame, eye_rows=eye_rows, config=config)
    if not prepared:
        return {}
    labels, _timing = runtime.infer_labels_batch([roi for _eye, roi in prepared])
    overlays: dict[str, np.ndarray] = {}
    for index, (eye, roi) in enumerate(prepared):
        _labels_color, overlay = render_qc_images(roi, labels[index])
        overlays[eye] = overlay
    return overlays


def _successful_eye_count(rows: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(
        str(row.get("ritnet_status") or "").strip().lower() == "success"
        for row in rows.values()
    )


def _selection_groups(
    ordered: list[QCSelection],
    eyes_by_key: Mapping[tuple[str, int, int], Mapping[str, Mapping[str, Any]]],
) -> Iterator[list[QCSelection]]:
    """Pack QC frames so one group never needs more than the fixed b16 eye slots."""
    group: list[QCSelection] = []
    eye_slots = 0
    for selection in ordered:
        needed = _successful_eye_count(eyes_by_key.get(selection.key, {}))
        if group and (eye_slots + needed > FIXED_BATCH_SIZE or len(group) >= QC_FRAME_GROUP_MAX):
            yield group
            group = []
            eye_slots = 0
        group.append(selection)
        eye_slots += needed
        if eye_slots == FIXED_BATCH_SIZE or len(group) >= QC_FRAME_GROUP_MAX:
            yield group
            group = []
            eye_slots = 0
    if group:
        yield group


def _read_qc_frame(
    cap: Any,
    target_frame: int,
    current_frame: int | None,
) -> tuple[np.ndarray | None, int | None]:
    """Read forward for nearby QC frames; seek only across large/backward gaps."""
    target_frame = int(target_frame)
    if (
        current_frame is None
        or target_frame < current_frame
        or target_frame - current_frame > QC_VIDEO_SEEK_GAP_THRESHOLD
    ):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        current_frame = target_frame

    frame: np.ndarray | None = None
    while current_frame is not None and current_frame <= target_frame:
        ok, decoded = cap.read()
        if not ok or decoded is None:
            frame = None
            break
        if current_frame == target_frame:
            frame = decoded
        current_frame += 1

    if frame is not None:
        return frame, current_frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, decoded = cap.read()
    if ok and decoded is not None:
        return decoded, target_frame + 1
    return None, None


def _index_rows(
    *,
    subject: str,
    subject_dir: Path,
    encoded_images: list[tuple[QCSelection, Path, bytes, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selection, path, payload, facts in sorted(
        encoded_images,
        key=lambda item: (item[0].phase, item[0].phase_segment, item[0].frame_idx),
    ):
        rows.append(
            {
                "qc_index_schema_version": QC_INDEX_SCHEMA_VERSION,
                "qc_selection_version": QC_SELECTION_VERSION,
                "qc_composite_version": QC_COMPOSITE_VERSION,
                "subject": subject,
                "phase": selection.phase,
                "phase_segment": selection.phase_segment,
                "frame_idx": selection.frame_idx,
                "coverage_status": facts["coverage_status"],
                "reasons": ";".join(selection.reasons),
                "eyes": ";".join(selection.eyes),
                "source_frame_available": facts["source_frame_available"],
                "left_overlay_available": facts["left_overlay_available"],
                "right_overlay_available": facts["right_overlay_available"],
                "image_path": path.relative_to(subject_dir).as_posix(),
                "image_sha256": hashlib.sha256(payload).hexdigest(),
                "image_size_bytes": len(payload),
            }
        )
    return rows


def produce_qc_artifacts(
    *,
    subject: str,
    subject_dir: Path,
    source_video: Path,
    config: Mapping[str, Any],
    eye_metrics_path: Path,
    frame_coverage_path: Path,
    device: str = "0",
) -> QCArtifacts:
    """Create bounded composite QC images plus ``qc_index.csv``.

    This is a second sparse RITnet pass over selected QC eyes only. YOLO is never
    rerun: boxes come exclusively from final ``eye_metrics`` provenance. QC eyes
    are packed across frames into fixed-b16 calls, but request only the hard
    ``labels`` ONNX output because probability/uncertainty maps are not needed to
    render QC overlays. Every ROI geometry is reproduced and checked against
    saved provenance before an overlay is accepted.
    """
    final_cfg = config.get("fullclass")
    if not isinstance(final_cfg, Mapping):
        raise ValueError("config.fullclass must be a mapping")
    image_limit = int(final_cfg.get("qc_image_max_count", 200))
    anomaly_limit = int(final_cfg.get("qc_anomaly_max_per_reason", 20))
    budget = int(final_cfg.get("qc_artifact_budget_bytes", 268435456))
    if budget <= 0:
        raise ValueError("qc_artifact_budget_bytes must be positive")
    roi_cfg = _roi_config(config)
    if roi_cfg["padding_mode"] != PADDING_MODE_REPLICATE:
        raise ValueError("final QC reproduction requires replicate artificial padding")

    coverage_rows = list(iter_csv_gz(frame_coverage_path))
    eye_rows = list(iter_csv_gz(eye_metrics_path))
    selections = build_qc_selections(
        frame_coverage_rows=coverage_rows,
        eye_metric_rows=eye_rows,
        anomaly_limit_per_reason_per_phase=anomaly_limit,
        max_image_count=image_limit,
    )
    coverage_by_key = {_frame_key(row): row for row in coverage_rows}
    eyes_by_key: dict[tuple[str, int, int], dict[str, Mapping[str, Any]]] = {}
    for row in eye_rows:
        key = _frame_key(row)
        eye = str(row.get("eye") or "")
        if eye in eyes_by_key.setdefault(key, {}):
            raise ValueError(f"duplicate eye metric QC key: {key + (eye,)}")
        eyes_by_key[key][eye] = row

    subject_dir = Path(subject_dir)
    qc_dir = subject_dir / "qc"
    images_dir = qc_dir / "images"
    index_path = qc_dir / "qc_index.csv"
    if index_path.exists() or (images_dir.exists() and any(images_dir.iterdir())):
        raise RuntimeError(
            "QC output already exists; completion orchestration must strict-skip a valid run "
            "or explicitly handle an incomplete run before producing new QC artifacts"
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    video = Path(source_video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video for QC: {video}")
    source_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    source_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if source_width <= 0 or source_height <= 0:
        cap.release()
        raise RuntimeError("source video dimensions are invalid for QC")

    model = _resolve_package_path(config["models"]["ritnet_fullclass_final"]).resolve()
    runtime: RitnetFullClassFinalRuntime | None = None
    encoded_images: list[tuple[QCSelection, Path, bytes, dict[str, Any]]] = []
    image_bytes = 0
    skipped_budget = 0
    current_frame: int | None = None
    try:
        ordered = sorted(
            selections,
            key=lambda item: (
                "fixed_anchor" not in item.reasons,
                item.frame_idx,
                item.phase,
                item.phase_segment,
            ),
        )
        for group in _selection_groups(ordered, eyes_by_key):
            works: list[dict[str, Any]] = []
            batch_rois: list[np.ndarray] = []
            batch_targets: list[tuple[int, str, np.ndarray]] = []

            for selection in group:
                key = selection.key
                coverage = coverage_by_key[key]
                metrics = eyes_by_key.get(key, {})
                frame, current_frame = _read_qc_frame(cap, selection.frame_idx, current_frame)
                work_index = len(works)
                works.append(
                    {
                        "selection": selection,
                        "coverage": coverage,
                        "metrics": metrics,
                        "frame": frame,
                        "overlays": {},
                    }
                )
                if frame is None:
                    continue
                for eye, roi in _prepare_eye_rois(
                    frame=frame,
                    eye_rows=metrics,
                    config=config,
                ):
                    batch_targets.append((work_index, eye, roi))
                    batch_rois.append(roi)

            if batch_rois:
                if len(batch_rois) > FIXED_BATCH_SIZE:
                    raise AssertionError(
                        f"QC selection group exceeded fixed RITnet batch: {len(batch_rois)}"
                    )
                if runtime is None:
                    if not model.is_file():
                        raise FileNotFoundError(model)
                    runtime = RitnetFullClassFinalRuntime(model, device=device)
                labels, _timing = runtime.infer_labels_batch(batch_rois)
                for output_index, (work_index, eye, roi) in enumerate(batch_targets):
                    _labels_color, overlay = render_qc_images(
                        roi,
                        labels[output_index],
                    )
                    works[work_index]["overlays"][eye] = overlay

            for work in works:
                selection = work["selection"]
                coverage = work["coverage"]
                metrics = work["metrics"]
                frame = work["frame"]
                overlays = work["overlays"]
                composite = render_qc_composite(
                    frame_bgr=frame,
                    selection=selection,
                    coverage_row=coverage,
                    eye_metric_rows=metrics,
                    eye_overlays=overlays,
                    fallback_frame_size=(source_width, source_height),
                )
                success, encoded = cv2.imencode(
                    ".png",
                    composite,
                    [cv2.IMWRITE_PNG_COMPRESSION, 6],
                )
                if not success:
                    raise RuntimeError(f"failed to encode QC composite: {selection.key}")
                payload = encoded.tobytes()
                mandatory = "fixed_anchor" in selection.reasons
                if image_bytes + len(payload) > budget:
                    if mandatory:
                        raise RuntimeError(
                            "mandatory fixed QC images exceed qc_artifact_budget_bytes; "
                            f"budget={budget}, attempted={image_bytes + len(payload)}"
                        )
                    skipped_budget += 1
                    continue

                path = qc_frame_image_path(images_dir, subject, selection)
                encoded_images.append(
                    (
                        selection,
                        path,
                        payload,
                        {
                            "coverage_status": str(coverage.get("coverage_status") or ""),
                            "source_frame_available": frame is not None,
                            "left_overlay_available": "frame_left" in overlays,
                            "right_overlay_available": "frame_right" in overlays,
                        },
                    )
                )
                image_bytes += len(payload)
    finally:
        cap.release()

    while True:
        index_rows = _index_rows(
            subject=subject,
            subject_dir=subject_dir,
            encoded_images=encoded_images,
        )
        index_payload = _encode_index(index_rows)
        total = image_bytes + len(index_payload)
        if total <= budget:
            break
        removable = next(
            (
                index
                for index in range(len(encoded_images) - 1, -1, -1)
                if "fixed_anchor" not in encoded_images[index][0].reasons
            ),
            None,
        )
        if removable is None:
            raise RuntimeError(
                "mandatory fixed QC images plus qc_index exceed qc_artifact_budget_bytes; "
                f"budget={budget}, attempted={total}"
            )
        _selection, _path, payload, _facts = encoded_images.pop(removable)
        image_bytes -= len(payload)
        skipped_budget += 1

    for _selection, path, payload, _facts in encoded_images:
        _atomic_write_bytes(path, payload)
    _atomic_write_bytes(index_path, index_payload)

    return QCArtifacts(
        qc_dir=qc_dir,
        images_dir=images_dir,
        index_path=index_path,
        selected_count=len(selections),
        saved_image_count=len(index_rows),
        skipped_for_budget_count=skipped_budget,
        image_bytes=image_bytes,
        index_bytes=len(index_payload),
        total_qc_bytes=image_bytes + len(index_payload),
    )