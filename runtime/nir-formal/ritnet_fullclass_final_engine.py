"""Numeric core of the final <=1 GiB RITnet full-class workflow.

This module is intentionally not the user-facing entry point yet. It produces
fixed-schema numeric artifacts using historical YOLO boxes + original AVI, exact
1.6 padded ROIs, the final five-output RITnet adapter, compact online uncertainty
summaries and gap-safe temporal facts. QC/integrity orchestration is layered on
before the canonical entry is switched to this engine.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import cv2

from ritnet_fullclass_contract import CLASS_MAPPING
from ritnet_fullclass_coverage import build_fixed_qc_anchor_keys, build_frame_coverage
from ritnet_fullclass_final_runtime import RitnetFullClassFinalRuntime
from ritnet_fullclass_io import atomic_write_csv_gz
from ritnet_fullclass_metric_adapter import summarize_final_hard_metrics
from ritnet_fullclass_roi import (
    PADDING_MODE_REPLICATE,
    ROI_ALGORITHM_VERSION,
    TARGET_ASPECT_RATIO,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    VALID_SOURCE_MASK_VERSION,
    crop_fixed_aspect_gray,
    fixed_aspect_roi_geometry,
    valid_source_analysis_mask,
)
from ritnet_fullclass_schema import (
    EYE_METRIC_FIELDS,
    EYE_METRICS_SCHEMA_VERSION,
    FRAME_COVERAGE_FIELDS,
    project_row,
)
from ritnet_fullclass_source import SourceFormalContext, load_source_context
from ritnet_fullclass_temporal import TEMPORAL_QC_VERSION, iter_temporal_facts
from ritnet_fullclass_uncertainty import (
    UNCERTAINTY_ALGORITHM_VERSION,
    UNCERTAINTY_DOMAIN_VERSION,
    summarize_uncertainty,
)
from ritnet_fullclass_workstore import FullClassWorkStore
from ritnet_label_store import sha256_file


PACKAGE_ROOT = Path(__file__).resolve().parent
CORE_VERSION = "fullclass-final-core-v2-valid-source-hard-domain"
VIDEO_SEEK_GAP_THRESHOLD = 64


@dataclass(frozen=True)
class CoreArtifacts:
    subject: str
    subject_dir: Path
    eye_metrics: Path
    frame_coverage: Path
    workstore: Path
    eye_row_count: int
    frame_row_count: int
    fixed_anchor_keys: frozenset[tuple[str, int, int]]
    source_context: SourceFormalContext
    work_identity: dict[str, Any]


def resolve_package_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PACKAGE_ROOT / path


def git_identity() -> tuple[str, str]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PACKAGE_ROOT, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PACKAGE_ROOT, text=True
    ).strip()
    if len(commit) != 40:
        raise RuntimeError(f"unexpected git commit: {commit!r}")
    return commit, branch


def _float_or_none(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _int(value: Any) -> int:
    return int(float(value))


def _source_base_row(subject: str, source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "eye_metrics_schema_version": EYE_METRICS_SCHEMA_VERSION,
        "subject": subject,
        "frame_idx": _int(source["frame_idx"]),
        "eye": str(source["eye"]),
        "phase": str(source["phase"]),
        "phase_segment": _int(source["phase_segment"]),
        "video_time_ms": _float_or_none(source.get("video_time_ms")),
        "unix_ms": _float_or_none(source.get("unix_ms")),
        "phase_time_ms": _float_or_none(source.get("phase_time_ms")),
        "source_frame_status": str(source.get("frame_status") or ""),
        "source_eye_status": str(source.get("status") or ""),
        "source_redetect_reason": str(source.get("redetect_reason") or ""),
        "source_yolo_batch_size": _int(source.get("yolo_batch_size") or 0),
        "yolo_confidence": float(source["anchor_yolo_confidence"]),
        "yolo_bbox_x1": float(source["bbox_x1"]),
        "yolo_bbox_y1": float(source["bbox_y1"]),
        "yolo_bbox_x2": float(source["bbox_x2"]),
        "yolo_bbox_y2": float(source["bbox_y2"]),
        "ritnet_status": None,
        "ritnet_failure_reason": None,
    }


def _final_roi_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return and validate the independent final full-class ROI contract.

    The historical top-level ``roi`` block belongs to the already-completed
    320x160 formal producer. Final full-class must not inherit its dimensions or
    expansion values implicitly, even when the current numeric values happen to
    match.
    """
    fullclass = config.get("fullclass")
    if not isinstance(fullclass, Mapping):
        raise ValueError("config.fullclass must be a mapping")
    roi = fullclass.get("roi")
    if not isinstance(roi, Mapping):
        raise ValueError("config.fullclass.roi must be a mapping")

    required = {
        "target_width",
        "target_height",
        "aspect_ratio",
        "expand_horizontal_each_side",
        "expand_vertical_each_side",
        "padding_mode",
    }
    missing = sorted(required - set(roi))
    if missing:
        raise ValueError(f"config.fullclass.roi missing required keys: {missing}")

    target_width = int(roi["target_width"])
    target_height = int(roi["target_height"])
    aspect_ratio = float(roi["aspect_ratio"])
    horizontal = float(roi["expand_horizontal_each_side"])
    vertical = float(roi["expand_vertical_each_side"])
    padding_mode = str(roi["padding_mode"])

    if (target_width, target_height) != (TARGET_WIDTH, TARGET_HEIGHT):
        raise ValueError(
            "final RITnet ROI target must remain 640x400; got "
            f"{target_width}x{target_height}"
        )
    if abs(aspect_ratio - TARGET_ASPECT_RATIO) > 1e-12:
        raise ValueError(
            f"final RITnet ROI aspect must remain {TARGET_ASPECT_RATIO}; got {aspect_ratio}"
        )
    if horizontal < 0 or vertical < 0:
        raise ValueError("final ROI expansion fractions must be non-negative")
    if padding_mode != PADDING_MODE_REPLICATE:
        raise ValueError(
            f"final ROI padding mode must remain {PADDING_MODE_REPLICATE!r}; got {padding_mode!r}"
        )

    return {
        "target_width": target_width,
        "target_height": target_height,
        "aspect_ratio": aspect_ratio,
        "expand_horizontal_each_side": horizontal,
        "expand_vertical_each_side": vertical,
        "padding_mode": padding_mode,
    }


def _group_remaining_rows(
    rows: tuple[dict[str, str], ...],
    start_ordinal: int,
) -> Iterator[tuple[int, list[tuple[int, dict[str, str]]]]]:
    current_frame: int | None = None
    group: list[tuple[int, dict[str, str]]] = []
    for ordinal in range(start_ordinal, len(rows)):
        row = rows[ordinal]
        frame = _int(row["frame_idx"])
        if current_frame is None:
            current_frame = frame
        if frame != current_frame:
            yield current_frame, group
            current_frame = frame
            group = []
        group.append((ordinal, row))
    if current_frame is not None and group:
        yield current_frame, group


def _prepared_items(
    *,
    context: SourceFormalContext,
    start_ordinal: int,
) -> Iterator[dict[str, Any]]:
    rows = context.eye_rows
    if start_ordinal >= len(rows):
        return
    roi_cfg = _final_roi_config(context.config)
    cap = cv2.VideoCapture(str(context.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {context.video}")
    frame_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    frame_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if frame_width <= 0 or frame_height <= 0:
        cap.release()
        raise RuntimeError("source video dimensions are invalid")

    current_frame: int | None = None
    try:
        for target_frame, group in _group_remaining_rows(rows, start_ordinal):
            if current_frame is None or target_frame < current_frame or target_frame - current_frame > VIDEO_SEEK_GAP_THRESHOLD:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                current_frame = target_frame
            frame = None
            while current_frame is not None and current_frame <= target_frame:
                ok, decoded = cap.read()
                if not ok or decoded is None:
                    raise RuntimeError(f"source video decode failed at frame {current_frame}")
                if current_frame == target_frame:
                    frame = decoded
                current_frame += 1
            if frame is None:
                raise RuntimeError(f"failed to obtain source frame {target_frame}")
            if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                raise RuntimeError(
                    f"source frame size changed at {target_frame}: {frame.shape[1]}x{frame.shape[0]} "
                    f"!= {frame_width}x{frame_height}"
                )

            for ordinal, source in group:
                base = _source_base_row(context.subject, source)
                try:
                    geometry = fixed_aspect_roi_geometry(
                        bbox=(
                            float(source["bbox_x1"]),
                            float(source["bbox_y1"]),
                            float(source["bbox_x2"]),
                            float(source["bbox_y2"]),
                        ),
                        frame_width=frame_width,
                        frame_height=frame_height,
                        expand_horizontal_each_side=float(roi_cfg["expand_horizontal_each_side"]),
                        expand_vertical_each_side=float(roi_cfg["expand_vertical_each_side"]),
                        padding_mode=str(roi_cfg["padding_mode"]),
                    )
                    roi = crop_fixed_aspect_gray(frame, geometry)
                    valid_source_mask = valid_source_analysis_mask(geometry)
                    base.update(geometry.as_dict())
                    yield {
                        "ordinal": ordinal,
                        "source": source,
                        "base": base,
                        "roi": roi,
                        "valid_source_mask": valid_source_mask,
                    }
                except ValueError as exc:
                    base["ritnet_status"] = "failed"
                    base["ritnet_failure_reason"] = f"roi_invalid:{type(exc).__name__}:{exc}"
                    yield {
                        "ordinal": ordinal,
                        "source": source,
                        "base": base,
                        "roi": None,
                        "valid_source_mask": None,
                    }
    finally:
        cap.release()


def _complete_batch(
    *,
    items: list[dict[str, Any]],
    runtime: RitnetFullClassFinalRuntime,
    boundary_band_px: int,
    low_max_probability_threshold: float | None,
) -> list[tuple[int, dict[str, Any]]]:
    successful_indices = [index for index, item in enumerate(items) if item["roi"] is not None]
    inferred: dict[int, dict[str, Any]] = {}
    if successful_indices:
        rois = [items[index]["roi"] for index in successful_indices]
        outputs, _timing = runtime.infer_batch(rois)
        for output_index, item_index in enumerate(successful_indices):
            labels = outputs["labels"][output_index]
            valid_source_mask = items[item_index].get("valid_source_mask")
            if valid_source_mask is None:
                raise RuntimeError(
                    "successful final RITnet item is missing valid_source_mask; refusing to compute "
                    "scientific hard metrics over synthetic padding"
                )
            hard = summarize_final_hard_metrics(
                labels,
                valid_source_mask=valid_source_mask,
            )
            uncertainty = summarize_uncertainty(
                labels=labels,
                valid_source_mask=valid_source_mask,
                soft_class_fraction=outputs["soft_class_fraction"][output_index],
                max_probability=outputs["max_probability"][output_index],
                top1_top2_margin=outputs["top1_top2_margin"][output_index],
                entropy=outputs["entropy"][output_index],
                boundary_band_px=boundary_band_px,
                low_max_probability_threshold=low_max_probability_threshold,
            )
            inferred[item_index] = {**hard, **uncertainty}

    completed: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(items):
        row = dict(item["base"])
        if index in inferred:
            row["ritnet_status"] = "success"
            row["ritnet_failure_reason"] = None
            row.update(inferred[index])
        completed.append((int(item["ordinal"]), row))
    return completed


def _work_identity(
    *,
    context: SourceFormalContext,
    config_path: Path,
    ritnet_model: Path,
    ritnet_external_data: Path,
) -> dict[str, Any]:
    git_commit, git_branch = git_identity()
    full_cfg = context.config.get("fullclass", {})
    roi_cfg = _final_roi_config(context.config)
    return {
        "core_version": CORE_VERSION,
        "subject": context.subject,
        "source_identity": context.source_identity,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "config_sha256": sha256_file(config_path),
        "ritnet_model_sha256": sha256_file(ritnet_model),
        "ritnet_external_data_sha256": sha256_file(ritnet_external_data),
        "ritnet_input": [TARGET_WIDTH, TARGET_HEIGHT],
        "ritnet_batch_size": 16,
        "ritnet_precision": "fp32",
        "class_mapping": {str(key): value for key, value in CLASS_MAPPING.items()},
        "roi_algorithm_version": ROI_ALGORITHM_VERSION,
        "valid_source_mask_version": VALID_SOURCE_MASK_VERSION,
        "roi_contract": dict(roi_cfg),
        "uncertainty_algorithm_version": UNCERTAINTY_ALGORITHM_VERSION,
        "uncertainty_domain_version": UNCERTAINTY_DOMAIN_VERSION,
        "temporal_qc_version": TEMPORAL_QC_VERSION,
        "eye_metrics_schema_version": EYE_METRICS_SCHEMA_VERSION,
        "qc_boundary_band_px": int(full_cfg.get("qc_boundary_band_px", 5)),
        "low_max_probability_threshold": full_cfg.get("low_max_probability_threshold"),
    }


def run_numeric_core(
    *,
    run_dir: Path,
    config_path: Path,
    device: str = "0",
) -> CoreArtifacts:
    context = load_source_context(run_dir, config_path)
    config = context.config
    final_cfg = config.get("fullclass", {})
    _final_roi_config(config)
    output_dirname = str(final_cfg.get("output_dirname") or "ritnet-fullclass-final")
    subject_dir = context.run_dir.parent / output_dirname / context.subject
    data_dir = subject_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    eye_metrics_path = data_dir / "eye_metrics.csv.gz"
    frame_coverage_path = data_dir / "frame_coverage.csv.gz"

    ritnet_model = resolve_package_path(config["models"]["ritnet_fullclass_final"]).resolve()
    ritnet_external = resolve_package_path(config["models"]["ritnet_fullclass_final_external_data"]).resolve()
    if not ritnet_model.is_file() or not ritnet_external.is_file():
        raise FileNotFoundError(
            "final RITnet uncertainty model/export data are missing; export and qualify them before running: "
            f"{ritnet_model} / {ritnet_external}"
        )

    identity = _work_identity(
        context=context,
        config_path=Path(config_path).resolve(),
        ritnet_model=ritnet_model,
        ritnet_external_data=ritnet_external,
    )
    work_root = context.run_dir.parent / ".ritnet-fullclass-work"
    workstore_path = work_root / f"{context.subject}.sqlite"
    runtime = RitnetFullClassFinalRuntime(ritnet_model, device=device)
    boundary_band_px = int(final_cfg.get("qc_boundary_band_px", 5))
    threshold_value = final_cfg.get("low_max_probability_threshold")
    threshold = None if threshold_value is None else float(threshold_value)

    with FullClassWorkStore(workstore_path, identity=identity) as store:
        start_ordinal = store.validate_prefix(context.eye_rows)
        pending: list[dict[str, Any]] = []
        for item in _prepared_items(context=context, start_ordinal=start_ordinal):
            pending.append(item)
            if len(pending) < runtime.FIXED_BATCH_SIZE:
                continue
            store.append_rows(
                _complete_batch(
                    items=pending,
                    runtime=runtime,
                    boundary_band_px=boundary_band_px,
                    low_max_probability_threshold=threshold,
                )
            )
            pending = []
        if pending:
            store.append_rows(
                _complete_batch(
                    items=pending,
                    runtime=runtime,
                    boundary_band_px=boundary_band_px,
                    low_max_probability_threshold=threshold,
                )
            )
        stored_rows = store.validate_prefix(context.eye_rows)
        if stored_rows != len(context.eye_rows):
            raise RuntimeError(f"numeric workstore incomplete: {stored_rows}/{len(context.eye_rows)}")

        eye_count = atomic_write_csv_gz(
            eye_metrics_path,
            (project_row(row, EYE_METRIC_FIELDS) for row in iter_temporal_facts(store.iter_rows())),
            EYE_METRIC_FIELDS,
        )
        if eye_count != len(context.eye_rows):
            raise RuntimeError(f"eye_metrics row count mismatch: {eye_count}/{len(context.eye_rows)}")

        fixed_anchors = build_fixed_qc_anchor_keys(
            context.frame_rows,
            interval_sec=float(final_cfg.get("qc_interval_sec", 30)),
        )
        coverage_rows = build_frame_coverage(
            subject=context.subject,
            source_frames=context.frame_rows,
            source_eye_rows=context.eye_rows,
            final_eye_rows=store.iter_rows(),
            fixed_anchor_keys=fixed_anchors,
        )
        frame_count = atomic_write_csv_gz(
            frame_coverage_path,
            coverage_rows,
            FRAME_COVERAGE_FIELDS,
        )
        if frame_count != len(context.frame_rows):
            raise RuntimeError(f"frame coverage row count mismatch: {frame_count}/{len(context.frame_rows)}")

    return CoreArtifacts(
        subject=context.subject,
        subject_dir=subject_dir,
        eye_metrics=eye_metrics_path,
        frame_coverage=frame_coverage_path,
        workstore=workstore_path,
        eye_row_count=eye_count,
        frame_row_count=frame_count,
        fixed_anchor_keys=frozenset(fixed_anchors),
        source_context=context,
        work_identity=identity,
    )
