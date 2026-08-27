"""Numeric core of the final <=1 GiB RITnet full-class workflow.

This module produces fixed-schema numeric artifacts using historical YOLO boxes +
original AVI, exact 1.6 padded ROIs, the final RITnet adapter, four-class
segmentation, pupil-only geometry, compact online uncertainty summaries and
gap-safe temporal facts.

The production loop is pipelined: source decode/ROI/preprocessing of batch N+1
overlaps DirectML inference of batch N, while CPU metric reduction of batch N-1
overlaps both.
"""
from __future__ import annotations

import subprocess
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import cv2
import numpy as np

from ritnet_fullclass_contract import CLASS_MAPPING
from ritnet_fullclass_coverage import build_fixed_qc_anchor_keys, build_frame_coverage
from ritnet_fullclass_final_runtime import RitnetFullClassFinalRuntime
from ritnet_fullclass_io import atomic_write_csv
from ritnet_fullclass_metric_adapter import ANALYSIS_DOMAIN_VERSION, summarize_final_hard_metrics
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
    FRAME_COVERAGE_SCHEMA_VERSION,
    project_row,
)
from ritnet_fullclass_source import SourceFormalContext, load_source_context
from ritnet_fullclass_temporal import TEMPORAL_QC_VERSION, iter_temporal_facts
from ritnet_fullclass_uncertainty import (
    COHORT_UNCERTAINTY_ALGORITHM_VERSION,
    COHORT_UNCERTAINTY_DOMAIN_VERSION,
    SOFT_CLASS_FRACTION_DOMAIN_VERSION,
    summarize_uncertainty,
)
from ritnet_fullclass_workstore import FullClassWorkStore
from ritnet_label_store import sha256_file


PACKAGE_ROOT = Path(__file__).resolve().parent
CORE_VERSION = "fullclass-final-core-v8-interface-safe-plain-csv"
VIDEO_SEEK_GAP_THRESHOLD = 64
DEFAULT_CHECKPOINT_ROWS = 128
DEFAULT_PROGRESS_EVERY_BATCHES = 100
DEFAULT_SUMMARY_WORKERS = 2
DEFAULT_MAX_PENDING_SUMMARIES = 2

FULL_SOURCE_VALID_MASK = np.ones((TARGET_HEIGHT, TARGET_WIDTH), dtype=bool)
FULL_SOURCE_VALID_MASK.setflags(write=False)


@dataclass(frozen=True)
class CoreArtifacts:
    subject: str
    subject_dir: Path
    eye_metrics: Path
    frame_coverage: Path
    workstore: Path
    eye_row_count: int
    frame_row_count: int
    eye_metric_rows: tuple[dict[str, Any], ...]
    frame_coverage_rows: tuple[dict[str, Any], ...]
    fixed_anchor_keys: frozenset[tuple[str, int, int]]
    source_context: SourceFormalContext
    work_identity: dict[str, Any]


@dataclass(frozen=True)
class PreparedBatch:
    items: list[dict[str, Any]]
    successful_indices: tuple[int, ...]
    tensor: Any | None
    valid_batch_size: int
    timing: dict[str, float]


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
        "source_detection_source": str(source.get("source") or ""),
        "source_frame_status": str(source.get("frame_status") or ""),
        "source_eye_status": str(source.get("status") or ""),
        "source_redetect_reason": str(source.get("redetect_reason") or ""),
        "source_yolo_batch_size": _int(source["yolo_batch_size"])
        if source.get("yolo_batch_size") not in (None, "")
        else None,
        "yolo_confidence": _float_or_none(source.get("anchor_yolo_confidence")),
        "yolo_bbox_x1": _float_or_none(source.get("bbox_x1")),
        "yolo_bbox_y1": _float_or_none(source.get("bbox_y1")),
        "yolo_bbox_x2": _float_or_none(source.get("bbox_x2")),
        "yolo_bbox_y2": _float_or_none(source.get("bbox_y2")),
    }


def _final_roi_config(config: Mapping[str, Any]) -> dict[str, Any]:
    full = config.get("fullclass")
    if not isinstance(full, Mapping):
        raise ValueError("config.fullclass must be a mapping")
    roi = full.get("roi")
    if not isinstance(roi, Mapping):
        raise ValueError("config.fullclass.roi must be a mapping")
    expected = {
        "target_width": TARGET_WIDTH,
        "target_height": TARGET_HEIGHT,
        "aspect_ratio": TARGET_ASPECT_RATIO,
        "padding_mode": PADDING_MODE_REPLICATE,
    }
    for key, value in expected.items():
        actual = roi.get(key)
        if isinstance(value, float):
            if not np.isclose(float(actual), value, rtol=0.0, atol=1e-12):
                raise ValueError(f"fullclass.roi.{key} must be {value}, got {actual}")
        elif actual != value:
            raise ValueError(f"fullclass.roi.{key} must be {value!r}, got {actual!r}")
    return {
        "target_width": TARGET_WIDTH,
        "target_height": TARGET_HEIGHT,
        "aspect_ratio": TARGET_ASPECT_RATIO,
        "expand_horizontal_each_side": float(roi["expand_horizontal_each_side"]),
        "expand_vertical_each_side": float(roi["expand_vertical_each_side"]),
        "padding_mode": PADDING_MODE_REPLICATE,
    }


def _failed_prepared_item(
    *,
    context: SourceFormalContext,
    ordinal: int,
    source: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    base = _source_base_row(context.subject, source)
    base["ritnet_status"] = "failed"
    base["ritnet_failure_reason"] = reason
    return {
        "ordinal": ordinal,
        "source": source,
        "base": base,
        "roi": None,
        "valid_source_mask": None,
    }


def _group_remaining_rows(
    rows: Sequence[Mapping[str, Any]],
    start_ordinal: int,
) -> Iterator[tuple[int, list[tuple[int, Mapping[str, Any]]]]]:
    current_frame: int | None = None
    group: list[tuple[int, Mapping[str, Any]]] = []
    for ordinal in range(start_ordinal, len(rows)):
        source = rows[ordinal]
        frame_idx = _int(source["frame_idx"])
        if current_frame is None:
            current_frame = frame_idx
        if frame_idx != current_frame:
            yield current_frame, group
            current_frame = frame_idx
            group = []
        group.append((ordinal, source))
    if group:
        assert current_frame is not None
        yield current_frame, group


def _iter_prepared_items(
    context: SourceFormalContext,
    start_ordinal: int,
) -> Iterator[dict[str, Any]]:
    rows = context.eye_rows
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
            if (
                current_frame is None
                or target_frame < current_frame
                or target_frame - current_frame > VIDEO_SEEK_GAP_THRESHOLD
            ):
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                current_frame = target_frame
            frame = None
            decode_failed_at: int | None = None
            while current_frame is not None and current_frame <= target_frame:
                ok, decoded = cap.read()
                if not ok or decoded is None:
                    decode_failed_at = int(current_frame)
                    break
                if current_frame == target_frame:
                    frame = decoded
                current_frame += 1

            if decode_failed_at is not None or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                retry_ok, retry_frame = cap.read()
                if retry_ok and retry_frame is not None:
                    frame = retry_frame
                    current_frame = target_frame + 1
                else:
                    first_failed = target_frame if decode_failed_at is None else decode_failed_at
                    reason = (
                        "source_video_decode_failed:"
                        f"target_frame={target_frame}:first_failed_frame={first_failed}"
                    )
                    for ordinal, source in group:
                        yield _failed_prepared_item(
                            context=context,
                            ordinal=ordinal,
                            source=source,
                            reason=reason,
                        )
                    current_frame = None
                    continue

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
                    has_padding = any(
                        (geometry.pad_left, geometry.pad_top, geometry.pad_right, geometry.pad_bottom)
                    )
                    valid_source_mask = (
                        valid_source_analysis_mask(geometry)
                        if has_padding
                        else FULL_SOURCE_VALID_MASK
                    )
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


def _next_prepared_batch(
    *,
    item_iterator: Iterator[dict[str, Any]],
    runtime: RitnetFullClassFinalRuntime,
) -> PreparedBatch | None:
    stage_started = time.perf_counter()
    items: list[dict[str, Any]] = []
    for _ in range(runtime.FIXED_BATCH_SIZE):
        try:
            items.append(next(item_iterator))
        except StopIteration:
            break
    if not items:
        return None

    source_prepare_ms = (time.perf_counter() - stage_started) * 1000.0
    successful_indices = tuple(
        index for index, item in enumerate(items) if item["roi"] is not None
    )
    tensor = None
    valid_batch_size = 0
    preprocess_ms = 0.0
    if successful_indices:
        rois = [items[index]["roi"] for index in successful_indices]
        tensor, valid_batch_size, prep_timing = runtime.prepare_batch(rois)
        preprocess_ms = float(prep_timing.get("preprocess_ms", 0.0))

    return PreparedBatch(
        items=items,
        successful_indices=successful_indices,
        tensor=tensor,
        valid_batch_size=int(valid_batch_size),
        timing={
            "source_prepare_ms": float(source_prepare_ms),
            "preprocess_ms": float(preprocess_ms),
            "producer_total_ms": float((time.perf_counter() - stage_started) * 1000.0),
        },
    )


def _summarize_outputs(
    *,
    prepared: PreparedBatch,
    outputs: dict[str, Any] | None,
) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, float]]:
    summary_started = time.perf_counter()
    hard_ms = 0.0
    uncertainty_ms = 0.0
    inferred: dict[int, dict[str, Any]] = {}

    if prepared.successful_indices:
        if outputs is None:
            raise RuntimeError("successful prepared batch is missing RITnet outputs")
        for output_index, item_index in enumerate(prepared.successful_indices):
            labels = outputs["labels"][output_index]
            valid_source_mask = prepared.items[item_index].get("valid_source_mask")
            if valid_source_mask is None:
                raise RuntimeError(
                    "successful final RITnet item is missing valid_source_mask; refusing to compute "
                    "scientific hard metrics over synthetic padding"
                )

            started = time.perf_counter()
            hard = summarize_final_hard_metrics(
                labels,
                valid_source_mask=valid_source_mask,
            )
            hard_ms += (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            uncertainty = summarize_uncertainty(
                labels=labels,
                valid_source_mask=valid_source_mask,
                class_probability=outputs["class_probability"][output_index],
                max_probability=outputs["max_probability"][output_index],
                top1_top2_margin=outputs["top1_top2_margin"][output_index],
                entropy=outputs["entropy"][output_index],
                inputs_validated=True,
            )
            uncertainty_ms += (time.perf_counter() - started) * 1000.0
            inferred[item_index] = {**hard, **uncertainty}

    completed: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(prepared.items):
        row = dict(item["base"])
        if index in inferred:
            row["ritnet_status"] = "success"
            row["ritnet_failure_reason"] = None
            row.update(inferred[index])
        completed.append((int(item["ordinal"]), row))

    return completed, {
        "hard_metric_ms": float(hard_ms),
        "uncertainty_ms": float(uncertainty_ms),
        "summary_total_ms": float((time.perf_counter() - summary_started) * 1000.0),
    }


def _work_identity(
    *,
    context: SourceFormalContext,
    config_path: Path,
    ritnet_model: Path,
    ritnet_external_data: Path,
) -> dict[str, Any]:
    git_commit, git_branch = git_identity()
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
        "analysis_domain_version": ANALYSIS_DOMAIN_VERSION,
        "uncertainty_algorithm_version": COHORT_UNCERTAINTY_ALGORITHM_VERSION,
        "uncertainty_domain_version": COHORT_UNCERTAINTY_DOMAIN_VERSION,
        "soft_class_fraction_domain_version": SOFT_CLASS_FRACTION_DOMAIN_VERSION,
        "temporal_qc_version": TEMPORAL_QC_VERSION,
        "eye_metrics_schema_version": EYE_METRICS_SCHEMA_VERSION,
        "frame_coverage_schema_version": FRAME_COVERAGE_SCHEMA_VERSION,
    }


def _accumulate_timing(total: dict[str, float], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0.0) + float(value)


def _flush_checkpoint(
    store: FullClassWorkStore,
    buffer: list[tuple[int, dict[str, Any]]],
) -> float:
    if not buffer:
        return 0.0
    started = time.perf_counter()
    store.append_rows(buffer)
    buffer.clear()
    return (time.perf_counter() - started) * 1000.0


def _report_progress(
    *,
    subject: str,
    start_ordinal: int,
    processed_session: int,
    total_rows: int,
    batch_count: int,
    wall_started: float,
    timing_total: Mapping[str, float],
) -> None:
    elapsed = max(1e-9, time.perf_counter() - wall_started)
    completed_total = start_ordinal + processed_session
    rate = processed_session / elapsed if processed_session else 0.0
    remaining = max(0, total_rows - completed_total)
    eta_sec = remaining / rate if rate > 0 else float("inf")
    eta_text = f"{eta_sec / 60.0:.1f}m" if eta_sec != float("inf") else "?"
    dml_ms = timing_total.get("gpu_and_transfer_ms", 0.0)
    summary_ms = timing_total.get("summary_total_ms", 0.0)
    producer_ms = timing_total.get("producer_total_ms", 0.0)
    print(
        f"[FULLCLASS] {subject} batch={batch_count} "
        f"eyes={completed_total}/{total_rows} "
        f"rate={rate:.2f} eyes/s ETA={eta_text} "
        f"stage_ms(prod={producer_ms:.0f},dml={dml_ms:.0f},summary={summary_ms:.0f})",
        flush=True,
    )


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
    eye_metrics_path = data_dir / "eye_metrics.csv"
    frame_coverage_path = data_dir / "frame_coverage.csv"

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

    checkpoint_rows = int(final_cfg.get("checkpoint_rows", DEFAULT_CHECKPOINT_ROWS))
    progress_every = int(final_cfg.get("progress_every_batches", DEFAULT_PROGRESS_EVERY_BATCHES))
    summary_workers = int(final_cfg.get("summary_workers", DEFAULT_SUMMARY_WORKERS))
    max_pending_summaries = int(
        final_cfg.get("max_pending_summaries", DEFAULT_MAX_PENDING_SUMMARIES)
    )
    if checkpoint_rows < RitnetFullClassFinalRuntime.FIXED_BATCH_SIZE:
        raise ValueError(
            f"fullclass.checkpoint_rows must be >= {RitnetFullClassFinalRuntime.FIXED_BATCH_SIZE}"
        )
    if progress_every <= 0:
        raise ValueError("fullclass.progress_every_batches must be positive")
    if summary_workers <= 0:
        raise ValueError("fullclass.summary_workers must be positive")
    if max_pending_summaries <= 0:
        raise ValueError("fullclass.max_pending_summaries must be positive")
    if max_pending_summaries < summary_workers:
        raise ValueError("fullclass.max_pending_summaries must be >= summary_workers")

    with FullClassWorkStore(workstore_path, identity=identity) as store:
        start_ordinal = store.validate_prefix(context.eye_rows)
        if start_ordinal == len(context.eye_rows):
            # Recovery/finalization from a complete checkpoint must not load the
            # DirectML session or allocate VRAM again.
            numeric_rows = list(iter_temporal_facts(store.iter_rows()))
        else:
            runtime = RitnetFullClassFinalRuntime(ritnet_model, device=device)
            wall_started = time.perf_counter()
            timing_total: dict[str, float] = {}
            item_iterator = _iter_prepared_items(context, start_ordinal)
            pending: deque[tuple[Future, int]] = deque()
            checkpoint_buffer: list[tuple[int, dict[str, Any]]] = []
            batch_count = 0
            processed_session = 0

            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="ritnet-producer",
            ) as producer_pool, ThreadPoolExecutor(
                max_workers=summary_workers,
                thread_name_prefix="ritnet-summary",
            ) as summary_pool:
                next_future: Future = producer_pool.submit(
                    _next_prepared_batch,
                    item_iterator=item_iterator,
                    runtime=runtime,
                )

                while True:
                    prepared = next_future.result()
                    if prepared is None:
                        break
                    next_future = producer_pool.submit(
                        _next_prepared_batch,
                        item_iterator=item_iterator,
                        runtime=runtime,
                    )
                    _accumulate_timing(timing_total, prepared.timing)

                    outputs = None
                    if prepared.successful_indices:
                        outputs, infer_timing = runtime.infer_prepared(
                            prepared.tensor,
                            prepared.valid_batch_size,
                        )
                        _accumulate_timing(timing_total, infer_timing)

                    future = summary_pool.submit(
                        _summarize_outputs,
                        prepared=prepared,
                        outputs=outputs,
                    )
                    pending.append((future, len(prepared.items)))
                    batch_count += 1

                    while len(pending) >= max_pending_summaries:
                        oldest, count = pending.popleft()
                        completed, summary_timing = oldest.result()
                        _accumulate_timing(timing_total, summary_timing)
                        checkpoint_buffer.extend(completed)
                        processed_session += count
                        if len(checkpoint_buffer) >= checkpoint_rows:
                            timing_total["sqlite_ms"] = timing_total.get("sqlite_ms", 0.0) + _flush_checkpoint(
                                store, checkpoint_buffer
                            )

                    if batch_count % progress_every == 0:
                        _report_progress(
                            subject=context.subject,
                            start_ordinal=start_ordinal,
                            processed_session=processed_session,
                            total_rows=len(context.eye_rows),
                            batch_count=batch_count,
                            wall_started=wall_started,
                            timing_total=timing_total,
                        )

                while pending:
                    future, count = pending.popleft()
                    completed, summary_timing = future.result()
                    _accumulate_timing(timing_total, summary_timing)
                    checkpoint_buffer.extend(completed)
                    processed_session += count
                    if len(checkpoint_buffer) >= checkpoint_rows:
                        timing_total["sqlite_ms"] = timing_total.get("sqlite_ms", 0.0) + _flush_checkpoint(
                            store, checkpoint_buffer
                        )
                timing_total["sqlite_ms"] = timing_total.get("sqlite_ms", 0.0) + _flush_checkpoint(
                    store, checkpoint_buffer
                )
                _report_progress(
                    subject=context.subject,
                    start_ordinal=start_ordinal,
                    processed_session=processed_session,
                    total_rows=len(context.eye_rows),
                    batch_count=batch_count,
                    wall_started=wall_started,
                    timing_total=timing_total,
                )

            numeric_rows = list(iter_temporal_facts(store.iter_rows()))

        if len(numeric_rows) != len(context.eye_rows):
            raise RuntimeError(
                f"final eye row count mismatch: {len(numeric_rows)} != {len(context.eye_rows)}"
            )

        eye_count = atomic_write_csv(
            eye_metrics_path,
            (project_row(row, EYE_METRIC_FIELDS) for row in numeric_rows),
            EYE_METRIC_FIELDS,
        )
        if eye_count != len(numeric_rows):
            raise RuntimeError(
                f"eye_metrics write count mismatch: {eye_count} != {len(numeric_rows)}"
            )

    fixed_anchor_keys = frozenset(
        build_fixed_qc_anchor_keys(
            context.frame_rows,
            interval_sec=float(final_cfg.get("qc_interval_sec", 30.0)),
        )
    )
    coverage_rows = build_frame_coverage(
        subject=context.subject,
        source_frames=context.frame_rows,
        source_eye_rows=context.eye_rows,
        final_eye_rows=numeric_rows,
        fixed_anchor_keys=set(fixed_anchor_keys),
    )
    frame_count = atomic_write_csv(
        frame_coverage_path,
        (project_row(row, FRAME_COVERAGE_FIELDS) for row in coverage_rows),
        FRAME_COVERAGE_FIELDS,
    )
    if frame_count != len(coverage_rows):
        raise RuntimeError(
            f"frame_coverage write count mismatch: {frame_count} != {len(coverage_rows)}"
        )

    return CoreArtifacts(
        subject=context.subject,
        subject_dir=subject_dir,
        eye_metrics=eye_metrics_path,
        frame_coverage=frame_coverage_path,
        workstore=workstore_path,
        eye_row_count=len(numeric_rows),
        frame_row_count=len(coverage_rows),
        eye_metric_rows=tuple(numeric_rows),
        frame_coverage_rows=tuple(coverage_rows),
        fixed_anchor_keys=fixed_anchor_keys,
        source_context=context,
        work_identity=identity,
    )
