"""Incremental, per-algorithm, high-parallel execution for the full-video benchmark.

Design (per user direction):

- Algorithms run ONE AT A TIME. Each algorithm gets its own output directory and
  is sharded across worker processes (large intra-algorithm parallelism).
- Every worker writes its results INCREMENTALLY (CSV appended + flushed per row),
  so partial results are durable and progress is visible while the run is live.
- A structured event log (``run_events.jsonl``) records every significant event:
  run start, per-algorithm start/chunk launches, periodic progress, chunk
  crashes, per-algorithm completion, and final completion -- all with wall-clock
  timestamps.
- Each worker keeps its own ``worker.log`` with per-N-frame progress lines.
- A crashing worker (e.g. a compiled-detector segfault) never loses the other
  chunks' results: every chunk writes to its own directory and the main process
  records the crash and resumes from the partial file if re-run.
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .core import normalize_phase, safe_ratio
from .runner import VideoFrameSource, detect_crop, run_crop_list
from .schema import ALGORITHM_SPECS, RESULT_COLUMNS


class EventLogger:
    """Append JSON-lines events with wall-clock timestamps."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _failure_detection(row: Mapping[str, Any], algorithm: str, error: Exception) -> dict[str, Any]:
    detection: dict[str, Any] = {
        "algorithm": algorithm,
        "algorithm_returned": False,
        "official_valid": False,
        "geometry_sane": False,
        "center_x": np.nan, "center_y": np.nan,
        "major_axis": np.nan, "minor_axis": np.nan,
        "angle_deg": np.nan, "diameter_geom": np.nan, "area": np.nan,
        "runtime_ms": np.nan,
        "native_confidence": np.nan, "outline_confidence": np.nan,
        "confidence_runtime_ms": np.nan,
        "failure": f"{type(error).__name__}: {error}",
    }
    return detection


def _chunk_worker(
    manifest_path: str,
    algorithm: str,
    out_dir: str,
    run_confidence: bool,
    mode: str,
) -> None:
    """Top-level worker: process a chunk and write results incrementally."""
    from .runner import assemble_row

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "worker.log"
    results_path = out / "results.csv"
    rows = pd.read_csv(manifest_path, low_memory=False).to_dict("records")

    def log(message: str) -> None:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")

    log(f"start algorithm={algorithm} mode={mode} rows={len(rows)}")
    started = time.perf_counter()

    if mode == "continuous":
        frame = run_crop_list(
            rows, [algorithm], crop_root=Path(""),
            run_confidence=run_confidence, mode="continuous", image_source="video",
        )
        frame.to_csv(results_path, index=False, encoding="utf-8-sig")
        elapsed = time.perf_counter() - started
        log(f"done rows={len(frame)} elapsed_s={elapsed:.1f}")
        return

    frame_source = VideoFrameSource()
    try:
        with open(results_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(RESULT_COLUMNS))
            writer.writeheader()
            for index, row in enumerate(rows):
                try:
                    image = frame_source.crop(row)
                    detection = detect_crop(image, algorithm, run_confidence=run_confidence)
                except Exception as exc:  # noqa: BLE001 - never let one frame kill the worker
                    detection = _failure_detection(row, algorithm, exc)
                writer.writerow(assemble_row(row, detection))
                if (index + 1) % 25 == 0 or index == len(rows) - 1:
                    handle.flush()
                    log(f"progress {index + 1}/{len(rows)}")
            elapsed = time.perf_counter() - started
            log(f"done rows={len(rows)} elapsed_s={elapsed:.1f}")
    finally:
        frame_source.close()


def _chunk_rows(rows: Sequence[Mapping[str, Any]], n: int) -> list[list[dict[str, Any]]]:
    if not rows:
        return []
    n = max(1, min(int(n), len(rows)))
    size = int(np.ceil(len(rows) / n))
    return [list(rows[i:i + size]) for i in range(0, len(rows), size)]


def _write_chunk_manifest(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with open(path, encoding="utf-8") as handle:
        return sum(1 for _ in handle) - 1  # minus header


def run_incremental(
    manifest: pd.DataFrame,
    algorithms: Sequence[str],
    *,
    run_dir: str | Path,
    workers: int = 8,
    run_confidence: bool = False,
    progress_interval_s: float = 15.0,
) -> dict[str, Any]:
    """Run every algorithm over the manifest incrementally and return the merged frame.

    Writes per-algorithm partial outputs under ``run_dir/algorithms/<algo>/`` and a
    structured ``run_dir/run_events.jsonl`` event log. Returns a frame with the
    RITnet agreement columns enriched (same schema as execute_manifest).
    """
    run_dir = Path(run_dir)
    events = EventLogger(run_dir / "run_events.jsonl")
    events.log("run_start", algorithms=list(algorithms), workers=workers)

    # Cap TBB/OMP per worker so N workers do not oversubscribe cores (Swirski2D
    # uses TBB internally). Environment is inherited by spawned worker processes.
    tbb_limit = max(1, (os.cpu_count() or 1) // max(1, int(workers)))
    os.environ["TBB_NUM_THREADS"] = str(tbb_limit)
    os.environ["OMP_NUM_THREADS"] = str(tbb_limit)
    events.log("thread_policy", tbb_per_worker=tbb_limit, workers=workers)

    ready = manifest[manifest["input_status"].eq("ready")].copy()
    tight = ready[ready["input_kind"].eq("production_tight_bbox")]
    continuous = ready[
        ready["input_kind"].eq("fixed_source_canvas_from_temporal_tight_bbox_union")
    ]
    events.log(
        "run_inputs", tight_rows=int(len(tight)), continuous_rows=int(len(continuous)),
    )

    all_parts: list[pd.DataFrame] = []
    for algorithm in algorithms:
        events.log("algorithm_start", algorithm=algorithm)
        algo_dir = run_dir / "algorithms" / algorithm
        algo_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[tuple[mp.Process, Path]] = []

        if not tight.empty:
            for index, chunk in enumerate(_chunk_rows(tight.to_dict("records"), workers)):
                chunk_dir = algo_dir / f"tight_{index:02d}"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = chunk_dir / "manifest.csv"
                _write_chunk_manifest(chunk, manifest_path)
                process = mp.Process(
                    target=_chunk_worker,
                    args=(str(manifest_path), algorithm, str(chunk_dir), run_confidence, "independent"),
                )
                process.start()
                chunks.append((process, chunk_dir))
                events.log("chunk_start", algorithm=algorithm, kind="tight", index=index,
                           rows=len(chunk), pid=process.pid)

        if not continuous.empty:
            chunk_dir = algo_dir / "continuous"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = chunk_dir / "manifest.csv"
            _write_chunk_manifest(continuous.to_dict("records"), manifest_path)
            process = mp.Process(
                target=_chunk_worker,
                args=(str(manifest_path), algorithm, str(chunk_dir), run_confidence, "continuous"),
            )
            process.start()
            chunks.append((process, chunk_dir))
            events.log("chunk_start", algorithm=algorithm, kind="continuous", rows=len(continuous),
                       pid=process.pid)

        events.log("chunks_launched", algorithm=algorithm, n=len(chunks))

        # Monitor loop: log progress from the incremental partial CSVs.
        previous_counts = {str(path): -1 for _, path in chunks}
        while any(process.is_alive() for process, _ in chunks):
            for process, chunk_dir in chunks:
                results = chunk_dir / "results.csv"
                count = _count_csv_rows(results)
                key = str(results)
                if count != previous_counts.get(key):
                    previous_counts[key] = count
                    events.log(
                        "chunk_progress", algorithm=algorithm,
                        chunk=chunk_dir.name, rows_written=count,
                    )
            time.sleep(progress_interval_s)

        for process, chunk_dir in chunks:
            process.join()
            if process.exitcode != 0:
                events.log(
                    "chunk_crash", algorithm=algorithm, chunk=chunk_dir.name,
                    exitcode=process.exitcode,
                )
            else:
                events.log(
                    "chunk_done", algorithm=algorithm, chunk=chunk_dir.name,
                    exitcode=0,
                )

        # Merge this algorithm's chunk partials.
        frames = []
        for _, chunk_dir in chunks:
            results = chunk_dir / "results.csv"
            if results.is_file():
                frames.append(pd.read_csv(results, low_memory=False))
        if frames:
            algorithm_frame = pd.concat(frames, ignore_index=True)
            algorithm_frame.to_csv(algo_dir / "results.csv", index=False, encoding="utf-8-sig")
            all_parts.append(algorithm_frame)
        events.log("algorithm_done", algorithm=algorithm, rows=int(len(algorithm_frame)) if frames else 0)

    events.log("all_algorithms_done", algorithms=list(algorithms), parts=len(all_parts))

    if not all_parts:
        empty = pd.DataFrame(columns=RESULT_COLUMNS)
        events.log("run_done", rows=0, status="no_parts")
        return empty

    from .formal import enrich_source_and_agreement

    merged = pd.concat(all_parts, ignore_index=True)
    enriched = enrich_source_and_agreement(merged)
    events.log("run_done", rows=int(len(enriched)), status="ok")
    return enriched
