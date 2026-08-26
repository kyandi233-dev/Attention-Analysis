from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from attention_pipeline.config import Config, load_config
from attention_pipeline.nir_behavior.alignment import (
    _add_trial_context,
    _behavior_window_features,
    _count_probes_in_window,
    _last_probe_before,
    _probe_times_by_block,
)
from attention_pipeline.nir_behavior.alignment_v12 import (
    _block_analysis_bounds,
    _estimate_sampling_rate_hz,
    _max_temporal_gap_sec,
)
from attention_pipeline.nir_behavior.behavior_qc import add_behavior_qc
from attention_pipeline.nir_behavior.contract import normalize_subject, parse_subject_list, parse_window_specs
from attention_pipeline.nir_behavior.discovery import load_behavior_trials, resolve_repo_path, sha256
from attention_pipeline.nir_behavior.features import summarize_signal

PIPELINE_VERSION = "nir-formal-analysis-tables-v1"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrackSpec:
    name: str
    value_column: str
    valid_column: str | None
    source_mode_column: str | None
    family: str


@dataclass(frozen=True)
class TrackIndex:
    subject: str
    block_num: int
    track: str
    family: str
    times_ms: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    source_modes: np.ndarray | None

    def bounds(self, start_ms: float, end_ms: float) -> tuple[int, int]:
        left = int(np.searchsorted(self.times_ms, start_ms, side="left"))
        right = int(np.searchsorted(self.times_ms, end_ms, side="left"))
        return left, right

    def count(self, start_ms: float, end_ms: float) -> int:
        left, right = self.bounds(start_ms, end_ms)
        return max(0, right - left)


def _git_commit(repo: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or None
    except Exception:
        return None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _analysis_ready_root(config: Config) -> Path:
    raw = config.section("paths").get("analysis_ready_root")
    if raw is None:
        raise KeyError("formal-analysis config missing paths.analysis_ready_root")
    return resolve_repo_path(config, raw)


def _output_root(config: Config) -> Path:
    raw = config.section("paths").get("output_root")
    if raw is None:
        raise KeyError("formal-analysis config missing paths.output_root")
    return resolve_repo_path(config, raw)


def _subject_frame_path(config: Config, subject: str) -> Path:
    subject = normalize_subject(subject)
    return (
        _analysis_ready_root(config)
        / "frame_level"
        / subject
        / f"{subject}_nir_analysis_ready.csv"
    )


def _subject_output_paths(config: Config, subject: str) -> dict[str, Path]:
    subject = normalize_subject(subject)
    root = _output_root(config)
    subject_dir = root / "subjects" / subject
    return {
        "subject_dir": subject_dir,
        "trial_level": subject_dir / f"{subject}_trial_level.csv",
        "trial_windows": subject_dir / f"{subject}_trial_pir_windows.csv",
        "probe_windows": subject_dir / f"{subject}_probe_pir_windows.csv",
        "time_on_task": subject_dir / f"{subject}_time_on_task_1s.csv",
        "trial_coverage": subject_dir / f"{subject}_trial_window_coverage.csv",
        "probe_coverage": subject_dir / f"{subject}_probe_window_coverage.csv",
        "manifest": subject_dir / f"{subject}_analysis_tables_manifest.json",
        "summary": subject_dir / f"{subject}_analysis_tables_summary.json",
        "completion": subject_dir / f"{subject}_analysis_tables_completion.json",
    }


def discover_subjects(config: Config) -> list[str]:
    root = _analysis_ready_root(config) / "frame_level"
    if not root.is_dir():
        return []
    result: list[str] = []
    for path in root.glob("sub-*"):
        if not path.is_dir():
            continue
        try:
            subject = normalize_subject(path.name)
        except ValueError:
            continue
        expected = path / f"{subject}_nir_analysis_ready.csv"
        if expected.is_file():
            result.append(subject)
    return sorted(set(result), key=lambda value: int(value.split("-")[1]))


def selected_subjects(config: Config, override: Iterable[str] | None = None) -> list[str]:
    if override:
        return parse_subject_list(list(override))
    raw = config.section("subjects").get("include", [])
    if raw:
        return parse_subject_list(raw)
    return discover_subjects(config)


def _parse_tracks(config: Config) -> list[TrackSpec]:
    raw = config.section("tracks").get("include", [])
    if not isinstance(raw, list) or not raw:
        raise ValueError("tracks.include must contain at least one track")
    tracks: list[TrackSpec] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("each tracks.include item must be a mapping")
        name = str(item["name"]).strip()
        if not name or name in seen:
            raise ValueError(f"duplicate/empty track name: {name!r}")
        seen.add(name)
        tracks.append(
            TrackSpec(
                name=name,
                value_column=str(item["value_column"]).strip(),
                valid_column=(
                    str(item["valid_column"]).strip()
                    if item.get("valid_column") not in (None, "")
                    else None
                ),
                source_mode_column=(
                    str(item["source_mode_column"]).strip()
                    if item.get("source_mode_column") not in (None, "")
                    else None
                ),
                family=str(item.get("family", "primary")).strip() or "primary",
            )
        )
    return tracks


def load_analysis_ready_frame(
    config: Config, subject: str, tracks: list[TrackSpec]
) -> pd.DataFrame:
    subject = normalize_subject(subject)
    path = _subject_frame_path(config, subject)
    if not path.is_file():
        raise FileNotFoundError(path)

    required = {
        "subject",
        "block",
        "phase",
        "phase_segment",
        "frame_idx",
        "unix_ms",
        "video_time_ms",
        "phase_time_ms",
    }
    for track in tracks:
        required.add(track.value_column)
        if track.valid_column:
            required.add(track.valid_column)
        if track.source_mode_column:
            required.add(track.source_mode_column)

    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    missing = sorted(required - set(header.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    df = pd.read_csv(
        path,
        usecols=sorted(required),
        encoding="utf-8-sig",
        low_memory=False,
    )
    df["subject"] = df["subject"].map(normalize_subject)
    actual = set(df["subject"].dropna().unique())
    if actual != {subject}:
        raise ValueError(f"{path}: unexpected subject identifiers {sorted(actual)}")

    for column in ("block", "phase_segment", "frame_idx", "unix_ms", "video_time_ms", "phase_time_ms"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[["block", "frame_idx", "unix_ms"]].isna().any(axis=1).any():
        raise ValueError(f"{path}: missing block/frame_idx/unix_ms")

    duplicate = df.duplicated(["subject", "block", "phase_segment", "frame_idx"], keep=False)
    if bool(duplicate.any()):
        examples = df.loc[
            duplicate,
            ["subject", "block", "phase_segment", "frame_idx", "unix_ms"],
        ].head(5)
        raise ValueError(f"{path}: duplicate analysis-ready time keys {examples.to_dict('records')}")

    return df.sort_values(["block", "unix_ms", "frame_idx"], kind="stable").reset_index(drop=True)


def build_track_indices(
    df: pd.DataFrame, subject: str, tracks: list[TrackSpec]
) -> dict[tuple[int, str], TrackIndex]:
    result: dict[tuple[int, str], TrackIndex] = {}
    for block_num in (1, 2):
        block = df[df["block"] == block_num].copy()
        if block.empty:
            continue
        block = block.sort_values("unix_ms", kind="stable").reset_index(drop=True)
        times = pd.to_numeric(block["unix_ms"], errors="coerce").to_numpy(dtype=float)
        for track in tracks:
            values = pd.to_numeric(block[track.value_column], errors="coerce").to_numpy(dtype=float)
            if track.valid_column:
                valid_series = block[track.valid_column]
                if pd.api.types.is_bool_dtype(valid_series):
                    valid = valid_series.fillna(False).to_numpy(dtype=bool)
                else:
                    valid = (
                        valid_series.astype(str).str.strip().str.lower()
                        .isin({"1", "true", "yes", "y"})
                        .to_numpy(dtype=bool)
                    )
            else:
                valid = np.isfinite(values)
            valid = valid & np.isfinite(values)

            modes = None
            if track.source_mode_column:
                modes = block[track.source_mode_column].astype(str).to_numpy(dtype=object)

            result[(block_num, track.name)] = TrackIndex(
                subject=subject,
                block_num=block_num,
                track=track.name,
                family=track.family,
                times_ms=times,
                values=values,
                valid=valid,
                source_modes=modes,
            )
    return result


def _source_mode_features(modes: np.ndarray | None, n_rows: int) -> dict[str, Any]:
    keys = ("binocular", "left_only", "right_only", "missing")
    result: dict[str, Any] = {}
    if modes is None or n_rows == 0:
        for key in keys:
            result[f"source_mode_{key}_fraction"] = None
        return result
    normalized = pd.Series(modes).fillna("missing").astype(str).str.strip().str.lower()
    for key in keys:
        result[f"source_mode_{key}_fraction"] = float(normalized.eq(key).mean())
    return result


def _window_features(
    index: TrackIndex | None,
    start_ms: float,
    end_ms: float,
) -> dict[str, Any]:
    if index is None:
        return {
            "n_nir_rows": 0,
            "n_pir_valid": 0,
            "pir_valid_fraction": None,
            **{
                f"pir_{suffix}": None
                for suffix in (
                    "median",
                    "mean",
                    "mad",
                    "iqr",
                    "sd",
                    "p10",
                    "p90",
                    "slope_per_sec",
                    "diff_mad",
                    "diff_rate_mad_per_sec",
                )
            },
            **_source_mode_features(None, 0),
        }

    left, right = index.bounds(start_ms, end_ms)
    times = index.times_ms[left:right]
    raw = index.values[left:right]
    valid = index.valid[left:right]
    values = np.where(valid, raw, np.nan)
    modes = index.source_modes[left:right] if index.source_modes is not None else None

    result: dict[str, Any] = {"n_nir_rows": int(max(0, right - left))}
    result.update(summarize_signal(times, values, "pir"))
    result["pir_valid_fraction"] = (
        float(np.isfinite(values).mean()) if len(values) else None
    )
    result.update(_source_mode_features(modes, len(values)))
    return result


def _window_availability_metadata(
    index: TrackIndex | None,
    requested_start: float,
    requested_end: float,
    block_start: float,
    block_end: float,
) -> dict[str, Any]:
    requested_duration_sec = max(0.0, (requested_end - requested_start) / 1000.0)
    available_start = max(requested_start, block_start)
    available_end = min(requested_end, block_end)
    available_duration_sec = max(0.0, (available_end - available_start) / 1000.0)
    available_fraction = (
        available_duration_sec / requested_duration_sec
        if requested_duration_sec > 0
        else None
    )

    rate = _estimate_sampling_rate_hz(index.times_ms) if index is not None else None
    expected_rows = (
        available_duration_sec * rate
        if rate is not None and available_duration_sec > 0
        else None
    )
    n_rows_available = (
        index.count(available_start, available_end)
        if index is not None and available_duration_sec > 0
        else 0
    )
    internal_coverage = (
        min(1.0, float(n_rows_available) / float(expected_rows))
        if expected_rows is not None and expected_rows > 0
        else None
    )
    max_gap = (
        _max_temporal_gap_sec(index.times_ms, available_start, available_end)
        if index is not None and available_duration_sec > 0
        else None
    )
    return {
        "requested_duration_sec": requested_duration_sec,
        "block_analysis_start_ms": block_start,
        "block_analysis_end_ms": block_end,
        "available_start_ms": available_start,
        "available_end_ms": available_end,
        "available_duration_sec": available_duration_sec,
        "available_duration_fraction": available_fraction,
        "window_truncated_by_block_start": bool(requested_start < block_start),
        "window_truncated_by_block_end": bool(requested_end > block_end),
        "sampling_rate_hz_estimate": rate,
        "expected_nir_rows_available": expected_rows,
        "n_nir_rows_available": int(n_rows_available),
        "internal_coverage_fraction": internal_coverage,
        "max_temporal_gap_sec": max_gap,
    }


def _add_trial_track_linkage(
    trials: pd.DataFrame,
    indices: dict[tuple[int, str], TrackIndex],
    tracks: list[TrackSpec],
) -> pd.DataFrame:
    result = trials.copy()
    for track in tracks:
        counts: list[int] = []
        valid_counts: list[int] = []
        for row in result.itertuples(index=False):
            block_num = int(row.block_num)
            onset = float(row.absolute_onset_time)
            next_onset = getattr(row, "next_trial_onset_time")
            if pd.isna(next_onset):
                next_onset = onset + 1150.0
            index = indices.get((block_num, track.name))
            if index is None:
                counts.append(0)
                valid_counts.append(0)
                continue
            left, right = index.bounds(onset, float(next_onset))
            counts.append(max(0, right - left))
            valid_counts.append(int(index.valid[left:right].sum()))
        result[f"nir_rows_trial__{track.name}"] = counts
        result[f"pir_valid_rows_trial__{track.name}"] = valid_counts
    return result


def build_trial_windows(
    trials: pd.DataFrame,
    indices: dict[tuple[int, str], TrackIndex],
    tracks: list[TrackSpec],
    specs: list[Any],
) -> pd.DataFrame:
    block_bounds = _block_analysis_bounds(trials)
    probe_times = _probe_times_by_block(trials)
    rows: list[dict[str, Any]] = []

    for trial in trials.itertuples(index=False):
        block_num = int(trial.block_num)
        onset = float(trial.absolute_onset_time)
        block_start, block_end = block_bounds[block_num]
        block_probes = probe_times.get(block_num, np.array([], dtype=float))
        previous_probe = _last_probe_before(block_probes, onset)
        for spec in specs:
            start_ms = onset + spec.start_offset_ms
            end_ms = onset + spec.end_offset_ms
            n_probes = _count_probes_in_window(block_probes, start_ms, end_ms)
            for track in tracks:
                index = indices.get((block_num, track.name))
                record: dict[str, Any] = {
                    "subject": trial.subject,
                    "block_num": block_num,
                    "trial_num": int(trial.trial_num),
                    "global_trial_index": int(trial.global_trial_index),
                    "trial_onset_ms": onset,
                    "track": track.name,
                    "track_family": track.family,
                    "window_family": spec.family,
                    "window_name": spec.name,
                    "window_start_offset_ms": spec.start_offset_ms,
                    "window_end_offset_ms": spec.end_offset_ms,
                    "window_start_ms": start_ms,
                    "window_end_ms": end_ms,
                    "n_probes_in_window": int(n_probes),
                    "window_crosses_probe": bool(n_probes > 0),
                    "last_probe_before_trial_ms": previous_probe,
                    "time_since_last_probe_sec": (
                        (onset - previous_probe) / 1000.0
                        if previous_probe is not None
                        else None
                    ),
                }
                record.update(_window_features(index, start_ms, end_ms))
                record.update(
                    _window_availability_metadata(
                        index, start_ms, end_ms, block_start, block_end
                    )
                )
                rows.append(record)
    return pd.DataFrame(rows)


def build_probe_windows(
    trials: pd.DataFrame,
    indices: dict[tuple[int, str], TrackIndex],
    tracks: list[TrackSpec],
    specs: list[Any],
) -> pd.DataFrame:
    block_bounds = _block_analysis_bounds(trials)
    probes = trials[
        pd.to_numeric(trials["is_probe"], errors="coerce").eq(1)
        & pd.to_numeric(trials["probe_onset_time"], errors="coerce").notna()
    ].copy()
    probes = probes.sort_values(["block_num", "probe_onset_time"]).reset_index(drop=True)
    probes["probe_index_global"] = np.arange(1, len(probes) + 1)
    probes["probe_index_in_block"] = probes.groupby("block_num").cumcount() + 1
    probes["previous_probe_onset_ms"] = probes.groupby("block_num")[
        "probe_onset_time"
    ].shift(1)

    trial_onsets = pd.to_numeric(trials["absolute_onset_time"], errors="coerce")
    rows: list[dict[str, Any]] = []

    for probe in probes.itertuples(index=False):
        block_num = int(probe.block_num)
        probe_onset = float(probe.probe_onset_time)
        previous_probe = getattr(probe, "previous_probe_onset_ms")
        previous_probe = None if pd.isna(previous_probe) else float(previous_probe)
        block_start, block_end = block_bounds[block_num]

        for spec in specs:
            start_ms = probe_onset + spec.start_offset_ms
            end_ms = probe_onset + spec.end_offset_ms
            behavior_frame = trials[
                (trials["block_num"] == block_num)
                & trial_onsets.ge(start_ms)
                & trial_onsets.lt(end_ms)
            ]
            behavior_features = _behavior_window_features(behavior_frame)
            crosses_previous = bool(
                previous_probe is not None and start_ms <= previous_probe < end_ms
            )

            for track in tracks:
                index = indices.get((block_num, track.name))
                record: dict[str, Any] = {
                    "subject": probe.subject,
                    "block_num": block_num,
                    "probe_index_global": int(probe.probe_index_global),
                    "probe_index_in_block": int(probe.probe_index_in_block),
                    "probe_trial_num": int(probe.trial_num),
                    "probe_onset_ms": probe_onset,
                    "probe_response": probe.probe_response,
                    "probe_rt": probe.probe_rt,
                    "probe_vigilance": probe.probe_vigilance,
                    "probe_vigilance_rt": probe.probe_vigilance_rt,
                    "previous_probe_onset_ms": previous_probe,
                    "seconds_since_previous_probe": (
                        (probe_onset - previous_probe) / 1000.0
                        if previous_probe is not None
                        else None
                    ),
                    "track": track.name,
                    "track_family": track.family,
                    "window_family": spec.family,
                    "window_name": spec.name,
                    "window_start_offset_ms": spec.start_offset_ms,
                    "window_end_offset_ms": spec.end_offset_ms,
                    "window_start_ms": start_ms,
                    "window_end_ms": end_ms,
                    "window_crosses_previous_probe": crosses_previous,
                    "seconds_of_window_before_previous_probe": (
                        max(0.0, (previous_probe - start_ms) / 1000.0)
                        if crosses_previous and previous_probe is not None
                        else None
                    ),
                }
                record.update(behavior_features)
                record.update(_window_features(index, start_ms, end_ms))
                record.update(
                    _window_availability_metadata(
                        index, start_ms, end_ms, block_start, block_end
                    )
                )
                rows.append(record)
    return pd.DataFrame(rows)


def build_time_on_task(
    trials: pd.DataFrame,
    indices: dict[tuple[int, str], TrackIndex],
    tracks: list[TrackSpec],
    *,
    bin_sec: float,
) -> pd.DataFrame:
    if bin_sec <= 0:
        raise ValueError("time_on_task.bin_sec must be > 0")
    block_bounds = _block_analysis_bounds(trials)
    rows: list[dict[str, Any]] = []

    for block_num, (block_start, block_end) in block_bounds.items():
        width_ms = bin_sec * 1000.0
        n_bins = int(np.ceil((block_end - block_start) / width_ms))
        for bin_index in range(n_bins):
            start_ms = block_start + bin_index * width_ms
            end_ms = min(block_end, start_ms + width_ms)
            for track in tracks:
                index = indices.get((block_num, track.name))
                record = {
                    "subject": str(trials["subject"].iloc[0]),
                    "block_num": block_num,
                    "track": track.name,
                    "track_family": track.family,
                    "bin_index_in_block": bin_index + 1,
                    "time_in_block_start_sec": (start_ms - block_start) / 1000.0,
                    "time_in_block_mid_sec": (
                        ((start_ms + end_ms) / 2.0) - block_start
                    ) / 1000.0,
                    "window_start_ms": start_ms,
                    "window_end_ms": end_ms,
                }
                record.update(_window_features(index, start_ms, end_ms))
                record.update(
                    _window_availability_metadata(
                        index, start_ms, end_ms, block_start, block_end
                    )
                )
                rows.append(record)
    return pd.DataFrame(rows)


def build_coverage_report(windows: pd.DataFrame, *, level: str) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame()
    group_cols = ["subject", "block_num", "track", "track_family", "window_name"]
    rows: list[dict[str, Any]] = []
    for key, frame in windows.groupby(group_cols, sort=True, dropna=False):
        subject, block_num, track, track_family, window_name = key
        row = {
            "subject": subject,
            "block_num": int(block_num),
            "track": track,
            "track_family": track_family,
            "window_name": window_name,
            "n_windows": int(len(frame)),
            "pir_valid_fraction_median": float(
                pd.to_numeric(frame["pir_valid_fraction"], errors="coerce").median()
            ),
            "available_duration_fraction_median": float(
                pd.to_numeric(frame["available_duration_fraction"], errors="coerce").median()
            ),
            "internal_coverage_fraction_median": float(
                pd.to_numeric(frame["internal_coverage_fraction"], errors="coerce").median()
            ),
            "boundary_truncated_fraction": float(
                (
                    frame["window_truncated_by_block_start"].fillna(False)
                    | frame["window_truncated_by_block_end"].fillna(False)
                ).mean()
            ),
            "max_temporal_gap_sec_p95": float(
                pd.to_numeric(frame["max_temporal_gap_sec"], errors="coerce").quantile(0.95)
            ),
            "level": level,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _identity(
    config: Config,
    subject: str,
    frame_path: Path,
    trials: pd.DataFrame,
) -> dict[str, Any]:
    behavior_files = sorted(
        {Path(value) for value in trials["source_file"].dropna().astype(str)}
    )
    return {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config_digest": config.digest,
        "analysis_ready_csv": str(frame_path),
        "analysis_ready_sha256": sha256(frame_path),
        "behavior_files": [
            {"path": str(path), "sha256": sha256(path)} for path in behavior_files
        ],
    }


def _completion_matches(path: Path, identity: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return current.get("status") == "complete" and current.get("identity") == identity


def run_subject(
    config: Config,
    subject: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    subject = normalize_subject(subject)
    tracks = _parse_tracks(config)
    frame_path = _subject_frame_path(config, subject)
    trials = load_behavior_trials(config, subject)
    trials = add_behavior_qc(
        trials,
        carryover_ms=float(
            config.section("behavior_qc").get("carryover_candidate_ms", 200)
        ),
    )
    trials = _add_trial_context(trials)

    identity = _identity(config, subject, frame_path, trials)
    paths = _subject_output_paths(config, subject)
    if not force and _completion_matches(paths["completion"], identity):
        return {"subject": subject, "status": "skipped", "reason": "validated_completion"}

    nir = load_analysis_ready_frame(config, subject, tracks)
    indices = build_track_indices(nir, subject, tracks)

    trial_specs = parse_window_specs(
        config.section("windows").get("trial", []), family="trial"
    )
    probe_specs = parse_window_specs(
        config.section("windows").get("probe", []), family="probe"
    )

    trial_level = _add_trial_track_linkage(trials, indices, tracks)
    trial_windows = build_trial_windows(trial_level, indices, tracks, trial_specs)
    probe_windows = build_probe_windows(trial_level, indices, tracks, probe_specs)
    time_on_task = build_time_on_task(
        trial_level,
        indices,
        tracks,
        bin_sec=float(config.section("time_on_task").get("bin_sec", 1.0)),
    )
    trial_coverage = build_coverage_report(trial_windows, level="trial")
    probe_coverage = build_coverage_report(probe_windows, level="probe")

    paths["subject_dir"].mkdir(parents=True, exist_ok=True)
    outputs = (
        (trial_level, paths["trial_level"]),
        (trial_windows, paths["trial_windows"]),
        (probe_windows, paths["probe_windows"]),
        (time_on_task, paths["time_on_task"]),
        (trial_coverage, paths["trial_coverage"]),
        (probe_coverage, paths["probe_coverage"]),
    )
    for frame, path in outputs:
        frame.to_csv(path, index=False, encoding="utf-8-sig")

    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "subject": subject,
        "created_at_utc": created_at,
        "identity": identity,
        "git_commit": _git_commit(config.path.parent.parent),
        "tracks": [track.__dict__ for track in tracks],
        "windows": {
            "trial": [spec.__dict__ for spec in trial_specs],
            "probe": [spec.__dict__ for spec in probe_specs],
        },
        "principles": {
            "analysis_ready_is_only_nir_input": True,
            "production_nir_read_directly": False,
            "primary_track": config.section("tracks").get("primary"),
            "coverage_thresholds_applied": False,
            "window_selected_from_outcome": False,
            "behavior_raw_scoring_overwritten": False,
            "time_on_task_bin_is_descriptive_base": True,
        },
        "outputs": {name: str(path) for name, path in paths.items() if name not in {"subject_dir"}},
    }
    _atomic_json(paths["manifest"], manifest)

    summary = {
        "subject": subject,
        "status": "complete",
        "trial_rows": int(len(trial_level)),
        "trial_window_rows": int(len(trial_windows)),
        "probe_window_rows": int(len(probe_windows)),
        "time_on_task_rows": int(len(time_on_task)),
        "trial_coverage_rows": int(len(trial_coverage)),
        "probe_coverage_rows": int(len(probe_coverage)),
        "tracks": [track.name for track in tracks],
        "missing_block_tracks": [
            f"block{block}-{track.name}"
            for block in (1, 2)
            for track in tracks
            if (block, track.name) not in indices
        ],
    }
    _atomic_json(paths["summary"], summary)
    _atomic_json(
        paths["completion"],
        {
            "status": "complete",
            "subject": subject,
            "created_at_utc": created_at,
            "identity": identity,
            "required_artifacts": [
                str(paths["trial_level"]),
                str(paths["trial_windows"]),
                str(paths["probe_windows"]),
                str(paths["time_on_task"]),
                str(paths["trial_coverage"]),
                str(paths["probe_coverage"]),
                str(paths["manifest"]),
                str(paths["summary"]),
            ],
        },
    )
    return summary


def run_cohort(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    selected = selected_subjects(config, subjects)
    if not selected:
        raise ValueError("No analysis-ready subjects selected")

    results: list[dict[str, Any]] = []
    for subject in selected:
        try:
            results.append(run_subject(config, subject, force=force))
        except Exception as exc:
            results.append(
                {
                    "subject": normalize_subject(subject),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    root = _output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config.path),
        "config_digest": config.digest,
        "analysis_ready_root": str(_analysis_ready_root(config)),
        "subjects": selected,
        "results": results,
    }
    _atomic_json(root / "cohort_manifest.json", manifest)

    failed = [item for item in results if item.get("status") == "failed"]
    return {
        "status": "complete" if not failed else "partial",
        "n_subjects_requested": len(selected),
        "n_subjects_failed": len(failed),
        "results": results,
        "manifest": str(root / "cohort_manifest.json"),
    }
