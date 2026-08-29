from __future__ import annotations

import hashlib
import json
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
from attention_pipeline.nir_behavior.contract import parse_window_specs
from attention_pipeline.nir_behavior.discovery import load_behavior_trials
from attention_pipeline.nir_behavior.features import summarize_signal

PIPELINE_VERSION = "nir-formal-analysis-tables-pupil-only-v2"
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class TrackSpec:
    name: str
    value_column: str
    valid_column: str | None
    source_mode_column: str | None
    family: str


@dataclass(frozen=True)
class TrackIndex:
    session_id: str
    block_num: int
    track: str
    family: str
    times_ms: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    source_modes: np.ndarray | None

    def bounds(self, start_ms: float, end_ms: float) -> tuple[int, int]:
        return (
            int(np.searchsorted(self.times_ms, start_ms, side="left")),
            int(np.searchsorted(self.times_ms, end_ms, side="left")),
        )

    def count(self, start_ms: float, end_ms: float) -> int:
        left, right = self.bounds(start_ms, end_ms)
        return max(0, right - left)


def _resolve(config: Config, key: str) -> Path:
    raw = config.section("paths").get(key)
    if raw in (None, ""):
        raise KeyError(f"formal pupil config missing paths.{key}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (config.path.parent.parent / path).resolve()
    return path


def _analysis_ready_root(config: Config) -> Path:
    return _resolve(config, "analysis_ready_root")


def _output_root(config: Config) -> Path:
    return _resolve(config, "output_root")


def _session_frame_path(config: Config, session_id: str) -> Path:
    return (
        _analysis_ready_root(config)
        / "frame_level"
        / session_id
        / f"{session_id}_nir_analysis_ready.csv"
    )


def _session_paths(config: Config, session_id: str) -> dict[str, Path]:
    root = _output_root(config) / "sessions" / session_id
    return {
        "root": root,
        "trial_level": root / f"{session_id}_trial_level.csv",
        "trial_windows": root / f"{session_id}_trial_pupil_windows.csv",
        "probe_windows": root / f"{session_id}_probe_pupil_windows.csv",
        "time_on_task": root / f"{session_id}_time_on_task_1s.csv",
        "trial_coverage": root / f"{session_id}_trial_window_coverage.csv",
        "probe_coverage": root / f"{session_id}_probe_window_coverage.csv",
        "dependency_audit": root / f"{session_id}_window_dependency_audit.csv",
        "manifest": root / f"{session_id}_analysis_tables_manifest.json",
        "completion": root / f"{session_id}_analysis_tables_completion.json",
    }


def discover_sessions(config: Config) -> list[str]:
    root = _analysis_ready_root(config) / "frame_level"
    if not root.is_dir():
        return []
    found: list[str] = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / f"{path.name}_nir_analysis_ready.csv").is_file():
            found.append(path.name)
    return found


def selected_sessions(config: Config, override: Iterable[str] | None = None) -> list[str]:
    if override:
        return [str(value).strip() for value in override if str(value).strip()]
    include = config.section("sessions").get("include", [])
    if include:
        return [str(value).strip() for value in include if str(value).strip()]
    return discover_sessions(config)


def _parse_tracks(config: Config) -> list[TrackSpec]:
    raw = config.section("tracks").get("include", [])
    if not isinstance(raw, list) or not raw:
        raise ValueError("tracks.include must be a non-empty list")
    tracks: list[TrackSpec] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("each track must be a mapping")
        name = str(item["name"]).strip()
        if not name or name in seen:
            raise ValueError(f"duplicate or empty track name: {name!r}")
        seen.add(name)
        tracks.append(
            TrackSpec(
                name=name,
                value_column=str(item["value_column"]),
                valid_column=(
                    str(item["valid_column"])
                    if item.get("valid_column") not in (None, "")
                    else None
                ),
                source_mode_column=(
                    str(item["source_mode_column"])
                    if item.get("source_mode_column") not in (None, "")
                    else None
                ),
                family=str(item.get("family", "primary")),
            )
        )
    return tracks


def load_analysis_ready_frame(
    config: Config, session_id: str, tracks: list[TrackSpec]
) -> pd.DataFrame:
    path = _session_frame_path(config, session_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {
        "session_id",
        "subject",
        "analysis_group_token",
        "repeat_group_size",
        "is_repeat_session",
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
        raise ValueError(f"{path}: missing pupil analysis-ready columns {missing}")
    frame = pd.read_csv(
        path,
        usecols=sorted(required),
        encoding="utf-8-sig",
        low_memory=False,
    )
    actual = set(frame["session_id"].dropna().astype(str).unique())
    if actual != {session_id}:
        raise ValueError(f"{path}: unexpected session identifiers {sorted(actual)}")
    for column in ("block", "phase_segment", "frame_idx", "unix_ms"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["block", "phase_segment", "frame_idx", "unix_ms"]].isna().any(axis=1).any():
        raise ValueError(f"{path}: missing analysis-ready identity/time values")
    duplicate = frame.duplicated(
        ["session_id", "block", "phase_segment", "frame_idx"], keep=False
    )
    if duplicate.any():
        raise ValueError(f"{path}: duplicate analysis-ready time keys")
    return frame.sort_values(["block", "unix_ms", "frame_idx"], kind="stable").reset_index(drop=True)


def _bool_values(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y"})
        .to_numpy(dtype=bool)
    )


def build_track_indices(
    frame: pd.DataFrame, session_id: str, tracks: list[TrackSpec]
) -> dict[tuple[int, str], TrackIndex]:
    result: dict[tuple[int, str], TrackIndex] = {}
    for block_num, block in frame.groupby("block", sort=True):
        block_num = int(block_num)
        block = block.sort_values("unix_ms", kind="stable").reset_index(drop=True)
        times = pd.to_numeric(block["unix_ms"], errors="coerce").to_numpy(dtype=float)
        for track in tracks:
            values = pd.to_numeric(block[track.value_column], errors="coerce").to_numpy(dtype=float)
            valid = (
                _bool_values(block[track.valid_column])
                if track.valid_column
                else np.isfinite(values)
            )
            valid = valid & np.isfinite(values)
            modes = (
                block[track.source_mode_column].astype("string").to_numpy(dtype=object)
                if track.source_mode_column
                else None
            )
            result[(block_num, track.name)] = TrackIndex(
                session_id=session_id,
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
    if modes is None or n_rows == 0:
        return {f"source_mode_{key}_fraction": None for key in keys}
    normalized = pd.Series(modes).fillna("missing").astype(str).str.strip().str.lower()
    return {
        f"source_mode_{key}_fraction": float(normalized.eq(key).mean()) for key in keys
    }


def _window_features(index: TrackIndex | None, start_ms: float, end_ms: float) -> dict[str, Any]:
    if index is None:
        empty = summarize_signal(np.array([], dtype=float), np.array([], dtype=float), "pupil")
        return {
            "n_nir_rows": 0,
            "n_pupil_valid": 0,
            "pupil_valid_fraction": None,
            **empty,
            **_source_mode_features(None, 0),
        }
    left, right = index.bounds(start_ms, end_ms)
    times = index.times_ms[left:right]
    raw = index.values[left:right]
    valid = index.valid[left:right]
    values = np.where(valid, raw, np.nan)
    result = summarize_signal(times, values, "pupil")
    result["n_nir_rows"] = int(max(0, right - left))
    result["n_pupil_valid"] = int(np.isfinite(values).sum())
    result["pupil_valid_fraction"] = (
        float(np.isfinite(values).mean()) if len(values) else None
    )
    modes = index.source_modes[left:right] if index.source_modes is not None else None
    result.update(_source_mode_features(modes, len(values)))
    return result


def _availability(
    index: TrackIndex | None,
    requested_start: float,
    requested_end: float,
    block_start: float,
    block_end: float,
) -> dict[str, Any]:
    requested_sec = max(0.0, (requested_end - requested_start) / 1000.0)
    available_start = max(requested_start, block_start)
    available_end = min(requested_end, block_end)
    available_sec = max(0.0, (available_end - available_start) / 1000.0)
    rate = _estimate_sampling_rate_hz(index.times_ms) if index is not None else None
    expected = available_sec * rate if rate and available_sec > 0 else None
    n_rows = (
        index.count(available_start, available_end)
        if index is not None and available_sec > 0
        else 0
    )
    internal = min(1.0, n_rows / expected) if expected and expected > 0 else None
    gap = (
        _max_temporal_gap_sec(index.times_ms, available_start, available_end)
        if index is not None and available_sec > 0
        else None
    )
    return {
        "requested_duration_sec": requested_sec,
        "available_duration_sec": available_sec,
        "available_duration_fraction": available_sec / requested_sec if requested_sec else None,
        "window_truncated_by_block_start": bool(requested_start < block_start),
        "window_truncated_by_block_end": bool(requested_end > block_end),
        "sampling_rate_hz_estimate": rate,
        "expected_nir_rows_available": expected,
        "n_nir_rows_available": int(n_rows),
        "internal_coverage_fraction": internal,
        "max_temporal_gap_sec": gap,
    }


def _hierarchy(frame: pd.DataFrame) -> dict[str, Any]:
    fields = ("session_id", "analysis_group_token", "repeat_group_size", "is_repeat_session")
    result: dict[str, Any] = {}
    for field in fields:
        values = frame[field].dropna().unique()
        if len(values) != 1:
            raise ValueError(f"analysis-ready {field} must be constant within session")
        result[field] = values[0]
    return result


def _add_trial_linkage(
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
            else:
                left, right = index.bounds(onset, float(next_onset))
                counts.append(max(0, right - left))
                valid_counts.append(int(index.valid[left:right].sum()))
        result[f"nir_rows_trial__{track.name}"] = counts
        result[f"pupil_valid_rows_trial__{track.name}"] = valid_counts
    return result


def build_trial_windows(
    trials: pd.DataFrame,
    indices: dict[tuple[int, str], TrackIndex],
    tracks: list[TrackSpec],
    specs: list[Any],
) -> pd.DataFrame:
    bounds = _block_analysis_bounds(trials)
    probe_times = _probe_times_by_block(trials)
    rows: list[dict[str, Any]] = []
    for trial in trials.itertuples(index=False):
        block_num = int(trial.block_num)
        onset = float(trial.absolute_onset_time)
        block_start, block_end = bounds[block_num]
        probes = probe_times.get(block_num, np.array([], dtype=float))
        previous_probe = _last_probe_before(probes, onset)
        for spec in specs:
            start_ms = onset + spec.start_offset_ms
            end_ms = onset + spec.end_offset_ms
            n_probes = _count_probes_in_window(probes, start_ms, end_ms)
            for track in tracks:
                index = indices.get((block_num, track.name))
                record = {
                    "session_id": trial.session_id,
                    "subject": trial.subject,
                    "analysis_group_token": trial.analysis_group_token,
                    "repeat_group_size": trial.repeat_group_size,
                    "is_repeat_session": trial.is_repeat_session,
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
                }
                record.update(_window_features(index, start_ms, end_ms))
                record.update(_availability(index, start_ms, end_ms, block_start, block_end))
                rows.append(record)
    return pd.DataFrame(rows)


def build_probe_windows(
    trials: pd.DataFrame,
    indices: dict[tuple[int, str], TrackIndex],
    tracks: list[TrackSpec],
    specs: list[Any],
) -> pd.DataFrame:
    bounds = _block_analysis_bounds(trials)
    probes = trials[
        pd.to_numeric(trials["is_probe"], errors="coerce").eq(1)
        & pd.to_numeric(trials["probe_onset_time"], errors="coerce").notna()
    ].copy()
    probes = probes.sort_values(["block_num", "probe_onset_time"], kind="stable").reset_index(drop=True)
    probes["probe_index_global"] = np.arange(1, len(probes) + 1)
    probes["probe_index_in_block"] = probes.groupby("block_num").cumcount() + 1
    probes["previous_probe_onset_ms"] = probes.groupby("block_num")["probe_onset_time"].shift(1)
    trial_onsets = pd.to_numeric(trials["absolute_onset_time"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for probe in probes.itertuples(index=False):
        block_num = int(probe.block_num)
        probe_onset = float(probe.probe_onset_time)
        previous_probe = getattr(probe, "previous_probe_onset_ms")
        previous_probe = None if pd.isna(previous_probe) else float(previous_probe)
        block_start, block_end = bounds[block_num]
        for spec in specs:
            start_ms = probe_onset + spec.start_offset_ms
            end_ms = probe_onset + spec.end_offset_ms
            # Strictly '< probe onset' for pre-probe windows. The anchoring probe
            # trial itself therefore cannot enter its own behavioral precursor window.
            behavior = trials[
                (pd.to_numeric(trials["block_num"], errors="coerce").eq(block_num))
                & trial_onsets.ge(start_ms)
                & trial_onsets.lt(end_ms)
            ]
            behavior = behavior[
                pd.to_numeric(behavior["absolute_onset_time"], errors="coerce").lt(probe_onset)
            ]
            behavior_features = _behavior_window_features(behavior)
            crosses_previous = bool(
                previous_probe is not None and start_ms <= previous_probe < end_ms
            )
            for track in tracks:
                index = indices.get((block_num, track.name))
                record = {
                    "session_id": probe.session_id,
                    "subject": probe.subject,
                    "analysis_group_token": probe.analysis_group_token,
                    "repeat_group_size": probe.repeat_group_size,
                    "is_repeat_session": probe.is_repeat_session,
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
                    "track": track.name,
                    "track_family": track.family,
                    "window_family": spec.family,
                    "window_name": spec.name,
                    "window_start_offset_ms": spec.start_offset_ms,
                    "window_end_offset_ms": spec.end_offset_ms,
                    "window_start_ms": start_ms,
                    "window_end_ms": end_ms,
                    "window_crosses_previous_probe": crosses_previous,
                    "anchoring_probe_trial_excluded": True,
                }
                record.update(behavior_features)
                record.update(_window_features(index, start_ms, end_ms))
                record.update(_availability(index, start_ms, end_ms, block_start, block_end))
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
    bounds = _block_analysis_bounds(trials)
    rows: list[dict[str, Any]] = []
    first = trials.iloc[0]
    width_ms = bin_sec * 1000.0
    for block_num, (block_start, block_end) in bounds.items():
        n_bins = int(np.ceil((block_end - block_start) / width_ms))
        for bin_index in range(n_bins):
            start_ms = block_start + bin_index * width_ms
            end_ms = min(block_end, start_ms + width_ms)
            for track in tracks:
                index = indices.get((block_num, track.name))
                record = {
                    "session_id": first["session_id"],
                    "subject": first["subject"],
                    "analysis_group_token": first["analysis_group_token"],
                    "repeat_group_size": first["repeat_group_size"],
                    "is_repeat_session": first["is_repeat_session"],
                    "block_num": int(block_num),
                    "track": track.name,
                    "track_family": track.family,
                    "bin_index_in_block": bin_index + 1,
                    "time_in_block_start_sec": (start_ms - block_start) / 1000.0,
                    "time_in_block_mid_sec": (((start_ms + end_ms) / 2.0) - block_start) / 1000.0,
                    "window_start_ms": start_ms,
                    "window_end_ms": end_ms,
                }
                record.update(_window_features(index, start_ms, end_ms))
                record.update(_availability(index, start_ms, end_ms, block_start, block_end))
                rows.append(record)
    return pd.DataFrame(rows)


def build_coverage_report(windows: pd.DataFrame, *, level: str) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame()
    group_cols = [
        "session_id",
        "analysis_group_token",
        "block_num",
        "track",
        "track_family",
        "window_name",
    ]
    rows: list[dict[str, Any]] = []
    for key, frame in windows.groupby(group_cols, sort=True, dropna=False):
        row = dict(zip(group_cols, key))
        row.update(
            {
                "n_windows": int(len(frame)),
                "pupil_valid_fraction_median": float(
                    pd.to_numeric(frame["pupil_valid_fraction"], errors="coerce").median()
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
                "level": level,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def audit_window_dependency(windows: pd.DataFrame) -> pd.DataFrame:
    """Quantify temporal overlap so overlapping windows are never treated as IID."""
    if windows.empty:
        return pd.DataFrame()
    group_cols = ["session_id", "analysis_group_token", "block_num", "track", "window_name"]
    rows: list[dict[str, Any]] = []
    for key, frame in windows.groupby(group_cols, sort=True, dropna=False):
        current = frame.sort_values("window_start_ms", kind="stable")
        starts = pd.to_numeric(current["window_start_ms"], errors="coerce").to_numpy(dtype=float)
        ends = pd.to_numeric(current["window_end_ms"], errors="coerce").to_numpy(dtype=float)
        previous_end = np.r_[-np.inf, ends[:-1]]
        overlap_ms = np.maximum(0.0, previous_end - starts)
        n_overlap = int(np.sum(overlap_ms > 0))
        row = dict(zip(group_cols, key))
        row.update(
            {
                "n_windows": int(len(current)),
                "n_overlap_with_previous": n_overlap,
                "overlap_with_previous_fraction": n_overlap / len(current) if len(current) else np.nan,
                "overlap_ms_total_pairwise_previous": float(np.nansum(overlap_ms)),
                "iid_trial_window_inference_allowed": bool(n_overlap == 0),
                "required_dependence_control": (
                    "analysis_group/session/block clustered or non-overlapping aggregation"
                    if n_overlap
                    else "session autocorrelation control still required"
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def run_session(config: Config, session_id: str, *, force: bool = False) -> dict[str, Any]:
    tracks = _parse_tracks(config)
    nir = load_analysis_ready_frame(config, session_id, tracks)
    hierarchy = _hierarchy(nir)
    behavior = load_behavior_trials(config, session_id)
    behavior = add_behavior_qc(
        behavior,
        carryover_ms=float(config.section("behavior_qc").get("carryover_candidate_ms", 200)),
    )
    behavior = _add_trial_context(behavior)
    behavior["session_id"] = session_id
    behavior["analysis_group_token"] = hierarchy["analysis_group_token"]
    behavior["repeat_group_size"] = hierarchy["repeat_group_size"]
    behavior["is_repeat_session"] = hierarchy["is_repeat_session"]
    indices = build_track_indices(nir, session_id, tracks)
    trial_specs = parse_window_specs(config.section("windows").get("trial", []), family="trial")
    probe_specs = parse_window_specs(config.section("windows").get("probe", []), family="probe")
    trial_level = _add_trial_linkage(behavior, indices, tracks)
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
    dependency = audit_window_dependency(trial_windows)

    paths = _session_paths(config, session_id)
    if paths["completion"].is_file() and not force:
        try:
            existing = json.loads(paths["completion"].read_text(encoding="utf-8"))
            if existing.get("status") == "complete" and existing.get("analysis_ready_sha256") == _digest_file(_session_frame_path(config, session_id)):
                return {"session_id": session_id, "status": "skipped", "reason": "validated_completion"}
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError, AttributeError):
            # A stale/malformed compatibility marker is safe to ignore and regenerate.
            # Filesystem/permission errors deliberately propagate instead of becoming
            # a silent cache miss.
            pass
    paths["root"].mkdir(parents=True, exist_ok=True)
    frames = {
        "trial_level": trial_level,
        "trial_windows": trial_windows,
        "probe_windows": probe_windows,
        "time_on_task": time_on_task,
        "trial_coverage": trial_coverage,
        "probe_coverage": probe_coverage,
        "dependency_audit": dependency,
    }
    for key, frame in frames.items():
        frame.to_csv(paths[key], index=False, encoding="utf-8-sig")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "signal_semantics": "pupil_geometry_only",
        "analysis_group_token_present": True,
        "repeat_session_grouping_preserved": True,
        "analysis_ready_sha256": _digest_file(_session_frame_path(config, session_id)),
        "tracks": [track.__dict__ for track in tracks],
        "windows": {
            "trial": [spec.__dict__ for spec in trial_specs],
            "probe": [spec.__dict__ for spec in probe_specs],
        },
        "dependence_policy": (
            "overlapping trial windows are descriptive features and are not IID inferential units; "
            "participant-group/session/block dependence must be controlled downstream"
        ),
        "probe_policy": "pre-probe behavioral windows exclude the anchoring probe trial and never cross block bounds",
        "outputs": {key: str(paths[key]) for key in frames},
    }
    _write_json(paths["manifest"], manifest)
    _write_json(
        paths["completion"],
        {
            "status": "complete",
            "pipeline_version": PIPELINE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "analysis_ready_sha256": manifest["analysis_ready_sha256"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        "session_id": session_id,
        "status": "complete",
        "n_trials": int(len(trial_level)),
        "n_trial_windows": int(len(trial_windows)),
        "n_probe_windows": int(len(probe_windows)),
    }


def run_cohort(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    selected = selected_sessions(config, subjects)
    if not selected:
        raise ValueError("No pupil analysis-ready sessions selected")
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for session_id in selected:
        try:
            result = run_session(config, session_id, force=force)
        except Exception as exc:
            result = {
                "session_id": session_id,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(result)
        results.append(result)
    root = _output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    failure_path = root / "failure_tables" / "analysis_table_session_failures.csv"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        failures, columns=["session_id", "status", "error_type", "error"]
    ).to_csv(failure_path, index=False, encoding="utf-8-sig")
    n_failed = len(failures)
    summary = {
        "status": "complete" if n_failed == 0 else "partial",
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "n_sessions_requested": len(selected),
        "n_sessions_completed": sum(row.get("status") == "complete" for row in results),
        "n_sessions_skipped_validated": sum(row.get("status") == "skipped" for row in results),
        "n_sessions_failed": n_failed,
        # compatibility key for existing CLI exit handling; it counts sessions, not participants.
        "n_subjects_failed": n_failed,
        "failure_table": str(failure_path),
        "results": results,
    }
    _write_json(root / "cohort_manifest.json", summary)
    return summary
