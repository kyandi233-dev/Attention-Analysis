from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from .alignment import (
    _add_trial_context,
    _atomic_json,
    _completion_matches,
    _git_commit,
    _source_identity,
    add_trial_nir_linkage,
    build_nir_indices,
    build_probe_windows,
    build_trial_windows,
    load_nir_frame,
)
from .behavior_qc import add_behavior_qc
from .contract import (
    EYES,
    OAR_COLUMN,
    PIR_COLUMN,
    normalize_subject,
    parse_window_specs,
    subject_output_paths,
)
from .coverage import build_window_coverage_report, coverage_overview
from .discovery import alignment_output_root, find_nir_source, load_behavior_trials


_NOMINAL_TRIAL_MS = 1150.0


def _block_analysis_bounds(trials: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Return behavior-defined analysis bounds for each formal block.

    Start prefers FocusWave's block_onset_time when present. End is inferred from
    the last formal trial onset plus the nominal 250-ms stimulus + 900-ms mask.
    These bounds represent design availability, not NIR availability.
    """
    result: dict[int, tuple[float, float]] = {}
    for block_num, frame in trials.groupby("block_num", sort=True):
        onsets = pd.to_numeric(frame["absolute_onset_time"], errors="coerce").dropna()
        if onsets.empty:
            continue
        if "block_onset_time" in frame.columns:
            starts = pd.to_numeric(frame["block_onset_time"], errors="coerce").dropna()
        else:
            starts = pd.Series(dtype=float)
        start_ms = float(starts.min()) if not starts.empty else float(onsets.min())
        end_ms = float(onsets.max()) + _NOMINAL_TRIAL_MS
        result[int(block_num)] = (start_ms, end_ms)
    return result


def _estimate_sampling_rate_hz(times_ms: np.ndarray) -> float | None:
    times = np.asarray(times_ms, dtype=float)
    times = times[np.isfinite(times)]
    if times.size < 2:
        return None
    diffs = np.diff(np.sort(times))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    median_ms = float(np.median(diffs))
    return 1000.0 / median_ms if median_ms > 0 else None


def _max_temporal_gap_sec(
    times_ms: np.ndarray,
    available_start_ms: float,
    available_end_ms: float,
) -> float | None:
    duration_ms = available_end_ms - available_start_ms
    if not np.isfinite(duration_ms) or duration_ms <= 0:
        return None
    times = np.asarray(times_ms, dtype=float)
    mask = (
        np.isfinite(times)
        & (times >= available_start_ms)
        & (times < available_end_ms)
    )
    points = np.concatenate(
        (
            np.array([available_start_ms], dtype=float),
            np.sort(times[mask]),
            np.array([available_end_ms], dtype=float),
        )
    )
    if points.size < 2:
        return duration_ms / 1000.0
    return float(np.max(np.diff(points)) / 1000.0)


def augment_window_metadata(
    windows: pd.DataFrame,
    trials: pd.DataFrame,
    indices: dict[tuple[int, str], Any],
) -> pd.DataFrame:
    """Add schema-v2 availability metadata to trial/probe window rows.

    This separates design/boundary truncation from internal NIR missingness.
    The historical name ``oar_valid_fraction`` is replaced by
    ``oar_available_fraction`` because a finite OAR value is not a validated
    blink/eye-state quality label.
    """
    frame = windows.copy()
    if "oar_valid_fraction" in frame.columns:
        frame = frame.rename(columns={"oar_valid_fraction": "oar_available_fraction"})

    block_bounds = _block_analysis_bounds(trials)
    rate_by_key: dict[tuple[int, str], float | None] = {}
    for key, index in indices.items():
        rate_by_key[key] = _estimate_sampling_rate_hz(index.times_ms)

    metadata: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        block_num = int(row.block_num)
        eye = str(row.eye)
        requested_start = float(row.window_start_ms)
        requested_end = float(row.window_end_ms)
        requested_duration_sec = max(0.0, (requested_end - requested_start) / 1000.0)
        block_start, block_end = block_bounds.get(
            block_num, (requested_start, requested_end)
        )
        available_start = max(requested_start, block_start)
        available_end = min(requested_end, block_end)
        available_duration_sec = max(0.0, (available_end - available_start) / 1000.0)
        available_fraction = (
            available_duration_sec / requested_duration_sec
            if requested_duration_sec > 0
            else None
        )
        rate = rate_by_key.get((block_num, eye))
        expected_rows = (
            available_duration_sec * rate
            if rate is not None and available_duration_sec > 0
            else None
        )
        index = indices.get((block_num, eye))
        n_rows_available = (
            index.count(available_start, available_end)
            if index is not None and available_duration_sec > 0
            else 0
        )
        internal_coverage = (
            min(1.0, float(n_rows_available) / expected_rows)
            if expected_rows is not None and expected_rows > 0
            else None
        )
        max_gap = (
            _max_temporal_gap_sec(index.times_ms, available_start, available_end)
            if index is not None and available_duration_sec > 0
            else None
        )
        metadata.append(
            {
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
        )

    meta = pd.DataFrame(metadata, index=frame.index)
    return pd.concat([frame, meta], axis=1)


def run_subject_alignment_v12(
    config: Config,
    subject: str,
    *,
    force: bool = False,
    make_diagnostics: bool = True,
) -> dict[str, Any]:
    subject = normalize_subject(subject)
    output_root = alignment_output_root(config)
    paths = subject_output_paths(output_root, subject)
    source = find_nir_source(config, subject)
    trials = load_behavior_trials(config, subject)
    trials = add_behavior_qc(
        trials,
        carryover_ms=float(
            config.section("behavior_qc").get("carryover_candidate_ms", 200)
        ),
    )
    trials = _add_trial_context(trials)

    identity = _source_identity(config, source, trials)
    if not force and _completion_matches(paths["completion"], identity):
        return {
            "subject": subject,
            "status": "skipped",
            "reason": "validated_completion",
        }

    nir = load_nir_frame(source)
    indices = build_nir_indices(nir, subject)
    missing_eye_blocks = [
        f"block{block}-{eye}"
        for block in (1, 2)
        for eye in EYES
        if (block, eye) not in indices
    ]

    trial_specs = parse_window_specs(
        config.section("windows").get("trial", []), family="trial"
    )
    probe_specs = parse_window_specs(
        config.section("windows").get("probe", []), family="probe"
    )

    trial_level = add_trial_nir_linkage(trials, indices)
    trial_windows = augment_window_metadata(
        build_trial_windows(trial_level, indices, trial_specs),
        trial_level,
        indices,
    )
    probe_windows = augment_window_metadata(
        build_probe_windows(trial_level, indices, probe_specs),
        trial_level,
        indices,
    )
    trial_coverage = build_window_coverage_report(trial_windows, level="trial")
    probe_coverage = build_window_coverage_report(probe_windows, level="probe")

    paths["subject_dir"].mkdir(parents=True, exist_ok=True)
    trial_level.to_csv(paths["trial_level"], index=False, encoding="utf-8-sig")
    trial_windows.to_csv(paths["trial_windows"], index=False, encoding="utf-8-sig")
    probe_windows.to_csv(paths["probe_windows"], index=False, encoding="utf-8-sig")
    trial_coverage.to_csv(paths["trial_coverage"], index=False, encoding="utf-8-sig")
    probe_coverage.to_csv(paths["probe_coverage"], index=False, encoding="utf-8-sig")

    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "subject": subject,
        "created_at_utc": created_at,
        "identity": identity,
        "git_commit": _git_commit(config.path.parent.parent),
        "nir_source_run_dir": str(source.run_dir),
        "nir_source_alternatives": [str(path) for path in source.alternatives],
        "windows": {
            "trial": [spec.__dict__ for spec in trial_specs],
            "probe": [spec.__dict__ for spec in probe_specs],
        },
        "outputs": {
            "trial_level": str(paths["trial_level"]),
            "trial_windows": str(paths["trial_windows"]),
            "probe_windows": str(paths["probe_windows"]),
            "trial_coverage": str(paths["trial_coverage"]),
            "probe_coverage": str(paths["probe_coverage"]),
            "qc_dir": str(paths["qc_dir"]),
        },
        "principles": {
            "raw_behavior_scoring_overwritten": False,
            "raw_pupil_pixels_used_as_primary": False,
            "pir_primary": PIR_COLUMN,
            "pir_requires_normalization_valid": True,
            "oar_primary": OAR_COLUMN,
            "oar_available_fraction_is_quality_gate": False,
            "roi_clipped_is_exclusion_gate": False,
            "left_right_fused": False,
            "coverage_thresholds_applied": False,
            "boundary_truncation_separated_from_internal_missingness": True,
        },
    }
    _atomic_json(paths["manifest"], manifest)

    probe_count = int(
        (
            pd.to_numeric(trial_level["is_probe"], errors="coerce").eq(1)
            & pd.to_numeric(trial_level["probe_onset_time"], errors="coerce").notna()
        ).sum()
    )
    summary = {
        "subject": subject,
        "status": "complete",
        "trial_rows": int(len(trial_level)),
        "trial_window_rows": int(len(trial_windows)),
        "probe_count": probe_count,
        "probe_window_rows": int(len(probe_windows)),
        "nir_rows_block1_block2": int(len(nir)),
        "missing_eye_blocks": missing_eye_blocks,
        "behavior_qc": {
            "multiple_keypress_trials": int(trial_level["multiple_keypress_flag"].sum()),
            "prestimulus_press_trials": int(trial_level["prestimulus_press_flag"].sum()),
            "ambiguous_omission_trials": int(
                trial_level["ambiguous_omission_flag"].sum()
            ),
            "carryover_candidate_trials": int(
                trial_level["carryover_candidate_flag"].sum()
            ),
            "rt_lt_100_trials": int(trial_level["rt_candidate_lt_100_flag"].sum()),
            "rt_lt_150_trials": int(trial_level["rt_candidate_lt_150_flag"].sum()),
            "rt_lt_200_trials": int(trial_level["rt_candidate_lt_200_flag"].sum()),
        },
        "coverage": {
            "trial": {
                "path": str(paths["trial_coverage"]),
                **coverage_overview(trial_coverage),
            },
            "probe": {
                "path": str(paths["probe_coverage"]),
                **coverage_overview(probe_coverage),
            },
            "note": (
                "Descriptive only; no exclusion threshold is frozen. "
                "Boundary truncation and internal NIR coverage are separated."
            ),
        },
    }

    if make_diagnostics:
        from .diagnostics import generate_diagnostics

        diagnostics_config = config.section("diagnostics")
        summary["diagnostics"] = generate_diagnostics(
            subject,
            nir,
            trial_level,
            probe_windows,
            paths,
            max_line_gap_sec=float(diagnostics_config.get("max_line_gap_sec", 2.5)),
        )

    _atomic_json(paths["summary"], summary)
    _atomic_json(
        paths["completion"],
        {
            "status": "complete",
            "subject": subject,
            "identity": identity,
            "created_at_utc": created_at,
            "required_artifacts": [
                str(paths["trial_level"]),
                str(paths["trial_windows"]),
                str(paths["probe_windows"]),
                str(paths["trial_coverage"]),
                str(paths["probe_coverage"]),
                str(paths["manifest"]),
                str(paths["summary"]),
            ],
        },
    )
    return summary
