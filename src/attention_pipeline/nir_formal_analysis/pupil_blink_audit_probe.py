from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.nir_behavior.features import robust_binned_slope_per_sec
from attention_pipeline.nir_formal_analysis.pupil_blink_audit_config import probe_block, probe_onset_ms
from attention_pipeline.nir_formal_analysis.pupil_blink_measurement import (
    AUDIT_SCHEMA_VERSION,
    fixed_probe_bins,
    select_probe_window,
    signal_values_and_mask,
    summarize_probe_signal,
)


def _legacy_slope(window: pd.DataFrame, signal: str, cleaning_track: str) -> float | None:
    values, valid = signal_values_and_mask(window, signal, cleaning_track)
    times = pd.to_numeric(window["unix_ms"], errors="coerce")
    mask = valid & np.isfinite(times) & np.isfinite(values)
    return robust_binned_slope_per_sec(times[mask].to_numpy(float), values[mask].to_numpy(float))


def probe_identity(row: pd.Series) -> dict[str, object]:
    result: dict[str, object] = {
        "session_id": str(row["session_id"]),
        "block_num": probe_block(row),
        "probe_onset_ms": probe_onset_ms(row),
    }
    for name in (
        "participant_group_id", "analysis_group_token", "probe_index_global",
        "probe_index_in_block", "probe_event_id", "probe_response", "probe_vigilance",
        "q1_nominal_4class", "q2_ordinal_4level",
    ):
        if name in row.index:
            result[name] = row[name]
    return result


def append_probe_track(
    *,
    out_rows: list[dict[str, object]],
    trajectory_rows: list[dict[str, object]],
    timepoints: pd.DataFrame,
    probe: pd.Series,
    signals: tuple[str, ...],
    tracks: tuple[str, ...],
    bin_widths: tuple[float, ...],
    window_sec: float,
    buffer_id: str,
    buffer_pre_ms: float | None,
    buffer_post_ms: float | None,
) -> None:
    identity = probe_identity(probe)
    window = select_probe_window(
        timepoints,
        session_id=str(identity["session_id"]),
        block_num=int(identity["block_num"]),
        probe_onset_ms=float(identity["probe_onset_ms"]),
        window_sec=window_sec,
    )
    for signal in signals:
        for track in tracks:
            for bin_width in bin_widths:
                summary = summarize_probe_signal(
                    window, signal=signal, cleaning_track=track,
                    window_sec=window_sec, bin_width_sec=bin_width,
                )
                summary["legacy_robust_binned_slope_per_sec"] = _legacy_slope(window, signal, track)
                summary.update(identity)
                summary.update({
                    "window_sec": float(window_sec), "buffer_id": buffer_id,
                    "buffer_pre_ms": buffer_pre_ms, "buffer_post_ms": buffer_post_ms,
                    "audit_schema_version": AUDIT_SCHEMA_VERSION,
                    "formal_parameter_status": "measurement_audit_candidate_not_frozen",
                })
                out_rows.append(summary)

                bins = fixed_probe_bins(
                    window, signal=signal, window_sec=window_sec,
                    bin_width_sec=bin_width, cleaning_track=track,
                )
                for _, bin_row in bins.iterrows():
                    trajectory_rows.append({
                        **identity, "window_sec": float(window_sec), "signal": signal,
                        "cleaning_track": track, "buffer_id": buffer_id,
                        "buffer_pre_ms": buffer_pre_ms, "buffer_post_ms": buffer_post_ms,
                        "bin_width_sec": float(bin_width), "bin_index": int(bin_row["bin_index"]),
                        "bin_start_sec": float(bin_row["bin_start_sec"]),
                        "bin_end_sec": float(bin_row["bin_end_sec"]),
                        "bin_center_sec": float(bin_row["bin_center_sec"]),
                        "n_valid_samples": int(bin_row["n_valid_samples"]),
                        "bin_value_median": bin_row["bin_value_median"],
                        "formal_parameter_status": "measurement_audit_candidate_not_frozen",
                    })
