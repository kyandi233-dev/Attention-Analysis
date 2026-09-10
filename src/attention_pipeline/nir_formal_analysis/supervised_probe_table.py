from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from attention_pipeline.nir_formal_analysis.supervised_features import (
    ALLOWED_WINDOW_SEC,
    BASE_SIGNAL,
    PRIMARY_WINDOW_SEC,
    SUPERVISED_NIR_INTERFACE_VERSION,
    build_raw_binocular_timepoints,
    select_preprobe_window,
)
from attention_pipeline.nir_formal_analysis.supervised_support import (
    audit_supervised_window_support,
)


PROBE_ID_COLUMNS = (
    "participant_group_id",
    "session_id",
    "block_num",
    "probe_index_global",
)


def _probe_onset_ms(row: pd.Series) -> float:
    for name in ("probe_onset_ms", "window_end_ms", "absolute_onset_time"):
        if name in row.index:
            value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
            if np.isfinite(value):
                return float(value)
    raise ValueError("probe row missing finite onset time")


def _participant_id(row: pd.Series, timepoints: pd.DataFrame) -> str:
    if "participant_group_id" in row.index and pd.notna(row["participant_group_id"]):
        return str(row["participant_group_id"])
    if "participant_group_id" in timepoints.columns:
        values = timepoints["participant_group_id"].dropna().astype(str).unique()
        if len(values) == 1:
            return str(values[0])
    raise ValueError(
        "formal supervised NIR output requires participant_group_id; "
        "analysis_group_token is compatibility metadata only"
    )


def build_supervised_probe_table(
    analysis_ready_frame: pd.DataFrame,
    probes: pd.DataFrame,
    *,
    windows_sec: Iterable[int] = (PRIMARY_WINDOW_SEC,),
) -> pd.DataFrame:
    """Build one supervised NIR row per probe × requested window.

    The function performs no imputation, standardization, analysis-set creation,
    empirical feature screening, or model fitting.  It only projects the legal
    pre-probe raw pupil signal into candidate summaries plus support/QC metadata.
    """

    windows = tuple(int(x) for x in windows_sec)
    invalid = sorted(set(windows) - set(ALLOWED_WINDOW_SEC))
    if invalid:
        raise ValueError(f"unsupported supervised NIR windows: {invalid}")

    required_probe = {"session_id", "block_num", "probe_index_global"}
    missing = sorted(required_probe - set(probes.columns))
    if missing:
        raise ValueError(f"probe table missing identity columns: {missing}")

    timepoints = build_raw_binocular_timepoints(analysis_ready_frame)
    rows: list[dict[str, object]] = []
    for _, probe in probes.iterrows():
        session_id = str(probe["session_id"])
        block_num = int(probe["block_num"])
        onset_ms = _probe_onset_ms(probe)
        participant_group_id = _participant_id(probe, timepoints)

        for window_sec in windows:
            selected = select_preprobe_window(
                timepoints,
                session_id=session_id,
                block_num=block_num,
                probe_onset_ms=onset_ms,
                window_sec=window_sec,
            )
            requested_start = onset_ms - 1000.0 * window_sec
            audit = audit_supervised_window_support(
                selected,
                requested_start_ms=requested_start,
                requested_end_ms=onset_ms,
            )
            record: dict[str, object] = {
                "participant_group_id": participant_group_id,
                "session_id": session_id,
                "block_num": block_num,
                "probe_index_global": int(probe["probe_index_global"]),
                "probe_onset_ms": onset_ms,
                "window_sec": int(window_sec),
                "window_role": "primary" if window_sec == PRIMARY_WINDOW_SEC else "window_sensitivity",
                "interface_version": SUPERVISED_NIR_INTERFACE_VERSION,
                "base_signal": BASE_SIGNAL,
            }
            for optional in (
                "probe_index_in_block",
                "probe_event_id",
                "probe_response",
                "probe_vigilance",
                "q1_nominal_4class",
                "q2_ordinal_4level",
                "analysis_group_token",
            ):
                if optional in probe.index:
                    record[optional] = probe[optional]
            record.update(audit)
            rows.append(record)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    key = ["participant_group_id", "session_id", "block_num", "probe_index_global", "window_sec"]
    if result.duplicated(key, keep=False).any():
        raise ValueError(f"duplicate supervised NIR probe-window key: {key}")
    return result.sort_values(
        ["participant_group_id", "session_id", "block_num", "probe_index_global", "window_sec"],
        kind="stable",
    ).reset_index(drop=True)


def supervised_nir_manifest() -> dict[str, object]:
    return {
        "interface_version": SUPERVISED_NIR_INTERFACE_VERSION,
        "base_signal": BASE_SIGNAL,
        "base_signal_unit": "px",
        "primary_window_sec": PRIMARY_WINDOW_SEC,
        "sensitivity_windows_sec": [10, 20],
        "window_interval": "[probe-window, probe)",
        "zero_calibration": True,
        "session_level_baseline_used": False,
        "participant_within_between_used": False,
        "imputation_performed": False,
        "standardization_performed": False,
        "analysis_set_id_generated": False,
        "automatic_coverage_drop": False,
        "head_motion_sensitivity_status": "not_validated",
        "scale_sensitivity_status": "not_validated",
    }
