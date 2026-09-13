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


def _finite_optional(row: pd.Series, names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in row.index:
            value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
            if np.isfinite(value):
                return float(value)
    return None


def _probe_onset_ms(row: pd.Series) -> float:
    """Resolve only fields whose semantics are explicitly probe-timed.

    ``probe_time_ms`` is the current Behavior authority. ``probe_onset_ms`` and
    ``window_end_ms`` are retained as explicit compatibility fields. SART trial
    ``absolute_onset_time`` is intentionally forbidden because it is stimulus onset,
    not Q1 probe onset.
    """
    value = _finite_optional(row, ("probe_time_ms", "probe_onset_ms", "window_end_ms"))
    if value is None:
        raise ValueError(
            "probe row missing finite explicit probe time (probe_time_ms/probe_onset_ms/window_end_ms)"
        )
    return value


def _unique_session_value(
    frame: pd.DataFrame,
    *,
    session_id: str,
    column: str,
) -> str | None:
    if column not in frame.columns:
        return None
    session = frame[frame["session_id"].astype(str).eq(str(session_id))]
    values = session[column].dropna().astype(str).str.strip()
    values = values[values.ne("")].unique()
    if len(values) > 1:
        raise ValueError(
            f"session {session_id} has conflicting {column} values in supervised NIR input: "
            f"{sorted(values.tolist())}"
        )
    return str(values[0]) if len(values) == 1 else None


def _probe_session_participant_map(probes: pd.DataFrame) -> dict[str, str]:
    if "participant_group_id" not in probes.columns:
        return {}
    mapping: dict[str, str] = {}
    for session_id, frame in probes.groupby("session_id", sort=False):
        values = frame["participant_group_id"].dropna().astype(str).str.strip()
        values = values[values.ne("")].unique()
        if len(values) > 1:
            raise ValueError(
                f"session {session_id} maps to multiple participant_group_id values: "
                f"{sorted(values.tolist())}"
            )
        if len(values) == 1:
            mapping[str(session_id)] = str(values[0])
    return mapping


def _participant_id(
    row: pd.Series,
    timepoints: pd.DataFrame,
    probe_session_map: dict[str, str],
) -> str:
    session_id = str(row["session_id"])
    explicit = None
    if "participant_group_id" in row.index and pd.notna(row["participant_group_id"]):
        value = str(row["participant_group_id"]).strip()
        explicit = value or None

    mapped = probe_session_map.get(session_id)
    nir_value = _unique_session_value(
        timepoints,
        session_id=session_id,
        column="participant_group_id",
    )

    candidates = [value for value in (explicit, mapped, nir_value) if value is not None]
    if len(set(candidates)) > 1:
        raise ValueError(
            f"participant_group_id mismatch for session {session_id}: {sorted(set(candidates))}"
        )
    if candidates:
        return candidates[0]
    raise ValueError(
        "formal supervised NIR output requires participant_group_id; "
        "analysis_group_token is compatibility metadata only"
    )


def _validate_compatibility_identity(row: pd.Series, timepoints: pd.DataFrame) -> None:
    if "analysis_group_token" not in row.index or pd.isna(row["analysis_group_token"]):
        return
    probe_token = str(row["analysis_group_token"]).strip()
    if not probe_token:
        return
    nir_token = _unique_session_value(
        timepoints,
        session_id=str(row["session_id"]),
        column="analysis_group_token",
    )
    if nir_token is not None and nir_token != probe_token:
        raise ValueError(
            f"analysis_group_token mismatch for session {row['session_id']}: "
            f"probe={probe_token}, nir={nir_token}"
        )


def _available_bounds(probe: pd.Series) -> tuple[float | None, float | None]:
    # Prefer behavior-defined block bounds because one probe table may be reused
    # for 10/20/30-s windows. Existing per-window available_* values are a safe
    # fallback; audit_supervised_window_support clamps them to each requested window.
    start = _finite_optional(probe, ("block_analysis_start_ms", "available_start_ms"))
    end = _finite_optional(probe, ("block_analysis_end_ms", "available_end_ms"))
    return start, end


def build_supervised_probe_table(
    analysis_ready_frame: pd.DataFrame,
    probes: pd.DataFrame,
    *,
    windows_sec: Iterable[int] = (PRIMARY_WINDOW_SEC,),
) -> pd.DataFrame:
    """Build one supervised NIR row per probe × requested window.

    The production input may be the accepted long-form candidate sidecar. The
    function performs no imputation, standardization, analysis-set creation,
    empirical feature screening, or model fitting.
    """

    windows = tuple(int(x) for x in windows_sec)
    invalid = sorted(set(windows) - set(ALLOWED_WINDOW_SEC))
    if invalid:
        raise ValueError(f"unsupported supervised NIR windows: {invalid}")

    required_probe = {"session_id", "block_num", "probe_index_global"}
    missing = sorted(required_probe - set(probes.columns))
    if missing:
        raise ValueError(f"probe table missing identity columns: {missing}")

    probe_session_map = _probe_session_participant_map(probes)
    timepoints = build_raw_binocular_timepoints(analysis_ready_frame)
    rows: list[dict[str, object]] = []
    for _, probe in probes.iterrows():
        session_id = str(probe["session_id"])
        block_num = int(probe["block_num"])
        onset_ms = _probe_onset_ms(probe)
        _validate_compatibility_identity(probe, timepoints)
        participant_group_id = _participant_id(probe, timepoints, probe_session_map)
        available_start_ms, available_end_ms = _available_bounds(probe)

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
                available_start_ms=available_start_ms,
                available_end_ms=available_end_ms,
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
        "production_input_schema": "formal_candidate_sidecar_long",
        "production_raw_column": f"{BASE_SIGNAL}__raw",
        "production_validity_column": f"{BASE_SIGNAL}__valid_primary",
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
