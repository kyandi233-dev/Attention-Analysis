from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.nir_formal_analysis.pupil_blink_measurement import (
    GEOMETRY_SIGNAL,
    RSEG_HARD_SIGNAL,
    RSEG_SOFT_SIGNAL,
    derive_eye_measurements,
)


def build_binocular_measurement_timepoints(frame: pd.DataFrame) -> pd.DataFrame:
    """Vectorized signal-specific binocular fusion for full-cohort audit runs."""
    eye = derive_eye_measurements(frame)
    key = ["session_id", "block", "unix_ms"]
    if "frame_idx" in eye.columns:
        key.append("frame_idx")
    duplicate_key = [*key, "eye"]
    if eye.duplicated(duplicate_key, keep=False).any():
        raise ValueError(f"duplicate eye-time key: {duplicate_key}")

    metadata_cols = [
        name
        for name in (
            "participant_group_id",
            "analysis_group_token",
            "subject",
            "phase",
            "phase_segment",
        )
        if name in eye.columns
    ]
    if metadata_cols:
        conflicts = eye.groupby(key, sort=False, dropna=False)[metadata_cols].nunique(dropna=True)
        if conflicts.gt(1).any().any():
            bad = conflicts.gt(1).stack()
            first = bad[bad].index[0]
            raise ValueError(f"conflicting {first[-1]} within binocular timepoint {first[:-1]}")

    signals = [GEOMETRY_SIGNAL, RSEG_HARD_SIGNAL]
    if RSEG_SOFT_SIGNAL in eye.columns:
        signals.append(RSEG_SOFT_SIGNAL)

    value_fields: list[str] = []
    for signal in signals:
        raw_value_col = signal if signal == GEOMETRY_SIGNAL else f"{signal}__raw_value"
        value_fields.extend(
            [
                raw_value_col,
                signal,
                f"{signal}__raw_computable",
                f"{signal}__audit_valid",
                f"{signal}__audit_invalid_reason",
            ]
        )
    value_fields = list(dict.fromkeys(value_fields))
    wide = eye.pivot(index=key, columns="eye", values=value_fields)
    out = wide.index.to_frame(index=False)

    if metadata_cols:
        metadata = (
            eye.groupby(key, sort=False, dropna=False)[metadata_cols]
            .first()
            .reindex(wide.index)
            .reset_index(drop=True)
        )
        out = pd.concat([out.reset_index(drop=True), metadata], axis=1)

    def column(field: str, eye_name: str, *, boolean: bool = False) -> pd.Series:
        column_key = (field, eye_name)
        if column_key not in wide.columns:
            return pd.Series(
                False if boolean else np.nan,
                index=range(len(wide)),
                dtype=bool if boolean else float,
            )
        series = wide[column_key].reset_index(drop=True)
        if boolean:
            return series.astype("boolean").fillna(False).astype(bool)
        return series

    def fuse(
        left: pd.Series,
        right: pd.Series,
        left_ok: pd.Series,
        right_ok: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        both = left_ok & right_ok
        left_only = left_ok & ~right_ok
        right_only = ~left_ok & right_ok
        values = np.select(
            [both, left_only, right_only],
            [(left + right) / 2.0, left, right],
            default=np.nan,
        ).astype(float)
        modes = np.select(
            [both, left_only, right_only],
            ["binocular", "left_only", "right_only"],
            default="missing",
        )
        return values, modes

    for signal in signals:
        raw_value_col = signal if signal == GEOMETRY_SIGNAL else f"{signal}__raw_value"
        left_raw = pd.to_numeric(column(raw_value_col, "left"), errors="coerce")
        right_raw = pd.to_numeric(column(raw_value_col, "right"), errors="coerce")
        left_qc = pd.to_numeric(column(signal, "left"), errors="coerce")
        right_qc = pd.to_numeric(column(signal, "right"), errors="coerce")
        left_raw_ok = column(f"{signal}__raw_computable", "left", boolean=True)
        right_raw_ok = column(f"{signal}__raw_computable", "right", boolean=True)
        left_valid = column(f"{signal}__audit_valid", "left", boolean=True)
        right_valid = column(f"{signal}__audit_valid", "right", boolean=True)

        raw_fused, raw_mode = fuse(left_raw, right_raw, left_raw_ok, right_raw_ok)
        qc_fused, qc_mode = fuse(left_qc, right_qc, left_valid, right_valid)
        out[f"{signal}__raw"] = raw_fused
        out[f"{signal}__raw_source_mode"] = raw_mode
        out[signal] = qc_fused
        out[f"{signal}__source_mode"] = qc_mode

        left_reason_key = (f"{signal}__audit_invalid_reason", "left")
        right_reason_key = (f"{signal}__audit_invalid_reason", "right")
        left_reason = (
            wide[left_reason_key].reset_index(drop=True).astype("string").fillna("eye_row_missing")
            if left_reason_key in wide.columns
            else pd.Series("eye_row_missing", index=range(len(wide)), dtype="string")
        )
        right_reason = (
            wide[right_reason_key].reset_index(drop=True).astype("string").fillna("eye_row_missing")
            if right_reason_key in wide.columns
            else pd.Series("eye_row_missing", index=range(len(wide)), dtype="string")
        )
        out[f"{signal}__left_invalid_reason"] = left_reason.to_numpy()
        out[f"{signal}__right_invalid_reason"] = right_reason.to_numpy()

    return out.sort_values(key, kind="stable").reset_index(drop=True)


def audit_binocular_source_modes(timepoints: pd.DataFrame) -> pd.DataFrame:
    """Summarize binocular/monocular/missing source composition without thresholds."""
    modes = ("binocular", "left_only", "right_only", "missing")
    signals = [GEOMETRY_SIGNAL, RSEG_HARD_SIGNAL]
    if f"{RSEG_SOFT_SIGNAL}__source_mode" in timepoints.columns:
        signals.append(RSEG_SOFT_SIGNAL)
    rows: list[dict[str, object]] = []
    for session_id, group in timepoints.groupby("session_id", sort=True):
        for signal in signals:
            for measurement_state, column in (
                ("raw_computable", f"{signal}__raw_source_mode"),
                ("nir_qc_valid", f"{signal}__source_mode"),
            ):
                counts = group[column].fillna("missing").astype(str).value_counts()
                total = int(len(group))
                for mode in modes:
                    count = int(counts.get(mode, 0))
                    rows.append(
                        {
                            "session_id": str(session_id),
                            "signal": signal,
                            "measurement_state": measurement_state,
                            "source_mode": mode,
                            "n_timepoints": count,
                            "fraction": float(count / total) if total else np.nan,
                        }
                    )
    return pd.DataFrame(rows)
