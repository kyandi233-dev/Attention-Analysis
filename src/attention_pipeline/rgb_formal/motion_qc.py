from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CONTEXT_COLUMNS = (
    "phase", "block", "trial_num", "cycle_num", "behavior_state", "trial_type",
    "is_probe", "probe_onset_time", "absolute_onset_time", "probe_response",
    "probe_vigilance",
)


def derive_motion_qc(motion: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build lightweight Motion Energy QC without conflating exposure change with body motion.

    The preserved motion producer remains the source of truth. This downstream layer only
    projects existing scalar fields and gives body-motion and exposure-change measurements
    separate names/statuses. No outcome- or mmWave-tuned threshold is introduced here.
    """
    if motion.empty:
        return pd.DataFrame(), {
            "status": "not_estimable",
            "reason": "motion_source_empty",
            "body_motion_status": "not_estimable",
            "exposure_control_status": "not_estimable",
        }
    if "unix_ms" not in motion.columns:
        raise ValueError("motion raw missing unix_ms")

    keep = [
        c
        for c in (
            "subject", "unix_ms", "video_frame_position", "capture_frame_idx", *CONTEXT_COLUMNS,
            "dt_ms", "gap_before", "irregular_dt", "motion_valid", "gray_mean",
            "gray_mean_delta", "changed_pixel_ratio", "global_motion_energy",
            "global_motion_energy_per_sec",
        )
        if c in motion.columns
    ]
    out = motion[keep].copy().sort_values("unix_ms").reset_index(drop=True)

    if "global_motion_energy_per_sec" in out.columns:
        body_raw = pd.to_numeric(out["global_motion_energy_per_sec"], errors="coerce")
        body_source = "global_motion_energy_per_sec"
    elif "global_motion_energy" in out.columns:
        body_raw = pd.to_numeric(out["global_motion_energy"], errors="coerce")
        body_source = "global_motion_energy"
    else:
        body_raw = pd.Series(np.nan, index=out.index, dtype=float)
        body_source = None

    producer_valid = pd.Series(True, index=out.index, dtype=bool)
    if "motion_valid" in out.columns:
        producer_valid &= out["motion_valid"].fillna(False).astype(bool)
    temporal_valid = pd.Series(True, index=out.index, dtype=bool)
    if "gap_before" in out.columns:
        temporal_valid &= ~out["gap_before"].fillna(False).astype(bool)
    if "irregular_dt" in out.columns:
        temporal_valid &= ~out["irregular_dt"].fillna(False).astype(bool)
    body_valid = body_raw.notna() & producer_valid & temporal_valid
    out["body_motion_energy"] = body_raw.where(body_valid)

    if "gray_mean_delta" in out.columns:
        exposure_raw = pd.to_numeric(out["gray_mean_delta"], errors="coerce")
        exposure_source = "gray_mean_delta"
    else:
        exposure_raw = pd.Series(np.nan, index=out.index, dtype=float)
        exposure_source = None
    exposure_valid = exposure_raw.notna() & temporal_valid
    out["exposure_change_signed"] = exposure_raw.where(exposure_valid)
    out["exposure_change_abs"] = exposure_raw.abs().where(exposure_valid)

    out["motion_producer_valid"] = producer_valid
    out["motion_temporal_valid"] = temporal_valid
    out["body_motion_observable"] = body_valid
    out["exposure_change_observable"] = exposure_valid
    out["motion_exposure_jointly_observable"] = body_valid & exposure_valid
    out["motion_exposure_separation_contract"] = "separate_tracks_no_combined_risk_score"

    return out, {
        "status": "generated" if body_valid.any() else "not_estimable",
        "reason": "" if body_valid.any() else "motion_energy_column_missing_or_all_invalid",
        "body_motion_status": "generated" if body_valid.any() else "not_estimable",
        "body_motion_source": body_source,
        "body_motion_valid_rows": int(body_valid.sum()),
        "producer_invalid_rows": int((~producer_valid).sum()),
        "temporal_invalid_rows": int((~temporal_valid).sum()),
        "exposure_control_status": "generated" if exposure_valid.any() else "not_estimable",
        "exposure_control_reason": "" if exposure_valid.any() else "gray_mean_delta_missing_or_all_invalid",
        "exposure_control_source": exposure_source,
        "exposure_control_valid_rows": int(exposure_valid.sum()),
        "combined_risk_score_generated": False,
    }
