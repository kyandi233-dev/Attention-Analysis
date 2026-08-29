from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.nir_analysis_ready.candidate_metrics import (
    PUPIL_CANDIDATE_METRICS,
    apply_candidate_standardization,
    compute_candidate_baselines,
)
from attention_pipeline.nir_formal_analysis.candidate_validation import (
    add_within_between,
    summarize_session_block_candidates,
)


def _adapted_session(session: str, group: str, offset: float = 0.0) -> pd.DataFrame:
    rows = []
    frame_idx = 0
    for phase in ("block1", "block2"):
        for eye, eye_shift in (("left", 0.0), ("right", 1.0)):
            for i in range(4):
                frame_idx += 1
                diameter = 20.0 + offset + eye_shift + i
                rows.append({
                    "session_id": session,
                    "subject": session,
                    "analysis_group_token": group,
                    "repeat_group_size": 1,
                    "is_repeat_session": False,
                    "phase": phase,
                    "phase_segment": 0,
                    "frame_idx": frame_idx,
                    "eye": eye,
                    "eye_raw": f"frame_{eye}",
                    "unix_ms": 1000.0 * frame_idx,
                    "video_time_ms": 1000.0 * frame_idx,
                    "phase_time_ms": 1000.0 * i,
                    "pupil_valid_primary": True,
                    "pupil_valid_strict": True,
                    "quality_track": "observed",
                    "roi_clipped": False,
                    "temporal_flagged": False,
                    "pupil_geom_mean_diameter": diameter,
                    "pupil_equivalent_diameter": diameter + 0.2,
                    "pupil_axis_a": diameter - 2.0,
                    "pupil_axis_b": diameter + 2.0,
                    "pupil_contour_area": 300.0 + diameter,
                    "pupil_ellipse_area": 310.0 + diameter,
                    "hard_pupil_fraction": 0.10 + i * 0.01,
                    "soft_pupil_fraction": 0.12 + i * 0.01,
                })
    return pd.DataFrame(rows)


def test_candidate_registry_is_pupil_only() -> None:
    assert "pupil_geom_mean_diameter" in PUPIL_CANDIDATE_METRICS
    assert "pupil_equivalent_diameter" in PUPIL_CANDIDATE_METRICS
    assert "pupil_contour_area" in PUPIL_CANDIDATE_METRICS
    assert "hard_pupil_fraction" in PUPIL_CANDIDATE_METRICS
    assert all("iris" not in name.lower() for name in PUPIL_CANDIDATE_METRICS)
    assert all("pir" not in name.lower() for name in PUPIL_CANDIDATE_METRICS)


def test_candidate_baselines_are_session_eye_specific_and_centered() -> None:
    first = _adapted_session("s1", "p1", 0.0)
    second = _adapted_session("s2", "p1", 10.0)
    frame = pd.concat([first, second], ignore_index=True)
    baselines = compute_candidate_baselines(frame)
    assert len(baselines) == 4  # 2 sessions x 2 eyes
    s1_left = baselines[(baselines["session_id"] == "s1") & (baselines["eye"] == "left")].iloc[0]
    s2_left = baselines[(baselines["session_id"] == "s2") & (baselines["eye"] == "left")].iloc[0]
    assert s2_left["pupil_geom_mean_diameter__primary__median"] > s1_left["pupil_geom_mean_diameter__primary__median"]

    standardized = apply_candidate_standardization(frame, baselines)
    for (session, eye), current in standardized.groupby(["session_id", "eye"]):
        centered = pd.to_numeric(
            current["pupil_geom_mean_diameter__centered_primary"], errors="coerce"
        ).dropna()
        assert np.isclose(float(centered.median()), 0.0)


def test_fraction_candidate_out_of_domain_is_rejected_without_changing_base_quality() -> None:
    frame = _adapted_session("s1", "p1")
    frame.loc[frame.index[0], "hard_pupil_fraction"] = 1.5
    baselines = compute_candidate_baselines(frame)
    out = apply_candidate_standardization(frame, baselines)
    assert bool(out.loc[out.index[0], "pupil_valid_primary"])
    assert not bool(out.loc[out.index[0], "hard_pupil_fraction__valid_primary"])
    assert np.isnan(out.loc[out.index[0], "hard_pupil_fraction__centered_primary"])


def test_candidate_summary_and_within_between_keep_group_structure() -> None:
    s1 = _adapted_session("s1", "p1", 0.0)
    s2 = _adapted_session("s2", "p1", 5.0)
    s3 = _adapted_session("s3", "p2", 20.0)
    frame = pd.concat([s1, s2, s3], ignore_index=True)
    baselines = compute_candidate_baselines(frame)
    sidecar = apply_candidate_standardization(frame, baselines)
    summary = summarize_session_block_candidates(sidecar)
    assert {"binocular_raw_median", "binocular_centered_median"}.issubset(summary.columns)
    assert set(summary["analysis_group_token"]) == {"p1", "p2"}

    decomposed = add_within_between(summary)
    p1 = decomposed[
        (decomposed["analysis_group_token"] == "p1")
        & (decomposed["metric"] == "pupil_geom_mean_diameter")
    ]
    assert p1["within_participant_status"].eq("estimable").all()
    assert np.isclose(float(p1["within_participant_deviation"].sum()), 0.0)
