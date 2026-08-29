from __future__ import annotations

import pandas as pd

from attention_pipeline.nir_formal_analysis.adjustment_audit import build_adjustment_comparison


def _effects(covariates: str) -> pd.DataFrame:
    rows = []
    for term in ("pupil_within", "pupil_between"):
        rows.append({
            "model_name": "go_correct_rt__unadjusted",
            "outcome": "rt",
            "adjusted": False,
            "pupil_term": term,
            "estimate": 1.0,
            "ci_low": 0.5,
            "ci_high": 1.5,
            "participant_group_n": 38,
            "session_n": 44,
            "n_rows": 1000,
            "covariates": "",
        })
        rows.append({
            "model_name": "go_correct_rt__adjusted",
            "outcome": "rt",
            "adjusted": True,
            "pupil_term": term,
            "estimate": 0.8,
            "ci_low": 0.3,
            "ci_high": 1.3,
            "participant_group_n": 38,
            "session_n": 44,
            "n_rows": 980,
            "covariates": covariates,
        })
    return pd.DataFrame(rows)


def test_visual_adjustment_requires_visual_table_and_visual_covariate() -> None:
    out = build_adjustment_comparison(
        _effects("time_in_block_z;previous_central_rel_lum_mean"),
        visual_status={"status": "available"},
    )
    assert out["formal_visual_adjustment_status"].eq("estimable").all()
    assert out["adjustment_set"].eq("visual_time_quality_adjusted").all()


def test_nonvisual_adjustment_is_not_mislabeled_as_visual() -> None:
    out = build_adjustment_comparison(
        _effects("time_in_block_z;pupil_valid_fraction"),
        visual_status={"status": "available"},
    )
    assert out["formal_visual_adjustment_status"].eq("not_estimable").all()
    assert out["adjustment_set"].eq("time_quality_adjusted_only").all()


def test_missing_visual_table_blocks_formal_visual_claim() -> None:
    out = build_adjustment_comparison(
        _effects("time_in_block_z;previous_central_rel_lum_mean"),
        visual_status={"status": "unavailable"},
    )
    assert out["formal_visual_adjustment_status"].eq("not_estimable").all()
    assert out["visual_table_available"].eq(False).all()
