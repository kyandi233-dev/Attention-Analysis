from pathlib import Path

import pandas as pd

from attention_pipeline.nir_formal_analysis.ocular_science_coverage import refresh_ocular_coverage
from attention_pipeline.nir_formal_analysis.ocular_science_freeze import build_frozen_ocular_science_output
from attention_pipeline.nir_formal_analysis.ocular_science_output import RSEG_HARD


def test_frozen_span_masks_dynamic_values(tmp_path: Path) -> None:
    g1 = tmp_path / "g1.csv"
    pd.DataFrame([{
        "session_id":"sub-001","participant_group_id":"p01","block_num":1,"probe_index_in_block":1,
        "signal":RSEG_HARD,"cleaning_track":"rgb_plus_nir_qc","buffer_id":"pre200_post200","bin_width_sec":2.0,
        "level_mean":0.4,"level_median":0.4,"variability_sd":0.02,"variability_mad":0.01,
        "linear_slope_per_sec":0.001,"quadratic_curvature_per_sec2":0.0001,"linear_status":"computable",
        "quadratic_status":"computable","temporal_span_sec":18.0,"has_early_support":True,"has_late_support":True,
    }]).to_csv(g1,index=False)
    rgb = tmp_path / "rgb.csv"
    pd.DataFrame([{"session_id":"sub-001","participant_group_id":"p01","block_id":"B1","probe_index_in_block":1,"blink_event_rate_per_min":4.0}]).to_csv(rgb,index=False)
    temporal = tmp_path / "temporal.csv"
    pd.DataFrame([{"signal":RSEG_HARD,"cleaning_track":"rgb_plus_nir_qc","buffer_id":"pre200_post200","bin_width_sec":2.0,"linear_span_ge_20s_fraction":0.86,"quadratic_span_ge_20s_fraction":0.86}]).to_csv(temporal,index=False)
    manifest = build_frozen_ocular_science_output(g1,tmp_path/"FormalScience",rgb_probe_features_path=rgb,temporal_support_summary_path=temporal)
    root = tmp_path / "FormalScience/Ocular"
    refresh_ocular_coverage(root)
    wide = pd.read_csv(root/"tables/ocular_probe_features_wide.csv")
    assert pd.notna(wide.loc[0,"ocular__rseg_hard__rgb_nir_qc__level_mean__2s"])
    assert pd.isna(wide.loc[0,"ocular__rseg_hard__rgb_nir_qc__linear_slope_per_sec__2s"])
    assert pd.isna(wide.loc[0,"ocular__rseg_hard__rgb_nir_qc__quadratic_curvature_per_sec2__2s"])
    assert manifest["technical_freeze"]["minimum_temporal_span_sec"] == 20.0
    assert manifest["temporal_support"]["formal_minimum_span_frozen"] is True
    assert manifest["temporal_support"]["formal_minimum_span_sec"] == 20.0
    assert manifest["temporal_support"]["status"] == "evidence_included_threshold_frozen"
    assert manifest["final_feature_registry_mutated"] is False
