from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from attention_pipeline.behavior_formal.science_output import (
    PRIMARY,
    build_behavior_feature_handoff,
    build_behavior_science_output,
)


def _probe() -> pd.DataFrame:
    rows = []
    for i in range(12):
        rows.append({
            "repeat_participant_id": f"p{i % 4}",
            "session_id": f"sub-{i % 6:03d}",
            "go_correct_rt_mean_ms": 400 + i * 8,
            "go_correct_rt_median_ms": 395 + i * 7,
            "go_correct_rt_cv": 0.15 + i * 0.01,
            "go_correct_rt_theilsen_slope_ms_per_s": -1.0 + i * 0.2,
            "raw_go_omission_rate": 0.01 * (i % 4),
            "commission_rate": 0.02 * (i % 3),
            "go_correct_rt_sd_ms": 50 + i,
            "go_correct_rt_mad_ms": 35 + i,
            "go_correct_rt_iqr_ms": 60 + i,
            "dprime_loglinear": 2.0 + i * 0.05,
            "clean_go_omission_rate": 0.005 * (i % 4),
            "timing_ambiguous_go_omission_rate": 0.005 * (i % 4),
        })
    return pd.DataFrame(rows)


def test_behavior_handoff_keeps_only_rt_level_as_researcher_choice():
    handoff = build_behavior_feature_handoff(_probe())
    ready = set(handoff.loc[handoff["registry_ready"], "predictor_column"])
    assert ready == set(PRIMARY)
    pending = set(handoff.loc[handoff["researcher_freeze_required"], "candidate_representation_id"])
    assert pending == {"rt_level_mean", "rt_level_median"}
    cv = handoff.loc[handoff["predictor_column"].eq("go_correct_rt_cv")].iloc[0]
    assert "minimum n=2" in cv["estimability_rule"]
    assert cv["temporal_anchor"] == "probe_time_ms"
    assert cv["temporal_scope"] == "pre_probe_only"
    assert cv["time_legality_status"] == "verified_pre_probe_only"
    assert cv["time_legality_evidence"]
    dprime = handoff.loc[handoff["predictor_column"].eq("dprime_loglinear")].iloc[0]
    assert not bool(dprime["registry_ready"])
    assert dprime["report_role"] == "sensitivity"


def test_behavior_science_output_is_question_driven_not_cartesian(tmp_path):
    formal = tmp_path / "Behavior" / "formal_v3"
    formal.mkdir(parents=True)
    _probe().to_csv(formal / "probe_primary_30s.csv", index=False)
    q1 = pd.DataFrame([
        {"predictor": "go_correct_rt_cv", "contrast_category": 2, "estimate_per_predictor_sd": .2,
         "ci_low": -.1, "ci_high": .5, "participant_group_n": 4, "session_n": 6, "n_rows": 12},
        {"predictor": "commission_rate", "contrast_category": 2, "estimate_per_predictor_sd": -.3,
         "ci_low": -.6, "ci_high": .05, "participant_group_n": 4, "session_n": 6, "n_rows": 12},
    ])
    q1.to_csv(formal / "q1_nominal_models.csv", index=False)
    q2 = pd.DataFrame([
        {"predictor": "go_correct_rt_cv", "estimate_per_predictor_sd": .15, "ci_low": -.05, "ci_high": .35,
         "participant_group_n": 4, "session_n": 6, "n_rows": 12},
    ])
    q2.to_csv(formal / "q2_ordinal_gee_models.csv", index=False)
    pd.DataFrame([
        {"metric": "go_correct_rt_cv", "estimate_b2_minus_b1": .02, "ci_low": -.01, "ci_high": .05,
         "participant_group_n": 4, "session_pair_n": 6},
        {"metric": "raw_go_omission_rate", "estimate_b2_minus_b1": -.01, "ci_low": -.03, "ci_high": .01,
         "participant_group_n": 4, "session_pair_n": 6},
        {"metric": "commission_rate", "estimate_b2_minus_b1": .01, "ci_low": -.02, "ci_high": .04,
         "participant_group_n": 4, "session_pair_n": 6},
    ]).to_csv(formal / "b1_b2_participant_cluster_bootstrap.csv", index=False)

    manifest = build_behavior_science_output(formal, tmp_path / "FormalScience")
    root = tmp_path / "FormalScience" / "Behavior"
    figures = pd.read_csv(root / "manifests" / "figure_manifest.csv")
    assert 1 <= len(figures) <= 8
    assert set(figures["purpose"]).issubset({"main", "qualification", "qc", "sensitivity"})
    assert figures["scientific_question"].astype(str).str.len().gt(10).all()
    assert figures["internal_title"].eq(False).all()
    assert figures["legend_frame"].eq(False).all()
    assert (root / "figures" / "main").is_dir()
    assert (root / "figures" / "qualification").is_dir()
    assert (root / "figures" / "qc").is_dir()
    assert not (root / "feature_registry.yaml").exists()
    assert manifest["formal_registry_mutated"] is False
    assert manifest["rt_level_freeze_pending"] is True
    stored = json.loads((root / "manifests" / "science_output_manifest.json").read_text(encoding="utf-8"))
    assert stored["feature_selection_policy"].startswith("Q1/Q2 significance")
