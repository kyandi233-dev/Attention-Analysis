from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import pandas as pd

import attention_pipeline.behavior_formal.science_v3_figures as wrapper


def _probe() -> pd.DataFrame:
    rows = []
    for i in range(8):
        rows.append({
            "repeat_participant_id": f"p{i % 3}",
            "session_id": f"s{i % 4}",
            "go_correct_rt_mean_ms": 420 + i * 4,
            "go_correct_rt_median_ms": 415 + i * 3,
            "go_correct_rt_cv": .18 + i * .01,
            "go_correct_rt_theilsen_slope_ms_per_s": -.3 + i * .1,
            "raw_go_omission_rate": .01 * (i % 3),
            "commission_rate": .02 * (i % 2),
        })
    return pd.DataFrame(rows)


def test_publication_contract_retires_cartesian_metric_pack() -> None:
    contract = wrapper.publication_figure_contract()
    assert contract["internal_title_allowed"] is False
    assert contract["in_image_language"] == "English"
    assert contract["font_family"] == "Times New Roman"
    assert contract["legend_frame"] is False
    assert contract["question_driven_allowlist_required"] is True
    assert contract["cartesian_metric_by_scale_pack_allowed"] is False
    assert contract["metric_scale_coverage_audit_required"] is False
    assert contract["current_output_layer"] == "FormalScience/Behavior"


def test_default_runner_figure_entrypoint_routes_to_science_layer(tmp_path) -> None:
    probe = _probe()
    probe.to_csv(tmp_path / "probe_primary_30s.csv", index=False)
    files = wrapper.generate_behavior_figures(
        pd.DataFrame(), probe, tmp_path / "figures", error_summary=pd.DataFrame()
    )
    root = tmp_path / "science_output" / "Behavior"
    manifest = pd.read_csv(root / "manifests" / "figure_manifest.csv")
    coverage = pd.read_csv(root / "tables" / "behavior_feature_coverage.csv")
    assert 1 <= len(manifest) <= 8
    assert len(files) == 2 * len(manifest)
    assert set(manifest["purpose"]).issubset({"main", "qualification", "qc", "sensitivity"})
    assert manifest["scientific_question"].astype(str).str.len().gt(10).all()
    assert manifest["internal_title"].eq(False).all()
    assert manifest["legend_frame"].eq(False).all()
    assert set(coverage["predictor_column"]).issuperset({
        "go_correct_rt_mean_ms", "go_correct_rt_median_ms", "go_correct_rt_cv",
        "go_correct_rt_theilsen_slope_ms_per_s", "raw_go_omission_rate", "commission_rate",
    })
