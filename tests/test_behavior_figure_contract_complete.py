from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import pandas as pd

import attention_pipeline.behavior_formal.science_v3_figures as wrapper


def _probe() -> pd.DataFrame:
    return pd.DataFrame({
        "repeat_participant_id": ["p1", "p1", "p2", "p2"],
        "session_id": ["s1", "s1", "s2", "s2"],
        "go_correct_rt_mean_ms": [420, 430, 410, 440],
        "go_correct_rt_median_ms": [415, 422, 408, 432],
        "go_correct_rt_cv": [.20, .22, .18, .25],
        "go_correct_rt_theilsen_slope_ms_per_s": [-.4, .2, .1, -.1],
        "raw_go_omission_rate": [.01, .02, .00, .03],
        "commission_rate": [.02, .01, .03, .00],
    })


def test_publication_contract_retires_cartesian_metric_pack() -> None:
    contract = wrapper.publication_figure_contract()
    assert contract["internal_title_allowed"] is False
    assert contract["in_image_language"] == "English"
    assert contract["font_family"] == "Times New Roman"
    assert contract["legend_frame"] is False
    assert contract["question_driven_allowlist_required"] is True
    assert contract["cartesian_metric_by_scale_pack_allowed"] is False
    assert contract["metric_scale_coverage_audit_required"] is False


def test_default_runner_figure_entrypoint_is_compact_and_question_driven(tmp_path) -> None:
    files = wrapper.generate_behavior_figures(
        pd.DataFrame(), _probe(), tmp_path / "figures", error_summary=pd.DataFrame()
    )
    manifest = pd.read_csv(tmp_path / "behavior_figure_manifest.csv")
    coverage = pd.read_csv(tmp_path / "behavior_figure_coverage_audit.csv")
    assert len(manifest) <= 5
    assert len(files) == 2 * len(manifest)
    assert set(manifest["purpose"]).issubset({"qualification", "qc"})
    assert manifest["scientific_question"].astype(str).str.len().gt(10).all()
    assert manifest["internal_title"].eq(False).all()
    assert manifest["legend_frame"].eq(False).all()
    assert set(coverage["predictor_column"]).issuperset({
        "go_correct_rt_mean_ms", "go_correct_rt_median_ms", "go_correct_rt_cv",
        "go_correct_rt_theilsen_slope_ms_per_s", "raw_go_omission_rate", "commission_rate",
    })
