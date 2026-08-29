import numpy as np
import pandas as pd
import pytest

from attention_pipeline.behavior_formal.reporting_contract import (
    ReportingContractError,
    assert_participant_disjoint,
    assert_single_forest_panel,
    binary_prediction_report,
    label_dependency_relationships,
    qc_count_table,
    validate_candidate_evidence,
)


def test_nine_to_one_accuracy_reports_majority_baseline_and_imbalance():
    y = [0] * 9 + [1]
    pred = [0] * 10
    score = [0.1] * 10
    report = binary_prediction_report(y, y_pred=pred, score=score)
    assert report["majority_accuracy"] == pytest.approx(0.9)
    assert report["accuracy"] == pytest.approx(0.9)
    assert report["balanced_accuracy"] == pytest.approx(0.5)
    assert report["accuracy_minus_majority"] == pytest.approx(0.0)
    assert report["class_imbalance_warning"] is True
    assert "auroc" in report and "pr_auc" in report


def test_participant_disjoint_audit_rejects_one_group_in_two_folds():
    rows = pd.DataFrame({
        "repeat_participant_id": ["p1", "p1", "p2"],
        "fold_id": [0, 1, 1],
    })
    with pytest.raises(ReportingContractError, match="multiple outer folds"):
        assert_participant_disjoint(rows)


def test_forest_axis_rejects_millisecond_and_proportion_mix():
    rows = pd.DataFrame([
        {"observation_unit": "probe", "estimate_unit": "ms"},
        {"observation_unit": "probe", "estimate_unit": "proportion"},
    ])
    with pytest.raises(ReportingContractError, match="cannot be plotted on one forest axis"):
        assert_single_forest_panel(rows)


def test_candidate_matrix_requires_sources_weights_and_prespecified_rule():
    bad = pd.DataFrame({"candidate": ["rt_cv"], "status": [1]})
    with pytest.raises(ReportingContractError, match="missing"):
        validate_candidate_evidence(bad)
    good = pd.DataFrame([
        {
            "candidate": "rt_cv",
            "evidence_source": "prespecified_behavior_v3_contract",
            "evidence_weight": 1.0,
            "prespecified_rule": "retain_if_available_and_nonredundant",
            "status": "candidate",
        }
    ])
    assert len(validate_candidate_evidence(good)) == 1


def test_qc_counts_always_carry_denominator():
    qc = qc_count_table({"included_sessions": (44, "registered_sessions", 44)})
    row = qc.iloc[0]
    assert row["denominator_name"] == "registered_sessions"
    assert row["denominator_count"] == 44
    assert row["fraction"] == pytest.approx(1.0)


def test_mathematical_complements_are_not_labeled_psychological_relationships():
    labels = label_dependency_relationships([
        ("omission_rate", "go_accuracy"),
        ("go_correct_rt_cv", "q2_ordinal_4level"),
    ])
    assert labels.iloc[0]["relationship_role"] == "measurement_mathematical_dependency"
    assert labels.iloc[1]["relationship_role"] == "empirical_association_requires_interpretation"
