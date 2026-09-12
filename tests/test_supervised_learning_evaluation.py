from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.supervised_learning.evaluation import (
    EvaluationContractError,
    evaluate_prediction_archive,
    fixed_oof_participant_bootstrap,
    paired_log_loss_increment,
    participant_log_loss,
)


def _model_archive(model_id: str, probabilities: dict[str, list[float]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for participant_index, (participant, probs) in enumerate(probabilities.items()):
        for probe_index, probability in enumerate(probs, start=1):
            q1_binary = probe_index % 2
            rows.append(
                {
                    "participant_group_id": participant,
                    "session_id": f"S-{participant_index:02d}",
                    "block_id": "B1",
                    "probe_event_id": f"{participant}|P{probe_index:02d}",
                    "analysis_set_id": "same-set",
                    "membership_type": "included_missing_aware",
                    "model_id": model_id,
                    "outer_fold_group": participant,
                    "q1_binary": q1_binary,
                    "p_q1_equals_1": float(probability),
                    "model_failed": False,
                }
            )
    return pd.DataFrame(rows)


def test_participant_equal_outer_loss_differs_from_pooled_probe_loss_when_counts_differ() -> None:
    archive = _model_archive(
        "m1",
        {
            "A": [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
            "B": [0.1, 0.9],
        },
    )
    participant, overall = participant_log_loss(archive)

    assert len(participant) == 2
    assert set(participant["membership_type"]) == {"included_missing_aware"}
    assert overall["membership_type"] == "included_missing_aware"
    assert participant.loc[participant["participant_group_id"].eq("A"), "n_probes"].item() == 8
    assert participant.loc[participant["participant_group_id"].eq("B"), "n_probes"].item() == 2
    assert overall["participant_equal_log_loss"] != pytest.approx(
        overall["pooled_probe_log_loss_descriptive"]
    )
    assert overall["participant_equal_log_loss"] == pytest.approx(participant["mean_log_loss"].mean())


def test_fixed_oof_bootstrap_is_reproducible_and_does_not_retrain() -> None:
    values = np.array([0.2, 0.4, 0.8, 1.0], dtype=float)
    first = fixed_oof_participant_bootstrap(values, replicates=1000, seed=20260830)
    second = fixed_oof_participant_bootstrap(values, replicates=1000, seed=20260830)

    assert first == second
    assert first["method"] == "fixed_oof_participant_cluster_percentile"
    assert first["retrain_within_bootstrap"] is False
    assert first["resampling_unit"] == "participant"
    assert first["replicates"] == 1000
    assert first["n_valid_replicates"] == 1000
    assert first["point_estimate"] == pytest.approx(values.mean())
    assert first["ci_lower"] <= first["point_estimate"] <= first["ci_upper"]


def test_archive_evaluation_marks_failed_model_not_estimable_without_dropping_rows() -> None:
    good = _model_archive("good", {"A": [0.8, 0.2], "B": [0.7, 0.3]})
    bad = _model_archive("bad", {"A": [0.8, 0.2], "B": [0.7, 0.3]})
    bad.loc[bad["participant_group_id"].eq("B"), "model_failed"] = True
    bad.loc[bad["participant_group_id"].eq("B"), "p_q1_equals_1"] = np.nan
    combined = pd.concat([good, bad], ignore_index=True)

    result = evaluate_prediction_archive(combined, replicates=50, seed=3)
    scores = result.model_scores.set_index("model_id")

    assert scores.loc["good", "status"] == "estimable"
    assert scores.loc["bad", "status"] == "not_estimable"
    assert "failed OOF rows" in scores.loc["bad", "reason"]
    assert set(result.participant_scores["model_id"]) == {"good"}
    assert {record["model_id"] for record in result.bootstrap_records} == {"good"}


def test_archive_evaluation_accepts_unambiguous_serialized_boolean_values() -> None:
    archive = _model_archive("m1", {"A": [0.8, 0.2], "B": [0.7, 0.3]})
    archive["model_failed"] = "False"
    participant, overall = participant_log_loss(archive)
    assert len(participant) == 2
    assert np.isfinite(overall["participant_equal_log_loss"])


def test_paired_increment_uses_exact_common_probe_set_and_positive_means_improvement() -> None:
    baseline = _model_archive(
        "behavior",
        {
            "A": [0.6, 0.4, 0.6, 0.4],
            "B": [0.6, 0.4],
            "C": [0.6, 0.4, 0.6],
        },
    )
    added = _model_archive(
        "behavior_plus_x",
        {
            "A": [0.9, 0.1, 0.9, 0.1],
            "B": [0.9, 0.1],
            "C": [0.9, 0.1, 0.9],
        },
    )
    result = paired_log_loss_increment(baseline, added, replicates=200, seed=20260830)

    assert result.overall_increment > 0
    assert result.membership_type == "included_missing_aware"
    assert result.bootstrap["paired_model_resampling"] is True
    assert result.bootstrap["increment_definition"] == "baseline_log_loss_minus_added_log_loss"
    assert result.bootstrap["point_estimate"] == pytest.approx(result.overall_increment)
    assert set(result.participant_increments["participant_group_id"]) == {"A", "B", "C"}


def test_paired_increment_rejects_different_analysis_set_membership_or_probe_membership() -> None:
    baseline = _model_archive("behavior", {"A": [0.6, 0.4], "B": [0.6, 0.4]})
    added = _model_archive("behavior_plus_x", {"A": [0.8, 0.2], "B": [0.8, 0.2]})

    wrong_set = added.copy()
    wrong_set["analysis_set_id"] = "other-set"
    with pytest.raises(EvaluationContractError, match="same analysis_set_id"):
        paired_log_loss_increment(baseline, wrong_set)

    wrong_membership = added.copy()
    wrong_membership["membership_type"] = "included_complete"
    with pytest.raises(EvaluationContractError, match="same membership_type"):
        paired_log_loss_increment(baseline, wrong_membership)

    missing_probe = added.iloc[:-1].copy()
    with pytest.raises(EvaluationContractError, match="probe count mismatch"):
        paired_log_loss_increment(baseline, missing_probe)


def test_paired_increment_rejects_outer_fold_or_label_mismatch() -> None:
    baseline = _model_archive("behavior", {"A": [0.6, 0.4], "B": [0.6, 0.4]})
    added = _model_archive("behavior_plus_x", {"A": [0.8, 0.2], "B": [0.8, 0.2]})

    wrong_fold = added.copy()
    wrong_fold.loc[0, "outer_fold_group"] = "B"
    with pytest.raises(EvaluationContractError, match="outer_fold_group"):
        paired_log_loss_increment(baseline, wrong_fold)

    wrong_label = added.copy()
    wrong_label.loc[0, "q1_binary"] = 1 - int(wrong_label.loc[0, "q1_binary"])
    with pytest.raises(EvaluationContractError, match="q1_binary"):
        paired_log_loss_increment(baseline, wrong_label)
