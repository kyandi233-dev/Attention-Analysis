from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import attention_pipeline.supervised_learning.runner as supervised_runner
from attention_pipeline.supervised_learning.feature_schemes import FeatureScheme
from attention_pipeline.supervised_learning.runner import run_nested_loso
from attention_pipeline.supervised_learning.task import SupervisedLearningContractError


def _probe_frame(seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for group_index in range(6):
        for visit in range(2):
            session = f"S-{group_index:02d}-{visit}"
            for probe in range(4):
                q1 = 1 if probe % 2 == 0 else 2 + (group_index + probe) % 3
                y = int(q1 == 1)
                probe_time = 1_700_000_000_000 + group_index * 100000 + visit * 10000 + probe * 1000
                rows.append(
                    {
                        "participant_group_id": f"P-{group_index:02d}",
                        "session_id": session,
                        "block_id": "B1" if probe < 2 else "B2",
                        "probe_event_id": f"{session}|probe|{probe}",
                        "probe_id": f"probe-{probe + 1:02d}",
                        "probe_order_in_block": probe % 2 + 1,
                        "probe_index_in_block": probe % 2 + 1,
                        "probe_index_global": probe + 1,
                        "probe_time_ms": probe_time,
                        "probe_onset_unix_ms": probe_time,
                        "window_name": "pre_30s",
                        "window_start_unix_ms": probe_time - 30000,
                        "window_effective_start_unix_ms": probe_time - 30000,
                        "window_end_unix_ms": probe_time,
                        "analysis_set_id": "synthetic-common-set",
                        "membership_type": "included_missing_aware",
                        "q1_nominal_4class": q1,
                        "signal": 3.0 * y + rng.normal(0, 0.15),
                        "noise": rng.normal(0, 1.0),
                        "constant": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def _schemes() -> dict[str, list[FeatureScheme]]:
    return {
        "behavior": [
            FeatureScheme("behavior_signal", ("signal",)),
            FeatureScheme("behavior_noise", ("noise",)),
        ]
    }


def _run(frame: pd.DataFrame, run_id: str):
    return run_nested_loso(
        frame,
        model_feature_schemes=_schemes(),
        inner_splits=3,
        c_candidates=[0.1, 1.0],
        run_id=run_id,
    )


def test_outer_loso_keeps_all_sessions_of_participant_together_and_preserves_analysis_set() -> None:
    frame = _probe_frame()
    result = _run(frame, "synthetic-run")

    assert len(result.predictions) == len(frame)
    assert result.failures.empty
    assert set(result.predictions["analysis_set_id"]) == {"synthetic-common-set"}
    assert set(result.predictions["membership_type"]) == {"included_missing_aware"}
    assert result.metadata["membership_type"] == "included_missing_aware"
    assert set(result.predictions["run_id"]) == {"synthetic-run"}
    assert (result.predictions["participant_group_id"] == result.predictions["outer_fold_group"]).all()
    for locator in (
        "probe_id",
        "probe_index_in_block",
        "probe_index_global",
        "probe_onset_unix_ms",
        "window_name",
        "window_start_unix_ms",
        "window_effective_start_unix_ms",
        "window_end_unix_ms",
    ):
        assert locator in result.predictions.columns

    for group, rows in result.predictions.groupby("participant_group_id"):
        assert rows["outer_fold_group"].unique().tolist() == [group]
        assert rows["session_id"].nunique() == 2

    for audit in result.fold_audits:
        assert audit["membership_type"] == "included_missing_aware"
        assert not (set(audit["outer_train_group_ids"]) & set(audit["outer_test_group_ids"]))
        assert audit["final_refit"]["preprocessing"]["fit_group_ids"] == audit["outer_train_group_ids"]


def test_held_out_q1_labels_do_not_affect_that_participants_predictions_or_selection() -> None:
    frame = _probe_frame()
    target = "P-00"
    base = _run(frame, "base")

    altered = frame.copy()
    mask = altered["participant_group_id"].eq(target)
    altered.loc[mask, "q1_nominal_4class"] = altered.loc[mask, "q1_nominal_4class"].map(
        lambda value: 2 if value == 1 else 1
    )
    changed = _run(altered, "changed")

    base_rows = base.predictions[base.predictions["outer_fold_group"].eq(target)].reset_index(drop=True)
    changed_rows = changed.predictions[changed.predictions["outer_fold_group"].eq(target)].reset_index(drop=True)
    np.testing.assert_allclose(base_rows["p_q1_equals_1"], changed_rows["p_q1_equals_1"])
    assert base_rows["feature_set_id"].tolist() == changed_rows["feature_set_id"].tolist()
    assert base_rows["selected_c"].tolist() == changed_rows["selected_c"].tolist()

    base_audit = next(a for a in base.fold_audits if a["outer_fold_group"] == target)
    changed_audit = next(a for a in changed.fold_audits if a["outer_fold_group"] == target)
    assert base_audit["selection"] == changed_audit["selection"]
    assert base_audit["final_refit"]["preprocessing"] == changed_audit["final_refit"]["preprocessing"]


def test_future_probe_feature_of_held_out_participant_cannot_change_current_probe_prediction() -> None:
    frame = _probe_frame()
    target = "P-00"
    target_rows = frame[frame["participant_group_id"].eq(target)].sort_values("probe_time_ms")
    current_event = target_rows.iloc[0]["probe_event_id"]
    future_event = target_rows.iloc[-1]["probe_event_id"]

    base = _run(frame, "base-future")
    altered = frame.copy()
    altered.loc[altered["probe_event_id"].eq(future_event), "signal"] = 1e12
    changed = _run(altered, "changed-future")

    base_current = base.predictions[
        base.predictions["probe_event_id"].eq(current_event)
        & base.predictions["outer_fold_group"].eq(target)
    ].iloc[0]
    changed_current = changed.predictions[
        changed.predictions["probe_event_id"].eq(current_event)
        & changed.predictions["outer_fold_group"].eq(target)
    ].iloc[0]

    assert base_current["feature_set_id"] == changed_current["feature_set_id"]
    assert base_current["selected_c"] == changed_current["selected_c"]
    np.testing.assert_allclose(
        [base_current["p_q1_equals_1"]],
        [changed_current["p_q1_equals_1"]],
        rtol=0,
        atol=0,
    )


def test_failed_model_fold_is_preserved_in_predictions_and_failure_table() -> None:
    frame = _probe_frame()
    result = run_nested_loso(
        frame,
        model_feature_schemes={
            "valid": [FeatureScheme("signal", ("signal",))],
            "invalid_constant": [FeatureScheme("constant_only", ("constant",))],
        },
        inner_splits=3,
        c_candidates=[1.0],
    )

    invalid = result.predictions[result.predictions["model_id"].eq("invalid_constant")]
    assert len(invalid) == len(frame)
    assert invalid["model_failed"].all()
    assert invalid["p_q1_equals_1"].isna().all()
    assert len(result.failures) == frame["participant_group_id"].nunique()
    assert result.failures["reason"].str.contains("no usable features").all()
    assert set(result.failures["membership_type"]) == {"included_missing_aware"}


def test_duplicate_probe_locator_or_missing_q1_fails_before_training() -> None:
    frame = _probe_frame()
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(SupervisedLearningContractError, match="duplicate probe locator"):
        run_nested_loso(duplicate, model_feature_schemes=_schemes(), inner_splits=3)

    missing_q1 = frame.copy()
    missing_q1.loc[0, "q1_nominal_4class"] = np.nan
    with pytest.raises(SupervisedLearningContractError, match="missing Q1"):
        run_nested_loso(missing_q1, model_feature_schemes=_schemes(), inner_splits=3)


def test_missing_or_mixed_membership_type_fails_before_training() -> None:
    missing = _probe_frame().drop(columns=["membership_type"])
    with pytest.raises(SupervisedLearningContractError, match="requires membership_type"):
        run_nested_loso(missing, model_feature_schemes=_schemes(), inner_splits=3)

    mixed = _probe_frame()
    mixed.loc[mixed.index[-1], "membership_type"] = "included_complete"
    with pytest.raises(SupervisedLearningContractError, match="one membership_type per run"):
        run_nested_loso(mixed, model_feature_schemes=_schemes(), inner_splits=3)


def test_unexpected_programming_error_in_outer_fold_propagates(monkeypatch) -> None:
    frame = _probe_frame()

    def _programmer_defect(*args, **kwargs):
        raise TypeError("simulated runner programmer defect")

    monkeypatch.setattr(supervised_runner, "select_logistic_model", _programmer_defect)
    with pytest.raises(TypeError, match="simulated runner programmer defect"):
        run_nested_loso(
            frame,
            model_feature_schemes=_schemes(),
            inner_splits=3,
            c_candidates=[1.0],
        )
