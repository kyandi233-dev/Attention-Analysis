from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.supervised_learning.feature_schemes import FeatureScheme
from attention_pipeline.supervised_learning.runner import run_nested_loso
from attention_pipeline.supervised_learning.task import SupervisedLearningContractError


def _frame(*, membership_type: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for participant_index in range(6):
        participant = f"P{participant_index:02d}"
        session = f"S{participant_index:02d}"
        for probe_index, q1 in enumerate((1, 2, 1, 3), start=1):
            rows.append(
                {
                    "participant_group_id": participant,
                    "session_id": session,
                    "block_id": "B1",
                    "probe_event_id": f"{session}|B1|probe|{probe_index}",
                    "analysis_set_id": "review-set",
                    "membership_type": membership_type,
                    "q1_nominal_4class": q1,
                    "signal": float(q1 == 1),
                }
            )
    return pd.DataFrame(rows)


def _schemes() -> dict[str, list[FeatureScheme]]:
    return {"model": [FeatureScheme("signal-v1", ("signal",))]}


def test_included_complete_rejects_missing_predictor_before_any_imputation() -> None:
    frame = _frame(membership_type="included_complete")
    frame.loc[0, "signal"] = np.nan

    with pytest.raises(
        SupervisedLearningContractError,
        match="included_complete cannot contain missing predictor values",
    ):
        run_nested_loso(
            frame,
            model_feature_schemes=_schemes(),
            inner_splits=5,
            c_candidates=[1.0],
        )


def test_missing_aware_still_allows_residual_missingness_for_train_only_imputation() -> None:
    frame = _frame(membership_type="included_missing_aware")
    frame.loc[0, "signal"] = np.nan

    result = run_nested_loso(
        frame,
        model_feature_schemes=_schemes(),
        inner_splits=5,
        c_candidates=[1.0],
    )

    assert not result.predictions.empty
    assert set(result.predictions["membership_type"]) == {"included_missing_aware"}
