from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.supervised_learning.task import (
    Q1_BINARY_SPEC,
    SupervisedLearningContractError,
    encode_q1_binary,
    positive_class_probability,
)


def test_q1_binary_mapping_is_explicit_and_missing_is_preserved() -> None:
    raw = pd.Series([1, 2, 3, 4, None], index=[10, 11, 12, 13, 14])
    encoded = encode_q1_binary(raw)

    assert encoded.index.tolist() == raw.index.tolist()
    assert encoded.iloc[:4].tolist() == [1, 0, 0, 0]
    assert pd.isna(encoded.iloc[4])
    assert encoded.dtype == "Int64"
    assert Q1_BINARY_SPEC.positive_probability_name == "p_q1_equals_1"


def test_q1_binary_mapping_rejects_unexpected_nonmissing_values() -> None:
    with pytest.raises(SupervisedLearningContractError, match="Unexpected q1_nominal_4class"):
        encode_q1_binary([1, 2, 5])

    with pytest.raises(SupervisedLearningContractError, match="Unexpected q1_nominal_4class"):
        encode_q1_binary([1, "off-task"])


def test_positive_probability_follows_class_labels_not_column_position() -> None:
    proba_01 = np.array([[0.8, 0.2], [0.3, 0.7]])
    proba_10 = np.array([[0.2, 0.8], [0.7, 0.3]])

    np.testing.assert_allclose(
        positive_class_probability(proba_01, np.array([0, 1])),
        np.array([0.2, 0.7]),
    )
    np.testing.assert_allclose(
        positive_class_probability(proba_10, np.array([1, 0])),
        np.array([0.2, 0.7]),
    )


def test_positive_probability_fails_closed_when_positive_class_is_absent() -> None:
    with pytest.raises(SupervisedLearningContractError, match="expected exactly one positive class"):
        positive_class_probability(np.array([[1.0]]), np.array([0]))
