"""Four-class task-contract tests (fail-closed encoding and class-probability alignment)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.supervised_learning.task import SupervisedLearningContractError
from attention_pipeline.supervised_learning.task_multiclass import (
    Q1_MULTICLASS_SPEC,
    MulticlassTaskSpec,
    aligned_class_probabilities,
    encode_q1_multiclass,
)


def test_spec_keeps_original_integer_class_values() -> None:
    """四个类别必须保留原始整数值 (1,2,3,4)，不得重映射到 0..3。"""
    assert Q1_MULTICLASS_SPEC.source_column == "q1_nominal_4class"
    assert Q1_MULTICLASS_SPEC.sorted_classes == (1, 2, 3, 4)
    assert Q1_MULTICLASS_SPEC.class_labels == (1, 2, 3, 4)
    assert Q1_MULTICLASS_SPEC.name == "q1_nominal_4class_multiclass"
    assert Q1_MULTICLASS_SPEC.probability_columns == (
        "p_q1_multiclass_1",
        "p_q1_multiclass_2",
        "p_q1_multiclass_3",
        "p_q1_multiclass_4",
    )


def test_encode_preserves_original_values_and_missingness() -> None:
    encoded = encode_q1_multiclass(pd.Series([1, 2, 3, 4, None, 2]))
    assert encoded.dtype.name == "Int64"
    assert encoded.tolist()[:4] == [1, 2, 3, 4]
    assert pd.isna(encoded.iloc[4])
    # 缺失必须保持缺失，不得插补成任何类别（尤其不得用众数 1 填充）。
    assert int(encoded.notna().sum()) == 5
    assert encoded.name == "q1_nominal_4class_multiclass"


def test_encode_accepts_integral_float_source_column() -> None:
    encoded = encode_q1_multiclass(pd.Series([1.0, 4.0, 3.0]))
    assert encoded.tolist() == [1, 4, 3]


@pytest.mark.parametrize("bad", [0, 5, -1, 9, 1.5, "3.5"])
def test_encode_raises_on_unexpected_values(bad: object) -> None:
    with pytest.raises(SupervisedLearningContractError, match="Unexpected"):
        encode_q1_multiclass(pd.Series([1, 2, bad]))


def test_encode_raises_on_non_numeric_text() -> None:
    with pytest.raises(SupervisedLearningContractError, match="Unexpected"):
        encode_q1_multiclass(pd.Series(["1", "focus", "3"]))


def test_encode_does_not_impute_a_fully_missing_series() -> None:
    encoded = encode_q1_multiclass(pd.Series([None, np.nan, None]))
    assert encoded.isna().all()


def test_aligned_class_probabilities_uses_explicit_model_labels() -> None:
    # 模型 classes_ 的顺序被打乱，且概率列随之一同打乱；
    # 对齐结果必须按 classes_ 标签取值，而不是按数组位置。
    probabilities = np.array([[0.10, 0.20, 0.30, 0.40]])
    classes = np.array([3, 1, 4, 2])
    aligned = aligned_class_probabilities(probabilities, classes)
    assert aligned.shape == (1, 4)
    assert aligned.tolist() == [[0.20, 0.40, 0.10, 0.30]]


def test_aligned_class_probabilities_raises_when_a_declared_class_is_absent() -> None:
    probabilities = np.array([[0.5, 0.5, 0.0]])
    classes = np.array([1, 2, 3])
    with pytest.raises(SupervisedLearningContractError, match="declared class 4"):
        aligned_class_probabilities(probabilities, classes)


def test_aligned_class_probabilities_raises_on_duplicate_declared_class() -> None:
    probabilities = np.array([[0.25, 0.25, 0.25, 0.25]])
    classes = np.array([1, 2, 3, 3])
    with pytest.raises(SupervisedLearningContractError, match="appears 2 times"):
        aligned_class_probabilities(probabilities, classes)


def test_aligned_class_probabilities_rejects_out_of_range_and_non_finite() -> None:
    classes = np.array([1, 2, 3, 4])
    with pytest.raises(SupervisedLearningContractError, match="within \\[0, 1\\]"):
        aligned_class_probabilities(np.array([[1.5, -0.5, 0.0, 0.0]]), classes)
    with pytest.raises(SupervisedLearningContractError, match="non-finite"):
        aligned_class_probabilities(np.array([[np.nan, 0.0, 0.0, 1.0]]), classes)


def test_aligned_class_probabilities_rejects_inconsistent_row_sums() -> None:
    classes = np.array([1, 2, 3, 4])
    with pytest.raises(SupervisedLearningContractError, match="sum to 1"):
        aligned_class_probabilities(np.array([[0.1, 0.1, 0.1, 0.1]]), classes)


def test_custom_spec_column_names_are_honoured() -> None:
    spec = MulticlassTaskSpec(
        name="custom",
        source_column="q1_nominal_4class",
        classes=(1, 2, 3, 4),
        probability_column_prefix="p_custom",
    )
    assert spec.probability_columns == (
        "p_custom_1",
        "p_custom_2",
        "p_custom_3",
        "p_custom_4",
    )
    assert spec.probability_column_name(3) == "p_custom_3"
