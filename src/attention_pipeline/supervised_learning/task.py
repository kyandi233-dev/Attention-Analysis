"""Task contracts for the current supervised-learning line.

The first production task is deliberately narrow: predict Q1=1 versus Q1=2/3/4
from a probe-preceding window while preserving explicit probability semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


class SupervisedLearningContractError(ValueError):
    """Raised when a supervised-learning input violates a frozen task contract."""


@dataclass(frozen=True)
class BinaryTaskSpec:
    """Explicit binary label contract used by training and reporting code."""

    name: str
    source_column: str
    positive_values: tuple[int, ...]
    negative_values: tuple[int, ...]
    positive_label: int = 1
    negative_label: int = 0
    positive_probability_name: str = "p_q1_equals_1"

    @property
    def class_labels(self) -> tuple[int, int]:
        return (self.negative_label, self.positive_label)


Q1_BINARY_SPEC = BinaryTaskSpec(
    name="q1_equals_1_vs_2_3_4",
    source_column="q1_nominal_4class",
    positive_values=(1,),
    negative_values=(2, 3, 4),
)


def encode_q1_binary(
    values: pd.Series | Sequence[object] | Iterable[object],
    *,
    spec: BinaryTaskSpec = Q1_BINARY_SPEC,
) -> pd.Series:
    """Encode raw Q1 values with a fail-closed mapping.

    Missing values remain missing.  Any non-missing value outside the declared
    positive/negative sets raises instead of being silently coerced into a class.
    """
    series = values.copy() if isinstance(values, pd.Series) else pd.Series(list(values))
    numeric = pd.to_numeric(series, errors="coerce")

    unexpected_text = series.notna() & numeric.isna()
    allowed = set(spec.positive_values) | set(spec.negative_values)
    unexpected_numeric = numeric.notna() & ~numeric.isin(allowed)
    unexpected = unexpected_text | unexpected_numeric
    if unexpected.any():
        bad = series.loc[unexpected].astype(str).drop_duplicates().tolist()
        raise SupervisedLearningContractError(
            f"Unexpected {spec.source_column} values for {spec.name}: {bad}"
        )

    encoded = pd.Series(pd.NA, index=series.index, dtype="Int64")
    encoded.loc[numeric.isin(spec.negative_values)] = spec.negative_label
    encoded.loc[numeric.isin(spec.positive_values)] = spec.positive_label
    encoded.name = f"{spec.source_column}_binary"
    return encoded


def positive_class_probability(
    probabilities: np.ndarray,
    classes: Sequence[object] | np.ndarray,
    *,
    spec: BinaryTaskSpec = Q1_BINARY_SPEC,
) -> np.ndarray:
    """Return P(positive class) using explicit model class labels.

    Never infer the positive column from array position.  This prevents a model
    whose ``classes_`` order changes from silently flipping probability meaning.
    """
    proba = np.asarray(probabilities, dtype=float)
    class_array = np.asarray(classes)
    if proba.ndim != 2:
        raise SupervisedLearningContractError("probabilities must be a 2D array")
    if proba.shape[1] != len(class_array):
        raise SupervisedLearningContractError(
            "probability column count does not match supplied class labels"
        )

    matches = np.flatnonzero(class_array == spec.positive_label)
    if len(matches) != 1:
        raise SupervisedLearningContractError(
            f"expected exactly one positive class {spec.positive_label}; got {class_array.tolist()}"
        )
    column = proba[:, int(matches[0])]
    if not np.isfinite(column).all():
        raise SupervisedLearningContractError("positive-class probability contains non-finite values")
    return column.astype(float, copy=False)
