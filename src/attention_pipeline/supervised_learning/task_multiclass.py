"""Nominal four-class Q1 task contract (additive to the frozen binary line).

文件：task_multiclass.py
版本：1.0.0
功能：为 `1.16.24 Q1 四分类正式分析与两条公平对照路线` 提供无序四分类标签合同。
      与二分类线（`task.py`）严格并存：本模块只新增对象，不修改 `BinaryTaskSpec`、
      `Q1_BINARY_SPEC`、`encode_q1_binary` 或 `positive_class_probability`。
用法：
    from attention_pipeline.supervised_learning.task_multiclass import (
        Q1_MULTICLASS_SPEC, encode_q1_multiclass, aligned_class_probabilities,
    )
依赖：numpy、pandas

冻结约束（预注册 §2）：
1. 四个类别是**无序名义类别**；保留原始整数值 (1, 2, 3, 4)，不得重映射到 0..3。
   重映射会诱使下游把类别当成有序注意水平或把数组位置当成类别语义。
2. 编码 fail-closed：任何非缺失但不在 {1,2,3,4} 的取值直接报错；缺失保持缺失（Int64），
   绝不插补标签。
3. 模型类别概率必须按模型自报的 `classes_` 标签对齐，绝不按数组位置取值。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .task import SupervisedLearningContractError


# ---------------------------------------------------------------------------- 冻结常量
#: 预注册 §2：来源列固定为 q1_nominal_4class。
Q1_MULTICLASS_SOURCE_COLUMN = "q1_nominal_4class"
#: 预注册 §2：保留原始整数值，不重映射。顺序仅用于确定性输出，不表示注意水平排序。
Q1_MULTICLASS_CLASSES: tuple[int, ...] = (1, 2, 3, 4)


@dataclass(frozen=True)
class MulticlassTaskSpec:
    """Explicit nominal four-class label contract used by training and reporting code.

    参数：
        name: 任务名，写入 manifest 的 ``task`` 字段。默认 ``q1_nominal_4class_multiclass``。
        source_column: 原始标签列名。
        classes: 无序类别元组。必须恰好等于 ``(1, 2, 3, 4)`` 的集合；顺序只影响输出列顺序。
        probability_column_prefix: 每类概率列名前缀，最终列名为
            ``{prefix}_{class}``（例如 ``p_q1_multiclass_1``）。
        predicted_column_name: 预测类别列名。
    返回：不可变 dataclass 实例。
    """

    name: str
    source_column: str
    classes: tuple[int, ...]
    probability_column_prefix: str
    predicted_column_name: str = "predicted_q1_multiclass"

    @property
    def sorted_classes(self) -> tuple[int, ...]:
        """返回确定性排序后的类别元组（仅用于列顺序与审计，不表示序数关系）。"""
        return tuple(sorted(int(value) for value in self.classes))

    @property
    def class_labels(self) -> tuple[int, ...]:
        """兼容别名：与 ``sorted_classes`` 相同。"""
        return self.sorted_classes

    @property
    def probability_columns(self) -> tuple[str, ...]:
        """四个类别概率列的固定顺序。"""
        return tuple(
            self.probability_column_name(label) for label in self.sorted_classes
        )

    def probability_column_name(self, label: int) -> str:
        """返回某一类别对应的概率列名。"""
        return f"{self.probability_column_prefix}_{int(label)}"


#: 正式四分类合同。原始整数值保持为 (1, 2, 3, 4)。
Q1_MULTICLASS_SPEC = MulticlassTaskSpec(
    name="q1_nominal_4class_multiclass",
    source_column=Q1_MULTICLASS_SOURCE_COLUMN,
    classes=Q1_MULTICLASS_CLASSES,
    probability_column_prefix="p_q1_multiclass",
)


def _numeric_series(values: pd.Series | Sequence[object] | Iterable[object]) -> pd.Series:
    """把任意输入统一转成 numeric Series，保留原始索引。

    为什么单独抽出来：编码与缺失检查必须共用**同一次**数值转换，否则文本型非法值
    （例如 ``"x"``）与数值型非法值（例如 ``9``）会被两条不一致的路径处理。
    """
    series = values.copy() if isinstance(values, pd.Series) else pd.Series(list(values))
    return pd.to_numeric(series, errors="coerce")


def encode_q1_multiclass(
    values: pd.Series | Sequence[object] | Iterable[object],
    *,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> pd.Series:
    """按无序四分类合同编码原始 Q1 值，fail-closed。

    参数：
        values: 原始标签序列（Series / 序列 / 可迭代对象）。
        spec: 四分类任务合同，默认 ``Q1_MULTICLASS_SPEC``。
    返回：
        dtype 为 ``Int64`` 的 Series；原始整数值**原样保留**（不重映射到 0..3），
        缺失保持为 ``pd.NA``。
    异常：
        SupervisedLearningContractError: 出现非缺失但不在 ``spec.classes`` 内的取值，
            或出现非数值文本。不静默归入任何一类。
    """
    series = values.copy() if isinstance(values, pd.Series) else pd.Series(list(values))
    numeric = _numeric_series(series)

    allowed = {int(value) for value in spec.classes}
    # 文本型非法值（例如 "3.5a"）与数值型越界值（例如 9）都必须报错；
    # 两者分开判定是为了在错误信息里保留原始取值，便于上游定位。
    unexpected_text = series.notna() & numeric.isna()
    unexpected_numeric = numeric.notna() & ~numeric.isin(sorted(allowed))
    unexpected = unexpected_text | unexpected_numeric
    if unexpected.any():
        bad = series.loc[unexpected].astype(str).drop_duplicates().tolist()
        raise SupervisedLearningContractError(
            f"Unexpected {spec.source_column} values for {spec.name}: {bad}; "
            f"expected one of {sorted(allowed)} or missing"
        )

    encoded = pd.Series(pd.NA, index=series.index, dtype="Int64")
    # 逐类赋值而不是 factorize/cat.codes：类别值本身就是权威标签，
    # 任何重映射都会在概率列与标签之间引入位置语义。
    for label in spec.sorted_classes:
        encoded.loc[numeric.eq(int(label))] = int(label)
    encoded.name = f"{spec.source_column}_multiclass"
    return encoded


def aligned_class_probabilities(
    probabilities: np.ndarray,
    classes: Sequence[object] | np.ndarray,
    *,
    spec: MulticlassTaskSpec = Q1_MULTICLASS_SPEC,
) -> np.ndarray:
    """按模型自报的 ``classes_`` 标签把概率矩阵对齐到 ``spec.sorted_classes``。

    参数：
        probabilities: ``(n, k)`` 概率矩阵。
        classes: 模型 ``classes_`` 标签序列，长度必须为 k。
        spec: 四分类任务合同。
    返回：
        ``(n, 4)`` float 矩阵，第 j 列对应 ``spec.sorted_classes[j]``。
    异常：
        SupervisedLearningContractError: 形状不匹配、概率非有限/越界，
            或某个已声明类别在 ``classes_`` 中缺失。
    """
    proba = np.asarray(probabilities, dtype=float)
    class_array = np.asarray(classes)
    if proba.ndim != 2:
        raise SupervisedLearningContractError("class probabilities must be a 2D array")
    if len(class_array) == 0:
        raise SupervisedLearningContractError("model class labels must be non-empty")
    if proba.shape[1] != len(class_array):
        raise SupervisedLearningContractError(
            "probability column count does not match supplied class labels: "
            f"{proba.shape[1]} vs {len(class_array)}"
        )
    if not np.isfinite(proba).all():
        raise SupervisedLearningContractError("class probabilities contain non-finite values")
    if np.any((proba < 0.0) | (proba > 1.0)):
        raise SupervisedLearningContractError("class probabilities must lie within [0, 1]")

    numeric_classes: list[int] = []
    for label in class_array.tolist():
        try:
            numeric_classes.append(int(label))
        except (TypeError, ValueError) as exc:
            raise SupervisedLearningContractError(
                f"model class label is not an integer: {label!r}"
            ) from exc

    order: list[int] = []
    for declared in spec.sorted_classes:
        matches = [index for index, value in enumerate(numeric_classes) if value == int(declared)]
        if len(matches) != 1:
            raise SupervisedLearningContractError(
                f"declared class {declared} appears {len(matches)} times in model classes "
                f"{numeric_classes}; every declared class must appear exactly once"
            )
        order.append(matches[0])

    aligned = proba[:, order]
    row_sums = aligned.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-6):
        raise SupervisedLearningContractError(
            "aligned class probabilities do not sum to 1 per row; refusing to report inconsistent probabilities"
        )
    return aligned.astype(float, copy=False)


__all__ = [
    "Q1_MULTICLASS_CLASSES",
    "Q1_MULTICLASS_SOURCE_COLUMN",
    "Q1_MULTICLASS_SPEC",
    "MulticlassTaskSpec",
    "aligned_class_probabilities",
    "encode_q1_multiclass",
]
