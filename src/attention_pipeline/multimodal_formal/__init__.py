"""Multimodal fusion formal analysis package (feature-level, LOSO prediction).

File: __init__.py
Version: multimodal-fusion-v1.0.0
Purpose:
    八组合阶梯（M0 行为基准 -> 三个单模态 -> 三个双模态 -> 完整三模态）
    对思维探针结局 Q1（四分类名义）/ Q2（四级有序）的预测性能比较。
    主模型为规则化多项逻辑回归（L2），Random Forest 为非线性补充；
    验证为按 participant_group_id 的留一参与者交叉验证（LOSO），
    标准化/缺失处理/特征筛选/参数选择只在训练折内拟合。

Scope:
    本包只做特征级可解释融合，只写独立输出目录，不修改任何单模态
    正式产物与已冻结端点；fusion 状态解锁动作不属于本包。

Usage:
    >>> from attention_pipeline.multimodal_formal.runner import run_multimodal_fusion
    >>> manifest = run_multimodal_fusion("configs/multimodal_fusion.yaml")

Dependencies:
    numpy, pandas, scikit-learn, scipy, joblib, PyYAML
"""
from __future__ import annotations

MULTIMODAL_FUSION_VERSION = "multimodal-fusion-v1.0.0"
