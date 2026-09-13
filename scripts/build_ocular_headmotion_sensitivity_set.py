"""Build the Ocular + head-motion analysis set for the 1.16.26 sensitivity.

方法权威：`分析设计/1.16.26-瞳孔头动协变量敏感性预注册_20260913.md`

冻结约束（不得在未改预注册的情况下变更）：
1. 协变量恰好两列，来自 `movement_feature_handoff.csv` 中 `report_role = sensitivity_auxiliary`
   的两个姿势方向列；不得增删。
2. 预测列 = 5 个注册 Ocular 特征列 + 2 个协变量；分析集合成员规则 = 这 7 列全部有限。
3. 不注册任何新特征、不改 registry、不把协变量升级为科学预测变量。
4. 键对齐必须归一化 `block_id` 大小写（行为表 `B1`，眼部表 `b1`）。
5. 眼部表有 `probe_event_id`，动作表没有，因此动作只能按
   `(participant_group_id, session_id, block_id, probe_order_in_block)` 对齐。

用法：
    python build_ocular_headmotion_set.py [--write]
不加 --write 只报告与断言；加 --write 写出分析集合 CSV。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BASE = Path(r"D:\Project\厚粲杯\11_数据\_FormalAnalysis")
BEHAVIOR_PRIMARY = BASE / "Behavior" / "formal_v3" / "probe_primary_30s.csv"
OCULAR_WIDE = BASE / "FormalScience" / "Ocular" / "tables" / "ocular_probe_features_wide.csv"
MOVEMENT_DESC = BASE / "FormalScience" / "Movement" / "tables" / "movement_probe_descriptive_source.csv"
OUT_ROOT = BASE / "SupervisedRunsOcularHeadmotionSensitivityV1" / "inputs"

#: 冻结注册表里的 5 个 Ocular 预测列（顺序取自注册表，不得改）。
OCULAR_COLUMNS = (
    "ocular__rseg_hard__rgb_nir_qc__level_mean__2s",
    "ocular__rseg_hard__rgb_nir_qc__variability_mad__2s",
    "ocular__rseg_hard__rgb_nir_qc__linear_slope_per_sec__2s",
    "ocular__rseg_hard__rgb_nir_qc__quadratic_curvature_per_sec2__2s",
    "ocular__blink_event_rate_per_min__pre30s",
)
#: 预注册 §3 固定的两个头动协变量（sensitivity_auxiliary）。
HEADMOTION_COLUMNS = (
    "pose_lateral_right_per_sec_median",
    "pose_vertical_up_per_sec_median",
)
PREDICTOR_COLUMNS = OCULAR_COLUMNS + HEADMOTION_COLUMNS
#: 冻结的 Q1 来源列（与 task.py 的 BinaryTaskSpec.source_column 一致）。
Q1_SOURCE_COLUMN = "q1_nominal_4class"
MEMBERSHIP_RULE = "all_five_registered_ocular_features_and_both_headmotion_covariates_finite"
ANALYSIS_SET_ID = "AS.ocular_headmotion_sensitivity"
MEMBERSHIP_TYPE = "included_missing_aware"

OCULAR_JOIN_KEYS = ("session_id", "block_id", "probe_event_id")
MOVEMENT_JOIN_KEYS = ("participant_group_id", "session_id", "block_id", "probe_order_in_block")


def _normalise(frame: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    """归一化连接键：block_id 统一小写，其余键统一字符串。"""
    out = frame.copy()
    if "block_id" in out.columns:
        out["block_id"] = out["block_id"].astype(str).str.lower()
    for key in keys:
        out[key] = out[key].astype(str)
    return out


def build(write: bool) -> dict[str, object]:
    behavior = _normalise(pd.read_csv(BEHAVIOR_PRIMARY, low_memory=False), OCULAR_JOIN_KEYS)
    ocular = _normalise(pd.read_csv(OCULAR_WIDE, low_memory=False), OCULAR_JOIN_KEYS)
    movement = _normalise(pd.read_csv(MOVEMENT_DESC, low_memory=False), MOVEMENT_JOIN_KEYS)

    report: dict[str, object] = {
        "method_authority": "分析设计/1.16.26-瞳孔头动协变量敏感性预注册_20260913.md",
        "analysis_set_id": ANALYSIS_SET_ID,
        "membership_rule": MEMBERSHIP_RULE,
        "predictor_columns": list(PREDICTOR_COLUMNS),
        "ocular_increment_columns": list(OCULAR_COLUMNS),
        "headmotion_covariate_columns": list(HEADMOTION_COLUMNS),
        "sources": {
            "behavior_labels_and_keys": str(BEHAVIOR_PRIMARY),
            "ocular_features": str(OCULAR_WIDE),
            "headmotion_covariates": str(MOVEMENT_DESC),
        },
        "source_row_counts": {
            "behavior": int(len(behavior)),
            "ocular": int(len(ocular)),
            "movement": int(len(movement)),
        },
    }

    # --- 眼部：按 (session_id, block_id, probe_event_id) 一对一连接 -------------------
    ocular_payload = ocular[list(OCULAR_JOIN_KEYS) + list(OCULAR_COLUMNS)]
    merged = behavior.merge(ocular_payload, on=list(OCULAR_JOIN_KEYS), how="left", validate="one_to_one")
    if len(merged) != len(behavior):
        raise AssertionError(f"ocular join changed row count: {len(merged)} vs {len(behavior)}")
    ocular_missing_rows = int(merged[list(OCULAR_COLUMNS)].isna().all(axis=1).sum())

    # --- 头动：动作表没有 probe_event_id，按四键连接 ---------------------------------
    # 两个来源的 probe_order_in_block dtype 不同（行为表 int、动作表 object），
    # 因此合并前必须把这一侧的四键也归一化成字符串，否则 pandas 直接拒绝连接。
    merged = _normalise(merged, MOVEMENT_JOIN_KEYS)
    movement_payload = movement[list(MOVEMENT_JOIN_KEYS) + list(HEADMOTION_COLUMNS)]
    duplicates = int(movement_payload.duplicated(subset=list(MOVEMENT_JOIN_KEYS)).sum())
    if duplicates:
        raise AssertionError(f"movement covariate source has {duplicates} duplicate join keys")
    merged = merged.merge(
        movement_payload, on=list(MOVEMENT_JOIN_KEYS), how="left", validate="one_to_one"
    )
    if len(merged) != len(behavior):
        raise AssertionError(f"movement join changed row count: {len(merged)} vs {len(behavior)}")
    movement_missing_rows = int(merged[list(HEADMOTION_COLUMNS)].isna().all(axis=1).sum())

    report["join_diagnostics"] = {
        "ocular_rows_with_no_ocular_values": ocular_missing_rows,
        "movement_rows_with_no_covariate_values": movement_missing_rows,
        "movement_duplicate_join_keys": duplicates,
    }

    # --- 成员规则 -------------------------------------------------------------------
    numeric = merged[list(PREDICTOR_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    member = merged[numeric.notna().all(axis=1)].copy()

    member["analysis_set_id"] = ANALYSIS_SET_ID
    member["membership_type"] = MEMBERSHIP_TYPE
    # 正式入口 `scripts/supervised_learning_analysis.py` 在任何拟合前调用
    # `validate_task_a_required_outcomes`，它要求分析集合保留 B 层的结果域声明；
    # 这里按冻结契约写成同一列名，不能省略。
    member["required_outcomes"] = json.dumps([Q1_SOURCE_COLUMN])

    q1 = member["q1_nominal_4class"].value_counts().sort_index()
    report["membership"] = {
        "rows_before_membership": int(len(merged)),
        "rows_after_membership": int(len(member)),
        "participant_groups": int(member["participant_group_id"].nunique()),
        "sessions": int(member["session_id"].nunique()),
        "q1_nominal_4class_counts": {str(int(k)): int(v) for k, v in q1.items()},
        "binary_positive_q1_equals_1": int((member["q1_nominal_4class"] == 1).sum()),
        "per_column_finite_counts": {
            column: int(pd.to_numeric(merged[column], errors="coerce").notna().sum())
            for column in PREDICTOR_COLUMNS
        },
    }

    # --- 与注册 Ocular 特征的覆盖对照（必须报告样本变化） ----------------------------
    ocular_only_member = merged[
        merged[list(OCULAR_COLUMNS)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    ]
    report["sample_change_vs_ocular_only"] = {
        "ocular_only_rows": int(len(ocular_only_member)),
        "ocular_plus_headmotion_rows": int(len(member)),
        "rows_lost": int(len(ocular_only_member) - len(member)),
        "participants_ocular_only": int(ocular_only_member["participant_group_id"].nunique()),
        "participants_ocular_plus_headmotion": int(member["participant_group_id"].nunique()),
    }

    if write:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        out_path = OUT_ROOT / f"{ANALYSIS_SET_ID}.csv"
        member.to_csv(out_path, index=False, encoding="utf-8-sig")
        report["written"] = str(out_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="写出分析集合 CSV。")
    args = parser.parse_args()
    print(json.dumps(build(write=bool(args.write)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
