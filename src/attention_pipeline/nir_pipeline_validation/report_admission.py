from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd

REPORT_GATE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class FigureContract:
    figure_id: str
    title_zh: str
    level: str
    n_unit: str
    purpose: str


FIGURE01_10: tuple[FigureContract, ...] = (
    FigureContract("Figure01", "NIR测量覆盖与质量概览", "session/eye", "session", "工程QC与测量覆盖"),
    FigureContract("Figure02", "场次内瞳孔状态与Block转换", "session/block", "session", "场次内描述"),
    FigureContract("Figure03", "Go遗漏与NoGo误按的行为端点", "event", "event", "行为端点定义与不平衡"),
    FigureContract("Figure04", "Probe前瞳孔状态：主窗口", "probe", "probe", "一probe一行主分析"),
    FigureContract("Figure05", "Probe窗口敏感性分析", "probe", "probe", "预注册窗口敏感性"),
    FigureContract("Figure06", "刺激亮度时间方向与历史覆盖", "event", "event", "混淆与时间泄漏审计"),
    FigureContract("Figure07", "瞳孔tonic/phasic/变化率/波动特征", "event/probe", "event", "特征分解"),
    FigureContract("Figure08", "左右眼、场次与参与者层级一致性", "participant/session/eye", "participant", "层级与敏感性"),
    FigureContract("Figure09", "行为/设计基线与NIR增量预测", "participant-exclusive CV", "participant", "增量效度候选分析"),
    FigureContract("Figure10", "层级推断、失败状态与报告准入", "model/report", "model", "正式准入汇总"),
)

REQUIRED_FAILURE_COLUMNS = (
    "stage",
    "error_type",
    "error_message",
    "input_unit",
    "admission_status",
)


def figure_registry_frame() -> pd.DataFrame:
    return pd.DataFrame([item.__dict__ for item in FIGURE01_10])


def validate_failure_table(failures: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_FAILURE_COLUMNS if column not in failures.columns]
    if missing:
        raise ValueError(f"failure table missing required columns: {missing}")
    if not failures.empty and failures["admission_status"].isna().any():
        raise ValueError("every failure requires explicit report-admission status")


def report_admission(
    *,
    failures: pd.DataFrame,
    model_results: pd.DataFrame,
    required_endpoints: Iterable[str] = ("go_omission", "nogo_commission"),
    brightness_audit_passed: bool,
    probe_unit_audit_passed: bool,
    behavior_v3_contract_frozen: bool,
) -> dict[str, Any]:
    validate_failure_table(failures)
    blocking = failures[
        failures["admission_status"].astype(str).str.lower().eq("blocked")
    ]
    required = set(required_endpoints)
    present = set(
        model_results.get("endpoint", pd.Series(dtype=str)).dropna().astype(str)
    )
    successful = (
        set(
            model_results.loc[
                model_results.get(
                    "status", pd.Series(index=model_results.index, dtype=str)
                )
                .astype(str)
                .eq("success"),
                "endpoint",
            ].astype(str)
        )
        if "endpoint" in model_results
        else set()
    )

    reasons: list[str] = []
    if not blocking.empty:
        reasons.append("blocking_failures_present")
    if not required.issubset(present) or not required.issubset(successful):
        reasons.append("required_endpoint_models_not_successful")
    if not brightness_audit_passed:
        reasons.append("brightness_time_direction_audit_failed")
    if not probe_unit_audit_passed:
        reasons.append("primary_probe_unit_audit_failed")
    if not behavior_v3_contract_frozen:
        reasons.append("behavior_v3_endpoint_contract_not_frozen")

    return {
        "schema_version": REPORT_GATE_SCHEMA_VERSION,
        "admitted": not reasons,
        "status": "admitted" if not reasons else "blocked",
        "reasons": reasons,
        "n_blocking_failures": int(len(blocking)),
        "required_endpoints": sorted(required),
        "figure_registry": [item.__dict__ for item in FIGURE01_10],
        "scientific_boundary": (
            "代码/工程验证不等于测量效度、独立预测能力或正式科学结论。"
        ),
    }


def chinese_figure_contract_ok(metadata: Mapping[str, Any]) -> bool:
    required = (
        "title_zh",
        "x_label_zh",
        "y_label_zh",
        "n_unit",
        "effect_label",
        "ci_label",
    )
    if any(not str(metadata.get(key, "")).strip() for key in required):
        return False

    def ascii_only(text: Any) -> bool:
        return str(text).isascii()

    return not (
        ascii_only(metadata["title_zh"])
        or ascii_only(metadata["x_label_zh"])
        or ascii_only(metadata["y_label_zh"])
    )
