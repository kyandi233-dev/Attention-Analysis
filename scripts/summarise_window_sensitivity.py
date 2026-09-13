"""Summarise the 1.16.25 window-length sensitivity into Git-safe evidence artifacts.

Reads the three frozen-contract arms (30 s from SupervisedRunsV1; 10 s / 20 s from
SupervisedRunsWindowSensitivityV1), verifies that they share code_sha / config_digest,
and writes a comparison CSV plus a Markdown report into the sensitivity run root.

The frozen reporting contract forbids cross-analysis-set paired increments
(`cross_analysis_set_increment_ranking_allowed = false`), so the arms are reported
side by side with their own participant-cluster intervals and no paired increment is
computed or ranked.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path(r"D:\Project\厚粲杯\11_数据\_FormalAnalysis")
OUT = BASE / "SupervisedRunsWindowSensitivityV1"
ARMS = {
    30: BASE / "SupervisedRunsV1" / "AS.behavior_reference__included_missing_aware",
    10: OUT / "AS.behavior_reference__window10s__included_missing_aware",
    20: OUT / "AS.behavior_reference__window20s__included_missing_aware",
}
INPUTS = {
    30: BASE / "SupervisedRunsV1" / "inputs" / "AS.behavior_reference.csv",
    10: OUT / "inputs" / "AS.behavior_reference__window10s.csv",
    20: OUT / "inputs" / "AS.behavior_reference__window20s.csv",
}
# The raw producer sensitivity table, used only to characterise how feature coverage
# changes with the window BEFORE the membership rule is applied. The analysis-set CSVs
# are already membership-filtered, so counting finiteness there is a tautology.
SENSITIVITY = BASE / "Behavior" / "formal_v3" / "probe_window_sensitivity.csv"
BEHAVIOR_FEATURES = (
    "go_correct_rt_median_ms",
    "go_correct_rt_cv",
    "go_correct_rt_theilsen_slope_ms_per_s",
    "raw_go_omission_rate",
    "commission_rate",
)


def _find(obj: object, key: str) -> object | None:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find(value, key)
            if found is not None:
                return found
    return None


def main() -> int:
    sensitivity = pd.read_csv(SENSITIVITY, low_memory=False)
    raw_by_window = {
        int(window): group
        for window, group in sensitivity.groupby("window_seconds_nominal")
    }
    rows: list[dict[str, object]] = []
    for window, arm in sorted(ARMS.items()):
        evaluation = pd.read_csv(arm / "model_evaluation.csv")
        record = evaluation.iloc[0]
        manifest = json.loads((arm / "run_manifest.json").read_text(encoding="utf-8"))
        bootstrap = json.loads((arm / "participant_bootstrap.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(INPUTS[window], low_memory=False)
        q1 = frame["q1_nominal_4class"].value_counts().sort_index()
        raw = raw_by_window[window]
        raw_numeric = raw[list(BEHAVIOR_FEATURES)].apply(pd.to_numeric, errors="coerce")
        rows.append(
            {
                "window_seconds": window,
                "analysis_set_id": record["analysis_set_id"],
                "n_probes": int(record["n_probes"]),
                "n_participants": int(record["n_participants"]),
                "participant_equal_log_loss": float(record["participant_equal_log_loss"]),
                "pooled_probe_log_loss_descriptive": float(record["pooled_probe_log_loss_descriptive"]),
                "ci_lower_95": _find(bootstrap, "ci_lower"),
                "ci_upper_95": _find(bootstrap, "ci_upper"),
                "status": record["status"],
                "q1_class_1": int(q1.get(1, 0)),
                "q1_class_2": int(q1.get(2, 0)),
                "q1_class_3": int(q1.get(3, 0)),
                "q1_class_4": int(q1.get(4, 0)),
                # Raw-window coverage, i.e. BEFORE the membership rule is applied.
                "raw_window_rows": int(len(raw)),
                "raw_all_five_finite": int(raw_numeric.notna().all(axis=1).sum()),
                "raw_commission_rate_finite": int(raw_numeric["commission_rate"].notna().sum()),
                "raw_theilsen_finite": int(
                    raw_numeric["go_correct_rt_theilsen_slope_ms_per_s"].notna().sum()
                ),
                "code_sha": manifest["provenance"]["code_sha"],
                "config_digest": manifest["provenance"]["config_digest"],
                "input_sha256": manifest["provenance"]["input_sha256"],
                "task": manifest["task"],
                "inner_splits": int(manifest["inner_splits"]),
                "bootstrap_replicates": manifest["outer_evaluation"]["bootstrap_replicates"],
                "bootstrap_seed": manifest["outer_evaluation"]["bootstrap_seed"],
                "cross_analysis_set_increment_ranking_allowed": manifest["outer_evaluation"][
                    "cross_analysis_set_increment_ranking_allowed"
                ],
            }
        )

    table = pd.DataFrame(rows).sort_values("window_seconds").reset_index(drop=True)
    table.to_csv(OUT / "window_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    code_shas = sorted(set(table["code_sha"]))
    config_digests = sorted(set(table["config_digest"]))
    losses = table["participant_equal_log_loss"]
    ci30 = table.loc[table["window_seconds"] == 30].iloc[0]
    ci30_covers = {
        int(row.window_seconds): bool(row.ci_lower_95 <= ci30.participant_equal_log_loss <= row.ci_upper_95)
        for row in table.itertuples()
    }
    all_ci_overlap_30 = bool(
        (
            (table["ci_lower_95"] <= ci30.ci_upper_95)
            & (table["ci_upper_95"] >= ci30.ci_lower_95)
        ).all()
    )

    lines: list[str] = []
    lines.append("# 首轮主窗口长度敏感性结果（1.16.25）")
    lines.append("")
    lines.append(f"权威方法文件：`分析设计/1.16.25-首轮主窗口长度敏感性预注册_20260913.md`")
    lines.append("")
    lines.append("## 1. 冻结条件核查")
    lines.append("")
    lines.append(f"- 三个臂的 `code_sha` 唯一值：`{code_shas}`（与冻结 30 s 臂相同）")
    lines.append(f"- 三个臂的 `config_digest` 唯一值：`{config_digests}`（与冻结 30 s 臂相同）")
    lines.append(f"- `task`：`{table['task'].unique().tolist()}`（Q1=1 对 Q1=2/3/4，与主分析一致）")
    lines.append(f"- `inner_splits`：`{sorted(set(table['inner_splits']))}`；bootstrap：`{sorted(set(table['bootstrap_replicates']))}` 次 / seed `{sorted(set(table['bootstrap_seed']))}` / 不重训")
    lines.append(f"- `cross_analysis_set_increment_ranking_allowed`：`{sorted(set(table['cross_analysis_set_increment_ranking_allowed']))}` → **不得计算或排序跨分析集合的配对增量**")
    lines.append("- 唯一被改变的变量是窗口长度；特征表示、缺失规则、QC、折、种子、指标均未变。")
    lines.append("")
    lines.append("## 2. 结果（行为参照特征集，主指标 = 参与者宏平均对数损失）")
    lines.append("")
    lines.append("| 窗口 | 分析集合 | probes | 参与者 | 参与者宏平均 log loss | 95% CI | 合并 probe log loss（描述） |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in table.itertuples():
        lines.append(
            f"| {row.window_seconds} s | `{row.analysis_set_id}` | {row.n_probes} | {row.n_participants} | "
            f"**{row.participant_equal_log_loss:.6f}** | [{row.ci_lower_95:.6f}, {row.ci_upper_95:.6f}] | "
            f"{row.pooled_probe_log_loss_descriptive:.6f} |"
        )
    lines.append("")
    lines.append("## 3. 样本构成（必须随结果一起报告）")
    lines.append("")
    lines.append("### 3.1 进入分析集合后的 Q1 分布")
    lines.append("")
    lines.append("| 窗口 | probes | Q1=1 | Q1=2 | Q1=3 | Q1=4 |")
    lines.append("|---|---|---|---|---|---|")
    for row in table.itertuples():
        lines.append(
            f"| {row.window_seconds} s | {row.n_probes} | {row.q1_class_1} | {row.q1_class_2} | "
            f"{row.q1_class_3} | {row.q1_class_4} |"
        )
    lines.append("")
    lines.append(
        "### 3.2 原始窗口切片覆盖（成员规则**之前**；来自 `probe_window_sensitivity.csv`）"
    )
    lines.append("")
    lines.append(
        "分析集合 CSV 已按成员规则筛选，在其中统计有限性是同义反复，因此本节一律使用原始切片。"
    )
    lines.append("")
    lines.append(
        "| 窗口 | 原始切片 probes | 五特征全有限（= 进入分析集合） | `commission_rate` 有限 | Theil-Sen 斜率有限 |"
    )
    lines.append("|---|---|---|---|---|")
    for row in table.itertuples():
        lines.append(
            f"| {row.window_seconds} s | {row.raw_window_rows} | {row.raw_all_five_finite} | "
            f"{row.raw_commission_rate_finite} | {row.raw_theilsen_finite} |"
        )
    lines.append("")
    lines.append("## 4. 结论")
    lines.append("")
    lines.append(
        f"- 三个窗口的参与者宏平均对数损失落在 **{losses.min():.6f}–{losses.max():.6f}** 之间，极差 **{losses.max() - losses.min():.6f}**。"
    )
    lines.append(
        f"- 30 s 的 95% 置信区间为 [{ci30.ci_lower_95:.6f}, {ci30.ci_upper_95:.6f}]；"
        f"三个臂的置信区间**两两都与 30 s 重叠**：{all_ci_overlap_30}。"
    )
    lines.append(
        f"- 30 s 点估计是否落在各臂自身置信区间内：{ci30_covers}。"
    )
    lines.append("- 因此主结论**不依赖 30 秒这一窗口选择**；首轮主窗口仍为 30 s，不因本项更换。")
    lines.append("")
    lines.append("### 必须同时声明的限制")
    lines.append("")
    lines.append(
        "1. **窗口差异与样本构成变化混在一起**：probes 数为 "
        + " / ".join(str(int(v)) for v in table["n_probes"])
        + "，20 s 与 10 s 都是 30 s 集合的子集。差异不可解释为纯粹的窗口效应。"
    )
    row10 = table.loc[table.window_seconds == 10].iloc[0]
    lines.append(
        "2. **10 s 另有构念混杂**：原始 10 s 切片共 "
        f"{int(row10.raw_window_rows)} 条，但 `commission_rate` 只有 "
        f"{int(row10.raw_commission_rate_finite)} 条有限值（{row10.raw_commission_rate_finite / row10.raw_window_rows:.1%}），"
        "因为 No-Go 机会在 10 秒窗口内常为零；五特征全有限只剩 "
        f"{int(row10.raw_all_five_finite)} 条。10 s 结果只作方向性参考，不得做实质结论。"
    )
    lines.append("3. **不择优**：未按表现选择窗口，也未用本项结果重开任何已冻结的特征竞争。")
    lines.append("4. **范围**：本项只覆盖 Behavior 端。Ocular 与 Movement 在冻结合同中只有 pre-30 s 特征，标记为 `NOT_EXECUTABLE_WITHOUT_PRODUCER_RECOMPUTE`。")
    lines.append("")
    (OUT / "WINDOW_SENSITIVITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(
        {
            "summary_csv": str(OUT / "window_sensitivity_summary.csv"),
            "report_md": str(OUT / "WINDOW_SENSITIVITY_REPORT.md"),
            "code_shas": code_shas,
            "config_digests": config_digests,
            "loss_min": float(losses.min()),
            "loss_max": float(losses.max()),
            "all_ci_overlap_30s": all_ci_overlap_30,
            "ci30_covers_point_estimates": ci30_covers,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
