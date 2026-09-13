"""Summarise the 1.16.24 Q1 four-class runs (route A and route B).

方法权威：`分析设计/1.16.24-Q1四分类正式分析与两条公平对照路线_20260913.md`

从每个运行的 `multiclass_evaluation.json` 读取：
- 每模型 `participant_macro_multiclass_log_loss`
- 逐外层折类别先验基线（同一参与者等权口径）
- `exceeds_no_information_baseline`
并输出汇总 CSV + Markdown 报告。

**预注册强制的报告义务**：两条路线都必须报告，即使其中一条未超过无信息基线；
必须同时报告逐折类别先验基线；不得用跨分析集合的配对增量排序。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Project\厚粲杯\11_数据\_FormalAnalysis\SupervisedRuns4ClassV1")
MEMBERSHIP = "included_missing_aware"
ROUTE_A_SUFFIX = f"__{MEMBERSHIP}__4classA"
ROUTE_B_SUFFIX = f"__{MEMBERSHIP}__4classB"


def _read_run(run_dir: Path) -> dict[str, object] | None:
    evaluation_path = run_dir / "multiclass_evaluation.json"
    manifest_path = run_dir / "run_manifest.json"
    if not evaluation_path.is_file() or not manifest_path.is_file():
        return None
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = payload["evaluation"]
    priors = evaluation["prior_baseline"]["per_model"]
    models: dict[str, dict[str, object]] = {}
    for model_id, block in evaluation["per_model"].items():
        summary = block["summary"]
        prior = priors.get(model_id, {})
        discrimination = block.get("discrimination", {})
        models[model_id] = {
            "participant_macro_multiclass_log_loss": float(
                summary["participant_macro_multiclass_log_loss"]
            ),
            "pooled_probe_multiclass_log_loss_descriptive": float(
                summary["pooled_probe_multiclass_log_loss_descriptive"]
            ),
            "prior_baseline_participant_macro_multiclass_log_loss": float(
                prior.get("overall_prior_baseline_participant_macro_multiclass_log_loss", float("nan"))
            ),
            "exceeds_no_information_baseline": bool(
                block.get("exceeds_no_information_baseline", False)
            ),
            "balanced_accuracy": discrimination.get("balanced_accuracy"),
            "macro_f1": discrimination.get("macro_f1"),
            "n_probes": discrimination.get("n_probes"),
            "n_participants": discrimination.get("n_participants"),
            "ovr_macro_auroc": discrimination.get("ovr_macro_auroc"),
            "per_class_log_loss": discrimination.get("per_class_log_loss"),
            "per_class_ovr_auroc": discrimination.get("per_class_ovr_auroc"),
            "class_support": discrimination.get("class_support"),
            "predicted_class_distribution": discrimination.get("predicted_class_distribution"),
            "probability_row_sum_max_abs_deviation": discrimination.get(
                "probability_row_sum_max_abs_deviation"
            ),
        }
    return {
        "run_id": payload.get("run_id"),
        "route": payload.get("route"),
        "analysis_set_id": payload.get("analysis_set_id"),
        "models": models,
        "manifest": manifest,
    }


def _forward_selection_summary(run_dir: Path) -> dict[str, object]:
    """路线 B：统计每个外层折选中的特征与步数。"""
    audits_path = run_dir / "fold_audits.json"
    if not audits_path.is_file():
        return {}
    audits = json.loads(audits_path.read_text(encoding="utf-8"))
    selection_counts: dict[str, int] = {}
    n_steps: list[int] = []
    empty_folds = 0
    for audit in audits:
        selection = audit.get("selection") or {}
        selected = selection.get("selected_feature_set") or []
        if not selected:
            empty_folds += 1
        for feature_id in selected:
            selection_counts[str(feature_id)] = selection_counts.get(str(feature_id), 0) + 1
        n_steps.append(int(selection.get("n_steps", 0) or 0))
    return {
        "n_outer_folds": len(audits),
        "empty_feature_set_folds": empty_folds,
        "mean_steps": (sum(n_steps) / len(n_steps)) if n_steps else None,
        "max_steps": max(n_steps) if n_steps else None,
        "feature_selected_in_n_folds": dict(
            sorted(selection_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def main() -> int:
    runs: list[dict[str, object]] = []
    for run_dir in sorted(path for path in ROOT.iterdir() if path.is_dir()):
        name = run_dir.name
        if name.endswith(ROUTE_A_SUFFIX):
            route = "A"
        elif name.endswith(ROUTE_B_SUFFIX):
            route = "B"
        else:
            continue
        record = _read_run(run_dir)
        if record is None:
            continue
        record["run_dir"] = str(run_dir)
        if route == "B":
            record["forward_selection"] = _forward_selection_summary(run_dir)
        runs.append(record)

    rows: list[dict[str, object]] = []
    for record in runs:
        for model_id, values in record["models"].items():
            rows.append(
                {
                    "route": record["route"],
                    "analysis_set_id": record["analysis_set_id"],
                    "model_id": model_id,
                    **values,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit("no four-class runs found; the batch has not produced results yet")
    table = table.sort_values(["route", "analysis_set_id", "model_id"]).reset_index(drop=True)
    table.to_csv(ROOT / "four_class_summary.csv", index=False, encoding="utf-8-sig")

    n_a = sum(1 for record in runs if record["route"] == "A")
    n_b = sum(1 for record in runs if record["route"] == "B")

    lines = [
        "# Q1 四分类结果（1.16.24）",
        "",
        "权威方法文件：`分析设计/1.16.24-Q1四分类正式分析与两条公平对照路线_20260913.md`",
        "",
        "## 0. 强制阅读边界",
        "",
        "四分类与二分类**标签不同、数值不可直接比较**。本报告只报告「Q1 自报类别的可推广预测信息」，",
        "**不得**写成「注意分类准确率」「注意水平预测」。四类为**无序类别**，不得解释为 1→4 的连续注意水平，",
        "不得把 2/3/4 统称「走神」。参与者级 AUROC 不可报（有参与者在外层折内只出现单一类别）。",
        "",
        "## 1. 运行清单",
        "",
        f"- route A：**{n_a}** 个分析集合",
        f"- route B：**{n_b}** 个运行（`AS.full` 上注册 11 特征池的前向逐步选择）",
        "",
        "## 2. route A：复用二分类已冻结特征方案的四分类结果",
        "",
        "| 分析集合 | 模型 | probes | 参与者宏平均多类 log loss | 逐折类别先验基线 | 超过基线 | balanced acc | macro-F1 |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    route_a = table[table["route"] == "A"]
    for row in route_a.itertuples():
        base = row.prior_baseline_participant_macro_multiclass_log_loss
        bal = row.balanced_accuracy
        f1 = row.macro_f1
        lines.append(
            f"| `{row.analysis_set_id}` | `{row.model_id}` | {row.n_probes} | "
            f"**{row.participant_macro_multiclass_log_loss:.6f}** | {base:.6f} | "
            f"{'是' if row.exceeds_no_information_baseline else '**否**'} | "
            f"{'–' if bal is None else f'{bal:.4f}'} | {'–' if f1 is None else f'{f1:.4f}'} |"
        )

    lines += ["", "## 3. route B：训练折内前向逐步选择", ""]
    route_b = table[table["route"] == "B"]
    if route_b.empty:
        lines.append("（尚未产出）")
    else:
        lines.append("| 分析集合 | 模型 | 参与者宏平均多类 log loss | 逐折类别先验基线 | 超过基线 |")
        lines.append("|---|---|---:|---:|---|")
        for row in route_b.itertuples():
            lines.append(
                f"| `{row.analysis_set_id}` | `{row.model_id}` | "
                f"**{row.participant_macro_multiclass_log_loss:.6f}** | "
                f"{row.prior_baseline_participant_macro_multiclass_log_loss:.6f} | "
                f"{'是' if row.exceeds_no_information_baseline else '**否**'} |"
            )
        for record in runs:
            if record["route"] != "B":
                continue
            selection = record.get("forward_selection") or {}
            lines += ["", "### 3.1 前向选择审计", ""]
            lines.append(f"- 外层折数：{selection.get('n_outer_folds')}")
            lines.append(f"- 终止于**空集**的折数：{selection.get('empty_feature_set_folds')}")
            lines.append(f"- 平均步数：{selection.get('mean_steps')}；最大步数：{selection.get('max_steps')}")
            counts = selection.get("feature_selected_in_n_folds") or {}
            if counts:
                lines.append("")
                lines.append("| 特征 | 被选中折数 |")
                lines.append("|---|---:|")
                for feature_id, count in counts.items():
                    lines.append(f"| `{feature_id}` | {count} |")
            else:
                lines.append("- **没有任何特征在任何外层折被选中**（全部退化到类别先验常数预测）。")

    lines += [
        "",
        "## 4. 一致性恒等式检查",
        "",
        "以下检查用**同一分析集合内、定义上等价的模型**互相印证，用于发现「跑错集合/模型接错列」这类",
        "静默错误。它们**不是**增量比较，也不跨分析集合做增量排序。",
        "",
    ]
    consistency: list[dict[str, object]] = []

    def compare(label: str, left: tuple[str, str], right: tuple[str, str]) -> None:
        def value(key: tuple[str, str]) -> float | None:
            subset = table[
                (table["analysis_set_id"] == key[0]) & (table["model_id"] == key[1])
            ]
            if subset.empty:
                return None
            return float(subset.iloc[0]["participant_macro_multiclass_log_loss"])

        a, b = value(left), value(right)
        if a is None or b is None:
            consistency.append({"check": label, "status": "NOT_AVAILABLE",
                                "left": list(left), "right": list(right)})
            return
        identical = abs(a - b) <= 1e-12
        consistency.append({"check": label, "status": "IN_ORDER" if identical else "MISMATCH",
                            "left": a, "right": b, "abs_diff": abs(a - b)})

    if not route_a.empty:
        # M0 = 仅行为；AS.behavior_reference 的 behavior_reference 也是仅行为。
        compare(
            "device M0 == behavior_reference",
            ("AS.device::M0", "M0"),
            ("AS.behavior_reference", "behavior_reference"),
        )
        # 同一集合内：动作只有 1 个注册特征，故模态条件增量 == 单特征增量。
        compare(
            "AS.behavior_plus_movement: behavior_plus_modality::movement == behavior_plus::<movement feature>",
            ("AS.behavior_plus_movement", "behavior_plus_modality::movement"),
            ("AS.behavior_plus_movement", "behavior_plus::movement.body_motion_energy.median.pre30s.v1"),
        )
        # M5 = 全部设备；与 full 应等价（与二分类线同一个恒等式）。
        compare(
            "device M5 == full",
            ("AS.device::M5", "M5"),
            ("AS.full", "full"),
        )

    for entry in consistency:
        if entry["status"] == "NOT_AVAILABLE":
            lines.append(f"- {entry['check']}：**尚未产出**（{entry['left']} / {entry['right']}）")
        elif entry["status"] == "IN_ORDER":
            lines.append(f"- {entry['check']}：**一致**（{entry['left']:.6f}）")
        else:
            lines.append(
                f"- {entry['check']}：**不一致**！左 {entry['left']:.6f} vs 右 {entry['right']:.6f}"
                f"（差 {entry['abs_diff']:.3e}）"
            )
    if not consistency:
        lines.append("- （无可用检查）")

    lines += [
        "",
        "## 5. 补充指标（预注册 §4.2 要求的报告项，不替代主指标）",
        "",
        "| 分析集合 | 模型 | 合并 OVR 宏平均 AUROC | balanced acc | macro-F1 | 预测为类别 1 的比例 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    per_class_rows: list[dict[str, object]] = []
    for row in route_a.itertuples():
        auroc = row.ovr_macro_auroc
        pred = row.predicted_class_distribution or {}
        total = sum(int(v) for v in pred.values()) if pred else 0
        share_class_1 = (int(pred.get("1", 0)) / total) if total else float("nan")
        lines.append(
            f"| `{row.analysis_set_id}` | `{row.model_id}` | "
            f"{'–' if auroc is None else f'{auroc:.4f}'} | "
            f"{'–' if row.balanced_accuracy is None else f'{row.balanced_accuracy:.4f}'} | "
            f"{'–' if row.macro_f1 is None else f'{row.macro_f1:.4f}'} | "
            f"{share_class_1:.4f} |"
        )
        for klass in ("1", "2", "3", "4"):
            per_class_rows.append(
                {
                    "analysis_set_id": row.analysis_set_id,
                    "model_id": row.model_id,
                    "class": klass,
                    "class_support": (row.class_support or {}).get(klass),
                    "predicted_n": (pred or {}).get(klass),
                    "per_class_log_loss": (row.per_class_log_loss or {}).get(klass),
                    "per_class_ovr_auroc": (row.per_class_ovr_auroc or {}).get(klass),
                }
            )
    pd.DataFrame(per_class_rows).to_csv(
        ROOT / "four_class_per_class_metrics.csv", index=False, encoding="utf-8-sig"
    )
    lines += [
        "",
        f"每类明细（支持数、预测数、每类对数损失、每类 OVR AUROC）见 `four_class_per_class_metrics.csv`。",
        "",
        "**读法警告**：balanced accuracy 的 4 类机会水平是 0.25，合并 OVR 宏平均 AUROC 的机会水平是 0.50。",
        "若某个模型的对数损失低于其逐折先验基线、但 balanced accuracy 与宏平均 AUROC 都在机会水平或以下，",
        "则该模型只是**在概率上略微优于常数先验**，**不具备类别判别力**，必须按此如实陈述。",
        "类别 3 的支持数很小，其每类损失与每类 AUROC 的抽样波动大，**不得**对类别 3 的细小差异做实质解释。",
    ]

    lines += [
        "",
        "## 6. 结论与判据",
        "",
        "- 超过无信息基线的判据是预注册 §4.4：参与者宏平均多类 log loss **低于**逐折类别先验基线。",
        "- 两条路线**都必须报告**，即使其中一条未超过基线；只报告较好的一条不成立。",
        "- 路线 B 与路线 A 的比较是**两种分析程序**的比较，**不是**某特征的四分类贡献；",
        "  跨分析集合的四分类增量排序禁止。",
        "- 主指标由类别 1 与类别 2 的区分主导（多数参与者只出现 1–2 个类别），",
        "  **不得**读作「四个类别被同等预测」；类别 3 仅 78 个探针，**不得**对其细小差异做实质解释。",
        "",
        "**产物**：`four_class_summary.csv`；逐运行目录下的 `multiclass_evaluation.json`、`run_manifest.json`、",
        "`fold_audits.json`（后两者为 local-only 或按 manifest 登记）。",
    ]
    (ROOT / "FOUR_CLASS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(
        {
            "runs": {"route_A": n_a, "route_B": n_b},
            "rows": int(len(table)),
            "summary_csv": str(ROOT / "four_class_summary.csv"),
            "report_md": str(ROOT / "FOUR_CLASS_REPORT.md"),
            "route_A_exceeds_baseline": int(route_a["exceeds_no_information_baseline"].sum()),
            "route_A_total": int(len(route_a)),
            "consistency": consistency,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
