"""Check the report text against the generated summary artifacts for transcription drift.

为什么需要：本工作线已经因为**手打数字**出过两次真实错误（手抄 AUROC 值；把合并探针口径
的基率直接与参与者等权主指标比较）。报告正文里的数字必须与运行产出的汇总文件逐一对应，
不能靠人眼复核。四分类多类概率诊断落地后本清单又扩了一组（布里尔分数与校准），
因为该项的正文数字同样是从汇总产物手抄进报告的。

检查方式：从汇总产物读出权威数值，再确认每个数值在指定文档中出现（同时接受 ASCII 连字符
与 U+2212 真负号两种写法——正文按 APA 用真负号，工具不能因此误报）。

用法：
    python verify_report_number_consistency.py
    python verify_report_number_consistency.py --formal-root "<默认分支的检出目录>"
退出码 0 = 全部命中；1 = 存在未被正文引用的权威数值。

``--formal-root`` 必须是**包含默认分支 `codex/code-fix-ledger` 上最新报告文本**的检出目录，
即 Formal PR #38 合并之后的状态；指向落后的功能分支检出会读到旧正文并产生假缺失。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# 默认指向本机同时包含全部新报告文本的检出（Formal PR #38 之后）；可用 --formal-root 覆盖。
DEFAULT_FORMAL = Path(r"D:\Project\.codex-worktrees\formal-fourclass-closure-20260913")
BASE = Path(r"D:\Project\厚粲杯\11_数据\_FormalAnalysis")
WINDOW_SUMMARY = BASE / "SupervisedRunsWindowSensitivityV1" / "window_sensitivity_summary.csv"
HEADMOTION_SUMMARY = (
    BASE / "SupervisedRunsOcularHeadmotionSensitivityV1" / "headmotion_sensitivity_summary.json"
)
MULTICLASS_DIAGNOSTICS = (
    BASE
    / "SupervisedRuns4ClassV1"
    / "multiclass_probability_diagnostics"
    / "multiclass_probability_diagnostics.csv"
)

#: 文档相对 `--formal-root` 的路径（键为报告中的简称，仅用于打印命中位置）。
DOCUMENT_PATHS = {
    "章节5.6": "国赛报告/章节草稿/5.6-各科学模态对新参与者Q1的预测能力.md",
    "完整结果/6": "国赛报告/完整结果/6-监督学习与多模态结果与资产.md",
    "完整结果/3": "国赛报告/完整结果/3-Ocular眼部结果与资产.md",
    "运行记录09-13-4": "运行记录与证据/09-13-4-首轮主窗口长度敏感性.md",
    "运行记录09-13-5": "运行记录与证据/09-13-5-瞳孔头动协变量敏感性.md",
    "运行记录09-13-7": "运行记录与证据/09-13-7-四分类多类概率诊断.md",
    "操作化登记1.16.27": "分析设计/1.16.27-多类折外概率诊断的操作化登记_20260913.md",
}

#: 四分类多类概率诊断中要核对的列与显示名。
#:
#: 每组后缀 ``u``（无符号 6 位小数）或 ``s``（带正负号 6 位小数，与正文写法一致）。
MULTICLASS_METRICS: tuple[tuple[str, str, str], ...] = (
    ("participant_macro_multiclass_brier", "布里尔分数", "u"),
    ("participant_macro_multiclass_brier_ci_lower", "布里尔 CI 下界", "u"),
    ("participant_macro_multiclass_brier_ci_upper", "布里尔 CI 上界", "u"),
    ("prior_baseline_participant_macro_multiclass_brier", "先验布里尔", "u"),
    ("brier_skill_score", "布里尔技能分数", "u"),
    ("participant_macro_brier_minus_prior", "配对布里尔差", "s"),
    ("participant_macro_brier_minus_prior_ci_lower", "配对布里尔差 CI 下界", "s"),
    ("participant_macro_brier_minus_prior_ci_upper", "配对布里尔差 CI 上界", "s"),
    ("pooled_probe_ovr_macro_auroc", "宏平均 AUROC", "u"),
    ("pooled_probe_ovr_macro_auroc_ci_lower", "宏平均 AUROC CI 下界", "u"),
    ("pooled_probe_ovr_macro_auroc_ci_upper", "宏平均 AUROC CI 上界", "u"),
    ("participant_macro_top_label_confidence", "平均置信度", "u"),
    ("participant_macro_top_label_accuracy", "实际正确率", "u"),
    ("participant_macro_top_label_calibration_in_the_large", "校准总体偏差", "s"),
    ("participant_macro_top_label_calibration_in_the_large_ci_lower", "校准总体偏差 CI 下界", "s"),
    ("participant_macro_top_label_calibration_in_the_large_ci_upper", "校准总体偏差 CI 上界", "s"),
    ("expected_calibration_error", "期望校准误差", "u"),
    ("calibration_slope", "校准斜率", "u"),
    ("calibration_slope_ci_lower", "校准斜率 CI 下界", "u"),
    ("calibration_slope_ci_upper", "校准斜率 CI 上界", "u"),
)

#: 两条公平对照路线的 (route, model_id) 与展示名。
MULTICLASS_HEADLINE: tuple[tuple[str, str, str], ...] = (
    ("A", "full", "路线A 完整组合"),
    ("B", "forward_selected_4class", "路线B 前向选择"),
)


def _multiclass_values(path: Path) -> list[tuple[str, str]]:
    """从四分类多类概率诊断表读出两条公平对照路线的权威数值。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少 {path}；请先运行 Attention-Analysis 的 "
            "scripts/build_probability_diagnostics_multiclass.py"
        )
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["analysis_set_id"] != "AS.full":
                continue
            rows[(row["route"], row["model_id"])] = row

    expected: list[tuple[str, str]] = []
    for route, model_id, label in MULTICLASS_HEADLINE:
        row = rows.get((route, model_id))
        if row is None:
            raise KeyError(f"诊断表中找不到 AS.full 的 {route}/{model_id}")
        for column, display, kind in MULTICLASS_METRICS:
            value = float(row[column])
            text = f"{value:+.6f}" if kind == "s" else f"{value:.6f}"
            expected.append((f"{label} {display}", text))
    # 输入规模：证明该层是在完整归档上跑的（用带千分位的行数，避免两位数泛匹配）
    manifest_path = path.parent / "multiclass_probability_diagnostics_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected.append(("多类概率诊断输入行数", f"{int(manifest['n_prediction_rows']):,}"))
    return expected


def _expected_values() -> list[tuple[str, str]]:
    """返回 (标签, 权威数值字符串) 列表。"""
    expected: list[tuple[str, str]] = []

    with WINDOW_SUMMARY.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            window = row["window_seconds"]
            expected.append((f"窗口 {window}s 参与者宏平均 log loss",
                             f"{float(row['participant_equal_log_loss']):.6f}"))
            expected.append((f"窗口 {window}s CI 下界", f"{float(row['ci_lower_95']):.6f}"))
            expected.append((f"窗口 {window}s CI 上界", f"{float(row['ci_upper_95']):.6f}"))
            expected.append((f"窗口 {window}s probes", f"{int(row['n_probes']):,}"))

    headmotion = json.loads(HEADMOTION_SUMMARY.read_text(encoding="utf-8"))
    for name, block in headmotion["models"].items():
        expected.append((f"{name} 参与者宏平均 log loss",
                         f"{float(block['participant_equal_log_loss']):.6f}"))
        expected.append((f"{name} CI 下界", f"{float(block['ci_lower_95']):.6f}"))
        expected.append((f"{name} CI 上界", f"{float(block['ci_upper_95']):.6f}"))
    increment = headmotion["paired_increment"]
    expected.append(("配对增量点估计", f"{float(increment['point_estimate']):+.6f}"))
    expected.append(("配对增量 CI 下界", f"{float(increment['ci_lower_95']):.6f}"))
    expected.append(("配对增量 CI 上界", f"{float(increment['ci_upper_95']):+.6f}"))

    expected.extend(_multiclass_values(MULTICLASS_DIAGNOSTICS))
    return expected


def _variants(value: str) -> tuple[str, ...]:
    """同一数值的可接受写法：ASCII 连字符与 U+2212 真负号。"""
    return (value, value.replace("-", "\u2212"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=DEFAULT_FORMAL,
        help=(
            "包含默认分支最新报告文本的检出目录；必须不早于 Formal PR #38 的合并提交，"
            "否则会读到旧正文并产生假缺失。"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    formal_root = Path(args.formal_root)
    missing_files = [
        path for path in DOCUMENT_PATHS.values() if not (formal_root / path).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"--formal-root {formal_root} 缺少以下文档：{missing_files}"
        )

    texts = {
        name: (formal_root / path).read_text(encoding="utf-8")
        for name, path in DOCUMENT_PATHS.items()
    }
    expected = _expected_values()

    missing: list[tuple[str, str]] = []
    print("=== 权威数值是否被正文引用 ===")
    for label, value in expected:
        hits = [
            name for name, text in texts.items()
            if any(variant in text for variant in _variants(value))
        ]
        print(f"  {label:38s} {value:>12s}  -> {hits if hits else '**缺失**'}")
        if not hits:
            missing.append((label, value))

    print(f"\n检查数值 {len(expected)} 个；未被任何文档引用 {len(missing)} 个")
    for label, value in missing:
        print(f"  缺失：{label} = {value}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
