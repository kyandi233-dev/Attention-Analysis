"""Check the report text against the generated summary artifacts for transcription drift.

为什么需要：本工作线已经因为**手打数字**出过两次真实错误（手抄 AUROC 值；把合并探针口径
的基率直接与参与者等权主指标比较）。报告正文里的数字必须与运行产出的汇总文件逐一对应，
不能靠人眼复核。

检查方式：从汇总产物读出权威数值，再确认每个数值在指定文档中出现（同时接受 ASCII 连字符
与 U+2212 真负号两种写法——正文按 APA 用真负号，工具不能因此误报）。

用法：
    python verify_report_number_consistency.py
退出码 0 = 全部命中；1 = 存在未被正文引用的权威数值。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

FORMAL = Path(r"D:\Project\.codex-worktrees\FocusWave-Formal-Analysis-q1-4class-20260913")
BASE = Path(r"D:\Project\厚粲杯\11_数据\_FormalAnalysis")
WINDOW_SUMMARY = BASE / "SupervisedRunsWindowSensitivityV1" / "window_sensitivity_summary.csv"
HEADMOTION_SUMMARY = (
    BASE / "SupervisedRunsOcularHeadmotionSensitivityV1" / "headmotion_sensitivity_summary.json"
)

DOCUMENTS = {
    "章节5.6": FORMAL / "国赛报告/章节草稿/5.6-各科学模态对新参与者Q1的预测能力.md",
    "完整结果/6": FORMAL / "国赛报告/完整结果/6-监督学习与多模态结果与资产.md",
    "完整结果/3": FORMAL / "国赛报告/完整结果/3-Ocular眼部结果与资产.md",
    "运行记录09-13-4": FORMAL / "运行记录与证据/09-13-4-首轮主窗口长度敏感性.md",
    "运行记录09-13-5": FORMAL / "运行记录与证据/09-13-5-瞳孔头动协变量敏感性.md",
}


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
    return expected


def _variants(value: str) -> tuple[str, ...]:
    """同一数值的可接受写法：ASCII 连字符与 U+2212 真负号。"""
    return (value, value.replace("-", "\u2212"))


def main() -> int:
    texts = {name: path.read_text(encoding="utf-8") for name, path in DOCUMENTS.items()}
    expected = _expected_values()

    missing: list[tuple[str, str]] = []
    print("=== 权威数值是否被正文引用 ===")
    for label, value in expected:
        hits = [
            name for name, text in texts.items()
            if any(variant in text for variant in _variants(value))
        ]
        print(f"  {label:34s} {value:>12s}  -> {hits if hits else '**缺失**'}")
        if not hits:
            missing.append((label, value))

    print(f"\n检查数值 {len(expected)} 个；未被任何文档引用 {len(missing)} 个")
    for label, value in missing:
        print(f"  缺失：{label} = {value}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
