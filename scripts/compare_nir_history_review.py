"""历史阶段4/4b与阶段5修复前后对比（开发诊断，不是生产入口）。"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.nir.benchmark import evaluation_context, rate_metrics


def benchmark_ungated(config, csv_path: Path, version: str) -> pd.DataFrame:
    context = evaluation_context(config)
    detections = pd.read_csv(csv_path)
    rows = []
    for (algorithm, preprocessing), group in detections.groupby(["algorithm", "preprocessing"]):
        metrics = rate_metrics(
            group,
            context["truth_lookup"], context["visible_truth_ids"], context["visible_all_ids"],
            context["invisible_roi_ids"], context["invisible_all_ids"],
            context["subject_of"], context["thresholds"], photometric_threshold=None,
        )
        rows.append({
            "version": version, "algorithm": algorithm, "preprocessing": preprocessing,
            **{name: metrics[name] for name in (
                "returned_n", "usable_n", "end_to_end_visible_rate",
                "wrong_among_returned_rate", "fp_rate_end_to_end", "median_iou",
                "median_center_error_px", "median_diameter_relative_error",
            )},
        })
    return pd.DataFrame(rows)


def old_sequence_table(report: Path) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"^\| (PuReST|PuRe) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| (\d+) \(([0-9.]+)\) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|$")
    for line in report.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        rows.append({
            "version": "before_fix", "algorithm": match.group(1),
            "observed_rate": float(match.group(2)), "visible_coverage": float(match.group(3)),
            "raw_fp_among_p80_closed": float(match.group(4)),
            "interpolated_frames": int(match.group(5)), "interpolated_fraction": float(match.group(6)),
            "diameter_log_jump_median": float(match.group(7)),
            "center_jump_norm_median": float(match.group(8)),
            "recovery_frames_ms_median": float(match.group(9)),
        })
    if len(rows) != 2:
        raise RuntimeError(f"cannot parse legacy sequence table: {report}")
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/preexperiment.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    old_benchmark = config.path_value("historical_benchmark_artifact_root")
    new_benchmark = config.path_value("benchmark_artifact_root")
    old_sequence = config.path_value("historical_sequence_artifact_root")
    new_sequence = config.path_value("sequence_artifact_root")

    benchmark = pd.concat([
        benchmark_ungated(config, old_benchmark / "default" / "detections.csv", "before_fix"),
        benchmark_ungated(config, new_benchmark / "default" / "detections.csv", "axis_fixed"),
    ], ignore_index=True).sort_values(["algorithm", "preprocessing", "version"])
    benchmark.to_csv(new_benchmark / "benchmark_comparison_ungated.csv", index=False, encoding="utf-8-sig")

    old_seq = old_sequence_table(old_sequence / "benchmark_report_sequence.md")
    new_seq = pd.read_csv(new_sequence / "sequence_summary.csv")
    fields = [
        "algorithm", "observed_rate", "visible_coverage", "raw_fp_among_p80_closed",
        "interpolated_frames", "interpolated_fraction", "diameter_log_jump_median",
        "center_jump_norm_median", "recovery_frames_ms_median",
    ]
    new_seq = new_seq[fields].copy()
    new_seq.insert(0, "version", "adapter_fixed")
    sequence = pd.concat([old_seq, new_seq], ignore_index=True).sort_values(["algorithm", "version"])
    sequence.to_csv(new_sequence / "sequence_comparison.csv", index=False, encoding="utf-8-sig")

    best = pd.read_csv(new_benchmark / "default" / "benchmark_summary.csv").sort_values(
        "end_to_end_visible_rate", ascending=False
    ).iloc[0]
    lines = [
        "# NIR历史结果修复复核摘要", "",
        "> 08-16（Asia/Shanghai）｜同一528眼与同一44×121历史序列；旧产物不覆盖。", "",
        "## 阶段4/4b", "",
        f"- 轴角与光度口径修复后，默认组最高端到端可用率为{best['end_to_end_visible_rate']:.3f}（{best['algorithm']}/{best['preprocessing']}），仍未达0.85准入门。",
        "- `benchmark_comparison_ungated.csv`使用同一版评价器比较旧/新检测，隔离轴角修复本身；带光度门的正式复核见`default/benchmark_summary.csv`。",
        "- 18组调优结果见`tuned/tuned_summary.csv`，不得继续引用旧best=22.8%或24.1%作为当前结论。", "",
        "## 阶段5", "",
        "- 新适配器保留算法原始返回与Python门控后接受，插值只写副轨。",
        "- `sequence_comparison.csv`给出修复前后覆盖、跳变、恢复和插值数量；正式参数仍需在新成像几何下标定。", "",
        "## 停止门", "",
        "历史结论已复核；可以开始正式ROI比较脚本的Block起点/时间戳与测速口径实现，但在60时点人工复核完成前不冻结ROI。",
    ]
    (new_sequence / "history_review_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(new_benchmark / "benchmark_comparison_ungated.csv")
    print(new_sequence / "sequence_comparison.csv")
    print(new_sequence / "history_review_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
