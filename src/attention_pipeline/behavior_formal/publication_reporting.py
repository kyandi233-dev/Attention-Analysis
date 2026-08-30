"""Publication-facing Behavior figures and a complete output/report catalogue.

This module redraws from already-computed formal tables.  It never reloads raw
behavior data and never refits a statistical model.  In-image text is English
to avoid platform-specific CJK font failures; filenames/report prose may remain
Chinese.  Figures use external captions, explicit units, participant-first
uncertainty where repeated rows exist, and both PNG (300 dpi) and SVG outputs.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .science_v3_metric_figures import ALL_FIGURE_METRICS

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})

LABELS = {
    "go_correct_rt_mean_ms": "Correct Go RT mean (ms)",
    "go_correct_rt_median_ms": "Correct Go RT median (ms)",
    "go_correct_rt_sd_ms": "Correct Go RT SD (ms)",
    "go_correct_rt_mad_ms": "Correct Go RT MAD (ms)",
    "go_correct_rt_iqr_ms": "Correct Go RT IQR (ms)",
    "go_correct_rt_cv": "Correct Go RT coefficient of variation",
    "go_correct_rt_theilsen_slope_ms_per_s": "Correct Go RT robust slope (ms/s)",
    "raw_go_omission_rate": "Raw Go omission rate",
    "clean_go_omission_rate": "Go omission rate without detected timing ambiguity",
    "timing_ambiguous_go_omission_rate": "Timing-ambiguous Go omission rate",
    "commission_rate": "No-Go commission rate",
    "dprime_loglinear": "d-prime (log-linear)",
    "criterion_c": "Criterion c",
    "beta": "Beta",
    "omission_prestimulus_only_ambiguity_rate": "Prestimulus-only ambiguity rate",
    "omission_carryover_only_ambiguity_rate": "Carry-over-only ambiguity rate",
    "omission_prestimulus_and_carryover_ambiguity_rate": "Prestimulus + carry-over ambiguity rate",
    "late_go_response_candidate_rate": "Late Go-response candidate rate",
    "anticipatory_go_response_candidate_rate": "Anticipatory Go-response candidate rate",
}

VIEWS = (
    "session_distribution",
    "block_pair",
    "cycle_trend",
    "probe_q1_nominal",
    "probe_q2_ordinal",
)


def _read(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False) if path.is_file() else pd.DataFrame()


def _safe(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_")


def _label(metric: str) -> str:
    return LABELS.get(metric, metric.replace("_", " "))


def _participant_col(frame: pd.DataFrame) -> str | None:
    for name in ("participant_group_id", "repeat_participant_id", "participant_key"):
        if name in frame.columns:
            return name
    return None


def _sem(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) >= 2 else math.nan


def _participant_summary(frame: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    pcol = _participant_col(frame)
    if pcol is None or metric not in frame:
        return pd.DataFrame()
    current = frame[[pcol, *group_cols, metric]].copy()
    current[metric] = pd.to_numeric(current[metric], errors="coerce")
    current = current.dropna(subset=[pcol, metric])
    if current.empty:
        return pd.DataFrame()
    participant = current.groupby([*group_cols, pcol], as_index=False, dropna=False)[metric].mean()
    summary = participant.groupby(group_cols, as_index=False, dropna=False)[metric].agg(mean="mean", participant_n="count")
    sem = participant.groupby(group_cols, dropna=False)[metric].apply(_sem).rename("sem").reset_index()
    return summary.merge(sem, on=group_cols, how="left", validate="one_to_one")


def _save(fig: plt.Figure, stem: Path) -> tuple[str, str]:
    for ax in fig.axes:
        ax.set_title("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")
    fig.tight_layout()
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return str(png), str(svg)


def _use_log(metric: str, values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    finite = finite[finite > 0]
    return metric == "beta" and len(finite) >= 2 and float(finite.max() / finite.min()) >= 10.0


def _figure_row(metric: str, view: str, status: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "metric": metric,
        "metric_label_en": _label(metric),
        "view": view,
        "status": status,
        "reason": reason,
        "internal_title": False,
        "in_image_language": "English",
        "caption_location": "external_manifest_and_report",
        **extra,
    }


def generate_behavior_publication_pack(formal_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    formal_root = Path(formal_root)
    out = formal_root / "figures_publication"
    session = _read(formal_root, "session_metrics.csv")
    block = _read(formal_root, "block_metrics.csv")
    cycle = _read(formal_root, "cycle_metrics.csv")
    probe = _read(formal_root, "probe_primary_30s.csv")
    frames = {
        "session_distribution": session,
        "block_pair": block,
        "cycle_trend": cycle,
        "probe_q1_nominal": probe,
        "probe_q2_ordinal": probe,
    }
    metrics = [m for m in ALL_FIGURE_METRICS if any(m in f.columns for f in frames.values())]
    manifest: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for metric in metrics:
        for view in VIEWS:
            frame = frames[view]
            if frame.empty or metric not in frame:
                audit.append(_figure_row(metric, view, "not_estimable", "source table empty or metric absent"))
                continue
            try:
                fig: plt.Figure
                ax: plt.Axes
                display_scale = "linear"
                uncertainty = "none"
                observation_unit = "session"
                caption_en = ""
                caption_zh = ""
                if view == "session_distribution":
                    values = pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy(float)
                    if len(values) < 2:
                        raise ValueError("fewer than two finite session values")
                    fig, ax = plt.subplots(figsize=(6.4, 4.2))
                    ax.hist(values, bins=min(20, max(5, int(np.sqrt(len(values))))), edgecolor="black")
                    ax.set_xlabel(_label(metric)); ax.set_ylabel("Sessions")
                    if _use_log(metric, values):
                        ax.set_xscale("log"); display_scale = "log-x"
                    caption_en = f"Session-level distribution of {_label(metric)}."
                    caption_zh = f"{_label(metric)} 的场次级分布；仅作描述性展示。"
                elif view == "block_pair":
                    if not {"session_id", "block_id"}.issubset(frame.columns):
                        raise ValueError("session_id/block_id unavailable")
                    wide = frame.pivot_table(index="session_id", columns="block_id", values=metric, aggfunc="first")
                    if not {"B1", "B2"}.issubset(wide.columns):
                        raise ValueError("B1/B2 pair unavailable")
                    wide = wide.dropna(subset=["B1", "B2"])
                    if len(wide) < 2:
                        raise ValueError("fewer than two complete session pairs")
                    fig, ax = plt.subplots(figsize=(5.8, 4.4))
                    for row in wide.itertuples(index=False):
                        ax.plot([1, 2], [float(row.B1), float(row.B2)], marker="o", linewidth=.7, alpha=.22)
                    ax.plot([1, 2], [wide["B1"].mean(), wide["B2"].mean()], marker="o", linewidth=2.2, label="Session-pair mean")
                    ax.set_xticks([1, 2], ["B1", "B2"]); ax.set_xlabel("Block"); ax.set_ylabel(_label(metric)); ax.legend(frameon=False)
                    vals = wide[["B1", "B2"]].to_numpy(float).ravel()
                    if _use_log(metric, vals):
                        ax.set_yscale("log"); display_scale = "log-y"
                    caption_en = f"Within-session B1/B2 paired trajectories for {_label(metric)}. Thin lines are sessions; the heavy line is the session-pair mean."
                    caption_zh = f"{_label(metric)} 的场次内 B1/B2 配对轨迹；细线为 session，粗线为场次配对均值。正式推断仍按参与者聚类。"
                elif view == "cycle_trend":
                    if not {"block_id", "cycle_bin"}.issubset(frame.columns):
                        raise ValueError("block_id/cycle_bin unavailable")
                    summary = _participant_summary(frame, ["block_id", "cycle_bin"], metric)
                    if summary.empty:
                        raise ValueError("participant-first cycle summary unavailable")
                    fig, ax = plt.subplots(figsize=(6.6, 4.3))
                    styles = ["o-", "s--", "^-."]
                    for idx, (block_id, cur) in enumerate(summary.groupby("block_id", sort=True)):
                        cur = cur.sort_values("cycle_bin")
                        ax.errorbar(cur["cycle_bin"], cur["mean"], yerr=cur["sem"], fmt=styles[idx % len(styles)], capsize=3, label=str(block_id))
                    ax.set_xlabel("Cycle within block"); ax.set_ylabel(_label(metric)); ax.legend(title="Block", frameon=False)
                    vals = pd.to_numeric(summary["mean"], errors="coerce").to_numpy(float)
                    if _use_log(metric, vals):
                        ax.set_yscale("log"); display_scale = "log-y"
                    uncertainty = "SEM after participant-first aggregation"; observation_unit = "participant group"
                    caption_en = f"Participant-first descriptive time-on-task profile of {_label(metric)}; error bars are SEM across participant groups."
                    caption_zh = f"{_label(metric)} 的区块内任务进程；先在参与者内汇总，误差线为参与者组间 SEM。"
                else:
                    category = "q1_nominal_4class" if view == "probe_q1_nominal" else "q2_ordinal_4level"
                    if category not in frame:
                        raise ValueError(f"{category} unavailable")
                    current = frame.dropna(subset=[category]).copy()
                    summary = _participant_summary(current, [category], metric)
                    if summary.empty:
                        raise ValueError("participant-first probe summary unavailable")
                    summary[category] = pd.to_numeric(summary[category], errors="coerce")
                    summary = summary.dropna(subset=[category]).sort_values(category)
                    if len(summary) < 2:
                        raise ValueError("fewer than two populated probe categories")
                    fig, ax = plt.subplots(figsize=(6.0, 4.2))
                    x = summary[category].to_numpy(float)
                    if view == "probe_q1_nominal":
                        ax.errorbar(x, summary["mean"], yerr=summary["sem"], fmt="o", linestyle="none", capsize=3)
                        ax.set_xlabel("Q1 category (nominal; 1-4)")
                        caption_en = f"{_label(metric)} in the 30-s pre-probe window by nominal Q1 category; categories are not connected as an ordinal trend."
                        caption_zh = f"{_label(metric)} 在探针前 30 秒主窗、按 Q1 名义四分类的参与者优先描述；类别不连线。"
                    else:
                        ax.errorbar(x, summary["mean"], yerr=summary["sem"], fmt="o-", capsize=3)
                        ax.set_xlabel("Q2 vigilance level (ordinal; 1-4)")
                        caption_en = f"{_label(metric)} in the 30-s pre-probe window across ordinal Q2 vigilance levels."
                        caption_zh = f"{_label(metric)} 在探针前 30 秒主窗、随 Q2 有序警觉等级的参与者优先描述。"
                    ax.set_xticks([1, 2, 3, 4]); ax.set_ylabel(_label(metric))
                    vals = pd.to_numeric(summary["mean"], errors="coerce").to_numpy(float)
                    if _use_log(metric, vals):
                        ax.set_yscale("log"); display_scale = "log-y"
                    uncertainty = "SEM after participant-first aggregation"; observation_unit = "participant group"

                stem = out / f"behavior_{_safe(metric)}__{view}"
                png, svg = _save(fig, stem)
                row = _figure_row(
                    metric, view, "generated", "", png=Path(png).name, svg=Path(svg).name,
                    caption_en=caption_en, caption_zh=caption_zh, display_scale=display_scale,
                    uncertainty=uncertainty, observation_unit=observation_unit,
                    display_transform_changes_analysis=False,
                )
                manifest.append(row); audit.append(row.copy())
            except Exception as exc:
                try:
                    plt.close(fig)
                except Exception:
                    pass
                audit.append(_figure_row(metric, view, "not_estimable", f"{type(exc).__name__}: {exc}"))

    manifest_df = pd.DataFrame(manifest)
    audit_df = pd.DataFrame(audit)
    manifest_df.to_csv(formal_root / "behavior_publication_figure_manifest.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(formal_root / "behavior_publication_figure_coverage.csv", index=False, encoding="utf-8-sig")
    return manifest_df, audit_df


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "_No rows._\n"
    d = frame[[c for c in columns if c in frame.columns]].copy()
    if limit is not None:
        d = d.head(limit)
    d = d.fillna("")
    headers = list(d.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in d.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in row) + " |")
    return "\n".join(lines) + "\n"


def write_complete_behavior_report(formal_root: Path, manifest: pd.DataFrame, audit: pd.DataFrame) -> Path:
    formal_root = Path(formal_root)
    run_manifest_path = formal_root / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8-sig")) if run_manifest_path.is_file() else {}
    csv_rows: list[dict[str, Any]] = []
    for path in sorted(formal_root.glob("*.csv")):
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            csv_rows.append({"file": path.name, "rows": len(frame), "columns": len(frame.columns)})
        except Exception as exc:
            csv_rows.append({"file": path.name, "rows": "unreadable", "columns": type(exc).__name__})
    inventory = pd.DataFrame(csv_rows)

    b12 = _read(formal_root, "b1_b2_participant_cluster_bootstrap.csv")
    q1 = _read(formal_root, "q1_nominal_models.csv")
    q2 = _read(formal_root, "q2_ordinal_gee_models.csv")
    failures = _read(formal_root, "model_failures.csv")
    endpoint = _read(formal_root, "behavior_endpoint_decisions.csv")

    topo = run_manifest.get("current_analysis_cohort_topology", {})
    lines = [
        "# Behavior 正式分析完整结果说明与图表映射",
        "",
        "本文件由已完成的正式分析表重新生成，不重新读取原始行为数据、不重新拟合模型。图片内部统一使用英文，避免中文字体依赖；中文解释与完整图注保留在本文件和 figure manifest 中。",
        "",
        "## 1. 分析范围与统计单位",
        "",
        f"- governed sessions: **{topo.get('sessions', 'unknown')}**",
        f"- participant groups: **{topo.get('analysis_groups', 'unknown')}**",
        f"- exactly-two-session groups: **{topo.get('double_session_repeat_groups', 'unknown')}**",
        "- session 是采集单位；重复参与者的正式推断与预测分折必须以 participant group 为簇。",
        "- Go omission 与 No-Go commission 使用不同机会数分母；Q1 是名义四分类，Q2 是有序四级。",
        "",
        "## 2. 输出表总目录",
        "",
        _markdown_table(inventory, ["file", "rows", "columns"]),
        "## 3. B1/B2 参与者聚类效应",
        "",
        _markdown_table(b12, ["metric", "estimate_b2_minus_b1", "ci_low", "ci_high", "participant_group_n", "session_pair_n", "status"]),
        "## 4. 主观探针模型",
        "",
        "### Q1 nominal models",
        _markdown_table(q1, ["model_name", "predictor", "contrast_category", "reference_category", "estimate_per_predictor_sd", "se", "ci_low", "ci_high", "participant_group_n", "session_n", "n_rows", "status"]),
        "### Q2 ordinal GEE models",
        _markdown_table(q2, ["model_name", "predictor", "estimate_per_predictor_sd", "se", "ci_low", "ci_high", "participant_group_n", "session_n", "n_rows", "status"]),
        "## 5. 模型失败与不可估计项目",
        "",
        _markdown_table(failures, list(failures.columns) if not failures.empty else ["status"]),
        "## 6. 候选指标/正式结局决策",
        "",
        _markdown_table(endpoint, list(endpoint.columns) if not endpoint.empty else ["status"]),
        "## 7. Publication figure manifest",
        "",
        "PNG 为 300 dpi 预览/报告图；SVG 为可编辑矢量版。图内无大标题，单位写入坐标轴。重复测量图的误差线在 manifest 中明确其统计含义。Beta 在动态范围超过 10 倍时使用对数显示轴，但分析值本身不变，也不删除极端值。",
        "",
        _markdown_table(manifest, ["metric", "view", "png", "svg", "display_scale", "observation_unit", "uncertainty", "caption_zh", "caption_en"]),
        "## 8. 未生成图的完整审计",
        "",
        _markdown_table(audit[audit.get("status", pd.Series(dtype=str)).ne("generated")], ["metric", "view", "status", "reason"]),
        "## 9. 解释边界",
        "",
        "- 描述图不替代正式模型；session-level thin lines 不等价于独立参与者。",
        "- `clean_go_omission` 仅表示未检测到预定义 motor-timing ambiguity，不等同于已证明的注意失败。",
        "- 候选指标不能按显著性筛选后再宣称为预设结局。",
        "- 显示尺度（例如 beta 的 log axis）只改变可视化，不改变保存的统计量或模型。",
    ]
    path = formal_root / "结果说明_完整.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def redraw_behavior_publication(formal_root: Path) -> dict[str, Any]:
    formal_root = Path(formal_root)
    required = ["session_metrics.csv", "block_metrics.csv", "cycle_metrics.csv", "probe_primary_30s.csv"]
    missing = [name for name in required if not (formal_root / name).is_file()]
    if missing:
        raise FileNotFoundError("completed Behavior formal tables missing: " + ", ".join(missing))
    manifest, audit = generate_behavior_publication_pack(formal_root)
    report = write_complete_behavior_report(formal_root, manifest, audit)
    return {
        "status": "complete",
        "formal_root": str(formal_root),
        "publication_figure_n": int(len(manifest)),
        "figure_audit_n": int(len(audit)),
        "report": str(report),
        "figure_root": str(formal_root / "figures_publication"),
        "analysis_refit": False,
        "raw_behavior_reload": False,
    }
