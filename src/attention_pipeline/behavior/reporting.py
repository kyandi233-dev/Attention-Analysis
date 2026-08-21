from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution

from ..config import Config
from ..contracts import OUTPUT_DIRS, PROBE_LABELS
from ..io import block_windows, subject_paths
from ..metadata import run_metadata, source_id
from .evidence import cohort_probe_evidence, cohort_rolling_evidence
from .extract import behavior_files, block_metrics, extract_trials


PROBE_COLORS = {
    1: "#2878B5",
    2: "#E69F00",
    3: "#7A5195",
    4: "#6B7280",
}
NEUTRAL = "#687386"
ACCENT = "#2878B5"
WARNING = "#D97706"
GRID = "#D9DEE7"


@dataclass(frozen=True)
class BehaviorReportPaths:
    root: Path
    reports: Path
    behavior: Path
    manifests: Path


def _configure_chinese_font() -> str:
    installed = {item.name for item in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        if candidate in installed:
            plt.rcParams["font.sans-serif"] = [candidate]
            plt.rcParams["axes.unicode_minus"] = False
            return candidate
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


def output_paths(config: Config, create: bool = False) -> BehaviorReportPaths:
    root = config.path_value("output_root")
    paths = BehaviorReportPaths(
        root=root,
        reports=root / "000-reports",
        behavior=root / "040-behavior",
        manifests=root / "090-manifests",
    )
    if create:
        for directory in (paths.root, paths.reports, paths.behavior, paths.manifests):
            directory.mkdir(parents=True, exist_ok=True)
    return paths


def load_cohort(config: Config) -> pd.DataFrame:
    frames = []
    for subject in config.data["subjects"]["include"]:
        frame = extract_trials(config, subject)
        frame.insert(0, "subject", subject)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["probe_response"] = pd.to_numeric(result["probe_response"], errors="coerce").astype("Int64")
    return result


def behavior_sources(config: Config) -> list[Path]:
    """List the exact config and 66 read-only behavior CSVs used by cohort reports."""
    return [
        config.path,
        *[
            source
            for subject in config.data["subjects"]["include"]
            for source in behavior_files(config, subject)
        ],
    ]


def _with_metadata(table: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    result = table.copy()
    result.insert(0, "pipeline_version", metadata["pipeline_version"])
    result.insert(1, "config_digest", metadata["config_digest"])
    result.insert(2, "generated_at", metadata["generated_at"])
    return result


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style_axis(axis, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def _category_rt(rt: pd.Series) -> pd.Categorical:
    return pd.cut(
        rt,
        bins=[-np.inf, 100, 150, 1000, 1150, np.inf],
        labels=["<100", "100–<150", "150–1000", ">1000–1150", ">1150"],
        right=False,
    )


def phase1_tables(config: Config, trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    subject_block = (
        trials.groupby(["subject", "block_num", "condition"], as_index=False)
        .agg(
            trials=("trial_num", "size"),
            probe_count=("is_probe", "sum"),
            rt_available=("rt", lambda values: int(values.notna().sum())),
            timestamp_qc_fail=("rt_qc_timestamp_inconsistent", "sum"),
            first_trial_ms=("absolute_onset_time", "min"),
            last_trial_ms=("absolute_onset_time", "max"),
        )
    )
    subject_block["observed_trial_span_sec"] = (
        subject_block["last_trial_ms"] - subject_block["first_trial_ms"]
    ) / 1000

    rt_rows = trials.loc[trials["rt"].notna(), ["subject", "block_num", "condition", "rt"]].copy()
    rt_rows["rt_qc_category"] = _category_rt(rt_rows["rt"])
    rt_qc = (
        rt_rows.groupby(["subject", "rt_qc_category"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    rt_qc["proportion_with_rt"] = rt_qc["count"] / rt_qc.groupby("subject")["count"].transform("sum")

    probes = trials.loc[trials["is_probe"].eq(1)].copy()
    probe_counts = (
        probes.groupby(["subject", "probe_response", "probe_state_label"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )

    timeline_rows = []
    raw_root = config.path_value("raw_root")
    for subject in config.data["subjects"]["include"]:
        paths = subject_paths(raw_root, subject)
        windows = block_windows(paths["master_timeline"])
        for window in windows:
            timeline_rows.append({
                "subject": subject,
                "block_num": window["block_num"],
                "condition": window["condition"],
                "block_duration_sec": (window["end_ms"] - window["start_ms"]) / 1000,
                "block_start_ms": window["start_ms"],
                "block_end_ms": window["end_ms"],
            })
    timelines = pd.DataFrame(timeline_rows)
    timelines = timelines.sort_values(["subject", "block_num"])
    timelines["rest_before_next_sec"] = (
        timelines.groupby("subject")["block_start_ms"].shift(-1) - timelines["block_end_ms"]
    ) / 1000

    summary = pd.DataFrame([
        {"metric": "subjects", "value": trials["subject"].nunique(), "unit": "人"},
        {"metric": "formal_trials", "value": len(trials), "unit": "试次"},
        {"metric": "probes", "value": int(trials["is_probe"].sum()), "unit": "次"},
        {"metric": "rt_available", "value": int(trials["rt"].notna().sum()), "unit": "试次"},
        {"metric": "rt_lt_100", "value": int(trials["rt_qc_lt_100"].sum()), "unit": "试次"},
        {"metric": "rt_lt_150", "value": int(trials["rt_qc_lt_150"].sum()), "unit": "试次"},
        {"metric": "rt_gt_1000", "value": int(trials["rt_qc_gt_1000"].sum()), "unit": "试次"},
        {"metric": "rt_gt_1150", "value": int(trials["rt_qc_gt_1150"].sum()), "unit": "试次"},
        {"metric": "timestamp_inconsistent", "value": int(trials["rt_qc_timestamp_inconsistent"].sum()), "unit": "试次"},
        {"metric": "duplicate_subject_block_trial", "value": int(trials.duplicated(["subject", "block_num", "trial_num"]).sum()), "unit": "行"},
    ])
    return {
        "summary": summary,
        "subject_block_audit": subject_block,
        "rt_qc": rt_qc,
        "probe_counts": probe_counts,
        "timeline": timelines,
    }


def plot_phase1_completeness(subject_block: pd.DataFrame, path: Path) -> None:
    pivot = subject_block.pivot(index="subject", columns="block_num", values="trials")
    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=216, cmap="Blues")
    axis.set_xticks(range(6), [f"Block {value}" for value in pivot.columns])
    axis.set_yticks(range(len(pivot)), pivot.index)
    axis.set_xlabel("正式 Block")
    axis.set_ylabel("被试")
    axis.set_title("正式行为试次完整性（期望每格216试次）")
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.iloc[row, column]
            axis.text(column, row, f"{int(value)}", ha="center", va="center", color="white" if value > 130 else "#1F2937", fontsize=8)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("试次数")
    _save_figure(fig, path)


def plot_phase1_rt_distribution(trials: pd.DataFrame, path: Path) -> None:
    values = trials.loc[trials["rt"].notna(), "rt"].to_numpy(dtype=float)
    ordered = np.sort(values)
    ecdf = np.arange(1, len(ordered) + 1) / len(ordered)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    axes[0].hist(values, bins=np.arange(0, 1201, 25), color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.25)
    for threshold, label in ((100, "100"), (150, "150"), (1000, "1000"), (1150, "1150")):
        axes[0].axvline(threshold, color=WARNING if threshold < 200 else NEUTRAL, linewidth=1.2, linestyle="--")
        axes[0].text(threshold, axes[0].get_ylim()[1] * 0.96, label, rotation=90, va="top", ha="right", fontsize=8)
    axes[0].set(xlabel="程序归属 RT（ms）", ylabel="试次数", title="全部已记录RT分布（不剔除）")
    _style_axis(axes[0])
    axes[1].plot(ordered, ecdf, color=ACCENT, linewidth=1.8)
    axes[1].set(xlabel="程序归属 RT（ms）", ylabel="累积比例", title="RT经验累积分布")
    axes[1].set_xlim(0, 1200)
    axes[1].set_ylim(0, 1)
    _style_axis(axes[1])
    fig.suptitle(f"N={len(values):,} 个有RT试次；阈值仅作QC标记")
    fig.tight_layout()
    _save_figure(fig, path)


def plot_phase1_rt_qc(rt_qc: pd.DataFrame, path: Path) -> None:
    pivot = rt_qc.pivot(index="subject", columns="rt_qc_category", values="proportion_with_rt").fillna(0)
    colors = ["#B91C1C", "#F59E0B", "#4C78A8", "#7C8A9A", "#374151"]
    fig, axis = plt.subplots(figsize=(10.5, 5.2))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for category, color in zip(pivot.columns, colors):
        values = pivot[category].to_numpy()
        axis.bar(x, values, bottom=bottom, label=f"{category} ms", color=color, width=0.72)
        bottom += values
    axis.set_xticks(x, pivot.index, rotation=35, ha="right")
    axis.set_xlabel("被试")
    axis.set_ylabel("有RT试次中的比例")
    axis.set_title("RT区间组成：短RT保留并用于固定周期策略分析")
    axis.set_ylim(0, 1)
    axis.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _style_axis(axis)
    fig.tight_layout()
    _save_figure(fig, path)


def plot_phase1_probe_counts(probe_counts: pd.DataFrame, path: Path) -> None:
    counts = probe_counts.pivot_table(index="subject", columns="probe_response", values="count", fill_value=0)
    counts = counts.reindex(columns=[1, 2, 3, 4], fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    bottom = np.zeros(len(counts))
    x = np.arange(len(counts))
    for state in [1, 2, 3, 4]:
        values = counts[state].to_numpy()
        axes[0].bar(x, values, bottom=bottom, color=PROBE_COLORS[state], label=f"{state}｜{PROBE_LABELS[state]}")
        bottom += values
    axes[0].set_xticks(x, counts.index, rotation=35, ha="right")
    axes[0].set(xlabel="被试", ylabel="探针次数", title="每名被试的四类探针分布")
    axes[0].set_ylim(0, 24)
    _style_axis(axes[0])
    totals = counts.sum(axis=0)
    bars = axes[1].bar([str(x) for x in totals.index], totals.values, color=[PROBE_COLORS[int(x)] for x in totals.index])
    for bar, value in zip(bars, totals.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 2, str(int(value)), ha="center", va="bottom")
    axes[1].set(xlabel="探针名义类别", ylabel="探针次数", title="全体分布（类别不是分数）")
    _style_axis(axes[1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    _save_figure(fig, path)


def plot_phase1_timeline(timeline: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    for subject, group in timeline.groupby("subject"):
        axes[0].plot(group["block_num"], group["block_duration_sec"], marker="o", alpha=0.35, linewidth=1)
    median = timeline.groupby("block_num")["block_duration_sec"].median()
    axes[0].plot(median.index, median.values, color="#111827", marker="o", linewidth=2.5, label="跨被试中位数")
    axes[0].set(xlabel="Block", ylabel="block_start至block_stop（秒）", title="正式Block时长")
    axes[0].legend(frameon=False)
    _style_axis(axes[0])
    rest = timeline.dropna(subset=["rest_before_next_sec"])
    for subject, group in rest.groupby("subject"):
        axes[1].plot(group["block_num"], group["rest_before_next_sec"], marker="o", alpha=0.35, linewidth=1)
    median_rest = rest.groupby("block_num")["rest_before_next_sec"].median()
    axes[1].plot(median_rest.index, median_rest.values, color="#111827", marker="o", linewidth=2.5)
    axes[1].set(xlabel="前一个Block编号", ylabel="至下一Block开始（秒）", title="Block间隔（含休息及过渡）")
    _style_axis(axes[1])
    fig.tight_layout()
    _save_figure(fig, path)


def _report_html(title: str, generated_at: str, sections: Iterable[tuple[str, str]], figures: list[tuple[str, str, str]]) -> str:
    section_html = "".join(f"<section><h2>{html.escape(heading)}</h2>{body}</section>" for heading, body in sections)
    figure_html = "".join(
        f'<figure><img src="{html.escape(filename)}" alt="{html.escape(alt)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        for filename, alt, caption in figures
    )
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{font-family:"Microsoft YaHei",system-ui,sans-serif;max-width:1120px;margin:32px auto;padding:0 22px;color:#172033;line-height:1.7}}h1{{line-height:1.25}}h2{{margin-top:32px;border-bottom:1px solid #d7dde7;padding-bottom:6px}}.meta{{color:#5b6575}}.notice{{border-left:4px solid #2878b5;padding:10px 14px;background:#eef6fb}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #d7dde7;padding:7px;text-align:left}}figure{{margin:28px 0}}img{{max-width:100%;height:auto}}figcaption{{color:#5b6575;font-size:0.92rem}}code{{background:#f1f3f6;padding:1px 4px}}</style></head><body><h1>{html.escape(title)}</h1><p class="meta">生成时间：{html.escape(generated_at)}</p>{section_html}<h2>图表</h2>{figure_html}</body></html>'''


def _markdown_table(table: pd.DataFrame) -> str:
    columns = list(table.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def write_directory_entries(paths: BehaviorReportPaths, metadata: dict) -> None:
    timestamp = metadata["generated_at"]
    entries = {
        paths.reports / "00-目录与映射.md": f"# 00｜报告目录与映射\n\n> {timestamp}｜行为可视化报告入口；当前不含专注分数。\n\n- `041-*`：数据质量与基础描述。\n- `042-*`：固定序列与Block效应。\n- `043-*`：窗口证据与四类探针。\n- `044-*`：实验程序修改建议与证据对照。\n- `045-*`：行为字段字典、缺失语义与冻结状态。\n",
        paths.behavior / "00-目录与映射.md": f"# 00｜行为数据目录与映射\n\n> {timestamp}｜逐试次、block、窗口和探针证据表；QC标记不执行删除。\n",
        paths.manifests / "00-目录与映射.md": f"# 00｜运行清单目录与映射\n\n> {timestamp}｜记录版本、配置摘要、生成时间与只读源文件标识。\n",
    }
    for path, text in entries.items():
        path.write_text(text, encoding="utf-8")


def generate_phase1(config: Config) -> dict:
    font = _configure_chinese_font()
    paths = output_paths(config, create=True)
    trials = load_cohort(config)
    metadata = run_metadata(config, behavior_sources(config))
    tables = phase1_tables(config, trials)
    write_directory_entries(paths, metadata)

    table_paths = {}
    for name, table in {"trials": trials, **tables}.items():
        destination = paths.behavior / f"041-{name}.csv"
        _with_metadata(table, metadata).to_csv(destination, index=False, encoding="utf-8-sig")
        table_paths[name] = str(destination)

    figures = [
        ("041-01-数据完整性热图.png", "被试和block的试次完整性热图", "每格应为216试次；用于发现缺block或截断记录。"),
        ("041-02-RT分布与ECDF.png", "全部程序归属RT的直方图和累积分布", "阈值只标记，不删除；图中保留极短和极长RT。"),
        ("041-03-被试RT区间组成.png", "各被试RT区间组成堆积图", "短RT的被试差异提示需要进一步检查固定周期策略。"),
        ("041-04-四类探针分布.png", "四类探针按被试与全体的频数", "四类为名义分类；数量明显不平衡，后续必须显示真实n。"),
        ("041-05-Block与间隔时长.png", "正式block和block间隔时长", "时间变量是设计与质量信息，不是专注真值。"),
    ]
    plot_phase1_completeness(tables["subject_block_audit"], paths.reports / figures[0][0])
    plot_phase1_rt_distribution(trials, paths.reports / figures[1][0])
    plot_phase1_rt_qc(tables["rt_qc"], paths.reports / figures[2][0])
    plot_phase1_probe_counts(tables["probe_counts"], paths.reports / figures[3][0])
    plot_phase1_timeline(tables["timeline"], paths.reports / figures[4][0])

    probe_totals = trials.loc[trials["is_probe"].eq(1), "probe_response"].value_counts().reindex([1, 2, 3, 4], fill_value=0)
    rt = trials["rt"].dropna()
    overview = pd.DataFrame([
        ["正式试次", f"{len(trials):,}"],
        ["探针", f"{int(trials['is_probe'].sum()):,}"],
        ["有RT试次", f"{len(rt):,}"],
        ["RT中位数", f"{rt.median():.2f} ms"],
        ["RT <100 ms", f"{int(trials['rt_qc_lt_100'].sum()):,}"],
        ["RT <150 ms", f"{int(trials['rt_qc_lt_150'].sum()):,}"],
        ["RT >1000 ms", f"{int(trials['rt_qc_gt_1000'].sum()):,}"],
        ["时间戳不一致", f"{int(trials['rt_qc_timestamp_inconsistent'].sum()):,}"],
    ], columns=["项目", "结果"])
    probe_table = pd.DataFrame({
        "探针类别": [f"{state}｜{PROBE_LABELS[state]}" for state in [1, 2, 3, 4]],
        "次数": [int(probe_totals[state]) for state in [1, 2, 3, 4]],
    })
    sections = [
        ("结论边界", '<p class="notice">本报告是预实验数据质量与描述性证据，不删除RT、不合并四类探针、不把时间或fatigue标签当作专注真值。</p>'),
        ("数据完整性", _markdown_table(overview).replace("\n", "<br>") if False else overview.to_html(index=False, border=0)),
        ("四类探针样本量", probe_table.to_html(index=False, border=0) + '<p>类别1占比很高，类别3/4样本很少；后续比较必须优先展示个体数据和不确定性。</p>'),
        ("需要进入下一段验证的问题", '<ul><li>大量小于150ms的RT是否集中在固定的周期位置；</li><li>A/B/C条件的No-Go机会数不同，原始commission次数不可直接横比；</li><li>探针类别不平衡是否来自真实体验、选项理解或界面偏好；</li><li>block与条件完全共线于ABCCBA位置，必须分开呈现。</li></ul>'),
    ]
    html_path = paths.reports / "041-行为数据质量与描述报告.html"
    html_path.write_text(_report_html("041｜行为数据质量与描述报告", metadata["generated_at"], sections, figures), encoding="utf-8")
    md_path = paths.reports / "041-行为数据质量与描述报告.md"
    md_path.write_text(
        "# 041｜行为数据质量与描述报告\n\n"
        f"> {metadata['generated_at']}｜11名被试正式行为数据完整；短RT与探针类别不平衡需进入结构性分析。\n\n"
        "## 结论边界\n\n不删除RT，不合并四类探针，不计算专注分数。\n\n"
        "## 数据概览\n\n" + _markdown_table(overview) + "\n\n"
        "## 四类探针样本量\n\n" + _markdown_table(probe_table) + "\n\n"
        "## 图表导航\n\n" + "\n".join(f"- `{name}`：{caption}" for name, _, caption in figures) + "\n",
        encoding="utf-8",
    )

    sources = []
    for subject in config.data["subjects"]["include"]:
        sources.extend(behavior_files(config, subject))
        sources.append(subject_paths(config.path_value("raw_root"), subject)["master_timeline"])
    manifest = {
        **metadata,
        "stage": "behavior_phase1",
        "font": font,
        "source_files": [{"path": str(path.resolve()), "source_id": source_id(path)} for path in sources],
        "outputs": [str(html_path), str(md_path), *table_paths.values(), *[str(paths.reports / name) for name, _, _ in figures]],
        "attention_score_created": False,
        "rt_rows_deleted": 0,
    }
    manifest_path = paths.manifests / "041-behavior-phase1-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "phase": 1,
        "report_html": str(html_path),
        "report_markdown": str(md_path),
        "figures": [str(paths.reports / name) for name, _, _ in figures],
        "tables": table_paths,
        "manifest": str(manifest_path),
        "rows": len(trials),
        "probe_counts": {int(key): int(value) for key, value in probe_totals.items()},
        "attention_score_created": False,
    }


def phase2_tables(trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    block_frames = []
    for subject, subject_trials in trials.groupby("subject", sort=True):
        metrics = block_metrics(subject_trials)
        metrics.insert(0, "subject", subject)
        block_frames.append(metrics)
    blocks = pd.concat(block_frames, ignore_index=True)

    keys = ["condition", "position_in_cycle"]
    position = (
        trials.groupby(keys, as_index=False)
        .agg(
            opportunities=("trial_num", "size"),
            is_no_go=("is_no_go", "first"),
            responses=("response", lambda value: int(pd.to_numeric(value, errors="coerce").fillna(0).sum())),
            rt_available=("rt", lambda value: int(value.notna().sum())),
            rt_median_ms=("rt", "median"),
            rt_lt_100=("rt_qc_lt_100", "sum"),
            rt_lt_150=("rt_qc_lt_150", "sum"),
            commissions=("commission", "sum"),
            omissions=("omission", "sum"),
        )
    )
    position["response_rate"] = position["responses"] / position["opportunities"]
    position["rt_lt_100_per_opportunity"] = position["rt_lt_100"] / position["opportunities"]
    position["rt_lt_150_per_opportunity"] = position["rt_lt_150"] / position["opportunities"]
    position["error_rate"] = np.where(
        position["is_no_go"].eq(1),
        position["commissions"] / position["opportunities"],
        position["omissions"] / position["opportunities"],
    )

    cycle_keys = ["subject", "block_num", "condition", "cycle_num"]
    base = trials.groupby(cycle_keys, as_index=False).agg(trials=("trial_num", "size"))
    go = (
        trials.loc[trials["is_no_go"].eq(0)]
        .groupby(cycle_keys, as_index=False)
        .agg(
            go_opportunities=("trial_num", "size"),
            go_rt_median_ms=("rt", "median"),
            go_rt_lt_150=("rt_qc_lt_150", "sum"),
            omissions=("omission", "sum"),
        )
    )
    nogo = (
        trials.loc[trials["is_no_go"].eq(1)]
        .groupby(cycle_keys, as_index=False)
        .agg(nogo_opportunities=("trial_num", "size"), commissions=("commission", "sum"))
    )
    cycles = base.merge(go, on=cycle_keys, how="left").merge(nogo, on=cycle_keys, how="left")
    cycles["go_rt_lt_150_rate"] = cycles["go_rt_lt_150"] / cycles["go_opportunities"]
    cycles["omission_rate"] = cycles["omissions"] / cycles["go_opportunities"]
    cycles["commission_rate"] = cycles["commissions"] / cycles["nogo_opportunities"]

    same_condition_pairs = blocks[[
        "subject", "block_num", "condition", "dprime_loglinear", "commission_rate",
        "omission_rate", "go_rt_median_ms",
    ]].copy()
    same_condition_pairs["condition_occurrence"] = same_condition_pairs.groupby(
        ["subject", "condition"]
    )["block_num"].rank(method="first").astype(int)
    return {
        "block_metrics": blocks,
        "position_metrics": position,
        "cycle_metrics": cycles,
        "same_condition_pairs": same_condition_pairs,
    }


def _plot_subject_trajectories(axis, table: pd.DataFrame, value: str, ylabel: str, title: str) -> None:
    for _, group in table.groupby("subject"):
        axis.plot(group["block_num"], group[value], color="#9CA3AF", marker="o", linewidth=0.9, alpha=0.48)
    summary = table.groupby("block_num")[value].agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    summary.columns = ["median", "q25", "q75"]
    axis.fill_between(summary.index, summary["q25"], summary["q75"], color=ACCENT, alpha=0.16, label="跨被试IQR")
    axis.plot(summary.index, summary["median"], color=ACCENT, marker="o", linewidth=2.4, label="跨被试中位数")
    axis.set(xlabel="Block（ABCCBA）", ylabel=ylabel, title=title)
    axis.set_xticks(range(1, 7), ["1 A", "2 B", "3 C", "4 C", "5 B", "6 A"])
    _style_axis(axis)


def plot_phase2_block_metrics(blocks: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.5))
    specs = [
        ("dprime_loglinear", "d′（loglinear）", "正确Go Hit与No-Go commission合成"),
        ("commission_rate", "No-Go commission率", "抑制失败率"),
        ("omission_rate", "Go omission率", "Go遗漏率"),
        ("go_rt_median_ms", "Go RT中位数（ms）", "Go反应速度"),
    ]
    for axis, (column, ylabel, title) in zip(axes.flat, specs):
        _plot_subject_trajectories(axis, blocks, column, ylabel, title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center")
    fig.suptitle("Block行为指标：个体轨迹与跨被试中位数/IQR")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    _save_figure(fig, path)


def plot_phase2_position_profiles(position: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.2, 7.8), sharex=True)
    colors = {"A": "#2878B5", "B": "#E69F00", "C": "#7A5195"}
    markers = {"A": "o", "B": "s", "C": "^"}
    for condition, group in position.groupby("condition"):
        group = group.sort_values("position_in_cycle")
        axes[0].plot(group["position_in_cycle"], group["response_rate"], color=colors[condition], marker=markers[condition], label=f"条件{condition}")
        axes[1].plot(group["position_in_cycle"], group["rt_lt_150_per_opportunity"], color=colors[condition], marker=markers[condition], label=f"条件{condition}")
        nogo = group.loc[group["is_no_go"].eq(1)]
        axes[0].scatter(nogo["position_in_cycle"], nogo["response_rate"], s=120, facecolors="none", edgecolors=colors[condition], linewidths=2)
        axes[1].scatter(nogo["position_in_cycle"], nogo["rt_lt_150_per_opportunity"], s=120, facecolors="none", edgecolors=colors[condition], linewidths=2)
    axes[0].set(ylabel="按键比例", title="每个固定位置的按键比例（空心圈标No-Go位置）")
    axes[1].set(xlabel="18试次周期中的位置", ylabel="RT<150ms / 全部机会", title="极短按键在固定周期位置的分布")
    axes[1].set_xticks(range(1, 19))
    for axis in axes:
        axis.set_ylim(bottom=0)
        _style_axis(axis)
    axes[0].legend(frameon=False, ncol=3)
    fig.suptitle("固定18试次周期产生强位置结构")
    fig.tight_layout()
    _save_figure(fig, path)


def plot_phase2_early_heatmap(position: pd.DataFrame, path: Path) -> None:
    pivot = position.pivot(index="condition", columns="position_in_cycle", values="rt_lt_150_per_opportunity").reindex(index=["A", "B", "C"])
    fig, axis = plt.subplots(figsize=(12, 3.4))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(0.22, float(pivot.max().max())))
    axis.set_xticks(range(18), range(1, 19))
    axis.set_yticks(range(3), ["条件A", "条件B", "条件C"])
    axis.set(xlabel="position_in_cycle", ylabel="条件", title="RT<150ms占全部试次机会的比例")
    for row in range(3):
        for column in range(18):
            value = pivot.iloc[row, column]
            axis.text(column, row, f"{value:.0%}", ha="center", va="center", fontsize=7, color="white" if value > 0.14 else "#1F2937")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("短RT比例")
    fig.tight_layout()
    _save_figure(fig, path)


def plot_phase2_cycle_trends(cycles: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), sharex=True)
    rt_summary = cycles.groupby("cycle_num")["go_rt_median_ms"].agg(
        median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
    )
    axes[0, 0].fill_between(rt_summary.index, rt_summary["q25"], rt_summary["q75"], color=ACCENT, alpha=0.18)
    axes[0, 0].plot(rt_summary.index, rt_summary["median"], color=ACCENT, marker="o", linewidth=2)
    axes[0, 0].set(ylabel="Go RT中位数（ms）", title="Go反应速度：被试×block中位数/IQR")

    rate_specs = [
        (axes[0, 1], "go_rt_lt_150", "go_opportunities", "Go短RT比例", "Go RT<150ms / Go机会"),
        (axes[1, 0], "commissions", "nogo_opportunities", "No-Go commission率", "commission / No-Go机会"),
        (axes[1, 1], "omissions", "go_opportunities", "Go omission率", "omission / Go机会"),
    ]
    for axis, success_column, opportunity_column, ylabel, title in rate_specs:
        grouped = cycles.groupby("cycle_num")[[success_column, opportunity_column]].sum()
        alpha = grouped[success_column] + 0.5
        beta_value = grouped[opportunity_column] - grouped[success_column] + 0.5
        grouped["posterior_mean"] = alpha / (alpha + beta_value)
        grouped["low"] = beta_distribution.ppf(0.025, alpha, beta_value)
        grouped["high"] = beta_distribution.ppf(0.975, alpha, beta_value)
        axis.fill_between(grouped.index, grouped["low"], grouped["high"], color=ACCENT, alpha=0.18)
        axis.plot(grouped.index, grouped["posterior_mean"], color=ACCENT, marker="o", linewidth=2)
        axis.set(ylabel=ylabel, title=title + "（Jeffreys 95%区间）")
    for axis in axes.flat:
        axis.set_xticks(range(1, 13))
        _style_axis(axis)
    axes[1, 0].set_xlabel("Block内cycle编号")
    axes[1, 1].set_xlabel("Block内cycle编号")
    fig.suptitle("Block内部12个cycle：RT分布与按实际机会数汇总的错误率")
    fig.tight_layout()
    _save_figure(fig, path)


def plot_phase2_same_condition(pairs: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.3))
    condition_blocks = {"A": (1, 6), "B": (2, 5), "C": (3, 4)}
    for axis, condition in zip(axes, ["A", "B", "C"]):
        group = pairs.loc[pairs["condition"].eq(condition)]
        pivot = group.pivot(index="subject", columns="condition_occurrence", values="dprime_loglinear")
        for _, row in pivot.iterrows():
            axis.plot([1, 2], row[[1, 2]], color="#9CA3AF", marker="o", alpha=0.6)
        medians = pivot.median()
        axis.plot([1, 2], medians[[1, 2]], color=ACCENT, marker="o", linewidth=3)
        first, second = condition_blocks[condition]
        axis.set_xticks([1, 2], [f"Block {first}", f"Block {second}"])
        axis.set(ylabel="d′（loglinear）" if condition == "A" else "", title=f"条件{condition}的前后重复")
        _style_axis(axis)
    fig.suptitle("同一条件在ABCCBA不同时间位置的d′变化（灰线=被试）")
    fig.tight_layout()
    _save_figure(fig, path)


def generate_phase2(config: Config) -> dict:
    _configure_chinese_font()
    paths = output_paths(config, create=True)
    trials = load_cohort(config)
    metadata = run_metadata(config, behavior_sources(config))
    tables = phase2_tables(trials)
    table_paths = {}
    for name, table in tables.items():
        destination = paths.behavior / f"042-{name}.csv"
        _with_metadata(table, metadata).to_csv(destination, index=False, encoding="utf-8-sig")
        table_paths[name] = str(destination)
    figures = [
        ("042-01-Block行为指标轨迹.png", "四个block行为指标的个体轨迹", "ABCCBA意味着条件和时间位置不能由单一曲线分离。"),
        ("042-02-固定周期位置曲线.png", "18试次固定周期的按键与短RT位置曲线", "空心标记为各条件No-Go位置。"),
        ("042-03-固定位置短RT热图.png", "条件和周期位置的短RT热图", "短RT并非随机散布，而与条件和周期位置共同变化。"),
        ("042-04-Cycle内趋势.png", "12个cycle中的行为指标趋势", "显示block内部学习或时间变化，不作为专注真值。"),
        ("042-05-同条件前后重复.png", "同条件在ABCCBA前后位置的d′配对", "用于展示条件效应与任务时间效应的混淆。"),
    ]
    plot_phase2_block_metrics(tables["block_metrics"], paths.reports / figures[0][0])
    plot_phase2_position_profiles(tables["position_metrics"], paths.reports / figures[1][0])
    plot_phase2_early_heatmap(tables["position_metrics"], paths.reports / figures[2][0])
    plot_phase2_cycle_trends(tables["cycle_metrics"], paths.reports / figures[3][0])
    plot_phase2_same_condition(tables["same_condition_pairs"], paths.reports / figures[4][0])

    blocks = tables["block_metrics"]
    block_summary = blocks.groupby(["block_num", "condition"], as_index=False).agg(
        dprime_median=("dprime_loglinear", "median"),
        commission_median=("commission_rate", "median"),
        omission_median=("omission_rate", "median"),
        go_rt_median_ms=("go_rt_median_ms", "median"),
    ).round(3)
    nogo_counts = trials.loc[trials["is_no_go"].eq(1)].groupby(["condition", "block_num"]).size().groupby("condition").first().div(11).astype(int)
    early_condition = trials.groupby("condition").agg(
        trials=("trial_num", "size"), rt_lt_150=("rt_qc_lt_150", "sum")
    )
    early_condition["rt_lt_150_per_trial"] = early_condition["rt_lt_150"] / early_condition["trials"]
    sections = [
        ("解释边界", '<p class="notice">A/B/C条件与ABCCBA时间位置部分混淆；固定周期位置是设计因素。以下结果用于识别策略学习和实验结构，不把短RT或时间趋势直接定义为不专注。</p>'),
        ("每个Block的No-Go机会", pd.DataFrame({"条件": nogo_counts.index, "每名被试每个该条件Block的No-Go机会": nogo_counts.values}).to_html(index=False, border=0) + '<p>A/B/C分别为48/24/12次，因此必须使用机会数或后验区间，不能比较原始commission次数。</p>'),
        ("Block描述指标", block_summary.to_html(index=False, border=0)),
        ("固定周期的直接证据", '<ul><li>No-Go位置固定：A为5/9/14/18，B为5/14，C仅为5；</li><li>短RT在条件B/C的部分Go位置明显聚集，No-Go位置本身短RT较少；</li><li>因此RT&lt;150ms包含可预测的周期策略成分，不能作为统一删除阈值或单独的注意失败指标。</li></ul>'),
    ]
    html_path = paths.reports / "042-固定序列与Block效应报告.html"
    html_path.write_text(_report_html("042｜固定序列与Block效应报告", metadata["generated_at"], sections, figures), encoding="utf-8")
    md_path = paths.reports / "042-固定序列与Block效应报告.md"
    md_path.write_text(
        "# 042｜固定序列与Block效应报告\n\n"
        f"> {metadata['generated_at']}｜固定18试次周期显著塑造按键与短RT分布，不能把阈值标记直接解释为注意失败。\n\n"
        "## 解释边界\n\nA/B/C条件与ABCCBA时间位置部分混淆；所有结果保持描述性。\n\n"
        "## Block描述指标\n\n" + _markdown_table(block_summary) + "\n\n"
        "## 固定周期结论\n\n- No-Go位置固定：A=5/9/14/18，B=5/14，C=5。\n- 短RT集中度随条件和周期位置变化。\n- 不建议将150ms或1000ms用作静默删除规则。\n\n"
        "## 图表导航\n\n" + "\n".join(f"- `{name}`：{caption}" for name, _, caption in figures) + "\n",
        encoding="utf-8",
    )
    manifest = {
        **metadata,
        "stage": "behavior_phase2",
        "outputs": [str(html_path), str(md_path), *table_paths.values(), *[str(paths.reports / name) for name, _, _ in figures]],
        "attention_score_created": False,
        "rt_rows_deleted": 0,
        "nogo_positions": {"A": [5, 9, 14, 18], "B": [5, 14], "C": [5]},
    }
    manifest_path = paths.manifests / "042-behavior-phase2-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "phase": 2,
        "report_html": str(html_path),
        "report_markdown": str(md_path),
        "figures": [str(paths.reports / name) for name, _, _ in figures],
        "tables": table_paths,
        "manifest": str(manifest_path),
        "attention_score_created": False,
    }


def phase3_tables(config: Config, trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rolling = cohort_rolling_evidence(config, trials)
    probes = cohort_probe_evidence(config, trials)
    rolling_status = (
        rolling.groupby(["time_window_sec", "nogo_window_target", "window_status"], as_index=False)
        .size()
        .rename(columns={"size": "windows"})
    )
    rolling_status["proportion"] = rolling_status["windows"] / rolling_status.groupby(
        ["time_window_sec", "nogo_window_target"]
    )["windows"].transform("sum")

    nogo_span = (
        rolling.groupby(["condition", "nogo_window_target"], as_index=False)
        .agg(
            windows=("window_end_ms", "size"),
            actual_opportunities_median=("nogo_opportunities_actual", "median"),
            actual_span_median_sec=("nogo_actual_span_sec", "median"),
            actual_span_q25_sec=("nogo_actual_span_sec", lambda x: x.quantile(0.25)),
            actual_span_q75_sec=("nogo_actual_span_sec", lambda x: x.quantile(0.75)),
            evidence_age_median_sec=("nogo_evidence_age_sec", "median"),
            full_evidence_rate=("window_status", lambda x: float(pd.Series(x).eq("full_evidence").mean())),
        )
    )

    state_counts = (
        trials.loc[trials["is_probe"].eq(1)]
        .groupby(["probe_response", "probe_state_label"], as_index=False)
        .agg(probes=("trial_num", "size"), subjects=("subject", "nunique"))
    )
    state_by_subject = (
        trials.loc[trials["is_probe"].eq(1)]
        .groupby(["subject", "probe_response", "probe_state_label"], as_index=False)
        .size()
        .rename(columns={"size": "probes"})
    )

    sensitivity_rows = []
    metrics = [
        "go_rt_median_ms",
        "go_rt_lt_150_rate",
        "omission_jeffreys_mean",
        "time_commission_jeffreys_mean",
        "commission_jeffreys_mean",
    ]
    for keys, group in probes.groupby(
        ["time_window_sec", "nogo_window_target", "probe_response", "probe_state_label"], sort=True
    ):
        time_window, nogo_target, state, label = keys
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            sensitivity_rows.append({
                "time_window_sec": int(time_window),
                "nogo_window_target": int(nogo_target),
                "probe_response": int(state),
                "probe_state_label": label,
                "metric": metric,
                "probe_rows": int(len(group)),
                "subjects": int(group["subject"].nunique()),
                "valid_values": int(len(values)),
                "median": float(values.median()) if len(values) else np.nan,
                "q25": float(values.quantile(0.25)) if len(values) else np.nan,
                "q75": float(values.quantile(0.75)) if len(values) else np.nan,
                "mean": float(values.mean()) if len(values) else np.nan,
            })
    sensitivity = pd.DataFrame(sensitivity_rows)
    return {
        "rolling_evidence": rolling,
        "probe_evidence": probes,
        "rolling_status": rolling_status,
        "nogo_span_summary": nogo_span,
        "probe_state_counts": state_counts,
        "probe_state_by_subject": state_by_subject,
        "probe_sensitivity_summary": sensitivity,
    }


def plot_phase3_status(status: pd.DataFrame, path: Path) -> None:
    statuses = ["insufficient_rt", "response_only", "insufficient_nogo", "full_evidence"]
    labels = ["无RT证据", "仅反应证据", "No-Go不足", "完整证据"]
    colors = ["#B91C1C", "#D97706", "#7C8A9A", "#2878B5"]
    combinations = [(time_window, nogo) for time_window in [30, 60, 90, 120] for nogo in [6, 8, 12]]
    pivot = status.pivot_table(index=["time_window_sec", "nogo_window_target"], columns="window_status", values="proportion", fill_value=0).reindex(combinations, fill_value=0)
    fig, axis = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    for state, label, color in zip(statuses, labels, colors):
        values = pivot[state].to_numpy() if state in pivot else np.zeros(len(pivot))
        axis.bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values
    axis.set_xticks(x, [f"{time}s\nNo-Go {nogo}" for time, nogo in combinations])
    axis.set(xlabel="时间窗 × No-Go机会窗", ylabel="窗口比例", title="候选证据窗状态组成（步长10秒且不跨Block）")
    axis.set_ylim(0, 1)
    axis.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _style_axis(axis)
    fig.tight_layout()
    _save_figure(fig, path)


def plot_phase3_nogo_span(span: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    offsets = {6: -0.22, 8: 0.0, 12: 0.22}
    colors = {6: "#2878B5", 8: "#E69F00", 12: "#7A5195"}
    conditions = ["A", "B", "C"]
    for target in [6, 8, 12]:
        group = span.loc[span["nogo_window_target"].eq(target)].set_index("condition").reindex(conditions)
        x = np.arange(3) + offsets[target]
        y = group["actual_span_median_sec"].to_numpy()
        low = y - group["actual_span_q25_sec"].to_numpy()
        high = group["actual_span_q75_sec"].to_numpy() - y
        axes[0].errorbar(x, y, yerr=[low, high], marker="o", capsize=3, color=colors[target], label=f"最近{target}次No-Go")
        axes[1].plot(x, group["full_evidence_rate"], marker="o", color=colors[target], label=f"最近{target}次No-Go")
    for axis in axes:
        axis.set_xticks(range(3), ["条件A\n4 No-Go/cycle", "条件B\n2 No-Go/cycle", "条件C\n1 No-Go/cycle"])
        _style_axis(axis)
    axes[0].set(ylabel="实际时间跨度（秒）", title="No-Go机会窗的实际跨度：中位数与IQR")
    axes[1].set(ylabel="完整证据窗比例", title="达到目标No-Go机会数的比例")
    axes[1].set_ylim(0, 1)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, path)


def plot_phase3_probe_subjects(counts: pd.DataFrame, path: Path) -> None:
    full_subjects = [f"sub-{index:03d}" for index in range(11)]
    pivot = counts.pivot_table(index="subject", columns="probe_response", values="probes", fill_value=0).reindex(index=full_subjects, columns=[1, 2, 3, 4], fill_value=0)
    fig, axis = plt.subplots(figsize=(8.3, 5.3))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="Purples", vmin=0, vmax=24)
    axis.set_xticks(range(4), [f"{state}\n{PROBE_LABELS[state]}" for state in [1, 2, 3, 4]])
    axis.set_yticks(range(11), full_subjects)
    axis.set(xlabel="探针名义类别", ylabel="被试", title="四类探针由多少被试贡献：每格为探针次数")
    for row in range(11):
        for column in range(4):
            value = int(pivot.iloc[row, column])
            axis.text(column, row, str(value), ha="center", va="center", color="white" if value > 10 else "#1F2937")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("探针次数")
    fig.tight_layout()
    _save_figure(fig, path)


def plot_phase3_probe_metrics(probes: pd.DataFrame, path: Path) -> None:
    selected = probes.loc[probes["nogo_window_target"].eq(8)].copy()
    fig, axes = plt.subplots(3, 3, figsize=(13.3, 10.8))
    metrics = [
        ("go_rt_median_ms", "Go RT中位数（ms）"),
        ("go_rt_lt_150_rate", "Go短RT比例"),
        ("time_commission_jeffreys_mean", "时间窗commission后验均值"),
    ]
    for row, (metric, ylabel) in enumerate(metrics):
        for column, duration in enumerate([30, 60, 90]):
            axis = axes[row, column]
            data = selected.loc[selected["time_window_sec"].eq(duration)]
            for state in [1, 2, 3, 4]:
                values = pd.to_numeric(data.loc[data["probe_response"].eq(state), metric], errors="coerce").dropna()
                if len(values):
                    jitter = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else np.array([0.0])
                    axis.scatter(np.full(len(values), state) + jitter, values, color=PROBE_COLORS[state], alpha=0.38, s=18)
                    median = values.median()
                    q25, q75 = values.quantile([0.25, 0.75])
                    axis.errorbar(state, median, yerr=[[median - q25], [q75 - median]], color=PROBE_COLORS[state], marker="D", markersize=6, linewidth=2, capsize=4)
                    axis.text(state, axis.get_ylim()[1] if axis.get_ylim()[1] else median, "", fontsize=1)
            axis.set_xticks([1, 2, 3, 4])
            axis.set_xlabel(f"探针类别（{duration}s窗；最近8次No-Go）")
            axis.set_ylabel(ylabel if column == 0 else "")
            axis.set_title(f"{duration}s：n=214/29/14/7")
            _style_axis(axis)
    fig.suptitle("探针前行为证据：散点=每次探针，菱形/IQR=类别中位数与四分位距")
    fig.tight_layout()
    _save_figure(fig, path)


def plot_phase3_sensitivity(sensitivity: pd.DataFrame, path: Path) -> None:
    metric = "go_rt_median_ms"
    subset = sensitivity.loc[sensitivity["metric"].eq(metric)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for axis, target in zip(axes, [6, 8, 12]):
        data = subset.loc[subset["nogo_window_target"].eq(target)]
        for state in [1, 2, 3, 4]:
            group = data.loc[data["probe_response"].eq(state)].sort_values("time_window_sec")
            axis.plot(group["time_window_sec"], group["median"], color=PROBE_COLORS[state], marker="o", label=f"{state}｜{PROBE_LABELS[state]}")
        axis.set(xlabel="探针前时间窗（秒）", ylabel="Go RT中位数（ms）" if target == 6 else "", title=f"最近{target}次No-Go定义")
        axis.set_xticks([30, 60, 90])
        _style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("窗口敏感性：四类探针的Go RT中位数随候选定义变化")
    fig.tight_layout()
    _save_figure(fig, path)


def generate_phase3(config: Config) -> dict:
    _configure_chinese_font()
    paths = output_paths(config, create=True)
    trials = load_cohort(config)
    metadata = run_metadata(config, behavior_sources(config))
    tables = phase3_tables(config, trials)
    table_paths = {}
    for name, table in tables.items():
        destination = paths.behavior / f"043-{name}.csv"
        _with_metadata(table, metadata).to_csv(destination, index=False, encoding="utf-8-sig")
        table_paths[name] = str(destination)
    figures = [
        ("043-01-候选窗口证据状态.png", "候选窗口的证据充分状态组成", "所有时间窗均不跨Block；不足状态不被伪装为完整证据。"),
        ("043-02-No-Go机会窗实际跨度.png", "不同条件和机会窗的实际跨度", "固定No-Go次数在A/B/C对应不同时间尺度。"),
        ("043-03-探针类别被试贡献.png", "被试对四类探针的贡献热图", "类别3仅5名被试、类别4仅3名被试，不能当作稳定群体常模。"),
        ("043-04-探针前行为证据.png", "三种探针前时间窗的行为证据散点", "每个点是一条探针记录；类别样本量为214/29/14/7。"),
        ("043-05-窗口敏感性.png", "窗口定义变化下的四类Go RT中位数", "用于判断候选窗口结论是否稳定，不据此选择评分权重。"),
    ]
    plot_phase3_status(tables["rolling_status"], paths.reports / figures[0][0])
    plot_phase3_nogo_span(tables["nogo_span_summary"], paths.reports / figures[1][0])
    plot_phase3_probe_subjects(tables["probe_state_by_subject"], paths.reports / figures[2][0])
    plot_phase3_probe_metrics(tables["probe_evidence"], paths.reports / figures[3][0])
    plot_phase3_sensitivity(tables["probe_sensitivity_summary"], paths.reports / figures[4][0])

    counts = tables["probe_state_counts"].copy()
    span = tables["nogo_span_summary"].round(2)
    sections = [
        ("解释边界", '<p class="notice">四类探针保持名义分类。当前数据是重复测量、类别高度不平衡且少数类别只来自少数被试；本报告只作敏感性和测量证据，不宣称因果、不计算专注分数。</p>'),
        ("类别样本量", counts.to_html(index=False, border=0) + '<p>类别1/2/3/4分别来自11/8/5/3名被试；少数类别的表观差异可能主要由个体构成造成。</p>'),
        ("No-Go机会窗", span[["condition", "nogo_window_target", "actual_span_median_sec", "evidence_age_median_sec", "full_evidence_rate"]].to_html(index=False, border=0) + '<p>同样的6/8/12次No-Go在A/B/C条件对应不同实际时间跨度，因此时间窗证据与机会窗证据应分轨保留。</p>'),
        ("当前可支持的结论", '<ul><li>窗口不足、证据年龄和实际跨度可以被量化，适合进入未来实时评分的置信度层；</li><li>四类探针比较必须使用被试内或分层模型，不能只比较合并均值；</li><li>当前类别3/4样本不足以冻结单独的阈值或权重；</li><li>30/60/90秒可用于探针敏感性分析，120秒保留在一般滚动证据中，但不跨block。</li></ul>'),
    ]
    html_path = paths.reports / "043-窗口证据与四类探针敏感性报告.html"
    html_path.write_text(_report_html("043｜窗口证据与四类探针敏感性报告", metadata["generated_at"], sections, figures), encoding="utf-8")
    md_path = paths.reports / "043-窗口证据与四类探针敏感性报告.md"
    md_path.write_text(
        "# 043｜窗口证据与四类探针敏感性报告\n\n"
        f"> {metadata['generated_at']}｜窗口证据可量化，但四类探针高度不平衡，尚不足以冻结评分阈值。\n\n"
        "## 解释边界\n\n四类保持名义分类；重复测量与被试构成必须在后续模型中处理。\n\n"
        "## 类别样本量\n\n" + _markdown_table(counts) + "\n\n"
        "## 当前结论\n\n- 时间窗和No-Go机会窗分轨保留。\n- 缺失、部分覆盖、证据年龄进入置信度层。\n- 类别3/4不足以冻结阈值。\n- 不输出专注分数。\n\n"
        "## 图表导航\n\n" + "\n".join(f"- `{name}`：{caption}" for name, _, caption in figures) + "\n",
        encoding="utf-8",
    )
    manifest = {
        **metadata,
        "stage": "behavior_phase3",
        "outputs": [str(html_path), str(md_path), *table_paths.values(), *[str(paths.reports / name) for name, _, _ in figures]],
        "attention_score_created": False,
        "probe_states_merged": False,
        "time_windows_sec": [30, 60, 90, 120],
        "probe_pre_windows_sec": [30, 60, 90],
        "nogo_opportunity_windows": [6, 8, 12],
    }
    manifest_path = paths.manifests / "043-behavior-phase3-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "phase": 3,
        "report_html": str(html_path),
        "report_markdown": str(md_path),
        "figures": [str(paths.reports / name) for name, _, _ in figures],
        "tables": table_paths,
        "manifest": str(manifest_path),
        "rolling_rows": len(tables["rolling_evidence"]),
        "probe_evidence_rows": len(tables["probe_evidence"]),
        "attention_score_created": False,
    }


def _program_recommendations() -> pd.DataFrame:
    """Return an auditable decision table; rows are proposals, not code changes."""
    rows = [
        ("REC-01", "可直接纳入下一版", "保留全部程序归属RT，阈值只生成QC/策略标记", "041：<100/<150/>1000/>1150ms均真实存在；042：短RT受固定周期位置影响", "分析与记录规则", "完全兼容旧数据", "批准后修改实验程序/分析规范"),
        ("REC-02", "可直接纳入下一版", "记录sequence_version、schedule_id、schedule_hash和software_version", "042：三条件的12个cycle在stimulus_name与No-Go位置上完全重复；当前数据无法标识序列版本", "新增溯源字段", "旧数据字段为空但仍可读取", "批准后修改实验程序"),
        ("REC-03", "可直接纳入下一版", "记录stimulus_flip、mask_flip、trial_end及响应归属规则版本", "正式程序在刺激flip后记录absolute_onset_time，并在刺激与mask阶段轮询按键；缺少各视觉阶段的显式时间戳", "新增事件时间字段", "只增列", "批准后修改实验程序"),
        ("REC-04", "可直接纳入下一版", "强化序列校验：列、trial/cycle/position、No-Go数量与间距、探针位置、文件hash", "正式程序check_sequence_files主要检查存在性、行数和探针范围，无法阻止结构被意外改动", "启动前校验", "不改变任务内容", "批准后修改实验程序"),
        ("REC-05", "可直接纳入下一版", "显式记录probe_number、probe_after_trial、cycle/position、题目与选项版本、确认时间", "043：四个探针固定在trial 30/82/137/191；当前题目版本和调度身份未进入逐试次表", "新增探针溯源字段", "只增列", "批准后修改实验程序"),
        ("REC-06", "可直接纳入下一版", "修正文档中探针仅有1/2/3类的过时说明", "正式show_probe文档字符串写1/2/3，实际按键和界面均为1/2/3/4", "文档修正", "无行为影响", "批准后修改实验程序"),
        ("REC-07", "需新实验版本验证", "跨被试平衡Block顺序，并记录block_order_id", "042：ABCCBA使A/B/C与非线性任务时间部分混淆", "新实验设计版本", "不可与旧版直接当同一设计", "需先讨论并批准"),
        ("REC-08", "需新实验版本验证", "在约束No-Go频率/间距的前提下准备多套序列并记录schedule_id", "042：每个条件12个cycle的刺激身份与No-Go位置完全重复，可能形成可预测策略", "新实验设计版本", "保留固定序列兼容分支", "需先确定SART设计目的"),
        ("REC-09", "需新实验版本验证", "在保持每Block四次与最小间距的前提下平衡/抖动探针位置", "043：探针总在固定cycle/position出现，探针前证据与位置及预期混淆", "新探针调度版本", "不可回写旧数据", "需先讨论并批准"),
        ("REC-10", "需操作性定义讨论", "把‘刚才’改成明确参考时段，或明确探针测量的是瞬时状态", "当前探针题干使用‘刚才’，而候选分析窗为30/60/90秒", "探针措辞版本", "改词会改变构念", "审批门3讨论"),
        ("REC-11", "需操作性定义讨论", "评估理解检查、置信度题或分层探针；若改变选项映射必须记录版本", "043：类别1/2/3/4为214/29/14/7，类别3/4仅来自5/3名被试；界面无默认选项", "探针测量设计", "增加负担且改变测量", "审批门3讨论"),
        ("REC-12", "当前不应实施", "不把类别2/3/4直接合并成不专注", "043：类别语义不同且样本高度不平衡，少数类别受被试构成影响", "保持四类名义状态", "维持现状", "等待新数据与定义"),
        ("REC-13", "当前不应实施", "不把fatigue_label、Block编号或任务时间当专注真值", "042：条件与时间混淆；时间变量只是设计因素", "解释边界", "维持现状", "无需改程序"),
        ("REC-14", "当前不应实施", "不冻结单一窗口、阈值、权重或总分", "043：窗口充分率依赖No-Go目标；条件间实际跨度不同；类别3/4样本不足", "保留候选证据", "维持多窗口输出", "等待审批门3"),
    ]
    return pd.DataFrame(rows, columns=[
        "recommendation_id", "decision_class", "recommendation", "evidence",
        "change_type", "compatibility", "approval",
    ])


def _design_audit() -> pd.DataFrame:
    return pd.DataFrame([
        ("Block顺序", "ABCCBA，所有11名被试相同", "条件与非线性任务时间部分混淆", "REC-07"),
        ("18试次周期", "每条件12个cycle的stimulus_name逐位置完全相同", "可形成位置/刺激预测策略", "REC-02/08"),
        ("No-Go位置", "A=5/9/14/18；B=5/14；C=5", "机会率和机会窗实际跨度随条件改变", "REC-08/14"),
        ("探针位置", "trial 30/82/137/191；对应cycle-position 2-12/5-10/8-11/11-11", "探针状态与固定位置及可预期性混淆", "REC-05/09"),
        ("探针界面", "必须先按1–4再Enter；current_choice初始为None", "类别1占优不能归因于代码默认选项", "REC-11"),
        ("探针措辞", "‘请选择最符合您刚才状态的表述’", "参考时间范围未定义", "REC-10"),
        ("响应归属", "每个trial开始清空事件；刺激250ms和mask 900ms内取首次space", "程序归属应保留，但需记录规则版本与视觉阶段时间", "REC-01/03"),
        ("序列校验", "检查文件、行数与探针范围，未冻结完整结构/hash", "序列被误改时难以审计", "REC-04"),
    ], columns=["audit_item", "observed_design", "analysis_consequence", "linked_recommendations"])


def _field_dictionary(trials: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    raw_definitions = {
        "subject": ("v2逐试次", "被试目录解析", "规范化被试标识；例如sub-000", "不得缺失"),
        "subject_id": ("原始逐试次", "正式程序", "原始行为文件中的被试标识", "不得缺失"),
        "block_num": ("原始逐试次", "正式程序", "正式Block编号1–6", "不得缺失"),
        "condition": ("原始逐试次", "正式程序", "SART条件A/B/C", "不得缺失"),
        "trial_num": ("原始逐试次", "正式程序", "Block内试次编号1–216", "不得缺失"),
        "cycle_num": ("原始逐试次", "正式程序", "Block内18试次周期编号1–12", "不得缺失"),
        "position_in_cycle": ("原始逐试次", "正式程序", "18试次周期中的位置1–18", "不得缺失"),
        "stimulus_name": ("原始逐试次", "正式程序", "当前刺激身份", "不得缺失"),
        "stimulus_size": ("原始逐试次", "正式程序", "刺激呈现尺寸", "不得缺失"),
        "is_no_go": ("原始逐试次", "正式程序", "1=No-Go机会，0=Go机会", "不得缺失"),
        "response": ("原始逐试次", "正式程序", "当前试次是否记录到space响应", "不得缺失"),
        "rt": ("原始逐试次", "正式程序", "程序归属到当前试次的首次响应时，单位ms", "无响应时缺失；不因QC阈值删除"),
        "response_time": ("原始逐试次", "正式程序", "首次响应Unix毫秒时间戳", "无响应时缺失"),
        "correct": ("原始逐试次", "正式程序", "该试次是否按任务规则正确", "不得缺失"),
        "commission": ("原始逐试次", "正式程序", "No-Go试次错误按键标记", "Go试次为0，不作为机会"),
        "omission": ("原始逐试次", "正式程序", "Go试次未按键标记", "No-Go试次为0，不作为机会"),
        "is_probe": ("原始逐试次", "正式程序", "该试次后是否呈现探针", "不得缺失"),
        "probe_response": ("原始逐试次", "正式程序", "探针名义类别1/2/3/4", "非探针行缺失；不得求1–4均值"),
        "probe_rt": ("原始逐试次", "正式程序", "探针出现至确认的反应时，单位ms", "非探针行缺失"),
        "probe_onset_time": ("原始逐试次", "正式程序", "探针出现Unix毫秒时间戳", "非探针行缺失"),
        "probe_response_time": ("原始逐试次", "正式程序", "探针确认Unix毫秒时间戳", "非探针行缺失"),
        "absolute_onset_time": ("原始逐试次", "正式程序", "刺激flip后记录的Unix毫秒跨设备锚点", "不得缺失"),
        "block_onset_time": ("原始逐试次", "正式程序", "Block开始的Unix毫秒时间戳", "不得缺失"),
        "raw_keypresses": ("原始逐试次", "正式程序", "该试次窗口内全部space时间戳，分号分隔", "无按键时为空"),
        "rest_duration": ("原始逐试次", "正式程序", "与休息相关的记录字段", "依原程序语义保留"),
        "source_file": ("v2逐试次", "v2读取器", "只读源CSV绝对路径", "不得缺失"),
        "source_row": ("v2逐试次", "v2读取器", "含表头偏移后的源CSV行号", "不得缺失"),
        "probe_state_label": ("v2逐试次", "v2映射", "四类探针中文名义标签", "非探针行缺失"),
        "condition_x_position": ("v2逐试次", "v2派生", "condition与position_in_cycle联合设计因子", "不得缺失"),
        "rt_timestamp_delta_ms": ("v2逐试次", "v2 QC", "response_time-absolute_onset_time-rt", "任一时间缺失时缺失"),
        "rt_qc_timestamp_inconsistent": ("v2逐试次", "v2 QC", "时间戳差绝对值是否超过配置容差", "仅标记，不删除"),
        "rt_qc_lt_100": ("v2逐试次", "v2 QC", "RT<100ms标记", "仅标记，不删除"),
        "rt_qc_lt_150": ("v2逐试次", "v2 QC", "RT<150ms标记", "仅标记，不删除"),
        "rt_qc_gt_1000": ("v2逐试次", "v2 QC", "RT>1000ms标记", "仅标记，不删除"),
        "rt_qc_gt_1150": ("v2逐试次", "v2 QC", "RT>标称1150ms标记", "仅标记，不删除"),
    }
    window_definitions = {
        "window_start_ms": "候选时间窗的标称起点",
        "window_end_ms": "候选时间窗终点/探针时点",
        "time_window_sec": "候选时间窗长度30/60/90/120秒",
        "nogo_window_target": "最近No-Go机会目标6/8/12次",
        "window_status": "insufficient_rt/insufficient_nogo/response_only/full_evidence",
        "window_actual_start_ms": "受Block边界截断后的实际起点",
        "window_actual_coverage_sec": "实际覆盖时长",
        "time_window_is_partial": "实际覆盖是否短于标称时间窗",
        "trial_count": "时间窗内全部试次数",
        "go_opportunities": "时间窗内Go机会数",
        "go_rt_count": "时间窗内有RT的Go试次数",
        "go_rt_median_ms": "时间窗内Go RT中位数",
        "go_rt_iqr_ms": "时间窗内Go RT四分位距",
        "go_rt_lt_150_count": "时间窗内Go RT<150ms数量；策略/QC证据",
        "go_rt_lt_150_rate": "Go短RT数除以Go机会数",
        "go_omissions": "时间窗内Go遗漏数",
        "omission_jeffreys_mean": "Go遗漏率Jeffreys后验均值",
        "omission_jeffreys_ci95_low": "Go遗漏率Jeffreys 95%区间下界",
        "omission_jeffreys_ci95_high": "Go遗漏率Jeffreys 95%区间上界",
        "time_nogo_opportunities": "时间窗内No-Go机会数",
        "time_nogo_commissions": "时间窗内No-Go commission数",
        "time_commission_jeffreys_mean": "时间窗commission率Jeffreys后验均值",
        "time_commission_jeffreys_ci95_low": "时间窗commission率95%区间下界",
        "time_commission_jeffreys_ci95_high": "时间窗commission率95%区间上界",
        "nogo_opportunities_actual": "最近机会窗实际取得的No-Go数",
        "nogo_commissions": "最近机会窗内commission数",
        "nogo_actual_span_sec": "最近No-Go机会从首个到末个的实际时间跨度",
        "nogo_evidence_age_sec": "窗口终点距最近No-Go机会的年龄",
        "commission_jeffreys_mean": "最近机会窗commission率Jeffreys后验均值",
        "commission_jeffreys_ci95_low": "最近机会窗commission率95%区间下界",
        "commission_jeffreys_ci95_high": "最近机会窗commission率95%区间上界",
        "probe_number_in_block": "Block内第1–4个探针",
        "probe_after_trial": "探针出现前的trial编号",
    }
    rows = []
    for field in trials.columns:
        layer, origin, definition, missing = raw_definitions.get(
            field, ("原始逐试次", "正式程序原样保留", "源文件字段；等待程序字段规范进一步确认", "按源程序语义")
        )
        rows.append((layer, field, str(trials[field].dtype), origin, definition, missing, "已实现"))
    for field in probes.columns:
        if field in trials.columns:
            continue
        definition = window_definitions.get(field, "探针前窗口的身份或分层字段")
        missing = "证据不足时保留缺失并结合window_status解释" if field not in {"subject", "block_num", "condition"} else "不得缺失"
        rows.append(("探针前证据窗", field, str(probes[field].dtype), "v2窗口派生", definition, missing, "候选字段，未冻结评分"))
    future = [
        ("建议新增", "sequence_version", "string", "正式程序", "序列规范版本", "旧数据缺失", "待批准"),
        ("建议新增", "schedule_id", "string", "正式程序", "本次使用的具体序列/调度身份", "旧数据缺失", "待批准"),
        ("建议新增", "schedule_hash", "string", "正式程序", "序列文件内容摘要", "旧数据缺失", "待批准"),
        ("建议新增", "software_version", "string", "正式程序", "实验程序版本", "旧数据缺失", "待批准"),
        ("建议新增", "block_order_id", "string", "正式程序", "Block顺序版本/平衡组", "旧数据缺失", "待批准"),
        ("建议新增", "probe_schedule_id", "string", "正式程序", "探针调度版本", "旧数据缺失", "待批准"),
        ("建议新增", "probe_wording_version", "string", "正式程序", "探针题干与选项版本", "旧数据缺失", "待批准"),
        ("建议新增", "stimulus_flip_time", "int64", "正式程序", "实际刺激flip时间戳", "旧数据缺失", "待批准"),
        ("建议新增", "mask_flip_time", "int64", "正式程序", "实际mask flip时间戳", "旧数据缺失", "待批准"),
        ("建议新增", "trial_end_time", "int64", "正式程序", "响应窗口结束时间戳", "旧数据缺失", "待批准"),
        ("建议新增", "response_assignment_rule_version", "string", "正式程序", "按键归属试次的规则版本", "旧数据缺失", "待批准"),
        ("建议新增", "probe_confirm_time", "int64", "正式程序", "探针最终确认时间；可与response_time一致但语义显式", "旧数据缺失", "待批准"),
    ]
    rows.extend(future)
    return pd.DataFrame(rows, columns=[
        "layer", "field", "dtype", "origin", "definition", "missing_semantics", "freeze_status",
    ])


def _within_subject_probe_contrasts(probes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = probes.loc[
        probes["time_window_sec"].eq(60) & probes["nogo_window_target"].eq(8)
    ].copy()
    metrics = [
        "go_rt_median_ms", "go_rt_lt_150_rate", "time_commission_jeffreys_mean",
        "omission_jeffreys_mean",
    ]
    subject_medians = selected.groupby(["subject", "probe_response"])[metrics].median().reset_index()
    rows = []
    for metric in metrics:
        pivot = subject_medians.pivot(index="subject", columns="probe_response", values=metric)
        for state in [2, 3, 4]:
            paired = pivot[[1, state]].dropna() if 1 in pivot and state in pivot else pd.DataFrame()
            if paired.empty:
                continue
            differences = paired[state] - paired[1]
            for subject, value in differences.items():
                rows.append((subject, state, PROBE_LABELS[state], metric, float(value)))
    contrasts = pd.DataFrame(rows, columns=["subject", "probe_response", "probe_state_label", "metric", "difference_from_state1"])
    summary = contrasts.groupby(["probe_response", "probe_state_label", "metric"], as_index=False).agg(
        paired_subjects=("subject", "nunique"),
        median_difference=("difference_from_state1", "median"),
        q25_difference=("difference_from_state1", lambda x: x.quantile(0.25)),
        q75_difference=("difference_from_state1", lambda x: x.quantile(0.75)),
    )
    return contrasts, summary


def phase4_tables(config: Config, trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    probes = cohort_probe_evidence(config, trials)
    contrasts, contrast_summary = _within_subject_probe_contrasts(probes)
    return {
        "program_recommendations": _program_recommendations(),
        "design_audit": _design_audit(),
        "probe_within_subject_contrasts": contrasts,
        "probe_within_subject_contrast_summary": contrast_summary,
        "field_dictionary": _field_dictionary(trials, probes),
    }


def plot_phase4_decision_map(recommendations: pd.DataFrame, path: Path) -> None:
    order = ["可直接纳入下一版", "需新实验版本验证", "需操作性定义讨论", "当前不应实施"]
    colors = {"可直接纳入下一版": "#2F855A", "需新实验版本验证": "#D97706", "需操作性定义讨论": "#7A5195", "当前不应实施": "#6B7280"}
    rows = recommendations.copy()
    rows["decision_class"] = pd.Categorical(rows["decision_class"], order, ordered=True)
    rows = rows.sort_values(["decision_class", "recommendation_id"])
    fig, axis = plt.subplots(figsize=(13.2, 8.2))
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.7, len(rows) - 0.3)
    axis.axis("off")
    for y, (_, row) in enumerate(rows.iloc[::-1].iterrows()):
        color = colors[str(row["decision_class"])]
        axis.add_patch(plt.Rectangle((0.01, y - 0.36), 0.17, 0.72, color=color, alpha=0.95))
        axis.text(0.095, y, f'{row["recommendation_id"]}\n{row["decision_class"]}', ha="center", va="center", color="white", fontsize=8.2)
        axis.text(0.2, y, row["recommendation"], ha="left", va="center", fontsize=9.2, wrap=True)
        if y > 0:
            axis.plot([0.01, 0.99], [y - 0.5, y - 0.5], color=GRID, linewidth=0.6)
    axis.set_title("行为证据到实验程序决策：建议不等于已修改", loc="left", fontsize=16, pad=14)
    axis.text(0.01, -0.62, "绿色=低风险记录/校验；橙色=新实验版本；紫色=需先冻结构念；灰色=当前避免", fontsize=9, color=NEUTRAL)
    _save_figure(fig, path)


def plot_phase4_within_subject(contrasts: pd.DataFrame, path: Path) -> None:
    specs = [
        ("go_rt_median_ms", "Go RT中位数差（ms）"),
        ("time_commission_jeffreys_mean", "commission后验均值差"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    for axis, (metric, ylabel) in zip(axes, specs):
        data = contrasts.loc[contrasts["metric"].eq(metric)]
        for state in [2, 3, 4]:
            values = data.loc[data["probe_response"].eq(state), "difference_from_state1"].to_numpy(dtype=float)
            if len(values) == 0:
                continue
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
            axis.scatter(np.full(len(values), state) + jitter, values, color=PROBE_COLORS[state], alpha=0.72, s=44)
            median = float(np.median(values))
            axis.plot([state - 0.18, state + 0.18], [median, median], color=PROBE_COLORS[state], linewidth=3)
            axis.text(state, max(values) if len(values) else median, f"n={len(values)}", ha="center", va="bottom", fontsize=8)
        axis.axhline(0, color="#111827", linewidth=1, linestyle="--")
        axis.set_xticks([2, 3, 4], ["2\n关注实验", "3\n任务无关", "4\n大脑空白"])
        axis.set(xlabel="探针类别（差值=该类被试中位数−状态1中位数）", ylabel=ylabel)
        _style_axis(axis)
    fig.suptitle("60秒 × 最近8次No-Go：被试内描述性差异（不用于冻结阈值）")
    fig.tight_layout()
    _save_figure(fig, path)


def generate_phase4(config: Config) -> dict:
    _configure_chinese_font()
    paths = output_paths(config, create=True)
    trials = load_cohort(config)
    metadata = run_metadata(config, behavior_sources(config))
    tables = phase4_tables(config, trials)
    write_directory_entries(paths, metadata)

    table_paths = {}
    for name, table in tables.items():
        prefix = "045" if name == "field_dictionary" else "044"
        destination = paths.behavior / f"{prefix}-{name}.csv"
        _with_metadata(table, metadata).to_csv(destination, index=False, encoding="utf-8-sig")
        table_paths[name] = str(destination)

    figures = [
        ("044-01-证据到决策分层图.png", "行为证据到程序决策的分层图", "建议分为低风险新增、需新版本验证、需定义讨论和当前避免；本阶段没有修改正式程序。"),
        ("044-02-探针被试内差异.png", "探针状态相对状态1的被试内差异", "60秒×最近8次No-Go的描述性对照；类别2/3/4仅有8/5/3名可配对被试。"),
    ]
    plot_phase4_decision_map(tables["program_recommendations"], paths.reports / figures[0][0])
    plot_phase4_within_subject(tables["probe_within_subject_contrasts"], paths.reports / figures[1][0])

    recommendations = tables["program_recommendations"]
    design = tables["design_audit"]
    direct = recommendations.loc[recommendations["decision_class"].eq("可直接纳入下一版")]
    redesign = recommendations.loc[recommendations["decision_class"].eq("需新实验版本验证")]
    discuss = recommendations.loc[recommendations["decision_class"].eq("需操作性定义讨论")]
    avoid = recommendations.loc[recommendations["decision_class"].eq("当前不应实施")]
    sections = [
        ("审批边界", '<p class="notice">本报告只提出修改建议并建立证据追溯；未写入或改动正式实验程序。用户批准具体条目后，才会另建程序版本实施。</p>'),
        ("程序与设计审计", design.to_html(index=False, border=0)),
        ("可直接纳入下一版的低风险项", direct[["recommendation_id", "recommendation", "evidence", "compatibility"]].to_html(index=False, border=0)),
        ("必须作为新实验版本验证", redesign[["recommendation_id", "recommendation", "evidence", "compatibility"]].to_html(index=False, border=0)),
        ("审批门3需要讨论", discuss[["recommendation_id", "recommendation", "evidence"]].to_html(index=False, border=0)),
        ("当前明确避免", avoid[["recommendation_id", "recommendation", "evidence"]].to_html(index=False, border=0)),
        ("对正式程序的关键核查", '<ul><li>探针没有默认选择：current_choice初始为None，必须按1–4再按Enter；</li><li>类别1占214/264不能归因为默认选项，但仍可能受题意、社会期许与样本状态分布影响；</li><li>程序在每个trial开始清空事件，并保留该trial窗口内全部space时间戳；因此程序归属RT全部保留，另做QC；</li><li>当前序列和探针位置高度固定，必须显式建模，下一版是否降低可预测性需由研究目的决定。</li></ul>'),
    ]
    html_path = paths.reports / "044-实验程序修改建议与证据对照.html"
    html_path.write_text(_report_html("044｜实验程序修改建议与证据对照", metadata["generated_at"], sections, figures), encoding="utf-8")
    md_path = paths.reports / "044-实验程序修改建议与证据对照.md"
    md_path.write_text(
        "# 044｜实验程序修改建议与证据对照\n\n"
        f"> {metadata['generated_at']}｜本文件把建议按审批风险分层；未修改正式实验程序，也未冻结专注评分。\n\n"
        "## 审批边界\n\n本报告是建议清单，不是实施记录。正式程序仍为只读。\n\n"
        "## 程序与设计审计\n\n" + _markdown_table(design) + "\n\n"
        "## 全部建议与依据\n\n" + _markdown_table(recommendations) + "\n\n"
        "## 需要你在审批门3决定的核心问题\n\n"
        "1. 探针测量的是确认前的瞬时状态，还是明确的30/60/90秒回顾状态？\n"
        "2. 固定18试次序列是希望保留的可预测持续注意范式，还是希望降低策略学习？\n"
        "3. 下一批实验是否允许跨被试平衡Block顺序和探针位置？\n"
        "4. 四类状态保持并列，还是采用分层提问；是否增加理解检查/置信度？\n\n"
        "## 图表导航\n\n" + "\n".join(f"- `{name}`：{caption}" for name, _, caption in figures) + "\n",
        encoding="utf-8",
    )

    dictionary = tables["field_dictionary"]
    dictionary_path = paths.reports / "045-行为分析字段字典.md"
    layer_counts = dictionary.groupby(["layer", "freeze_status"], as_index=False).size().rename(columns={"size": "fields"})
    dictionary_path.write_text(
        "# 045｜行为分析字段字典\n\n"
        f"> {metadata['generated_at']}｜统一说明原始字段、v2派生证据、缺失语义和建议新增字段；候选证据不等于评分字段。\n\n"
        "## 字段层级速查\n\n" + _markdown_table(layer_counts) + "\n\n"
        "## 状态与缺失约定\n\n"
        "- `insufficient_rt`：窗内没有可计算的Go RT证据。\n"
        "- `response_only`：有RT但没有No-Go机会。\n"
        "- `insufficient_nogo`：有No-Go机会但少于目标6/8/12次。\n"
        "- `full_evidence`：有RT且达到目标No-Go机会数。\n"
        "- QC标记不删除原始RT；Jeffreys区间与实际机会数、跨度、年龄一同解释。\n\n"
        "## 完整字段表\n\n" + _markdown_table(dictionary) + "\n",
        encoding="utf-8",
    )

    outputs = [
        str(html_path), str(md_path), str(dictionary_path), *table_paths.values(),
        *[str(paths.reports / name) for name, _, _ in figures],
    ]
    manifest = {
        **metadata,
        "stage": "behavior_phase4_gate3_package",
        "outputs": outputs,
        "formal_experiment_modified": False,
        "attention_score_created": False,
        "probe_states_merged": False,
        "approval_required_next": "gate3_attention_definition",
    }
    manifest_path = paths.manifests / "044-behavior-phase4-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "phase": 4,
        "report_html": str(html_path),
        "report_markdown": str(md_path),
        "field_dictionary": str(dictionary_path),
        "figures": [str(paths.reports / name) for name, _, _ in figures],
        "tables": table_paths,
        "manifest": str(manifest_path),
        "recommendations": len(recommendations),
        "formal_experiment_modified": False,
        "attention_score_created": False,
        "stopped_at": "gate3_attention_definition",
    }
