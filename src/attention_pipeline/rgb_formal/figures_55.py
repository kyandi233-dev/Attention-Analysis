# -*- coding: utf-8 -*-
"""
figures_55.py
版本: v1.0 (2026-08-31)
功能: 正式报告 5.5 节 RGB 可见行为结果的数据图。

图表 contract (050 文档第 6 节 + 任务规格):
  - 图内无标题; 完整题名与科学解释写外部 caption manifest;
  - 中文坐标轴/图例 (SimSun 优先, Microsoft YaHei/SimHei 兜底, 数字 Arial,
    unicode_minus=False; 参照 nir_pipeline_validation/figure_style.py);
  - 每个指标 × 视图输出 generated 或 not_estimable + reason, 写入 audit 表。

图清单:
  1. 运动能量与曝光变化的 B1/B2 配对与 block×cycle 轨迹 (2×2 面板);
  2. 左右/上下/前后方向分布 (3 面板; 前后为复合方向候选, 非物理位移);
  3. 眨眼候选事件分布 (场次级频率/时长/间隔, 3 面板);
  4. 场次级覆盖 (各轨可观测帧比例)。

用法: 由 scripts/rgb_55_analysis.py 调用; 数据路径由调用方注入。
依赖: matplotlib, numpy, pandas
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from attention_pipeline.formal_analysis.publication_style import (
    finalize_publication_figure,
)

# ---------------------------------------------------------------------------
# 字体与样式 (集中声明)
# ---------------------------------------------------------------------------
# 中文出图字体: SimSun 优先, 依次回退 Microsoft YaHei / SimHei;
# 数字与西文使用 Arial。检测块写法与 NIR figure_style.py 一致。
def _detect_cjk_font() -> str:
    """检测本机可用的中文字体, 返回字体族名。"""
    for name in ("SimSun", "Microsoft YaHei", "SimHei"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "SimSun"


FIGURE_FONT = _detect_cjk_font()

# 色盲友好且灰度可辨的 B1/B2 配色 (与 NIR figure_style PALETTE 一致)
BLOCK_COLORS = {"B1": "#2F5597", "B2": "#C55A11"}
BLOCK_LINESTYLES = {"B1": "-", "B2": "--"}

# 各轨覆盖图配色 (色盲友好)
COVERAGE_COLORS = {
    "body_motion_observable_ratio": "#2F5597",
    "exposure_change_observable_ratio": "#C55A11",
    "pose_shoulders_observable_ratio": "#59A14F",
    "left_eye_observable_ratio": "#4C78A8",
    "right_eye_observable_ratio": "#B279A2",
    "bilateral_consistent_ratio": "#E45756",
}
COVERAGE_LABELS_ZH = {
    "body_motion_observable_ratio": "身体运动可观测帧比例",
    "exposure_change_observable_ratio": "曝光变化可观测帧比例",
    "pose_shoulders_observable_ratio": "肩部可观测帧比例",
    "left_eye_observable_ratio": "左眼可观测帧比例",
    "right_eye_observable_ratio": "右眼可观测帧比例",
    "bilateral_consistent_ratio": "双眼一致性比例",
}


def configure_cjk_style() -> None:
    """应用 5.5 图表统一样式: 中文字体 + Arial 数字 + 图内无标题。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FIGURE_FONT, "Arial"],
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "lines.linewidth": 1.25,
        "lines.markersize": 4.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(fig: plt.Figure, path: Path) -> str:
    """保存图 (PNG 300 dpi + SVG 矢量); 保存前强制移除图内标题。"""
    finalize_publication_figure(fig, remove_titles=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    png = path.with_suffix(".png")
    svg = path.with_suffix(".svg")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return str(png)


def _participant_cell_mean(table: pd.DataFrame, metric: str, group_cols: Sequence[str]) -> pd.DataFrame:
    """参与者优先描述: 先取每参与者每 cell 均值, 再跨参与者求 mean ± SEM。

    参数:
        table: 含 participant_group_id / metric / group_cols 的表
        metric: 指标列
        group_cols: 分组列 (如 block_id / cycle_bin)
    返回:
        每 cell 的 participant_n / mean / sem DataFrame
    """
    d = table[["participant_group_id", metric, *group_cols]].copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=[metric, "participant_group_id"])
    if d.empty:
        return pd.DataFrame()
    per_participant = d.groupby(["participant_group_id", *group_cols], dropna=False, sort=True)[metric].mean().reset_index()
    grouped = per_participant.groupby(list(group_cols), dropna=False, sort=True)[metric].agg(
        participant_n="count", mean="mean", sem=lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) >= 2 else np.nan
    ).reset_index()
    return grouped


# ---------------------------------------------------------------------------
# 图 1: 运动能量与曝光变化
# ---------------------------------------------------------------------------
def figure_motion_exposure(probe_features: pd.DataFrame, cycle_table: pd.DataFrame, output_root: Path) -> tuple[bool, str, str]:
    """2×2 面板: 运动能量与曝光变化的 B1/B2 配对 (probe 窗口) 与 block×cycle 轨迹。

    面板 A/B: body_motion_energy_median 的 B1-B2 配对与 cycle 轨迹;
    面板 C/D: exposure_change_abs_median 的 B1-B2 配对与 cycle 轨迹。
    B1/B2 配对为参与者级描述 (每参与者两 block 的探针窗口均值配对);
    cycle 轨迹为参与者均值 ± SEM。

    返回: (generated, reason, png 路径)
    """
    configure_cjk_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    fig.subplots_adjust(wspace=0.32, hspace=0.5)
    metrics = (
        ("body_motion_energy_median", "整体运动能量中位数"),
        ("exposure_change_abs_median", "曝光变化绝对值中位数"),
    )
    generated_any = False
    for row_idx, (metric, label_zh) in enumerate(metrics):
        # 左列: B1/B2 配对 (参与者级)
        ax_pair = axes[row_idx, 0]
        pairs: list[tuple[float, float]] = []
        if not probe_features.empty and metric in probe_features:
            d = probe_features[["participant_group_id", "block_id", metric]].copy()
            d[metric] = pd.to_numeric(d[metric], errors="coerce")
            d = d.dropna()
            block_means = d.groupby(["participant_group_id", "block_id"], sort=True)[metric].mean().unstack()
            for b1, b2 in zip(block_means.get("B1", pd.Series(dtype=float)), block_means.get("B2", pd.Series(dtype=float))):
                if np.isfinite(b1) and np.isfinite(b2):
                    pairs.append((float(b1), float(b2)))
        if len(pairs) >= 2:
            left = [p[0] for p in pairs]
            right = [p[1] for p in pairs]
            for a, b in pairs:
                ax_pair.plot([1, 2], [a, b], marker="o", linewidth=0.7, alpha=0.4, color="#999999")
            ax_pair.plot([1, 2], [float(np.mean(left)), float(np.mean(right))], marker="o", linewidth=2.0, color="#333333")
            ax_pair.set_xticks([1, 2], ["B1", "B2"])
            ax_pair.set_xlabel("区块")
            ax_pair.set_ylabel(label_zh)
            ax_pair.set_xlim(0.8, 2.2)
            generated_any = True
        else:
            ax_pair.text(0.5, 0.5, "数据不足", transform=ax_pair.transAxes, ha="center", va="center", color="#888888")
            ax_pair.set_xticks([]); ax_pair.set_yticks([])

        # 右列: block×cycle 轨迹 (参与者均值 ± SEM)
        ax_cycle = axes[row_idx, 1]
        cells = _participant_cell_mean(cycle_table, metric, ["block_id", "cycle_bin"])
        if not cells.empty and cells["cycle_bin"].nunique() >= 2:
            for block_id in ("B1", "B2"):
                cur = cells[cells["block_id"].astype(str).eq(block_id)].sort_values("cycle_bin")
                if cur.empty:
                    continue
                ax_cycle.errorbar(
                    cur["cycle_bin"], cur["mean"], yerr=cur["sem"],
                    color=BLOCK_COLORS[block_id], linestyle=BLOCK_LINESTYLES[block_id],
                    capsize=2.5, label=block_id,
                )
            ax_cycle.set_xlabel("区块内时间段（cycle bin）")
            ax_cycle.set_ylabel(label_zh)
            ax_cycle.legend(frameon=False)
            generated_any = True
        else:
            ax_cycle.text(0.5, 0.5, "数据不足", transform=ax_cycle.transAxes, ha="center", va="center", color="#888888")
            ax_cycle.set_xticks([]); ax_cycle.set_yticks([])
    png = str(output_root / "fig_55_1_motion_exposure.png")
    if generated_any:
        save_figure(fig, Path(png))
        return True, "", png
    plt.close(fig)
    return False, "motion/exposure probe or cycle data insufficient for trajectory panels", ""


# ---------------------------------------------------------------------------
# 图 2: 姿态方向分布
# ---------------------------------------------------------------------------
def figure_pose_direction(probe_features: pd.DataFrame, output_root: Path) -> tuple[bool, str, str]:
    """3 面板: 探针窗口左右 / 上下 / 复合前后方向候选中位数的分布直方图。

    前后方向为多分量符号一致性候选 (图像内, 无量纲), 不是物理位移。
    返回: (generated, reason, png 路径)
    """
    configure_cjk_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    fig.subplots_adjust(wspace=0.34)
    specs = (
        ("pose_lateral_right_per_sec_median", "左右方向（正=右，/s）"),
        ("pose_vertical_up_per_sec_median", "上下方向（正=上，/s）"),
        ("pose_radial_proximity_direction_score_median", "复合前后方向候选（图像内无量纲）"),
    )
    generated_any = False
    for ax, (column, label_zh) in zip(axes, specs):
        values = pd.to_numeric(probe_features.get(column), errors="coerce").dropna() if not probe_features.empty else pd.Series(dtype=float)
        if len(values) >= 3:
            ax.hist(values.to_numpy(float), bins=min(20, max(6, int(np.sqrt(len(values))))), edgecolor="white", color="#2F5597")
            ax.axvline(0, color="#999999", linewidth=0.8)
            ax.set_xlabel(label_zh)
            ax.set_ylabel("探针窗口数")
            generated_any = True
        else:
            ax.text(0.5, 0.5, "数据不足", transform=ax.transAxes, ha="center", va="center", color="#888888")
            ax.set_xticks([]); ax.set_yticks([])
    png = str(output_root / "fig_55_2_pose_direction.png")
    if generated_any:
        save_figure(fig, Path(png))
        return True, "", png
    plt.close(fig)
    return False, "pose direction probe medians insufficient for distribution", ""


# ---------------------------------------------------------------------------
# 图 3: 眨眼候选事件分布
# ---------------------------------------------------------------------------
def figure_blink_events(session_coverage: pd.DataFrame, output_root: Path) -> tuple[bool, str, str]:
    """3 面板: 场次级眨眼候选事件频率 / 时长 / 间隔分布直方图。

    事件为算法定义候选 (未经人工视频验证), 不是已验证的眨眼结局。
    返回: (generated, reason, png 路径)
    """
    configure_cjk_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    fig.subplots_adjust(wspace=0.34)
    specs = (
        ("blink_event_rate_per_min", "候选事件频率（次/分钟）"),
        ("blink_event_duration_median_ms", "事件时长中位数（ms）"),
        ("blink_ibi_median_ms", "事件间隔中位数（ms）"),
    )
    generated_any = False
    for ax, (column, label_zh) in zip(axes, specs):
        values = pd.to_numeric(session_coverage.get(column), errors="coerce").dropna() if not session_coverage.empty else pd.Series(dtype=float)
        if len(values) >= 3:
            ax.hist(values.to_numpy(float), bins=min(20, max(6, int(np.sqrt(len(values))))), edgecolor="white", color="#C55A11")
            ax.set_xlabel(label_zh)
            ax.set_ylabel("场次数")
            generated_any = True
        else:
            ax.text(0.5, 0.5, "数据不足", transform=ax.transAxes, ha="center", va="center", color="#888888")
            ax.set_xticks([]); ax.set_yticks([])
    png = str(output_root / "fig_55_3_blink_events.png")
    if generated_any:
        save_figure(fig, Path(png))
        return True, "", png
    plt.close(fig)
    return False, "blink event session-level statistics insufficient for distribution", ""


# ---------------------------------------------------------------------------
# 图 4: 场次级覆盖
# ---------------------------------------------------------------------------
def figure_coverage(session_coverage: pd.DataFrame, output_root: Path) -> tuple[bool, str, str]:
    """场次级覆盖图: 每场各轨可观测帧比例 (按场排序的散点线)。

    六轨: 身体运动 / 曝光变化 / 肩部 / 左眼 / 右眼 / 双眼一致性。
    返回: (generated, reason, png 路径)
    """
    configure_cjk_style()
    if session_coverage.empty:
        return False, "session coverage table is empty", ""
    plot_cols = [c for c in COVERAGE_LABELS_ZH if c in session_coverage.columns]
    if len(plot_cols) < 2:
        return False, "session coverage lacks observable-ratio columns", ""
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    order = session_coverage.sort_values("session_id")["session_id"].astype(str).tolist()
    x = np.arange(1, len(order) + 1)
    any_plotted = False
    for column in plot_cols:
        values = pd.to_numeric(session_coverage.set_index("session_id").reindex(order)[column], errors="coerce").to_numpy(float)
        finite = np.isfinite(values)
        if finite.sum() >= 2:
            ax.plot(x[finite], values[finite], marker="o", markersize=2.2, linewidth=0.8,
                    color=COVERAGE_COLORS[column], label=COVERAGE_LABELS_ZH[column])
            any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return False, "no coverage series has at least two finite values", ""
    ax.set_xlabel("场次（按编号排序）")
    ax.set_ylabel("可观测帧比例")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="lower left", fontsize=6.5)
    png = str(output_root / "fig_55_4_coverage.png")
    save_figure(fig, Path(png))
    return True, "", png


# ---------------------------------------------------------------------------
# 图包编排 + 外部 caption manifest + 覆盖审计
# ---------------------------------------------------------------------------
FIGURE_SPECS = (
    ("fig_55_1_motion_exposure", figure_motion_exposure,
     ("probe_features", "cycle_table"),
     "运动能量与曝光变化的区块配对与任务时间轨迹",
     "B1/B2 配对与 block×cycle 轨迹（参与者均值 ± SEM）；运动与曝光两条轨分开呈现。",
     "B1/B2 pairing and block-by-cycle trajectories (participant mean ± SEM) for motion energy and exposure change; the two tracks are kept separate."),
    ("fig_55_2_pose_direction", figure_pose_direction,
     ("probe_features",),
     "姿态方向候选的探针窗口分布",
     "左右、上下方向与复合前后方向候选（图像内无量纲，非物理位移）在探针前 30 s 窗口的分布。",
     "Distributions of lateral, vertical, and composite radial direction candidates (dimensionless, in-image, not physical displacement) over pre-probe 30 s windows."),
    ("fig_55_3_blink_events", figure_blink_events,
     ("session_coverage",),
     "眨眼候选事件的场次级分布",
     "算法定义眨眼候选事件的场次级频率、时长与间隔分布；候选事件未经人工视频验证。",
     "Session-level distributions of algorithm-defined blink-candidate event rate, duration, and interval; candidates lack manual video validation."),
    ("fig_55_4_coverage", figure_coverage,
     ("session_coverage",),
     "各轨场次级可观测帧比例",
     "身体运动、曝光变化、肩部、左眼、右眼与双眼一致性在各场次的可观测帧比例。",
     "Per-session observable-frame ratios for body motion, exposure change, shoulders, left/right eyes, and bilateral consistency."),
)


def build_figure_pack(
    *,
    probe_features: pd.DataFrame,
    cycle_table: pd.DataFrame,
    session_coverage: pd.DataFrame,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成 5.5 全部数据图并返回 (figure_manifest, coverage_audit)。

    manifest 是外部题名/科学解释的唯一来源 (图内无标题);
    audit 记录每个图 generated 或 not_estimable + reason, 缺失图不可与遗忘图混淆。

    参数:
        probe_features: 探针窗口特征表
        cycle_table: block×cycle 聚合表
        session_coverage: 场次级覆盖表
        output_root: 图输出目录
    返回:
        (figure_manifest, coverage_audit) 两个 DataFrame
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    data = {"probe_features": probe_features, "cycle_table": cycle_table, "session_coverage": session_coverage}
    for figure_id, fn, arg_names, name_zh, caption_zh, caption_en in FIGURE_SPECS:
        args = tuple(data[name] for name in arg_names)
        try:
            generated, reason, png = fn(*args, root)
        except Exception as exc:  # 图失败必须可审计, 不得静默
            generated, reason, png = False, f"{type(exc).__name__}: {exc}", ""
        audit_rows.append({
            "figure_id": figure_id, "status": "generated" if generated else "not_estimable",
            "reason": reason, "internal_title_present": False,
            "caption_external": True, "in_image_language": "Chinese",
            "font_family": FIGURE_FONT, "legend_frame": False,
        })
        if generated:
            manifest_rows.append({
                "figure_id": figure_id,
                "file": Path(png).name,
                "title_zh": name_zh,
                "caption_zh": caption_zh,
                "caption_en": caption_en,
                "status": "generated",
                "caption_location": "external_manifest",
                "inference_note": "图为描述性可视化；正式推断使用参与者聚类模型，以 95% CI 是否包含 0 判断。",
            })
    return pd.DataFrame(manifest_rows), pd.DataFrame(audit_rows)
