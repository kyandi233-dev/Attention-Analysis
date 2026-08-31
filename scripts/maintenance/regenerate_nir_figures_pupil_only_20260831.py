# -*- coding: utf-8 -*-
"""
文件名：regenerate_nir_figures_pupil_only_20260831.py
版本：v3.0
功能：复用 GitHub 正式样式重画 NIR 瞳孔结果图，统一 pupil-only 口径，采用「原始瞳孔几何直径（px）」非中心化。
      图7 数据质量、图8 视觉控制、图9 探针状态(30s窗口分布)、图18 探针前轨迹、图10 试次级效应分解、
      图19 六轨道稳健性、趋势图(A/B/C)、热力图。
      样式：线细、图例无框且不挡、字号紧凑、箱线离群小点、比例尺自然；趋势图用 frame 级原始逐点(30ms)不平滑。
用法：python regenerate_nir_figures_pupil_only_20260831.py
"""

from __future__ import annotations

import os
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from attention_pipeline.nir_pipeline_validation.figure_style import (
    configure_publication_style,
    make_figure,
    clean_axis,
    panel_label,
    finalize_layout,
    save_figure,
)

# =============================================================================
# 硬编码参数
# =============================================================================
NIR_TABLES = r"D:/Project/厚粲杯/11_数据/_FormalAnalysis/NIR/11_analysis_tables"
FRAME_ROOT = r"D:/Project/厚粲杯/11_数据/_FormalAnalysis/NIR/10_analysis_ready/frame_level"
OUT_DIR = r"D:/Project/厚粲杯/11_数据/_FormalAnalysis/new_figure"

PUPIL_AXIS_ZH = "瞳孔几何平均直径（px）"
LOW, HIGH = -0.9, 0.9  # 箱线图默认 y 轴范围（比例尺自然，用数据驱动时忽略）
FORMATS = ["png", "svg"]

Q1_LABELS = {1.0: "完全专注", 2.0: "在任务上没想目标", 3.0: "走神", 4.0: "大脑空白"}
Q2_LABELS = {1.0: "非常困倦", 2.0: "比较困倦", 3.0: "比较清醒", 4.0: "非常清醒"}
OUTCOME_LABEL_ZH = {"rt": "正确 Go 反应时", "omission": "Go 遗漏", "commission": "No-Go 误按"}

BEHAVIOR_METRICS = [
    ("dprime", "辨别力 d′"), ("go_rt_median_ms", "正确 Go RT 中位数"), ("rt_cv", "RT 变异系数"),
    ("commission_rate", "No-Go 误按率"), ("clean_omission_rate", "真遗漏率"),
    ("program_omission_rate", "预判遗漏率"), ("ambiguous_omission_rate", "时序模糊遗漏率"),
]

# 细线样式（覆盖 figure_style 默认 1.25）+ Paul Tol muted 柔和配色
LW = {"line": 0.8, "box": 0.5, "err": 0.7, "frame": 0.6}
BOX_FACE = "#E8ECF3"
BOX_EDGE = "#4477AA"
SCATTER_C = "#4477AA"
HIST_C = "#4477AA"
FLIER = dict(marker=".", markersize=1.6, markeredgewidth=0, alpha=0.45)


def _mask_outliers(series):
    """剔除 3×MAD 极端值。"""
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if series.size == 0:
        return series
    med = np.nanmedian(series)
    mad = np.nanmedian(np.abs(series - med)) * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return series
    return series[np.abs(series - med) <= 3 * mad]


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return save_figure(fig, Path(OUT_DIR) / name, FORMATS)


# =============================================================================
# 数据加载
# =============================================================================
def _frame_binocular_raw(df: pd.DataFrame) -> np.ndarray:
    """frame 级双眼原始几何直径 = 左右眼原始值按有效标志加权平均（px）。"""
    lv = df["left_pupil_valid_primary"].astype(bool)
    rv = df["right_pupil_valid_primary"].astype(bool)
    num = df["left_raw_pupil_diameter"] * lv + df["right_raw_pupil_diameter"] * rv
    den = lv.astype(int) + rv.astype(int)
    return (num / den.replace(0, np.nan)).to_numpy()


def _load_probe_windows(tracks=None):
    files = sorted(glob(os.path.join(NIR_TABLES, "sessions", "sub-*", "*_probe_pupil_windows.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df = df[df["window_name"] == "pre_30s"]
        if tracks:
            df = df[df["track"] == tracks]
        else:
            df = df[df["track_family"].isin(["primary_reference", "eye_preserved", "strict_sensitivity"])]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_visual_conditions():
    return pd.read_csv(os.path.join(NIR_TABLES, "tables", "publication_analysis", "stimulus_condition_summary.csv"))


def _load_trial_effects():
    return pd.read_csv(os.path.join(NIR_TABLES, "reference_adjusted_models", "trial_unadjusted_adjusted_effects.csv"))


def _load_heatmap_data():
    cand = pd.read_csv(os.path.join(NIR_TABLES, "candidate_validation", "nir_candidate_session_block_metrics.csv"))
    cand = cand[cand["metric"].isin(["pupil_geom_mean_diameter", "hard_pupil_fraction"])]
    pupil_wide = cand.pivot_table(index=["session_id", "block_num", "analysis_group_token"],
                                  columns="metric", values="binocular_raw_median").reset_index()
    beh = pd.read_csv(os.path.join(NIR_TABLES, "tables", "publication_analysis", "advanced_behavior_subject_block.csv"))
    merged = pupil_wide.merge(beh, left_on=["session_id", "block_num"], right_on=["subject", "block_num"], how="inner")
    collapsed = merged.groupby(["analysis_group_token", "block_num"])[
        ["pupil_geom_mean_diameter", "hard_pupil_fraction"] + [m for m, _ in BEHAVIOR_METRICS]].mean().reset_index()
    left_row = [pd.Series(collapsed["pupil_geom_mean_diameter"]).corr(collapsed[m], method="spearman") for m, _ in BEHAVIOR_METRICS]
    right_row = [pd.Series(collapsed["hard_pupil_fraction"]).corr(collapsed[m], method="spearman") for m, _ in BEHAVIOR_METRICS]
    return np.array(left_row), np.array(right_row), [l for _, l in BEHAVIOR_METRICS]


def _load_trend_data(grid_ms: int = 30) -> pd.DataFrame:
    """frame 级原始几何直径，按 30ms 网格聚合每个探针前 30s 的逐点中位数（不平滑）。"""
    grid = np.arange(0, 30000 + grid_ms, grid_ms)
    n_grid = len(grid)
    files = sorted(glob(os.path.join(NIR_TABLES, "sessions", "sub-*", "*_probe_pupil_windows.csv")))
    rows = []
    for pf in files:
        sid = os.path.basename(pf).replace("_probe_pupil_windows.csv", "")
        probe_df = pd.read_csv(pf)
        probe_df = probe_df[(probe_df["track"] == "binocular_primary") & (probe_df["window_name"] == "pre_30s")]
        fpath = os.path.join(FRAME_ROOT, sid, f"{sid}_nir_analysis_ready.csv")
        if not os.path.exists(fpath):
            continue
        frame = pd.read_csv(fpath, usecols=["left_raw_pupil_diameter", "left_pupil_valid_primary",
                                            "right_raw_pupil_diameter", "right_pupil_valid_primary", "unix_ms"],
                            low_memory=False)
        frame = frame.sort_values("unix_ms").reset_index(drop=True)
        unix = frame["unix_ms"].to_numpy()
        raw = _frame_binocular_raw(frame)
        for _, pr in probe_df.iterrows():
            onset = pr["probe_onset_ms"]
            left = int(np.searchsorted(unix, onset - 30000, side="left"))
            right = int(np.searchsorted(unix, onset, side="left"))
            if right <= left:
                continue
            rel_ms = onset - unix[left:right]  # 0..30000
            v = raw[left:right]
            bin_idx = (rel_ms // grid_ms).astype(int)
            ok = (bin_idx >= 0) & (bin_idx < n_grid) & np.isfinite(v)
            if not ok.any():
                continue
            grp = pd.Series(v[ok]).groupby(bin_idx[ok]).median()
            vals = grp.reindex(range(n_grid)).to_numpy()
            rows.append({"session_id": sid, "analysis_group_token": pr["analysis_group_token"],
                         "is_repeat_session": pr["is_repeat_session"], "block_num": pr["block_num"],
                         "probe_response": pr["probe_response"], "probe_vigilance": pr["probe_vigilance"],
                         "vals": vals})
    return pd.DataFrame(rows)


def _load_tracks():
    full = _load_probe_windows()
    track_map = {"binocular_primary": "双眼", "left_primary": "左眼", "right_primary": "右眼",
                 "binocular_strict": "双眼·严格", "left_strict": "左眼·严格", "right_strict": "右眼·严格"}
    full = full[full["track"].isin(track_map)][["session_id", "block_num", "probe_index_global", "track", "pupil_median"]]
    wide = full.pivot_table(index=["session_id", "block_num", "probe_index_global"], columns="track", values="pupil_median").reset_index()
    wide.columns = [track_map.get(c, c) for c in wide.columns]
    return wide


def _trend_arrays(vals_list):
    """将一组探针的 vals 数组展平为 (x_sec, y) 并按 30ms 网格取中位数，用于画原始趋势（不平滑）。"""
    n = len(vals_list[0])
    mat = np.vstack([np.asarray(v, dtype=float) for v in vals_list])
    med = np.nanmedian(mat, axis=0)
    q25 = np.nanpercentile(mat, 25, axis=0)
    q75 = np.nanpercentile(mat, 75, axis=0)
    x = -np.arange(n) / 1000.0 * 30  # 0..n → -30..0 s（30ms 步长，近似）
    x = np.linspace(-30, 0, n)
    return x, med, q25, q75


# =============================================================================
# 图 7：数据质量
# =============================================================================
def fig07_data_quality(df):
    fig, axes = make_figure(width="full", height_cm=4.6, nrows=1, ncols=3)
    panels = [("pupil_valid_fraction", "有效瞳孔帧比例", "A"), ("internal_coverage_fraction", "窗口内部时间覆盖", "B"),
              ("source_mode_binocular_fraction", "双眼轨道可用比例", "C")]
    for ax, (col, xlabel, lbl) in zip(axes, panels):
        vals = df[col].dropna().to_numpy()
        ax.hist(vals, bins=40, color="#4C78A8", edgecolor="white", linewidth=0.3)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel("观测数", fontsize=7)
        panel_label(ax, lbl)
        clean_axis(ax)
    finalize_layout(fig, left=0.08, right=0.985, bottom=0.18, top=0.90, wspace=0.34, hspace=0.36)
    return _save(fig, "fig07_pupil_data_quality")[0]


# =============================================================================
# 图 8：视觉控制
# =============================================================================
def fig08_visual_controls(vis):
    fig, axes = make_figure(width="full", height_cm=4.4, nrows=1, ncols=4)
    ax = axes[0]
    sizes = sorted(vis["stimulus_size"].dropna().unique())
    data_by_size = [vis.loc[vis["stimulus_size"] == s, "pupil_median"].dropna().to_numpy() for s in sizes]
    ax.boxplot(data_by_size, tick_labels=[f"{s:.2f}" for s in sizes], widths=0.55, patch_artist=True,
               boxprops=dict(linewidth=LW["box"], facecolor=BOX_FACE, edgecolor=BOX_EDGE),
               capprops=dict(linewidth=LW["box"]), whiskerprops=dict(linewidth=LW["box"]),
               medianprops=dict(linewidth=LW["box"], color="#C55A11"), flierprops=FLIER)
    ax.set_xlabel("刺激尺寸", fontsize=7)
    ax.set_ylabel(PUPIL_AXIS_ZH, fontsize=7)
    panel_label(ax, "A")
    clean_axis(ax)
    scatter_panels = [("current_central_rel_lum_mean", "数字相对亮度", "B"), ("current_central_rms_contrast", "RMS 对比度", "C"),
                      ("current_fruit_visible_area_fraction_central_roi", "刺激可见面积比例", "D")]
    for ax, (col, xlabel, lbl) in zip(axes[1:], scatter_panels):
        x = vis[col].dropna().to_numpy()
        y = vis.loc[np.isfinite(vis[col].to_numpy()), "pupil_median"].to_numpy()
        ax.scatter(x, y, s=4, alpha=0.35, color="#2F5597", edgecolors="none", linewidths=0)
        ax.set_xlabel(xlabel, fontsize=7)
        if lbl == "B":
            ax.set_ylabel(PUPIL_AXIS_ZH, fontsize=7)
        panel_label(ax, lbl)
        clean_axis(ax)
    finalize_layout(fig, left=0.07, right=0.985, bottom=0.18, top=0.90, wspace=0.44, hspace=0.36)
    return _save(fig, "fig08_visual_controls")[0]


# =============================================================================
# 图 9（新）：探针前 30s 原始瞳孔几何直径分组分布
# =============================================================================
def fig09_probe_states(trend):
    fig, axes = make_figure(width="full", height_cm=4.6, nrows=1, ncols=2)
    labels_map = [("probe_response", Q1_LABELS, "Q1 注意内容", "A"), ("probe_vigilance", Q2_LABELS, "Q2 警觉程度", "B")]
    for ax, (col, labels, title, lbl) in zip(axes, labels_map):
        series_by_code = []
        ticks = []
        for code in sorted(trend[col].dropna().unique()):
            sub = trend[trend[col] == code]["vals"]
            arr = np.vstack(sub.to_list())
            # 取每探针 30s 窗内的中位数（原始几何直径）
            med_per_probe = np.nanmedian(arr, axis=1)
            series_by_code.append(_mask_outliers(med_per_probe))
            ticks.append(labels.get(code, str(code)))
        ax.boxplot(series_by_code, tick_labels=ticks, widths=0.5, patch_artist=True,
                   boxprops=dict(linewidth=LW["box"], facecolor=BOX_FACE, edgecolor=BOX_EDGE),
                   capprops=dict(linewidth=LW["box"]), whiskerprops=dict(linewidth=LW["box"]),
                   medianprops=dict(linewidth=LW["box"], color="#C55A11"), flierprops=FLIER)
        ax.set_ylabel(PUPIL_AXIS_ZH, fontsize=7)
        ax.tick_params(axis="x", rotation=20, labelsize=7)
        ax.margins(x=0.06)
        panel_label(ax, lbl)
        clean_axis(ax)
    finalize_layout(fig, left=0.09, right=0.985, bottom=0.24, top=0.90, wspace=0.38, hspace=0.36)
    return _save(fig, "fig09_probe_states")[0]


# Paul Tol muted 柔和色（色盲友好，科研汇报常用）
TREND_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]


# =============================================================================
# 趋势图（独立单图，全宽横坐标长）：整体 / 按 Q1 / 按 Q2
# =============================================================================
def fig_trend_overall(trend):
    """整体探针前 30s 原始瞳孔趋势（一条中位数 + IQR 带）。"""
    fig, ax = make_figure(width="full", height_cm=5.4, nrows=1, ncols=1)
    x, med, q25, q75 = _trend_arrays(trend["vals"].to_list())
    ax.plot(x, med, color=TREND_COLORS[0], linewidth=LW["line"], label="中位数")
    ax.fill_between(x, q25, q75, color=TREND_COLORS[0], alpha=0.15, linewidth=0, label="IQR 四分位距")
    ax.axhline(np.nanmedian(med), color="#888888", linewidth=0.6, linestyle=":")
    band = np.concatenate([q25, q75])
    ax.set_ylim(np.nanpercentile(band, 5), np.nanpercentile(band, 95))
    ax.set_xlabel("距探针时间（s）", fontsize=8)
    ax.set_ylabel(PUPIL_AXIS_ZH, fontsize=8)
    ax.tick_params(labelsize=7)
    # 图例移到图外右侧，避免与曲线重叠且不被裁切
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, frameon=False)
    panel_label(ax, "A")
    clean_axis(ax)
    finalize_layout(fig, left=0.10, right=0.80, bottom=0.18, top=0.92, wspace=0.3, hspace=0.36)
    return _save(fig, "fig_trend_overall")[0]


def _trend_by_group(trend, col, labels, name):
    """按指定列（Q1/Q2）分组的探针前 30s 原始瞳孔趋势。"""
    fig, ax = make_figure(width="full", height_cm=5.4, nrows=1, ncols=1)
    band_all = []
    for i, code in enumerate(sorted(trend[col].dropna().unique())):
        sub = trend[trend[col] == code]["vals"]
        if sub.empty:
            continue
        x, med, q25, q75 = _trend_arrays(sub.to_list())
        band_all.append(np.concatenate([q25, q75]))
        c = TREND_COLORS[i % len(TREND_COLORS)]
        ax.plot(x, med, color=c, linewidth=LW["line"], label=f"{labels.get(code, code)} (n={len(sub)})")
        ax.fill_between(x, q25, q75, color=c, alpha=0.12, linewidth=0)
    if band_all:
        yall = np.concatenate(band_all)
        ax.set_ylim(np.nanpercentile(yall, 5), np.nanpercentile(yall, 95))
    ax.set_xlabel("距探针时间（s）", fontsize=8)
    ax.set_ylabel(PUPIL_AXIS_ZH, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, frameon=False, ncol=1)
    panel_label(ax, "A")
    clean_axis(ax)
    finalize_layout(fig, left=0.10, right=0.80, bottom=0.18, top=0.92, wspace=0.3, hspace=0.36)
    return _save(fig, name)[0]


def fig_trend_q1(trend):
    """按 Q1 注意内容分组的探针前 30s 原始瞳孔趋势。"""
    return _trend_by_group(trend, "probe_response", Q1_LABELS, "fig_trend_q1")


def fig_trend_q2(trend):
    """按 Q2 警觉程度分组的探针前 30s 原始瞳孔趋势。"""
    return _trend_by_group(trend, "probe_vigilance", Q2_LABELS, "fig_trend_q2")


# =============================================================================
# 图 10：试次级效应分解
# =============================================================================
def fig10_trial_effect_forest(effects):
    fig, axes = make_figure(width="full", height_cm=5.0, nrows=1, ncols=3)
    rows = [("pupil_within", False), ("pupil_within", True), ("pupil_between", False), ("pupil_between", True)]
    row_labels = ["参与者内·未调整", "参与者内·调整后", "参与者间·未调整", "参与者间·调整后"]
    y_pos = list(range(len(rows)))[::-1]
    for ax, outcome, lbl in zip(axes, ["rt", "omission", "commission"], ["A", "B", "C"]):
        sub = effects[effects["outcome"] == outcome]
        for y, (term, adj) in zip(y_pos, rows):
            r = sub[(sub["pupil_term"] == term) & (sub["adjusted"] == adj)]
            if r.empty:
                continue
            r = r.iloc[0]
            est, lo, hi = r["estimate"], r["ci_low"], r["ci_high"]
            is_between = term == "pupil_between"
            ax.errorbar(est, y, xerr=[[est - lo], [hi - est]], fmt="o" if is_between else "s",
                        color="#2F5597", markersize=3.5, markerfacecolor="#2F5597" if is_between else "white",
                        markeredgewidth=0.7, elinewidth=LW["err"], capsize=2.0)
        ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(row_labels, fontsize=6.5)
        ax.set_xlabel("效应估计（95% CI）", fontsize=7)
        panel_label(ax, lbl)
        clean_axis(ax)
    finalize_layout(fig, left=0.22, right=0.985, bottom=0.18, top=0.90, wspace=0.34, hspace=0.36)
    return _save(fig, "fig10_trial_effect_forest")[0]


# =============================================================================
# 图 19（旧）：六轨道稳健性
# =============================================================================
def fig19_tracks(wide):
    fig, axes = make_figure(width="full", height_cm=5.0, nrows=1, ncols=3)
    track_cols = ["双眼", "左眼", "右眼", "双眼·严格", "左眼·严格", "右眼·严格"]
    wide = wide.dropna(subset=track_cols)
    ax = axes[0]
    corr = wide[track_cols].corr(method="pearson").fillna(0)
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=0.7, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(track_cols))); ax.set_xticklabels(track_cols, rotation=30, ha="right", fontsize=6)
    ax.set_yticks(range(len(track_cols))); ax.set_yticklabels(track_cols, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    panel_label(ax, "A")
    clean_axis(ax)
    others = [c for c in track_cols if c != "双眼"]
    ax = axes[1]
    r = [wide[c].corr(wide["双眼"], method="pearson") for c in others]
    ax.barh([c.replace("·", "\n") for c in others], r, color="#4C78A8", height=0.55)
    ax.set_xlabel("与双眼主轨道 Pearson r", fontsize=7)
    panel_label(ax, "B")
    clean_axis(ax)
    ax = axes[2]
    diffs = [np.nanmedian(np.abs(wide[c] - wide["双眼"])) for c in others]
    ax.barh([c.replace("·", "\n") for c in others], diffs, color="#C55A11", height=0.55)
    ax.set_xlabel("与主轨道中位数绝对差", fontsize=7)
    panel_label(ax, "C")
    clean_axis(ax)
    finalize_layout(fig, left=0.13, right=0.985, bottom=0.20, top=0.90, wspace=0.38, hspace=0.36)
    return _save(fig, "fig19_tracks")[0]




# =============================================================================
# 热力图
# =============================================================================
def fig_heatmap_pupil_behavior(left_row, right_row, behavior_labels):
    # 合并为一张：2 行（瞳孔几何平均直径 / 瞳孔面积比例）× 7 列（行为指标）
    fig, ax = make_figure(width="full", height_cm=3.8, nrows=1, ncols=1)
    mat = np.vstack([left_row, right_row])
    row_labels = ["瞳孔几何\n平均直径", "瞳孔\n面积比例"]
    vmax = max(0.2, np.nanmax(np.abs(mat)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(2))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xticks(range(len(behavior_labels)))
    ax.set_xticklabels(behavior_labels, rotation=30, ha="right", fontsize=6)
    for i in range(2):
        for j in range(len(behavior_labels)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="black" if abs(v) < 0.5 * vmax else "white")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    clean_axis(ax)
    finalize_layout(fig, left=0.13, right=0.90, bottom=0.30, top=0.90)
    return _save(fig, "fig_heatmap_pupil_behavior")[0]


def main():
    configure_publication_style()
    # 全局细线
    plt.rcParams.update({"lines.linewidth": LW["line"], "lines.markersize": 2.5, "axes.linewidth": LW["frame"],
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    print("加载数据...")
    probe_df = _load_probe_windows(tracks="binocular_primary")
    vis_df = _load_visual_conditions()
    effects_df = _load_trial_effects()
    left_row, right_row, beh_labels = _load_heatmap_data()
    print("重算探针前 30s 原始瞳孔趋势（30ms 逐点）...")
    trend_df = _load_trend_data()
    print(f"  趋势探针数: {len(trend_df)}")
    wide_tracks = _load_tracks()
    outputs = [
        fig07_data_quality(probe_df), fig08_visual_controls(vis_df), fig09_probe_states(trend_df),
        fig10_trial_effect_forest(effects_df), fig19_tracks(wide_tracks),
        fig_trend_overall(trend_df), fig_trend_q1(trend_df), fig_trend_q2(trend_df),
        fig_heatmap_pupil_behavior(left_row, right_row, beh_labels),
    ]
    print("已生成图（new_figure/）：")
    for p in outputs:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
