"""Formal RGB figure pack with external captions and metric coverage auditing.

Figures intentionally contain no internal title. Every candidate metric receives
an explicit generated/not-estimable row for every prespecified view so a missing
plot cannot be confused with a forgotten plot.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VIEW_SPECS = (
    "session_distribution",
    "block_b1_b2",
    "time_on_task",
    "probe_q1_nominal",
    "probe_q2_ordinal",
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_")[:160] or "metric"


def _ylabel(metric: str) -> str:
    mapping = {
        "ear_mean": "平均眼睑纵横比（EAR）",
        "ear_left": "左眼眼睑纵横比（EAR）",
        "ear_right": "右眼眼睑纵横比（EAR）",
        "eye_openness_norm": "相对眼睑开放度",
        "closure_fraction": "眼睑闭合比例",
        "global_motion_energy": "全局运动能量",
        "global_motion_energy_per_sec": "单位时间全局运动能量",
        "changed_pixel_ratio": "变化像素比例",
        "gray_mean": "画面平均灰度",
        "gray_mean_delta": "画面平均灰度变化",
        "pose_visibility_mean": "姿态关键点平均可见度",
        "shoulder_line_angle_rad": "肩线角度（弧度）",
    }
    if metric in mapping:
        return mapping[metric]
    if "speed_per_sec" in metric:
        return f"运动速度（{metric}）"
    if str(metric).lower().startswith("au") or "__AU" in metric:
        return f"动作单元强度（{metric}）"
    return f"指标值（{metric}）"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _session_distribution(summary: pd.DataFrame, metric: str, path: Path) -> tuple[bool, str]:
    data = summary[(summary["scale"].eq("session")) & summary["metric"].eq(metric)].copy()
    values = pd.to_numeric(data.get("median"), errors="coerce").dropna()
    if len(values) < 2:
        return False, "session scale has fewer than 2 valid values"
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(values, bins=min(12, max(4, int(np.sqrt(len(values))))), edgecolor="black")
    ax.set_xlabel(_ylabel(metric))
    ax.set_ylabel("场次数")
    ax.grid(axis="y", alpha=.2)
    _save(fig, path)
    return True, ""


def _block_pair(summary: pd.DataFrame, metric: str, path: Path) -> tuple[bool, str]:
    data = summary[(summary["scale"].eq("block")) & summary["metric"].eq(metric)].copy()
    if "block" not in data or data.empty:
        return False, "block scale unavailable"
    data["block"] = pd.to_numeric(data["block"], errors="coerce")
    data["median"] = pd.to_numeric(data["median"], errors="coerce")
    wide = data.pivot_table(index="session_id", columns="block", values="median", aggfunc="first")
    if 1 not in wide or 2 not in wide:
        return False, "both B1 and B2 are not available"
    paired = wide[[1, 2]].dropna()
    if len(paired) < 2:
        return False, "fewer than 2 complete B1/B2 session pairs"
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for row in paired.itertuples(index=False):
        ax.plot([1, 2], [row[0], row[1]], marker="o", linewidth=.8, alpha=.45)
    ax.plot([1, 2], paired.mean(axis=0).to_numpy(float), marker="o", linewidth=2.2, label="场次均值")
    ax.set_xticks([1, 2], ["区块1", "区块2"])
    ax.set_xlabel("区块")
    ax.set_ylabel(_ylabel(metric))
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    _save(fig, path)
    return True, ""


def _time_plot(time_bins: pd.DataFrame, metric: str, path: Path) -> tuple[bool, str]:
    data = time_bins[time_bins["metric"].eq(metric)].copy() if not time_bins.empty else pd.DataFrame()
    if data.empty:
        return False, "no time-on-task bins"
    data["time_sec"] = pd.to_numeric(data["time_sec"], errors="coerce")
    data["median"] = pd.to_numeric(data["median"], errors="coerce")
    # Session-level traces are descriptive. Inference remains participant-clustered elsewhere.
    grouped = data.dropna(subset=["time_sec", "median"]).groupby("time_sec")["median"].agg(["median", "count"]).reset_index()
    if len(grouped) < 3:
        return False, "fewer than 3 populated time bins"
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(grouped["time_sec"] / 60.0, grouped["median"], marker="o", linewidth=1.4, label="跨场次中位数")
    ax.set_xlabel("任务进行时间（分钟）")
    ax.set_ylabel(_ylabel(metric))
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    _save(fig, path)
    return True, ""


def _probe_category(probes: pd.DataFrame, metric: str, response_col: str, xlabels: str, path: Path) -> tuple[bool, str]:
    if probes.empty or response_col not in probes:
        return False, f"{response_col} unavailable"
    data = probes[probes["metric"].eq(metric)].copy()
    data["median"] = pd.to_numeric(data.get("median"), errors="coerce")
    data[response_col] = pd.to_numeric(data[response_col], errors="coerce")
    data = data.dropna(subset=["median", response_col])
    cats = sorted(data[response_col].unique())
    if len(cats) < 2:
        return False, "fewer than 2 populated response categories"
    arrays = [data.loc[data[response_col].eq(cat), "median"].to_numpy(float) for cat in cats]
    if sum(len(a) for a in arrays) < 4:
        return False, "fewer than 4 valid probe observations"
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    # labels= is used for compatibility with matplotlib 3.8.
    ax.boxplot(arrays, labels=[str(int(c)) if float(c).is_integer() else str(c) for c in cats], showfliers=False)
    ax.set_xlabel(xlabels)
    ax.set_ylabel(_ylabel(metric))
    ax.grid(axis="y", alpha=.2)
    _save(fig, path)
    return True, ""


def generate_rgb_figure_pack(
    summary: pd.DataFrame,
    probe_summary: pd.DataFrame,
    time_bins: pd.DataFrame,
    validation: pd.DataFrame,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate figures and a full metric×view audit.

    The manifest is the external-caption source of truth. No image title is set.
    """
    fig_root = Path(output_root) / "figures"
    fig_root.mkdir(parents=True, exist_ok=True)
    metrics = sorted(set(validation.get("metric", pd.Series(dtype=str)).dropna().astype(str)))
    if not metrics:
        metrics = sorted(set(summary.get("metric", pd.Series(dtype=str)).dropna().astype(str)))
    audit_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for metric in metrics:
        stem = _safe_name(metric)
        specs = [
            ("session_distribution", _session_distribution, (summary, metric, fig_root / f"{stem}__session_distribution.png"), "场次分布"),
            ("block_b1_b2", _block_pair, (summary, metric, fig_root / f"{stem}__block_b1_b2.png"), "区块1与区块2配对比较"),
            ("time_on_task", _time_plot, (time_bins, metric, fig_root / f"{stem}__time_on_task.png"), "随任务进行时间的变化"),
            ("probe_q1_nominal", _probe_category, (probe_summary, metric, "q1_nominal_4class", "Q1 类别（名义四分类）", fig_root / f"{stem}__probe_q1_nominal.png"), "探针Q1名义四分类下的分布"),
            ("probe_q2_ordinal", _probe_category, (probe_summary, metric, "q2_ordinal_4level", "Q2 等级（有序四级）", fig_root / f"{stem}__probe_q2_ordinal.png"), "探针Q2有序四级下的分布"),
        ]
        for view, fn, args, caption_tail in specs:
            path = args[-1]
            try:
                generated, reason = fn(*args)
            except Exception as exc:  # figure failures are auditable, not silent
                generated, reason = False, f"{type(exc).__name__}: {exc}"
            audit_rows.append({
                "metric": metric, "view": view, "status": "generated" if generated else "not_estimable",
                "reason": reason, "internal_title_present": False,
                "caption_external": True, "file": str(path.relative_to(output_root)) if generated else "",
            })
            if generated:
                manifest_rows.append({
                    "figure_id": f"rgb_{stem}_{view}", "metric": metric, "view": view,
                    "file": str(path.relative_to(output_root)),
                    "caption_zh": f"{_ylabel(metric)}：{caption_tail}。",
                    "internal_title_present": False,
                    "caption_location": "external_manifest",
                    "inference_note": "图为描述性可视化；正式推断使用参与者聚类结构。",
                })
    return pd.DataFrame(manifest_rows), pd.DataFrame(audit_rows)
