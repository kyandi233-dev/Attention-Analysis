"""阶段5：连续序列评估（主环境 3.13）。

对 44×121 序列 + PuRe/PuReST 检测，做：
- classify_status（8 状态：no_face/openness_missing/closed_gate/interpolated/accepted/low_outline/detector_missing）
- 真实插值：两端 observed、缺口≤max_gap_ms、端点直径比≤tol 的缺口线性填充并标 is_interpolated
- 指标：observed_rate / visible_coverage / 闭眼原始误检 / 连续指标（diameter_log_jump、
  center_jump_norm÷160、recovery_frames_ms）/ interpolated 占比
- 报告 benchmark_report_sequence.md + 图

插值帧不计 observed、不进 FP；不跨 closed/no_face/序列边界。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from .benchmark import geometry_plausible


def classify_status(face_detected, openness, openness_visible, geom_photo_ok, outline_ok, is_interpolated) -> str:
    """8 状态状态机（v1 口径）。geom_photo_ok=返回且过几何/光度门；outline_ok=轮廓置信≥min。"""
    if not face_detected:
        return "no_face"
    if openness is None or math.isnan(openness):
        return "openness_missing"
    if openness < openness_visible:
        return "closed_gate"
    if is_interpolated:
        return "interpolated"
    if geom_photo_ok:
        return "accepted" if outline_ok else "low_outline"
    return "detector_missing"


def apply_interpolation(
    frame_df: pd.DataFrame,
    max_gap_ms: float,
    endpoint_diameter_tol: float,
    endpoint_center_tol_norm: float = 0.12,
    center_span_px: float = 160.0,
    openness_visible: float = 0.55,
    openness_closed: float = 0.20,
) -> pd.DataFrame:
    """只在安全条件满足时生成插值副轨，且不覆盖原始检测列。"""
    out = frame_df.copy().reset_index(drop=True)
    out["is_interpolated"] = 0
    for column in ("major_diameter", "minor_diameter", "center_x", "center_y"):
        out[f"{column}_interpolated"] = out[column]
    if "angle_deg" in out.columns:
        out["angle_deg_interpolated"] = out["angle_deg"]
    if "sequence_id" in out.columns and out["sequence_id"].nunique(dropna=False) > 1:
        return out
    if not {"face_detected", "roi_path", "visible_proxy", "p80_closed_proxy"}.issubset(out.columns):
        return out

    observed = out["observed"].astype(bool).to_numpy()
    times = out["unix_ms"].to_numpy(dtype=float)
    frame_offsets = out["frame_offset"].to_numpy(dtype=float) if "frame_offset" in out.columns else None
    major = out["major_diameter"].to_numpy(dtype=float)
    minor = out["minor_diameter"].to_numpy(dtype=float)
    cx = out["center_x"].to_numpy(dtype=float)
    cy = out["center_y"].to_numpy(dtype=float)
    n = len(out)
    idx = 0
    while idx < n:
        if observed[idx]:
            idx += 1
            continue
        gap_start = idx
        while idx < n and not observed[idx]:
            idx += 1
        gap_end = idx
        left = gap_start - 1
        right = gap_end
        if left < 0 or right >= n:
            continue
        span_times = times[left:right + 1]
        if not np.isfinite(span_times).all() or not np.all(np.diff(span_times) > 0):
            continue
        if times[right] - times[left] > max_gap_ms:
            continue
        if frame_offsets is not None:
            span_offsets = frame_offsets[left:right + 1]
            if not np.isfinite(span_offsets).all() or not np.all(np.diff(span_offsets) == 1):
                continue
        gap = out.iloc[gap_start:gap_end]
        if not gap["face_detected"].fillna(0).astype(bool).all():
            continue
        if not gap["roi_path"].fillna("").astype(str).str.strip().ne("").all():
            continue
        if not gap["visible_proxy"].fillna(0).astype(bool).all():
            continue
        if gap["p80_closed_proxy"].fillna(0).astype(bool).any():
            continue
        if "openness" in gap.columns:
            openness = pd.to_numeric(gap["openness"], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(openness).all() or np.any(openness < openness_visible) or np.any(openness < openness_closed):
                continue
        endpoint_values = np.array([
            major[left], major[right], minor[left], minor[right],
            cx[left], cx[right], cy[left], cy[right],
        ], dtype=float)
        if not np.isfinite(endpoint_values).all() or np.any(endpoint_values[:4] <= 0):
            continue
        major_rel = abs(major[right] / major[left] - 1.0)
        minor_rel = abs(minor[right] / minor[left] - 1.0)
        center_delta = math.hypot(cx[right] - cx[left], cy[right] - cy[left]) / center_span_px
        if major_rel > endpoint_diameter_tol or minor_rel > endpoint_diameter_tol or center_delta > endpoint_center_tol_norm:
            continue
        n_fill = gap_end - gap_start
        angle = pd.to_numeric(out["angle_deg"], errors="coerce").to_numpy(dtype=float) if "angle_deg" in out.columns else None
        for j, k in enumerate(range(gap_start, gap_end)):
            f = (j + 1) / (n_fill + 1)
            out.loc[k, "major_diameter_interpolated"] = major[left] + f * (major[right] - major[left])
            out.loc[k, "minor_diameter_interpolated"] = minor[left] + f * (minor[right] - minor[left])
            out.loc[k, "center_x_interpolated"] = cx[left] + f * (cx[right] - cx[left])
            out.loc[k, "center_y_interpolated"] = cy[left] + f * (cy[right] - cy[left])
            if angle is not None and np.isfinite(angle[left]) and np.isfinite(angle[right]):
                out.loc[k, "angle_deg_interpolated"] = angle[left] + f * (angle[right] - angle[left])
            out.loc[k, "is_interpolated"] = 1
    return out

def continuity_metrics(rows: pd.DataFrame) -> dict:
    """连续指标：diameter_log_jump / center_jump_norm / recovery_frames_ms。

    按 (sequence_id, eye) 分组，避免跨序列边界配对。相邻有效对 = 前一帧与当前帧
    都 observed 且 visible_proxy=1。recovery = 每个 p80_closed 段尾 → 首次 visible
    → 首次 observed 的时间差（ms）。
    """
    jumps, centers, recovery_ms = [], [], []
    for (sequence_id, eye), sub in rows.groupby(["sequence_id", "eye"]):
        sub = sub.sort_values("frame_offset")
        visible_proxy = sub["visible_proxy"].fillna(0).astype(int)
        p80_closed_proxy = sub["p80_closed_proxy"].fillna(0).astype(int)
        valid = sub["observed"].astype(bool) & (visible_proxy == 1)
        adj = valid & valid.shift(1)
        diam = sub["major_diameter"].to_numpy(dtype=float)
        cx = sub["center_x"].to_numpy(dtype=float)
        cy = sub["center_y"].to_numpy(dtype=float)
        times = sub["unix_ms"].to_numpy(dtype=float)
        for i in np.where(adj.to_numpy())[0]:
            if diam[i] > 0 and diam[i - 1] > 0:
                jumps.append(abs(math.log(diam[i] / diam[i - 1])))
            centers.append(math.hypot(cx[i] - cx[i - 1], cy[i] - cy[i - 1]) / 160.0)

        # recovery：闭眼段结束 → 首次 visible → 首次 observed
        closed = (p80_closed_proxy == 1).to_numpy()
        visible = (visible_proxy == 1).to_numpy()
        observed_arr = sub["observed"].astype(bool).to_numpy()
        i = 0
        n = len(sub)
        while i < n:
            if closed[i]:
                while i < n and closed[i]:
                    i += 1
                seg_end = i
                if seg_end >= n:
                    break
                vis_idx = next((k for k in range(seg_end, n) if visible[k]), None)
                if vis_idx is None:
                    break
                obs_idx = next((k for k in range(vis_idx, n) if observed_arr[k]), None)
                if obs_idx is not None:
                    recovery_ms.append(times[obs_idx] - times[vis_idx])
            else:
                i += 1

    return {
        "diameter_log_jump_median": float(np.median(jumps)) if jumps else np.nan,
        "center_jump_norm_median": float(np.median(centers)) if centers else np.nan,
        "recovery_frames_ms_median": float(np.median(recovery_ms)) if recovery_ms else np.nan,
        "n_recovery_events": len(recovery_ms),
    }


def evaluate_sequences(config: Config, tag: str, sequence_tag: str | None = None) -> tuple[pd.DataFrame, dict]:
    """合并序列 manifest + 检测结果，逐算法算指标。"""
    sequence_root = config.path_value("sequence_artifact_root")
    manifest = pd.read_csv(sequence_root / "sequence_manifest.csv")
    det = pd.read_csv(sequence_root / "detections_sequence.csv")
    seq_cfg = config.section("nir")["sequence"]
    photo = float(config.section("nir")["benchmark"].get("photometric_threshold", 0.02))
    outline_min = float(seq_cfg["outline_min"])
    openness_visible = float(seq_cfg["openness_visible"])
    max_gap_ms = float(max(config.section("nir")["interpolation_gaps_ms"]))

    merged = manifest.merge(
        det[["sequence_id", "eye", "frame_offset", "algorithm", "algorithm_returned", "returned",
             "quality_status", "detector_status", "session_state", "center_x", "center_y",
             "major_diameter", "minor_diameter", "angle_deg", "confidence",
             "outline_confidence", "photometric_contrast"]],
        on=["sequence_id", "eye", "frame_offset"], how="left",
    )

    summary_rows = []
    per_sequence = []
    for algorithm, group in merged.groupby("algorithm"):
        group = group.copy()
        group["raw_geom_photo_ok"] = group.apply(
            lambda r: (
                int(r["algorithm_returned"]) == 1
                and geometry_plausible((r["center_x"], r["center_y"]), r["major_diameter"], r["minor_diameter"])
                and pd.notna(r["photometric_contrast"]) and float(r["photometric_contrast"]) > photo
            ),
            axis=1,
        )
        group["geom_photo_ok"] = group.apply(
            lambda r: (
                int(r["returned"]) == 1
                and geometry_plausible((r["center_x"], r["center_y"]), r["major_diameter"], r["minor_diameter"])
                and pd.notna(r["photometric_contrast"]) and float(r["photometric_contrast"]) > photo
            ),
            axis=1,
        )
        group["outline_ok"] = group.apply(
            lambda r: pd.notna(r["outline_confidence"]) and float(r["outline_confidence"]) >= outline_min,
            axis=1,
        )
        group["detector_ok"] = group["geom_photo_ok"].astype(bool) & group["outline_ok"].astype(bool)
        group["raw_detector_ok"] = group["raw_geom_photo_ok"].astype(bool) & group["outline_ok"].astype(bool)
        # 插值（按 sequence×eye 分组后合并回）
        interpolated = []
        for (sid, eye), sub in group.groupby(["sequence_id", "eye"]):
            sub = sub.sort_values("frame_offset").copy()
            sub["observed"] = (
                sub["detector_ok"].astype(bool)
                & sub["face_detected"].fillna(0).astype(bool)
                & sub["roi_path"].fillna("").astype(str).str.strip().ne("")
                & sub["visible_proxy"].fillna(0).astype(int).eq(1)
                & sub["p80_closed_proxy"].fillna(0).astype(int).eq(0)
                & pd.to_numeric(sub["openness"], errors="coerce").ge(openness_visible)
            )
            filled = apply_interpolation(
                sub,
                max_gap_ms,
                float(seq_cfg["endpoint_diameter_tol"]),
                endpoint_center_tol_norm=float(seq_cfg.get("endpoint_center_tol_norm", 0.12)),
                center_span_px=float(seq_cfg.get("center_span_px", 160.0)),
                openness_visible=openness_visible,
                openness_closed=float(seq_cfg.get("openness_closed", 0.20)),
            )
            interpolated.append(filled)
        group = pd.concat(interpolated, ignore_index=True)
        group["observed"] = group["observed"].astype(bool) & (group["is_interpolated"] == 0)
        group["status"] = group.apply(
            lambda r: classify_status(
                int(r["face_detected"]), r["openness"], openness_visible,
                bool(r["geom_photo_ok"]), bool(r["outline_ok"]), int(r["is_interpolated"]) == 1,
            ),
            axis=1,
        )

        total = len(group)
        visible = group[group["visible_proxy"].fillna(0).astype(int) == 1]
        closed = group[group["p80_closed_proxy"].fillna(0).astype(int) == 1]
        raw_fp_among_closed = float(group["raw_detector_ok"][group["p80_closed_proxy"].fillna(0).astype(int) == 1].mean()) if len(closed) else np.nan
        observed_rate = float(group["observed"].mean()) if total else np.nan
        visible_coverage = float(visible["observed"].mean()) if len(visible) else np.nan
        interp_frames = int(group["is_interpolated"].sum())

        cm = continuity_metrics(group)

        summary_rows.append({
            "algorithm": algorithm,
            "total_frames": total,
            "observed_frames": int(group["observed"].sum()),
            "observed_rate": observed_rate,
            "visible_coverage": visible_coverage,
            "raw_fp_among_p80_closed": raw_fp_among_closed,
            "closed_frames": len(closed),
            "interpolated_frames": interp_frames,
            "interpolated_fraction": interp_frames / total if total else np.nan,
            "closed_gate_frames": int((group["status"] == "closed_gate").sum()),
            "visibility_rejected_frames": int((group["quality_status"] == "visibility_rejected").sum()),
            "diameter_rejected_frames": int((group["quality_status"] == "diameter_rejected").sum()),
            "session_reset_frames": int((group["session_state"] == "session_reset").sum()),
            "status_counts": group["status"].value_counts().to_dict(),
            **cm,
        })
        per_sequence.append(group.assign(algorithm=algorithm))

    summary = pd.DataFrame(summary_rows)
    return summary, {"per_sequence": per_sequence, "config": seq_cfg}


def _plot_timeline(output: Path, group: pd.DataFrame, sid: str, eye: str, title: str, openness_visible: float = 0.55, openness_closed: float = 0.20) -> Path:
    """连续追踪时间线：瞳孔直径 vs 帧，叠加门控/闭眼/插值/漏检色带。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, (algorithm, g) in zip(axes, group.groupby("algorithm")):
        sub = g[(g["sequence_id"] == sid) & (g["eye"] == eye)].sort_values("frame_offset")
        x = sub["frame_offset"].values
        openness = sub["openness"].fillna(0).values
        closed = openness < openness_visible
        p80 = sub["p80_closed_proxy"].fillna(0).astype(int).values == 1
        nf = sub["face_detected"].fillna(0).astype(int).values == 0
        ax.fill_between(x, 0, 60, where=nf, color="gray", alpha=0.5, label="no_face")
        ax.fill_between(x, 0, 60, where=closed & ~nf, color="#f1c40f", alpha=0.35, label=f"openness<{openness_visible:g}(门控)")
        ax.fill_between(x, 0, 60, where=p80 & ~nf, color="#e67e22", alpha=0.5, label="p80闭眼<0.20")
        obs = sub["observed"].astype(bool)
        interp = sub["is_interpolated"] == 1
        ax.plot(x[obs], sub.loc[obs, "major_diameter"], "o", ms=3, color="#2ecc71", label="accepted(观测)")
        if interp.any():
            interp_column = (
                "major_diameter_interpolated"
                if "major_diameter_interpolated" in sub.columns
                else "major_diameter"
            )
            ax.plot(x[interp], sub.loc[interp, interp_column], "s", ms=3, color="#3498db", label="interpolated(副轨)")
        dm = ~obs & ~interp & ~closed & ~nf
        ax.plot(x[dm], np.full(dm.sum(), -2), "x", ms=2, color="#95a5a6", label="detector_missing")
        ax.set_ylabel(f"{algorithm} 瞳孔直径(px)")
        ax.set_ylim(-5, 60)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
    axes[0].set_title(title)
    axes[-1].set_xlabel("帧号（0-120，~4秒）")
    fig.tight_layout()
    path = output / f"timeline_{sid}_{eye}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_sequence_report(config: Config, tag: str, summary: pd.DataFrame, context: dict, sequence_tag: str | None = None) -> Path:
    """生成 benchmark_report_sequence.md + 图（状态时间线抽样、恢复事件）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    sequence_root = config.path_value("sequence_artifact_root")
    output = config.path_value("sequence_artifact_root")
    output.mkdir(parents=True, exist_ok=True)
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # 图1：逐算法状态组成（堆叠条形）
    statuses = ["accepted", "interpolated", "closed_gate", "low_outline", "detector_missing", "openness_missing", "no_face"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(summary))
    colors = {"accepted": "#2ecc71", "interpolated": "#3498db", "closed_gate": "#f1c40f",
              "low_outline": "#e67e22", "detector_missing": "#95a5a6", "openness_missing": "#7f8c8d", "no_face": "#bdc3c7"}
    for status in statuses:
        values = [row.get("status_counts", {}).get(status, 0) for _, row in summary.iterrows()]
        ax.bar(summary["algorithm"], values, bottom=bottom, label=status, color=colors.get(status))
        bottom += np.array(values)
    ax.set_ylabel("帧数")
    ax.legend(fontsize=8)
    ax.set_title("逐算法逐帧状态组成")
    fig.tight_layout()
    p1 = plots / "01_status_waterfall.png"
    fig.savefig(p1, dpi=130); plt.close(fig)

    # 图2：关键指标对比
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(summary))
    ax.bar(x - 0.25, summary["observed_rate"], 0.25, label="observed_rate")
    ax.bar(x, summary["visible_coverage"], 0.25, label="visible_coverage")
    ax.bar(x + 0.25, summary["raw_fp_among_p80_closed"], 0.25, label="raw FP among closed")
    ax.set_xticks(x, summary["algorithm"])
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("observed / visible_coverage / 闭眼原始误检")
    fig.tight_layout()
    p2 = plots / "02_rates.png"
    fig.savefig(p2, dpi=130); plt.close(fig)

    # 时间线图（用 per_sequence 分组，PuRe/PuReST 双栏）
    combined = pd.concat(context["per_sequence"], ignore_index=True) if context.get("per_sequence") else None
    if combined is not None:
        for sid, eye, title in [
            ("sub-000_blink_transition", "eye_right", "sub-000 眨眼转换段（右眼）：闭眼→门控→恢复"),
            ("sub-001_block3_uniform", "eye_right", "sub-001 Block3 均匀段（右眼）：稳定追踪"),
        ]:
            _plot_timeline(plots, combined, sid, eye, title, float(context["config"]["openness_visible"]), float(context["config"].get("openness_closed", 0.20)))

    # 报告
    summary.to_csv(output / "sequence_summary.csv", index=False, encoding="utf-8-sig")

    lines = []
    lines.append(f"# 阶段5｜PuReST 连续序列评估报告")
    lines.append("")
    lines.append(f"> {pd.Timestamp.now().isoformat(timespec='seconds')}（Asia/Shanghai）｜44 段×121 帧×双眼；门控 openness≥{context['config']['openness_visible']} 可见、<{context['config']['openness_closed']} 闭眼；插值 ≤{max(config.section('nir')['interpolation_gaps_ms'])}ms 且端点直径比≤{context['config']['endpoint_diameter_tol']}。")
    lines.append("")
    lines.append("## 指标")
    lines.append("")
    lines.append("![状态组成](plots/01_status_waterfall.png)")
    lines.append("")
    lines.append("![关键率](plots/02_rates.png)")
    lines.append("")
    lines.append("## 连续追踪时间线")
    lines.append("")
    lines.append("![眨眼段](plots/timeline_sub-000_blink_transition_eye_right.png)")
    lines.append("")
    lines.append("![均匀段](plots/timeline_sub-001_block3_uniform_eye_right.png)")
    lines.append("")
    lines.append(
        f"时间线说明：绿点=成功检测(观测)、蓝方块=插值副轨、"
        f"黄带=openness<{context['config']['openness_visible']:g}门控、"
        f"橙带=p80闭眼、灰带=no_face、灰叉=漏检。"
        "眨眼段中的恢复差异只作为历史序列证据，不能直接冻结正式数据参数。"
    )
    lines.append("")
    lines.append("| 算法 | observed_rate | visible_coverage | 闭眼原始误检 | 插值帧数/占比 | diameter_log_jump中位 | center_jump中位 | 恢复帧(ms)中位 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        raw_fp = r["raw_fp_among_p80_closed"]
        raw_fp_str = f"{raw_fp:.3f}" if pd.notna(raw_fp) else "—"
        recovery = r["recovery_frames_ms_median"]
        recovery_str = f"{recovery:.0f}" if pd.notna(recovery) else "—"
        lines.append(
            f"| {r['algorithm']} | {r['observed_rate']:.3f} | {r['visible_coverage']:.3f} | "
            f"{raw_fp_str} | "
            f"{r['interpolated_frames']} ({r['interpolated_fraction']:.3f}) | "
            f"{r['diameter_log_jump_median']:.3f} | {r['center_jump_norm_median']:.3f} | "
            f"{recovery_str} |"
        )
    lines.append("")
    lines.append("## 结论与边界")
    lines.append("")
    lines.append("- 门控校准结果见 gate_calibration.csv（openness vs 528 眼人工可见性，只报告不重调）。")
    lines.append("- PuReST使用普通连续调用；`pupil_min_mm_compat`仅为内部首帧/重检兼容设置，生产接受严格由Python层px/几何/光度/outline/可见性门决定。")
    lines.append("- 插值只写`*_interpolated`副轨，原始检测列不覆盖，且不计observed/FP。")
    lines.append("")
    report = output / "benchmark_report_sequence.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report




