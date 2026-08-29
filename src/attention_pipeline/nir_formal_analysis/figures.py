from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from attention_pipeline.config import load_config
from attention_pipeline.nir_analysis_ready.candidate_metrics import PUPIL_CANDIDATE_METRICS
from .pupil_tables import selected_sessions

FIGURE_PIPELINE_VERSION = "nir-formal-figures-v1"

WINDOW_FEATURES = (
    "pupil_median",
    "pupil_mean",
    "pupil_mad",
    "pupil_iqr",
    "pupil_sd",
    "pupil_p10",
    "pupil_p90",
    "pupil_slope_per_sec",
    "pupil_diff_mad",
    "pupil_diff_rate_mad_per_sec",
    "pupil_peak_to_trough",
    "pupil_dilation_velocity_median_per_sec",
    "pupil_constriction_velocity_median_per_sec",
)
WINDOW_QC_FEATURES = (
    "pupil_valid_fraction",
    "internal_coverage_fraction",
    "available_duration_fraction",
    "max_temporal_gap_sec",
)
EVENT_FEATURES = (
    "baseline_median",
    "baseline_mad",
    "dilation_peak_amplitude",
    "dilation_peak_latency_ms",
    "constriction_peak_amplitude",
    "constriction_peak_latency_ms",
    "dominant_peak_amplitude_abs",
    "dominant_peak_latency_ms",
    "recovery_tolerance",
    "recovery_time_after_onset_ms",
    "late_recovery_residual",
    "late_recovery_abs_residual",
)

PUPIL_LABELS_ZH = {
    "pupil_geom_mean_diameter": "瞳孔椭圆轴几何均值直径",
    "pupil_equivalent_diameter": "瞳孔等效直径",
    "pupil_axis_a": "瞳孔椭圆长/短轴 A",
    "pupil_axis_b": "瞳孔椭圆长/短轴 B",
    "pupil_contour_area": "瞳孔轮廓面积",
    "pupil_ellipse_area": "瞳孔椭圆面积",
    "hard_pupil_fraction": "硬分割瞳孔面积比例",
    "soft_pupil_fraction": "软分割瞳孔面积比例",
}
FEATURE_LABELS_ZH = {
    "pupil_median": "瞳孔中位数",
    "pupil_mean": "瞳孔均值",
    "pupil_mad": "瞳孔中位数绝对偏差",
    "pupil_iqr": "瞳孔四分位距",
    "pupil_sd": "瞳孔标准差",
    "pupil_p10": "瞳孔 P10",
    "pupil_p90": "瞳孔 P90",
    "pupil_slope_per_sec": "瞳孔稳健斜率（/s）",
    "pupil_diff_mad": "瞳孔相邻差值 MAD",
    "pupil_diff_rate_mad_per_sec": "瞳孔变化率 MAD（/s）",
    "pupil_peak_to_trough": "瞳孔峰谷幅度",
    "pupil_dilation_velocity_median_per_sec": "瞳孔扩张速度中位数（/s）",
    "pupil_constriction_velocity_median_per_sec": "瞳孔收缩速度中位数（/s）",
    "pupil_valid_fraction": "瞳孔有效比例",
    "internal_coverage_fraction": "窗口内部覆盖比例",
    "available_duration_fraction": "理论可用时长比例",
    "max_temporal_gap_sec": "最大时间缺口（s）",
    "baseline_median": "事件前局部参考中位数",
    "baseline_mad": "事件前局部参考 MAD",
    "dilation_peak_amplitude": "扩张峰幅度",
    "dilation_peak_latency_ms": "扩张峰潜伏期（ms）",
    "constriction_peak_amplitude": "收缩峰幅度",
    "constriction_peak_latency_ms": "收缩峰潜伏期（ms）",
    "dominant_peak_amplitude_abs": "主导峰绝对幅度",
    "dominant_peak_latency_ms": "主导峰潜伏期（ms）",
    "recovery_tolerance": "恢复判据容差",
    "recovery_time_after_onset_ms": "刺激后恢复时间（ms）",
    "late_recovery_residual": "晚期恢复残差",
    "late_recovery_abs_residual": "晚期恢复绝对残差",
}


def _resolve(config, key: str) -> Path:
    raw = config.section("paths").get(key)
    if raw in (None, ""):
        raise KeyError(f"formal pupil config missing paths.{key}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (config.path.parent.parent / path).resolve()
    return path


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(value)).strip("_")


def _save(fig: plt.Figure, path: Path) -> str:
    # User/report contract: the publication title lives in the external caption.
    for axis in fig.axes:
        axis.set_title("")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _read_concat(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p, encoding="utf-8-sig", low_memory=False) for p in paths if p.is_file()]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _session_tables(config, sessions: list[str], kind: str) -> pd.DataFrame:
    suffixes = {
        "trial": "trial_pupil_windows.csv",
        "probe": "probe_pupil_windows.csv",
        "time": "time_on_task_1s.csv",
    }
    suffix = suffixes[kind]
    root = _resolve(config, "output_root") / "sessions"
    return _read_concat(root / s / f"{s}_{suffix}" for s in sessions)


def _participant_first(frame: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    if "analysis_group_token" not in frame.columns or value_col not in frame.columns:
        return pd.DataFrame()
    d = frame[["analysis_group_token", *group_cols, value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=["analysis_group_token", value_col])
    if d.empty:
        return d
    return d.groupby([*group_cols, "analysis_group_token"], as_index=False, dropna=False)[value_col].median()


def _counts(frame: pd.DataFrame) -> tuple[int, int]:
    p = int(frame["analysis_group_token"].dropna().astype(str).nunique()) if "analysis_group_token" in frame else 0
    s = int(frame["session_id"].dropna().astype(str).nunique()) if "session_id" in frame else 0
    return p, s


def _row(
    *, metric: str, family: str, scale: str, status: str, reason: str,
    frame: pd.DataFrame | None = None, filename: str = "", caption: str = "",
    report_layer: str = "support",
) -> dict[str, Any]:
    p, s = _counts(frame) if frame is not None else (0, 0)
    return {
        "metric": metric,
        "metric_label_zh": PUPIL_LABELS_ZH.get(metric, FEATURE_LABELS_ZH.get(metric, metric)),
        "figure_family": family,
        "analysis_scale": scale,
        "status": status,
        "reason": reason,
        "filename": filename,
        "caption_zh": caption,
        "internal_title_allowed": False,
        "caption_is_external": True,
        "participant_group_n": p,
        "session_n": s,
        "report_layer": report_layer,
    }


def _histogram(
    frame: pd.DataFrame, value_col: str, xlabel: str, path: Path, caption: str,
) -> tuple[str, str]:
    values = pd.to_numeric(frame[value_col], errors="coerce")
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("fewer_than_two_finite_values")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bins = min(24, max(5, int(np.sqrt(len(values)))))
    ax.hist(values.to_numpy(dtype=float), bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("观察数")
    return _save(fig, path), caption


def generate_nir_figure_pack(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    sessions = selected_sessions(config, subjects)
    root = _resolve(config, "output_root")
    output_dir = root / "formal_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = root / "candidate_validation" / "nir_candidate_session_block_metrics.csv"
    candidate = pd.read_csv(candidate_path, encoding="utf-8-sig", low_memory=False) if candidate_path.is_file() else pd.DataFrame()
    trial = _session_tables(config, sessions, "trial")
    probe = _session_tables(config, sessions, "probe")
    time = _session_tables(config, sessions, "time")
    event_path = root / "event_response_candidates" / "trial_event_response_candidates.csv"
    event = pd.read_csv(event_path, encoding="utf-8-sig", low_memory=False) if event_path.is_file() else pd.DataFrame()
    effects_path = root / "reference_adjusted_models" / "trial_unadjusted_adjusted_effects.csv"
    effects = pd.read_csv(effects_path, encoding="utf-8-sig", low_memory=False) if effects_path.is_file() else pd.DataFrame()

    files: list[str] = []
    audit: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    # 1) Every pupil geometry/segmentation candidate: distribution, B1/B2 pairing,
    # and left-right agreement. These are admission/support figures, not winner claims.
    for metric, spec in PUPIL_CANDIDATE_METRICS.items():
        label = PUPIL_LABELS_ZH.get(metric, metric)
        current = candidate[candidate.get("metric", pd.Series(dtype=str)).astype(str).eq(metric)].copy() if not candidate.empty else pd.DataFrame()
        families = ("candidate_distribution", "candidate_B1B2_pair", "candidate_left_right")
        for family in families:
            if current.empty:
                audit.append(_row(metric=metric, family=family, scale="session_block", status="not_estimable", reason="candidate_session_block_table_missing_or_empty"))
                continue
            try:
                if family == "candidate_distribution":
                    filename = f"NIR候选_{_safe(metric)}_场次区块分布.png"
                    path, caption = _histogram(current, "binocular_raw_median", f"{label}（{spec['unit']}）", output_dir / filename, f"{label}在场次×区块层面的原始中位数分布。")
                elif family == "candidate_B1B2_pair":
                    wide = current.pivot_table(index="session_id", columns="block_num", values="binocular_raw_median", aggfunc="first")
                    if not {1, 2}.issubset(wide.columns):
                        raise ValueError("B1_B2_not_both_available")
                    wide = wide.dropna(subset=[1, 2])
                    if len(wide) < 2:
                        raise ValueError("fewer_than_two_complete_session_pairs")
                    fig, ax = plt.subplots(figsize=(6.3, 4.2))
                    for r in wide.itertuples(index=False):
                        ax.plot([1, 2], [float(r[0]), float(r[1])], marker="o", alpha=0.25, linewidth=0.8)
                    ax.plot([1, 2], [float(wide[1].mean()), float(wide[2].mean())], marker="o", linewidth=2.0, label="场次配对均值")
                    ax.set_xticks([1, 2]); ax.set_xticklabels(["B1", "B2"])
                    ax.set_xlabel("区块"); ax.set_ylabel(f"{label}（{spec['unit']}）"); ax.legend(title="描述性汇总")
                    filename = f"NIR候选_{_safe(metric)}_B1B2配对.png"
                    path = _save(fig, output_dir / filename)
                    caption = f"{label}在同一场次 B1 与 B2 间的配对轨迹。"
                else:
                    d = current[["left_raw_median", "right_raw_median"]].apply(pd.to_numeric, errors="coerce").dropna()
                    if len(d) < 3:
                        raise ValueError("fewer_than_three_binocular_pairs")
                    fig, ax = plt.subplots(figsize=(5.2, 5.0))
                    ax.scatter(d["left_raw_median"], d["right_raw_median"], alpha=0.6)
                    low = float(np.nanmin(d.to_numpy(dtype=float))); high = float(np.nanmax(d.to_numpy(dtype=float)))
                    ax.plot([low, high], [low, high], linestyle="--", linewidth=1)
                    ax.set_xlabel(f"左眼 {label}（{spec['unit']}）"); ax.set_ylabel(f"右眼 {label}（{spec['unit']}）")
                    filename = f"NIR候选_{_safe(metric)}_左右眼一致性.png"
                    path = _save(fig, output_dir / filename)
                    caption = f"{label}的左右眼场次×区块中位数一致性；虚线为 y=x。"
                files.append(path)
                row = _row(metric=metric, family=family, scale="session_block", status="generated", reason="generated", frame=current, filename=filename, caption=caption)
                audit.append(row); manifest.append(row.copy())
            except Exception as exc:
                audit.append(_row(metric=metric, family=family, scale="session_block", status="not_estimable", reason=f"{type(exc).__name__}:{exc}", frame=current))

    primary_track = str(config.section("tracks").get("primary", "binocular_primary"))
    feature_sets = [
        (trial, "trial_window", "window_name", (*WINDOW_FEATURES, *WINDOW_QC_FEATURES)),
        (probe, "probe_window", "window_name", (*WINDOW_FEATURES, *WINDOW_QC_FEATURES)),
    ]
    for frame, scale, category_col, features in feature_sets:
        primary = frame[frame.get("track", pd.Series(dtype=str)).astype(str).eq(primary_track)].copy() if not frame.empty else pd.DataFrame()
        for feature in features:
            family = f"{scale}_across_windows"
            label = FEATURE_LABELS_ZH.get(feature, feature)
            if primary.empty or feature not in primary.columns or category_col not in primary.columns:
                audit.append(_row(metric=feature, family=family, scale=scale, status="not_estimable", reason="primary_track_or_feature_missing", frame=primary))
                continue
            try:
                p = _participant_first(primary, [category_col], feature)
                if p.empty:
                    raise ValueError("participant_first_table_empty")
                names = list(dict.fromkeys(p[category_col].astype(str).tolist()))
                data = [pd.to_numeric(p.loc[p[category_col].astype(str).eq(name), feature], errors="coerce").dropna().to_numpy(dtype=float) for name in names]
                if sum(len(x) for x in data) < 2:
                    raise ValueError("insufficient_finite_participant_values")
                fig, ax = plt.subplots(figsize=(max(7.0, 0.75 * len(names)), 4.5))
                ax.boxplot(data, tick_labels=names, showfliers=False)
                ax.set_xlabel("窗口"); ax.set_ylabel(label)
                ax.tick_params(axis="x", rotation=35)
                filename = f"NIR窗口_{_safe(feature)}_{scale}.png"
                caption = f"{label}在不同{ '试次' if scale=='trial_window' else '探针前' }窗口中的参与者优先分布（主参考轨道 {primary_track}）。"
                path = _save(fig, output_dir / filename); files.append(path)
                row = _row(metric=feature, family=family, scale=scale, status="generated", reason="generated", frame=primary, filename=filename, caption=caption)
                audit.append(row); manifest.append(row.copy())
            except Exception as exc:
                audit.append(_row(metric=feature, family=family, scale=scale, status="not_estimable", reason=f"{type(exc).__name__}:{exc}", frame=primary))

    # 3) All window features get a time-on-task view on the primary track.
    primary_time = time[time.get("track", pd.Series(dtype=str)).astype(str).eq(primary_track)].copy() if not time.empty else pd.DataFrame()
    for feature in (*WINDOW_FEATURES, *WINDOW_QC_FEATURES):
        family = "time_on_task"
        label = FEATURE_LABELS_ZH.get(feature, feature)
        if primary_time.empty or feature not in primary_time.columns or "time_in_block_mid_sec" not in primary_time.columns:
            audit.append(_row(metric=feature, family=family, scale="time_on_task", status="not_estimable", reason="time_on_task_feature_missing", frame=primary_time))
            continue
        try:
            p = _participant_first(primary_time, ["block_num", "time_in_block_mid_sec"], feature)
            if p.empty:
                raise ValueError("participant_first_time_table_empty")
            summary = p.groupby(["block_num", "time_in_block_mid_sec"], as_index=False)[feature].mean()
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            for block, d in summary.groupby("block_num", sort=True):
                d = d.sort_values("time_in_block_mid_sec")
                ax.plot(d["time_in_block_mid_sec"], d[feature], label=f"B{int(block)}")
            ax.set_xlabel("区块内时间（s）"); ax.set_ylabel(label); ax.legend(title="区块")
            filename = f"NIR时间进程_{_safe(feature)}.png"
            caption = f"{label}随区块内任务时间的参与者优先描述性变化（主参考轨道 {primary_track}）。"
            path = _save(fig, output_dir / filename); files.append(path)
            row = _row(metric=feature, family=family, scale="time_on_task", status="generated", reason="generated", frame=primary_time, filename=filename, caption=caption)
            audit.append(row); manifest.append(row.copy())
        except Exception as exc:
            audit.append(_row(metric=feature, family=family, scale="time_on_task", status="not_estimable", reason=f"{type(exc).__name__}:{exc}", frame=primary_time))

    # 4) Every configured track receives sensitivity coverage for every window feature.
    reference_window = "pre_30s"
    reference_trial = trial[trial.get("window_name", pd.Series(dtype=str)).astype(str).eq(reference_window)].copy() if not trial.empty else pd.DataFrame()
    for feature in (*WINDOW_FEATURES, *WINDOW_QC_FEATURES):
        family = "track_sensitivity_pre30s"
        if reference_trial.empty or feature not in reference_trial.columns or "track" not in reference_trial.columns:
            audit.append(_row(metric=feature, family=family, scale="trial_window", status="not_estimable", reason="pre_30s_or_feature_missing", frame=reference_trial))
            continue
        try:
            p = _participant_first(reference_trial, ["track"], feature)
            tracks = list(dict.fromkeys(p["track"].astype(str).tolist()))
            data = [pd.to_numeric(p.loc[p["track"].astype(str).eq(name), feature], errors="coerce").dropna().to_numpy(dtype=float) for name in tracks]
            if sum(len(x) for x in data) < 2:
                raise ValueError("insufficient_track_values")
            fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(tracks)), 4.5))
            ax.boxplot(data, tick_labels=tracks, showfliers=False)
            ax.set_xlabel("瞳孔轨道"); ax.set_ylabel(FEATURE_LABELS_ZH.get(feature, feature)); ax.tick_params(axis="x", rotation=30)
            filename = f"NIR轨道敏感性_{_safe(feature)}_pre30s.png"
            caption = f"{FEATURE_LABELS_ZH.get(feature, feature)}在 pre-30s 状态窗中的左右眼、双眼与 strict 轨道敏感性比较。"
            path = _save(fig, output_dir / filename); files.append(path)
            row = _row(metric=feature, family=family, scale="trial_window", status="generated", reason="generated", frame=reference_trial, filename=filename, caption=caption)
            audit.append(row); manifest.append(row.copy())
        except Exception as exc:
            audit.append(_row(metric=feature, family=family, scale="trial_window", status="not_estimable", reason=f"{type(exc).__name__}:{exc}", frame=reference_trial))

    # 5) Every event-response feature receives its own distribution audit figure.
    for feature in EVENT_FEATURES:
        family = "event_response_distribution"
        if event.empty or feature not in event.columns:
            audit.append(_row(metric=feature, family=family, scale="trial_event", status="not_estimable", reason="event_response_feature_missing", frame=event))
            continue
        try:
            filename = f"NIR事件响应_{_safe(feature)}_分布.png"
            path, caption = _histogram(event, feature, FEATURE_LABELS_ZH.get(feature, feature), output_dir / filename, f"{FEATURE_LABELS_ZH.get(feature, feature)}的可估试次分布。")
            files.append(path)
            row = _row(metric=feature, family=family, scale="trial_event", status="generated", reason="generated", frame=event, filename=filename, caption=caption)
            audit.append(row); manifest.append(row.copy())
        except Exception as exc:
            audit.append(_row(metric=feature, family=family, scale="trial_event", status="not_estimable", reason=f"{type(exc).__name__}:{exc}", frame=event))

    # 6) Adjustment results are shown by outcome; no p-value-based figure selection.
    expected_outcomes = ("go_correct_rt", "go_omission", "nogo_commission")
    for outcome in expected_outcomes:
        family = "unadjusted_vs_adjusted_effect"
        d = effects[effects.get("outcome", pd.Series(dtype=str)).astype(str).eq(outcome)].copy() if not effects.empty else pd.DataFrame()
        if d.empty:
            audit.append(_row(metric=outcome, family=family, scale="trial_model", status="not_estimable", reason="effect_table_outcome_missing", frame=d, report_layer="core_after_endpoint_freeze"))
            continue
        needed = {"adjustment", "estimate", "ci_low", "ci_high"}
        if not needed.issubset(d.columns):
            audit.append(_row(metric=outcome, family=family, scale="trial_model", status="not_estimable", reason="effect_columns_missing", frame=d, report_layer="core_after_endpoint_freeze"))
            continue
        try:
            d = d.dropna(subset=["estimate", "ci_low", "ci_high"]).reset_index(drop=True)
            if d.empty:
                raise ValueError("no_finite_effect_rows")
            y = np.arange(len(d)); est = pd.to_numeric(d["estimate"], errors="coerce").to_numpy(dtype=float)
            lo = pd.to_numeric(d["ci_low"], errors="coerce").to_numpy(dtype=float); hi = pd.to_numeric(d["ci_high"], errors="coerce").to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(7.2, max(3.8, 0.55 * len(d) + 1.5)))
            ax.errorbar(est, y, xerr=np.vstack([est-lo, hi-est]), fmt="o", capsize=3); ax.axvline(0, linestyle="--", linewidth=1)
            ax.set_yticks(y); ax.set_yticklabels(d["adjustment"].astype(str).tolist())
            ax.set_xlabel("效应估计及 95% 置信区间"); ax.set_ylabel("调整方案")
            filename = f"NIR模型_{_safe(outcome)}_调整前后.png"
            caption = f"{outcome} 的瞳孔效应在未调整与协变量调整模型中的比较；是否达到视觉调整标准以 adjustment audit 为准。"
            path = _save(fig, output_dir / filename); files.append(path)
            row = _row(metric=outcome, family=family, scale="trial_model", status="generated", reason="generated", frame=d, filename=filename, caption=caption, report_layer="core_after_endpoint_freeze")
            audit.append(row); manifest.append(row.copy())
        except Exception as exc:
            audit.append(_row(metric=outcome, family=family, scale="trial_model", status="not_estimable", reason=f"{type(exc).__name__}:{exc}", frame=d, report_layer="core_after_endpoint_freeze"))

    manifest_df = pd.DataFrame(manifest)
    audit_df = pd.DataFrame(audit)
    manifest_df.to_csv(root / "nir_figure_manifest.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(root / "nir_figure_coverage_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "pipeline_version": FIGURE_PIPELINE_VERSION,
        "status": "complete",
        "n_generated_figures": int(len(files)),
        "n_coverage_rows": int(len(audit_df)),
        "n_not_estimable": int(audit_df["status"].ne("generated").sum()) if not audit_df.empty else 0,
        "internal_title_allowed": False,
        "caption_is_external": True,
        "coverage_contract": "every prespecified pupil candidate/window feature/QC feature/event feature/model outcome is generated or receives an explicit not-estimable reason",
        "figure_files": files,
    }
    (root / "nir_figure_manifest.json").write_text(
        pd.Series(summary).to_json(force_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
