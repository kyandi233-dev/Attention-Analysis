from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .science_v3 import CANONICAL_METRICS
from .behavior_error_taxonomy import TAXONOMY_RATE_METRICS
from attention_pipeline.formal_analysis.publication_style import (
    FONT_FAMILY,
    configure_publication_style,
    finalize_publication_figure,
)

configure_publication_style()

METRIC_LABELS_EN = {
    "go_correct_rt_mean_ms": "Correct Go RT mean (ms)",
    "go_correct_rt_median_ms": "Correct Go RT median (ms)",
    "go_correct_rt_sd_ms": "Correct Go RT SD (ms)",
    "go_correct_rt_mad_ms": "Correct Go RT MAD (ms)",
    "go_correct_rt_iqr_ms": "Correct Go RT IQR (ms)",
    "go_correct_rt_cv": "Correct Go RT coefficient of variation",
    "go_correct_rt_theilsen_slope_ms_per_s": "Correct Go RT robust slope (ms/s)",
    "omission_rate": "Raw Go omission rate (compatibility alias)",
    "raw_go_omission_rate": "Raw Go omission rate",
    "clean_go_omission_rate": "Go omission rate without detected timing ambiguity",
    "timing_ambiguous_go_omission_rate": "Timing-ambiguous Go omission rate",
    "commission_rate": "No-Go commission rate",
    "dprime_loglinear": "d′",
    "criterion_c": "Criterion c",
    "beta": "β",
    "omission_no_detected_motor_timing_ambiguity_rate": "Go omission without detected timing ambiguity (alias)",
    "omission_motor_timing_ambiguous_rate": "Timing-ambiguous Go omission (alias)",
    "omission_prestimulus_only_ambiguity_rate": "Prestimulus-only ambiguity rate",
    "omission_carryover_only_ambiguity_rate": "Carry-over-only ambiguity rate",
    "omission_prestimulus_and_carryover_ambiguity_rate": "Prestimulus plus carry-over ambiguity rate",
    "late_go_response_candidate_rate": "Late Go-response candidate rate",
    "anticipatory_go_response_candidate_rate": "Anticipatory Go-response candidate rate",
}

# ``omission_rate`` is the historical compatibility alias of
# ``raw_go_omission_rate``.  The systematic publication pack renders the new
# formal field once rather than creating two visually duplicate figure families.
_BASE_FIGURE_METRICS = tuple(m for m in CANONICAL_METRICS if m != "omission_rate")
ALL_FIGURE_METRICS = tuple(dict.fromkeys([*_BASE_FIGURE_METRICS, *TAXONOMY_RATE_METRICS]))
FIGURE_FAMILIES = (
    "session_distribution",
    "block_pair",
    "cycle_trend",
    "probe_q1_nominal",
    "probe_q2_ordinal",
)


def _metric_label(metric: str) -> str:
    return METRIC_LABELS_EN.get(metric, metric.replace("_", " "))


def _safe_name(metric: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", metric).strip("_")


def _participant_col(frame: pd.DataFrame) -> str | None:
    for column in ("participant_group_id", "repeat_participant_id", "participant_key"):
        if column in frame.columns:
            return column
    return None


def _counts(frame: pd.DataFrame) -> tuple[int, int]:
    participant = _participant_col(frame)
    p_n = int(frame[participant].dropna().astype(str).nunique()) if participant else 0
    s_n = int(frame["session_id"].dropna().astype(str).nunique()) if "session_id" in frame else 0
    return p_n, s_n


def _sem(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) >= 2 else math.nan


def _use_log(metric: str, values: np.ndarray) -> bool:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    return metric == "beta" and len(finite) >= 2 and float(finite.max() / finite.min()) >= 10.0


def _participant_first(frame: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    participant = _participant_col(frame)
    if participant is None or metric not in frame.columns:
        return pd.DataFrame()
    columns = [participant, *group_cols, metric]
    current = frame[columns].copy()
    current[metric] = pd.to_numeric(current[metric], errors="coerce")
    current = current.dropna(subset=[participant, metric])
    if current.empty:
        return pd.DataFrame()
    participant_level = (
        current.groupby([*group_cols, participant], as_index=False, dropna=False)[metric].mean()
    )
    summary = participant_level.groupby(group_cols, as_index=False, dropna=False)[metric].agg(mean="mean", participant_n="count")
    sem = participant_level.groupby(group_cols, dropna=False)[metric].apply(_sem).rename("sem").reset_index()
    return summary.merge(sem, on=group_cols, how="left", validate="one_to_one")


def _save(fig: plt.Figure, path: Path) -> str:
    # Publication contract: figure titles are external captions, never pixels.
    finalize_publication_figure(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _audit_row(
    *, metric: str, family: str, scale: str, status: str, reason: str,
    filename: str = "", caption_zh: str = "", frame: pd.DataFrame | None = None,
) -> dict[str, object]:
    p_n, s_n = _counts(frame) if frame is not None else (0, 0)
    return {
        "metric": metric,
        "metric_label_zh": _metric_label(metric),
        "metric_label_en": _metric_label(metric),
        "figure_family": family,
        "analysis_scale": scale,
        "status": status,
        "reason": reason,
        "filename": filename,
        "caption_zh": caption_zh,
        "internal_title_allowed": False,
        "caption_is_external": True,
        "in_image_language": "English",
        "font_family": FONT_FAMILY,
        "legend_frame": False,
        "participant_n": p_n,
        "session_n": s_n,
        "report_layer": "support_candidate_until_endpoint_freeze",
    }


def generate_complete_metric_figure_pack(
    *,
    session: pd.DataFrame | None,
    block: pd.DataFrame | None,
    cycle: pd.DataFrame | None,
    probe: pd.DataFrame | None,
    output_dir: Path,
    metrics: Iterable[str] = ALL_FIGURE_METRICS,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Generate systematic descriptive figure families and an explicit coverage ledger.

    Not every metric must be estimable at every scale, but every metric/family
    combination receives a machine-readable generated/not-estimable reason. This
    prevents silent figure omissions and keeps report selection separate from
    analysis completeness.
    """
    output_dir = Path(output_dir)
    files: list[str] = []
    audit: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    frames = {
        "session_distribution": (session, "session"),
        "block_pair": (block, "block"),
        "cycle_trend": (cycle, "cycle"),
        "probe_q1_nominal": (probe, "probe"),
        "probe_q2_ordinal": (probe, "probe"),
    }

    for metric in tuple(metrics):
        label = _metric_label(metric)
        safe = _safe_name(metric)
        for family in FIGURE_FAMILIES:
            frame, scale = frames[family]
            if frame is None or frame.empty:
                audit.append(_audit_row(metric=metric, family=family, scale=scale, status="not_estimable", reason="source_table_empty", frame=frame))
                continue
            if metric not in frame.columns:
                audit.append(_audit_row(metric=metric, family=family, scale=scale, status="not_estimable", reason="metric_column_absent_at_scale", frame=frame))
                continue
            values = pd.to_numeric(frame[metric], errors="coerce")
            if int(np.isfinite(values).sum()) < 2:
                audit.append(_audit_row(metric=metric, family=family, scale=scale, status="not_estimable", reason="fewer_than_two_finite_values", frame=frame))
                continue

            filename = ""
            caption = ""
            reason = "generated"
            try:
                if family == "session_distribution":
                    finite = values[np.isfinite(values)]
                    fig, ax = plt.subplots(figsize=(6.6, 4.3))
                    bins = min(20, max(5, int(np.sqrt(len(finite)))))
                    ax.hist(finite.to_numpy(dtype=float), bins=bins)
                    ax.set_xlabel(label)
                    ax.set_ylabel("Sessions")
                    if _use_log(metric, finite.to_numpy(dtype=float)):
                        ax.set_xscale("log")
                    caption = f"{label}的场次级原始分布。"
                    filename = f"行为指标_{safe}_场次分布.png"

                elif family == "block_pair":
                    required = {"session_id", "block_id", metric}
                    if not required.issubset(frame.columns):
                        raise ValueError("missing_session_or_block_key")
                    wide = frame.pivot_table(index="session_id", columns="block_id", values=metric, aggfunc="first")
                    if not {"B1", "B2"}.issubset(wide.columns):
                        raise ValueError("B1_B2_not_both_available")
                    wide = wide.dropna(subset=["B1", "B2"])
                    if len(wide) < 2:
                        raise ValueError("fewer_than_two_complete_session_pairs")
                    fig, ax = plt.subplots(figsize=(6.4, 4.3))
                    for row in wide.itertuples(index=False):
                        ax.plot([1, 2], [float(row.B1), float(row.B2)], marker="o", alpha=0.25, linewidth=0.8)
                    ax.plot([1, 2], [float(wide["B1"].mean()), float(wide["B2"].mean())], marker="o", linewidth=2.0, label="Session-pair mean")
                    ax.set_xticks([1, 2]); ax.set_xticklabels(["B1", "B2"])
                    ax.set_xlabel("Block")
                    ax.set_ylabel(label)
                    ax.legend(frameon=False)
                    if _use_log(metric, wide[["B1", "B2"]].to_numpy(dtype=float).ravel()):
                        ax.set_yscale("log")
                    caption = f"{label}在同一场次 B1 与 B2 间的配对轨迹；推断区间另以参与者为簇计算。"
                    filename = f"行为指标_{safe}_B1B2配对.png"

                elif family == "cycle_trend":
                    if "cycle_bin" not in frame.columns or "block_id" not in frame.columns:
                        raise ValueError("cycle_or_block_key_absent")
                    summary = _participant_first(frame, ["block_id", "cycle_bin"], metric)
                    if summary.empty:
                        raise ValueError("participant_first_summary_empty")
                    fig, ax = plt.subplots(figsize=(6.8, 4.4))
                    for block_id, current in summary.groupby("block_id", sort=True):
                        current = current.sort_values("cycle_bin")
                        ax.errorbar(current["cycle_bin"], current["mean"], yerr=current["sem"], marker="o", capsize=3, label=str(block_id))
                    ax.set_xlabel("Cycle within block")
                    ax.set_ylabel(label)
                    ax.legend(title="Block", frameon=False)
                    if _use_log(metric, summary["mean"].to_numpy(dtype=float)):
                        ax.set_yscale("log")
                    caption = f"{label}随区块内任务进程的参与者优先描述性变化。"
                    filename = f"行为指标_{safe}_任务时间进程.png"

                elif family in {"probe_q1_nominal", "probe_q2_ordinal"}:
                    category = "q1_nominal_4class" if family == "probe_q1_nominal" else "q2_ordinal_4level"
                    if category not in frame.columns:
                        raise ValueError("probe_category_absent")
                    summary = _participant_first(frame.dropna(subset=[category]), [category], metric)
                    if summary.empty:
                        raise ValueError("participant_first_probe_summary_empty")
                    summary[category] = pd.to_numeric(summary[category], errors="coerce")
                    summary = summary.dropna(subset=[category]).sort_values(category)
                    if len(summary) < 2:
                        raise ValueError("fewer_than_two_observed_probe_categories")
                    fig, ax = plt.subplots(figsize=(6.5, 4.3))
                    x = summary[category].to_numpy(dtype=float)
                    if family == "probe_q1_nominal":
                        ax.errorbar(x, summary["mean"], yerr=summary["sem"], fmt="o", linestyle="none", capsize=3)
                        ax.set_xlabel("Q1 category (nominal; 1-4)")
                        caption = f"{label}在 Q1 名义四分类下的探针前 30 秒参与者优先描述性分布；类别之间不连接为有序趋势。"
                        filename = f"行为指标_{safe}_Q1探针.png"
                    else:
                        ax.errorbar(x, summary["mean"], yerr=summary["sem"], marker="o", capsize=3)
                        ax.set_xlabel("Q2 vigilance level (ordinal; 1-4)")
                        caption = f"{label}随 Q2 有序警觉程度的探针前 30 秒参与者优先描述性变化。"
                        filename = f"行为指标_{safe}_Q2探针.png"
                    ax.set_ylabel(label)
                    ax.set_xticks([1, 2, 3, 4])
                    if _use_log(metric, summary["mean"].to_numpy(dtype=float)):
                        ax.set_yscale("log")
                else:
                    raise ValueError("unknown_figure_family")

                path = output_dir / filename
                files.append(_save(fig, path))
                row = _audit_row(metric=metric, family=family, scale=scale, status="generated", reason=reason, filename=filename, caption_zh=caption, frame=frame)
                audit.append(row)
                manifest.append(row.copy())
            except Exception as exc:
                if "fig" in locals():
                    try:
                        plt.close(fig)
                    except Exception:
                        pass
                audit.append(_audit_row(metric=metric, family=family, scale=scale, status="not_estimable", reason=f"{type(exc).__name__}:{exc}", frame=frame))

    audit_df = pd.DataFrame(audit)
    manifest_df = pd.DataFrame(manifest)
    return files, manifest_df, audit_df
