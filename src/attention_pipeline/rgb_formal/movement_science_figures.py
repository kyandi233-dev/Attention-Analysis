from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import font_manager

from attention_pipeline.formal_analysis.publication_style import finalize_publication_figure


PRIMARY = "body_motion_energy_median"
EXPOSURE = "exposure_change_abs_median"
AUXILIARY = {
    "pose_lateral_right_per_sec_median",
    "pose_vertical_up_per_sec_median",
    "pose_radial_proximity_direction_score_median",
}

MODEL_TABLES = {
    "movement_task_progression.csv": {
        "figure_id": "movement_task_progression_coefficients",
        "figure_name": "movement_task_progression_coefficients.png",
        "purpose": "main_science",
        "short_caption": "身体动作随区块与任务时间推进的参与者聚类 GEE（广义估计方程）系数及 95% CI（置信区间）。",
    },
    "movement_q1_models.csv": {
        "figure_id": "movement_q1_relationships",
        "figure_name": "movement_q1_relationships.png",
        "purpose": "main_science",
        "short_caption": "身体动作与 Q1 四分类的参与者聚类多项逻辑回归系数及 95% CI（置信区间）。",
    },
    "movement_q2_models.csv": {
        "figure_id": "movement_q2_relationships",
        "figure_name": "movement_q2_relationships.png",
        "purpose": "main_science",
        "short_caption": "身体动作与 Q2 有序等级的参与者聚类有序 GEE（广义估计方程）系数及 95% CI（置信区间）。",
    },
    "movement_behavior_links.csv": {
        "figure_id": "movement_behavior_relationships",
        "figure_name": "movement_behavior_relationships.png",
        "purpose": "sensitivity",
        "short_caption": "身体动作与近期行为指标的参与者聚类 GEE（广义估计方程）关系系数及 95% CI（置信区间）。",
    },
}

FIGURE_MANIFEST_COLUMNS = [
    "figure_id",
    "modality",
    "status",
    "purpose",
    "source_tables",
    "output_path",
    "short_caption",
    "participant_group_n",
    "session_n",
    "probe_or_model_row_n",
    "analysis_set_note",
    "audit_basis",
]

FIGURE_AUDIT_COLUMNS = [
    "figure_id",
    "generation_status",
    "reason",
    "source_tables",
    "output_path_png",
    "output_path_svg",
    "internal_title_present",
]


def _detect_cjk_font() -> str:
    installed = {item.name for item in font_manager.fontManager.ttflist}
    for name in ("SimSun", "Microsoft YaHei", "SimHei", "Noto Serif CJK SC"):
        if name in installed:
            return name
    return "DejaVu Sans"


def _configure_style() -> None:
    import matplotlib.pyplot as plt

    cjk = _detect_cjk_font()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", cjk, "DejaVu Serif"],
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _first_existing(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _feature_mask(frame: pd.DataFrame, wanted: set[str]) -> pd.Series:
    keep = pd.Series(False, index=frame.index)
    for column in frame.columns:
        keep = keep | frame[column].astype("string").str.strip().isin(wanted)
    return keep


def _row_label(frame: pd.DataFrame) -> pd.Series:
    label_cols = [
        name
        for name in (
            "predictor",
            "metric",
            "term",
            "contrast_category",
            "reference_category",
            "outcome",
            "behavior_metric",
            "level",
        )
        if name in frame.columns
    ]
    if not label_cols:
        return pd.Series([f"row_{i + 1}" for i in range(len(frame))], index=frame.index)
    labels = []
    for _, row in frame[label_cols].iterrows():
        parts: list[str] = []
        for name in label_cols:
            value = row[name]
            if pd.isna(value):
                continue
            text = str(value).strip()
            if name == "contrast_category":
                text = f"Q1={text}"
            elif name == "reference_category":
                text = f"ref={text}"
            if text and text not in parts:
                parts.append(text)
        labels.append(" | ".join(parts) if parts else "model term")
    return pd.Series(labels, index=frame.index)


def _save_figure(fig: Any, path: Path) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    finalize_publication_figure(fig, remove_titles=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    png = path.with_suffix(".png")
    svg = path.with_suffix(".svg")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    participant_n = (
        int(frame["participant_group_id"].dropna().astype(str).nunique())
        if "participant_group_id" in frame.columns
        else 0
    )
    session_n = (
        int(frame["session_id"].dropna().astype(str).nunique())
        if "session_id" in frame.columns
        else 0
    )
    return {
        "participant_group_n": participant_n,
        "session_n": session_n,
        "probe_or_model_row_n": int(len(frame)),
    }


def _model_counts(frame: pd.DataFrame, fallback: pd.DataFrame) -> dict[str, int]:
    fallback_counts = _counts(fallback)
    result = dict(fallback_counts)
    for source, target in (
        ("participant_group_n", "participant_group_n"),
        ("session_n", "session_n"),
        ("n_rows", "probe_or_model_row_n"),
    ):
        if source not in frame.columns:
            continue
        values = pd.to_numeric(frame[source], errors="coerce")
        values = values[np.isfinite(values)]
        if len(values):
            result[target] = int(values.max())
    return result


def _primary_distribution_plot(probe: pd.DataFrame, path: Path) -> tuple[bool, str]:
    import matplotlib.pyplot as plt

    if PRIMARY not in probe.columns:
        return False, f"{PRIMARY} missing"
    values = pd.to_numeric(probe[PRIMARY], errors="coerce")
    values = values[np.isfinite(values)].sort_values()
    if len(values) < 3:
        return False, "fewer than 3 finite primary Movement values"
    _configure_style()
    y = np.arange(1, len(values) + 1, dtype=float) / float(len(values))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.step(values.to_numpy(float), y, where="post")
    ax.set_xlabel("身体动作能量中位数（无量纲）")
    ax.set_ylabel("经验累积分布")
    _save_figure(fig, path)
    return True, ""


def _participant_cycle_summary(cycle: pd.DataFrame) -> pd.DataFrame:
    required = {"participant_group_id", "block_id", "cycle_bin", PRIMARY}
    if cycle.empty or not required.issubset(cycle.columns):
        return pd.DataFrame()
    d = cycle[["participant_group_id", "block_id", "cycle_bin", PRIMARY]].copy()
    d[PRIMARY] = pd.to_numeric(d[PRIMARY], errors="coerce")
    d["cycle_bin"] = pd.to_numeric(d["cycle_bin"], errors="coerce")
    d = d.dropna(subset=["participant_group_id", "block_id", "cycle_bin", PRIMARY])
    if d.empty:
        return pd.DataFrame()
    per_participant = (
        d.groupby(["participant_group_id", "block_id", "cycle_bin"], sort=True)[PRIMARY]
        .mean()
        .reset_index()
    )
    return (
        per_participant.groupby(["block_id", "cycle_bin"], sort=True)[PRIMARY]
        .agg(
            participant_n="count",
            mean="mean",
            sem=lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) >= 2 else np.nan,
        )
        .reset_index()
    )


def _movement_pair_task_progression_plot(
    probe: pd.DataFrame,
    cycle: pd.DataFrame,
    path: Path,
) -> tuple[bool, str]:
    import matplotlib.pyplot as plt

    required = {"participant_group_id", "block_id", PRIMARY}
    if probe.empty or not required.issubset(probe.columns):
        return False, "probe table lacks participant/block/body-motion fields"
    d = probe[["participant_group_id", "block_id", PRIMARY]].copy()
    d[PRIMARY] = pd.to_numeric(d[PRIMARY], errors="coerce")
    d["block_id"] = d["block_id"].astype(str).str.upper()
    d = d.dropna(subset=["participant_group_id", PRIMARY])
    block_means = d.groupby(["participant_group_id", "block_id"], sort=True)[PRIMARY].mean().unstack()
    if not {"B1", "B2"}.issubset(block_means.columns):
        return False, "B1/B2 paired body-motion values are not both available"
    paired = block_means[["B1", "B2"]].dropna()
    cells = _participant_cycle_summary(cycle)
    if len(paired) < 2 or cells.empty or cells["cycle_bin"].nunique() < 2:
        return False, "insufficient participant pairs or task-time cells"

    _configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    ax_pair, ax_time = axes
    for row in paired.itertuples():
        ax_pair.plot([1, 2], [row.B1, row.B2], marker="o", linewidth=0.7, alpha=0.35)
    ax_pair.plot(
        [1, 2],
        [float(paired["B1"].mean()), float(paired["B2"].mean())],
        marker="o",
        linewidth=2.0,
    )
    ax_pair.set_xticks([1, 2], ["B1", "B2"])
    ax_pair.set_xlabel("区块")
    ax_pair.set_ylabel("身体动作能量中位数（无量纲）")

    for block, linestyle in (("B1", "-"), ("B2", "--")):
        cur = cells[cells["block_id"].astype(str).str.upper().eq(block)].sort_values("cycle_bin")
        if cur.empty:
            continue
        ax_time.errorbar(
            cur["cycle_bin"],
            cur["mean"],
            yerr=cur["sem"],
            marker="o",
            linestyle=linestyle,
            capsize=2.5,
            label=block,
        )
    ax_time.set_xlabel("区块内任务时间段（cycle bin）")
    ax_time.set_ylabel("参与者均值 ± SEM")
    ax_time.legend(frameon=False)
    _save_figure(fig, path)
    return True, ""


def _exposure_qc_plot(probe: pd.DataFrame, path: Path) -> tuple[bool, str]:
    import matplotlib.pyplot as plt

    if PRIMARY not in probe.columns or EXPOSURE not in probe.columns:
        return False, "body-motion or exposure field missing"
    x = pd.to_numeric(probe[EXPOSURE], errors="coerce")
    y = pd.to_numeric(probe[PRIMARY], errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return False, "fewer than 3 finite movement-exposure pairs"
    _configure_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    if int(mask.sum()) >= 100:
        ax.hexbin(x[mask], y[mask], gridsize=32, mincnt=1)
    else:
        ax.scatter(x[mask], y[mask], s=12, alpha=0.4)
    ax.set_xlabel("曝光变化绝对值中位数（QC）")
    ax.set_ylabel("身体动作能量中位数（无量纲）")
    _save_figure(fig, path)
    return True, ""


def _pose_distribution_plot(probe: pd.DataFrame, path: Path) -> tuple[bool, str]:
    import matplotlib.pyplot as plt

    specs = (
        ("pose_lateral_right_per_sec_median", "横向方向（正=右，/s）"),
        ("pose_vertical_up_per_sec_median", "纵向方向（正=上，/s）"),
        ("pose_radial_proximity_direction_score_median", "径向方向代理（无量纲）"),
    )
    available = []
    for column, label in specs:
        if column not in probe.columns:
            available.append((column, label, pd.Series(dtype=float)))
            continue
        values = pd.to_numeric(probe[column], errors="coerce")
        available.append((column, label, values[np.isfinite(values)]))
    if not any(len(values) >= 3 for _, _, values in available):
        return False, "pose direction values are not estimable"
    _configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for ax, (_, label, values) in zip(axes, available):
        if len(values) >= 3:
            ax.hist(values.to_numpy(float), bins=min(20, max(6, int(np.sqrt(len(values))))))
            ax.axvline(0.0, linewidth=0.8)
            ax.set_xlabel(label)
            ax.set_ylabel("探针窗口数")
        else:
            ax.text(0.5, 0.5, "数据不足", transform=ax.transAxes, ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
    _save_figure(fig, path)
    return True, ""


def _coverage_plot(session: pd.DataFrame, path: Path) -> tuple[bool, str]:
    import matplotlib.pyplot as plt

    if session.empty or "session_id" not in session.columns:
        return False, "session coverage table is empty or lacks session_id"
    specs = (
        ("body_motion_observable_ratio", "身体动作可观测帧比例"),
        ("exposure_change_observable_ratio", "曝光变化可观测帧比例"),
        ("pose_shoulders_observable_ratio", "肩部姿态可观测帧比例"),
    )
    columns = [(column, label) for column, label in specs if column in session.columns]
    if not columns:
        return False, "no Movement/RGB coverage ratio columns"
    order = session.sort_values("session_id").reset_index(drop=True)
    _configure_style()
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    x = np.arange(1, len(order) + 1)
    plotted = False
    for column, label in columns:
        values = pd.to_numeric(order[column], errors="coerce").to_numpy(float)
        finite = np.isfinite(values)
        if int(finite.sum()) < 2:
            continue
        ax.plot(x[finite], values[finite], marker="o", markersize=2.2, linewidth=0.8, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False, "no coverage series has at least two finite values"
    ax.set_xlabel("场次（按编号排序）")
    ax.set_ylabel("可观测帧比例")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)
    _save_figure(fig, path)
    return True, ""


def _coefficient_plot(frame: pd.DataFrame, path: Path) -> tuple[bool, str]:
    import matplotlib.pyplot as plt

    estimate_col = _first_existing(
        frame,
        ("estimate_per_predictor_sd", "estimate", "coef", "coefficient", "beta", "B"),
    )
    if estimate_col is None:
        return False, "no recognized estimate column"
    estimates = pd.to_numeric(frame[estimate_col], errors="coerce")
    mask = np.isfinite(estimates)
    data = frame.loc[mask].copy()
    estimates = estimates.loc[mask]
    if data.empty:
        return False, "no finite coefficient rows"
    low_col = _first_existing(data, ("ci_low", "conf_low", "lower_ci", "ci_95_low", "lower"))
    high_col = _first_existing(data, ("ci_high", "conf_high", "upper_ci", "ci_95_high", "upper"))
    labels = _row_label(data)
    if len(data) > 28:
        data = data.iloc[:28].copy()
        estimates = estimates.iloc[:28]
        labels = labels.iloc[:28]
    _configure_style()
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9.2, max(4.8, 0.38 * len(data) + 1.8)))
    if low_col and high_col:
        low = pd.to_numeric(data[low_col], errors="coerce")
        high = pd.to_numeric(data[high_col], errors="coerce")
        valid_ci = np.isfinite(low) & np.isfinite(high)
        est_arr = estimates.to_numpy(float)
        low_arr = low.to_numpy(float)
        high_arr = high.to_numpy(float)
        xerr = np.vstack(
            [
                np.where(valid_ci, np.maximum(est_arr - low_arr, 0.0), 0.0),
                np.where(valid_ci, np.maximum(high_arr - est_arr, 0.0), 0.0),
            ]
        )
        ax.errorbar(estimates, y, xerr=xerr, fmt="o", capsize=3)
    else:
        ax.scatter(estimates, y)
    ax.axvline(0.0, linewidth=1)
    ax.set_yticks(y, labels=labels.tolist())
    ax.set_xlabel("标准化模型系数（每预测变量 1 SD）")
    _save_figure(fig, path)
    return True, ""


def _append_generated(
    manifest_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    *,
    figure_id: str,
    purpose: str,
    source_tables: list[str],
    output_png: str,
    output_svg: str,
    short_caption: str,
    counts: dict[str, int],
) -> None:
    manifest_rows.append(
        {
            "figure_id": figure_id,
            "modality": "movement",
            "status": "candidate",
            "purpose": purpose,
            "source_tables": "|".join(source_tables),
            "output_path": output_png,
            "short_caption": short_caption,
            **counts,
            "analysis_set_note": "existing authoritative RGB 5.5 producer analysis set; P4 performs no refit",
            "audit_basis": "existing RGB 5.5 producer outputs; no Q1/Q2-based feature selection",
        }
    )
    audit_rows.append(
        {
            "figure_id": figure_id,
            "generation_status": "generated",
            "reason": "",
            "source_tables": "|".join(source_tables),
            "output_path_png": output_png,
            "output_path_svg": output_svg,
            "internal_title_present": False,
        }
    )


def _append_not_estimable(
    audit_rows: list[dict[str, Any]],
    *,
    figure_id: str,
    source_tables: list[str],
    reason: str,
) -> None:
    audit_rows.append(
        {
            "figure_id": figure_id,
            "generation_status": "not_estimable",
            "reason": reason,
            "source_tables": "|".join(source_tables),
            "output_path_png": "",
            "output_path_svg": "",
            "internal_title_present": False,
        }
    )


def _record_plot(
    *,
    root: Path,
    generated: list[str],
    manifest_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    figure_id: str,
    purpose: str,
    source_tables: list[str],
    path: Path,
    short_caption: str,
    counts: dict[str, int],
    result: tuple[bool, str],
) -> None:
    ok, reason = result
    if not ok:
        _append_not_estimable(
            audit_rows,
            figure_id=figure_id,
            source_tables=source_tables,
            reason=reason,
        )
        return
    png = path.with_suffix(".png")
    svg = path.with_suffix(".svg")
    out_png = str(png.relative_to(root))
    out_svg = str(svg.relative_to(root))
    generated.append(out_png)
    _append_generated(
        manifest_rows,
        audit_rows,
        figure_id=figure_id,
        purpose=purpose,
        source_tables=source_tables,
        output_png=out_png,
        output_svg=out_svg,
        short_caption=short_caption,
        counts=counts,
    )


def build_movement_science_figures(movement_root: str | Path) -> list[str]:
    root = Path(movement_root).expanduser().resolve()
    tables = root / "tables"
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    probe_source = tables / "movement_probe_descriptive_source.csv"
    cycle_source = tables / "rgb_block_cycle_source.csv"
    session_source = tables / "rgb_session_coverage_source.csv"
    probe = pd.read_csv(probe_source, encoding="utf-8-sig", low_memory=False) if probe_source.exists() else pd.DataFrame()
    cycle = pd.read_csv(cycle_source, encoding="utf-8-sig", low_memory=False) if cycle_source.exists() else pd.DataFrame()
    session = pd.read_csv(session_source, encoding="utf-8-sig", low_memory=False) if session_source.exists() else pd.DataFrame()

    generated: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    probe_counts = _counts(probe)

    plot_specs = [
        (
            "movement_primary_distribution",
            "feature_qualification",
            [probe_source.name],
            root / "figures/qualification/movement_primary_distribution.png",
            "Movement 主候选身体动作能量在合法探针前 30 s 窗口中的经验累积分布。",
            probe_counts,
            lambda path: _primary_distribution_plot(probe, path),
        ),
        (
            "movement_block_pair_task_progression",
            "main_science",
            [probe_source.name, cycle_source.name],
            root / "figures/main/movement_block_pair_task_progression.png",
            "身体动作能量的参与者级 B1/B2 配对与区块内任务时间轨迹；曝光变化不进入该主科学图。",
            probe_counts,
            lambda path: _movement_pair_task_progression_plot(probe, cycle, path),
        ),
        (
            "movement_exposure_qc",
            "qc",
            [probe_source.name],
            root / "figures/qc/movement_exposure_qc.png",
            "身体动作能量与曝光变化的关系，仅用于检查亮度/曝光混杂，不作为注意特征。",
            probe_counts,
            lambda path: _exposure_qc_plot(probe, path),
        ),
        (
            "movement_pose_direction_distribution",
            "sensitivity",
            [probe_source.name],
            root / "figures/sensitivity/movement_pose_direction_distribution.png",
            "横向、纵向与无量纲径向姿态候选在探针前 30 s 窗口的分布；仅作敏感性展示。",
            probe_counts,
            lambda path: _pose_distribution_plot(probe, path),
        ),
        (
            "movement_source_coverage",
            "qc",
            [session_source.name],
            root / "figures/qc/movement_source_coverage.png",
            "Movement 相关 RGB（可见光视频）动作、曝光与姿态轨在各场次的可观测帧比例。",
            _counts(session),
            lambda path: _coverage_plot(session, path),
        ),
    ]
    for figure_id, purpose, source_tables, path, caption, counts, function in plot_specs:
        try:
            result = function(path)
        except Exception as exc:
            result = (False, f"{type(exc).__name__}: {exc}")
        _record_plot(
            root=root,
            generated=generated,
            manifest_rows=manifest_rows,
            audit_rows=audit_rows,
            figure_id=figure_id,
            purpose=purpose,
            source_tables=source_tables,
            path=path,
            short_caption=caption,
            counts=counts,
            result=result,
        )

    # Producer-side model estimates are consumed read-only. P4 visualizes them
    # without refitting and without selecting features from Q1/Q2 results.
    for table_name, spec in MODEL_TABLES.items():
        source_path = tables / table_name
        figure_id = str(spec["figure_id"])
        if not source_path.exists() or source_path.stat().st_size == 0:
            _append_not_estimable(
                audit_rows,
                figure_id=figure_id,
                source_tables=[table_name],
                reason="source table missing or empty",
            )
            continue
        try:
            frame = pd.read_csv(source_path, encoding="utf-8-sig", low_memory=False)
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
        if frame.empty:
            _append_not_estimable(
                audit_rows,
                figure_id=figure_id,
                source_tables=[table_name],
                reason="source table has no rows",
            )
            continue

        primary = frame.loc[_feature_mask(frame, {PRIMARY})].copy()
        purpose_dir = "main" if str(spec["purpose"]) == "main_science" else str(spec["purpose"])
        figure_path = root / "figures" / purpose_dir / str(spec["figure_name"])
        try:
            result = _coefficient_plot(primary, figure_path)
        except Exception as exc:
            result = (False, f"{type(exc).__name__}: {exc}")
        _record_plot(
            root=root,
            generated=generated,
            manifest_rows=manifest_rows,
            audit_rows=audit_rows,
            figure_id=figure_id,
            purpose=str(spec["purpose"]),
            source_tables=[table_name],
            path=figure_path,
            short_caption=str(spec["short_caption"]),
            counts=_model_counts(primary, probe),
            result=result,
        )

        auxiliary = frame.loc[_feature_mask(frame, AUXILIARY)].copy()
        aux_id = f"{figure_id}_pose_sensitivity"
        aux_path = root / "figures/sensitivity" / str(spec["figure_name"]).replace(".png", "_pose_sensitivity.png")
        try:
            aux_result = _coefficient_plot(auxiliary, aux_path)
        except Exception as exc:
            aux_result = (False, f"{type(exc).__name__}: {exc}")
        _record_plot(
            root=root,
            generated=generated,
            manifest_rows=manifest_rows,
            audit_rows=audit_rows,
            figure_id=aux_id,
            purpose="sensitivity",
            source_tables=[table_name],
            path=aux_path,
            short_caption="姿态/径向辅助候选对应的同源模型系数及 95% CI（置信区间）；仅作敏感性展示。",
            counts=_model_counts(auxiliary, probe),
            result=aux_result,
        )

    pd.DataFrame(manifest_rows, columns=FIGURE_MANIFEST_COLUMNS).to_csv(
        manifests / "figure_manifest.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(audit_rows, columns=FIGURE_AUDIT_COLUMNS).to_csv(
        manifests / "figure_audit.csv", index=False, encoding="utf-8-sig"
    )
    return generated
