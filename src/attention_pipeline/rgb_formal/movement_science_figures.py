from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from attention_pipeline.formal_analysis.publication_style import finalize_publication_figure
from attention_pipeline.rgb_formal.figures_55 import (
    figure_coverage,
    figure_motion_exposure,
    figure_pose_direction,
)


PRIMARY = "body_motion_energy_median"
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
        "short_caption": "身体动作随区块与任务时间推进的参与者聚类 GEE 系数及 95% CI。",
    },
    "movement_q1_models.csv": {
        "figure_id": "movement_q1_relationships",
        "figure_name": "movement_q1_relationships.png",
        "purpose": "main_science",
        "short_caption": "身体动作与 Q1 四分类的参与者聚类多项逻辑回归系数及 95% CI。",
    },
    "movement_q2_models.csv": {
        "figure_id": "movement_q2_relationships",
        "figure_name": "movement_q2_relationships.png",
        "purpose": "main_science",
        "short_caption": "身体动作与 Q2 有序等级的参与者聚类有序 GEE 系数及 95% CI。",
    },
    "movement_behavior_links.csv": {
        "figure_id": "movement_behavior_relationships",
        "figure_name": "movement_behavior_relationships.png",
        "purpose": "sensitivity",
        "short_caption": "身体动作与近期行为指标的参与者聚类 GEE 关系系数及 95% CI。",
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
        parts = []
        for name in label_cols:
            value = row[name]
            if pd.notna(value):
                text = str(value).strip()
                if text and text not in parts:
                    if name == "contrast_category":
                        text = f"Q1={text}"
                    elif name == "reference_category":
                        text = f"ref={text}"
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


def _primary_distribution_plot(probe: pd.DataFrame, path: Path) -> tuple[bool, str]:
    import matplotlib.pyplot as plt

    if PRIMARY not in probe.columns:
        return False, f"{PRIMARY} missing"
    values = pd.to_numeric(probe[PRIMARY], errors="coerce")
    values = values[np.isfinite(values)].sort_values()
    if len(values) < 3:
        return False, "fewer than 3 finite primary Movement values"
    y = np.arange(1, len(values) + 1, dtype=float) / float(len(values))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.step(values.to_numpy(float), y, where="post")
    ax.set_xlabel("身体动作能量中位数（无量纲）")
    ax.set_ylabel("经验累积分布")
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


def _move_legacy_pair(
    *,
    root: Path,
    source_png: str,
    destination_stem: Path,
) -> tuple[str, str]:
    src_png = Path(source_png)
    src_svg = src_png.with_suffix(".svg")
    dst_png = destination_stem.with_suffix(".png")
    dst_svg = destination_stem.with_suffix(".svg")
    destination_stem.parent.mkdir(parents=True, exist_ok=True)
    src_png.replace(dst_png)
    if src_svg.exists():
        src_svg.replace(dst_svg)
    return str(dst_png.relative_to(root)), str(dst_svg.relative_to(root))


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
            "audit_basis": "existing RGB 5.5 producer outputs; no refit in P4 materializer",
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


def build_movement_science_figures(movement_root: str | Path) -> list[str]:
    root = Path(movement_root).expanduser().resolve()
    tables = root / "tables"
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    probe_source = tables / "movement_probe_descriptive_source.csv"
    cycle_source = tables / "rgb_block_cycle_source.csv"
    session_source = tables / "rgb_session_coverage_source.csv"

    probe = (
        pd.read_csv(probe_source, encoding="utf-8-sig", low_memory=False)
        if probe_source.exists()
        else pd.DataFrame()
    )
    cycle = (
        pd.read_csv(cycle_source, encoding="utf-8-sig", low_memory=False)
        if cycle_source.exists()
        else pd.DataFrame()
    )
    session = (
        pd.read_csv(session_source, encoding="utf-8-sig", low_memory=False)
        if session_source.exists()
        else pd.DataFrame()
    )

    primary_dist_id = "movement_primary_distribution"
    primary_dist_path = root / "figures/qualification/movement_primary_distribution.png"
    ok, reason = _primary_distribution_plot(probe, primary_dist_path)
    if ok:
        out_png = str(primary_dist_path.relative_to(root))
        out_svg = str(primary_dist_path.with_suffix(".svg").relative_to(root))
        generated.append(out_png)
        _append_generated(
            manifest_rows,
            audit_rows,
            figure_id=primary_dist_id,
            purpose="feature_qualification",
            source_tables=[probe_source.name],
            output_png=out_png,
            output_svg=out_svg,
            short_caption="Movement 主候选身体动作能量在合法探针前 30 s 窗口中的经验累积分布。",
        )
    else:
        _append_not_estimable(
            audit_rows,
            figure_id=primary_dist_id,
            source_tables=[probe_source.name],
            reason=reason,
        )

    # Participant-first B1/B2 + task-time figure, reusing the already validated
    # RGB 5.5 descriptive implementation. Historical fig_55 names are renamed
    # here so the scientific-output layer is not chapter-number based.
    try:
        ok, reason, png = figure_motion_exposure(probe, cycle, root / "figures/main")
    except Exception as exc:
        ok, reason, png = False, f"{type(exc).__name__}: {exc}", ""
    if ok:
        out_png, out_svg = _move_legacy_pair(
            root=root,
            source_png=png,
            destination_stem=root / "figures/main/movement_block_pair_task_progression",
        )
        generated.append(out_png)
        _append_generated(
            manifest_rows,
            audit_rows,
            figure_id="movement_block_pair_task_progression",
            purpose="main_science",
            source_tables=[probe_source.name, cycle_source.name],
            output_png=out_png,
            output_svg=out_svg,
            short_caption="身体动作能量与曝光变化的参与者级 B1/B2 配对及 block×cycle 任务时间轨迹。",
        )
    else:
        _append_not_estimable(
            audit_rows,
            figure_id="movement_block_pair_task_progression",
            source_tables=[probe_source.name, cycle_source.name],
            reason=reason,
        )

    try:
        ok, reason, png = figure_pose_direction(probe, root / "figures/sensitivity")
    except Exception as exc:
        ok, reason, png = False, f"{type(exc).__name__}: {exc}", ""
    if ok:
        out_png, out_svg = _move_legacy_pair(
            root=root,
            source_png=png,
            destination_stem=root / "figures/sensitivity/movement_pose_direction_distribution",
        )
        generated.append(out_png)
        _append_generated(
            manifest_rows,
            audit_rows,
            figure_id="movement_pose_direction_distribution",
            purpose="sensitivity",
            source_tables=[probe_source.name],
            output_png=out_png,
            output_svg=out_svg,
            short_caption="横向、纵向与无量纲径向姿态候选在探针前 30 s 窗口的分布。",
        )
    else:
        _append_not_estimable(
            audit_rows,
            figure_id="movement_pose_direction_distribution",
            source_tables=[probe_source.name],
            reason=reason,
        )

    try:
        ok, reason, png = figure_coverage(session, root / "figures/qc")
    except Exception as exc:
        ok, reason, png = False, f"{type(exc).__name__}: {exc}", ""
    if ok:
        out_png, out_svg = _move_legacy_pair(
            root=root,
            source_png=png,
            destination_stem=root / "figures/qc/movement_source_coverage",
        )
        generated.append(out_png)
        _append_generated(
            manifest_rows,
            audit_rows,
            figure_id="movement_source_coverage",
            purpose="qc",
            source_tables=[session_source.name],
            output_png=out_png,
            output_svg=out_svg,
            short_caption="Movement 相关 RGB 轨在各场次的可观测帧比例。",
        )
    else:
        _append_not_estimable(
            audit_rows,
            figure_id="movement_source_coverage",
            source_tables=[session_source.name],
            reason=reason,
        )

    # Formal coefficient plots use producer-side participant-clustered model
    # estimates. P4 only visualizes them; no model refit or feature selection.
    for table_name, spec in MODEL_TABLES.items():
        path = tables / table_name
        figure_id = str(spec["figure_id"])
        source_tables = [table_name]
        if not path.exists() or path.stat().st_size == 0:
            _append_not_estimable(
                audit_rows,
                figure_id=figure_id,
                source_tables=source_tables,
                reason="source table missing or empty",
            )
            continue
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except pd.errors.EmptyDataError:
            _append_not_estimable(
                audit_rows,
                figure_id=figure_id,
                source_tables=source_tables,
                reason="source table has no rows",
            )
            continue
        primary = frame.loc[_feature_mask(frame, {PRIMARY})].copy()
        purpose_dir = str(spec["purpose"]).replace("main_science", "main")
        figure_path = root / "figures" / purpose_dir / str(spec["figure_name"])
        ok, reason = _coefficient_plot(primary, figure_path)
        if ok:
            png = figure_path.with_suffix(".png")
            svg = figure_path.with_suffix(".svg")
            out_png = str(png.relative_to(root))
            out_svg = str(svg.relative_to(root))
            generated.append(out_png)
            _append_generated(
                manifest_rows,
                audit_rows,
                figure_id=figure_id,
                purpose=str(spec["purpose"]),
                source_tables=source_tables,
                output_png=out_png,
                output_svg=out_svg,
                short_caption=str(spec["short_caption"]),
            )
        else:
            _append_not_estimable(
                audit_rows,
                figure_id=figure_id,
                source_tables=source_tables,
                reason=reason,
            )

        auxiliary = frame.loc[_feature_mask(frame, AUXILIARY)].copy()
        aux_id = f"{figure_id}_pose_sensitivity"
        aux_name = str(spec["figure_name"]).replace(".png", "_pose_sensitivity.png")
        aux_path = root / "figures/sensitivity" / aux_name
        ok, reason = _coefficient_plot(auxiliary, aux_path)
        if ok:
            png = aux_path.with_suffix(".png")
            svg = aux_path.with_suffix(".svg")
            out_png = str(png.relative_to(root))
            out_svg = str(svg.relative_to(root))
            generated.append(out_png)
            _append_generated(
                manifest_rows,
                audit_rows,
                figure_id=aux_id,
                purpose="sensitivity",
                source_tables=source_tables,
                output_png=out_png,
                output_svg=out_svg,
                short_caption="姿态/径向辅助候选对应的同源模型系数及 95% CI；仅作敏感性展示。",
            )
        else:
            _append_not_estimable(
                audit_rows,
                figure_id=aux_id,
                source_tables=source_tables,
                reason=reason,
            )

    pd.DataFrame(manifest_rows, columns=FIGURE_MANIFEST_COLUMNS).to_csv(
        manifests / "figure_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(audit_rows, columns=FIGURE_AUDIT_COLUMNS).to_csv(
        manifests / "figure_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return generated
