"""Publication figures and figure audit for frozen-post Ocular analyses."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from attention_pipeline.formal_analysis.publication_style import (
    configure_publication_style,
    finalize_publication_figure,
)
from .ocular_postfreeze_analysis import FROZEN_OCULAR_FEATURES, SCHEMA_VERSION

FIGURE_MANIFEST_COLUMNS = (
    "figure_id", "purpose", "scientific_question", "source_table", "analysis_unit",
    "participant_group_n", "session_n", "probe_n", "inference_status", "caption_zh",
    "png_path", "svg_path", "code_contract",
)
FIGURE_AUDIT_COLUMNS = (
    "figure_id", "purpose", "expected", "generated_png", "generated_svg",
    "source_row_n", "status", "reason",
)


def _save(fig, root: Path, purpose: str, figure_id: str) -> tuple[str, str]:
    configure_publication_style()
    folder = root / "figures" / purpose
    folder.mkdir(parents=True, exist_ok=True)
    finalize_publication_figure(fig, remove_titles=True)
    fig.tight_layout()
    png = folder / f"{figure_id}.png"
    svg = folder / f"{figure_id}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return str(png.relative_to(root)), str(svg.relative_to(root))


def _counts(frame: pd.DataFrame) -> tuple[Any, Any, Any]:
    def _maxint(col: str):
        if col not in frame or frame.empty:
            return pd.NA
        x = pd.to_numeric(frame[col], errors="coerce").dropna()
        return int(x.max()) if len(x) else pd.NA
    return _maxint("participant_group_n"), _maxint("session_n"), _maxint("n_rows")


def _forest(
    frame: pd.DataFrame,
    root: Path,
    purpose: str,
    figure_id: str,
    labels: list[str],
    *,
    xlabel: str,
    source_table: str,
    question: str,
    caption_zh: str,
) -> dict[str, Any] | None:
    if frame.empty:
        return None
    d = frame.copy()
    for c in ("estimate", "ci_low", "ci_high"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    mask = d[["estimate", "ci_low", "ci_high"]].notna().all(axis=1)
    d = d.loc[mask].copy()
    labels = [label for label, keep in zip(labels, mask.tolist()) if keep]
    if d.empty:
        return None
    y = np.arange(len(d))
    est = d["estimate"].to_numpy(float)
    lo = d["ci_low"].to_numpy(float)
    hi = d["ci_high"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.4, max(4.5, 0.32 * len(d) + 1.7)))
    ax.errorbar(est, y, xerr=np.vstack([est - lo, hi - est]), fmt="o", capsize=2)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.invert_yaxis()
    png, svg = _save(fig, root, purpose, figure_id)
    pg, ss, pn = _counts(d)
    return {
        "figure_id": figure_id, "purpose": purpose, "scientific_question": question,
        "source_table": source_table, "analysis_unit": "probe",
        "participant_group_n": pg, "session_n": ss, "probe_n": pn,
        "inference_status": "95% CI from formal participant-clustered model; significance never changes feature freeze",
        "caption_zh": caption_zh, "png_path": png, "svg_path": svg,
        "code_contract": SCHEMA_VERSION,
    }


def _coverage_figure(coverage: pd.DataFrame, root: Path, purpose: str, figure_id: str):
    if coverage.empty:
        return None
    d = coverage.copy()
    d["finite_fraction"] = pd.to_numeric(d["finite_fraction"], errors="coerce")
    d = d.dropna(subset=["finite_fraction"])
    if d.empty:
        return None
    labels = [
        FROZEN_OCULAR_FEATURES.get(str(fid), {}).get("label", str(fid))
        for fid in d["feature_id"]
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = np.arange(len(d))
    ax.barh(y, d["finite_fraction"].to_numpy(float))
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Finite probe fraction")
    ax.set_ylabel("")
    ax.invert_yaxis()
    png, svg = _save(fig, root, purpose, figure_id)
    return {
        "figure_id": figure_id, "purpose": purpose,
        "scientific_question": "What is the valid-probe coverage of each frozen first-round Ocular feature?",
        "source_table": "ocular_feature_analysis_coverage.csv", "analysis_unit": "feature",
        "participant_group_n": int(pd.to_numeric(d["participant_group_n"], errors="coerce").max()),
        "session_n": int(pd.to_numeric(d["session_n"], errors="coerce").max()),
        "probe_n": int(pd.to_numeric(d["probe_total_n"], errors="coerce").max()),
        "inference_status": "descriptive qualification only",
        "caption_zh": "五个第一轮冻结眼部特征的探针级有效覆盖率。缺失保持为缺失，不补零、不插补。",
        "png_path": png, "svg_path": svg, "code_contract": SCHEMA_VERSION,
    }


def build_ocular_postfreeze_figures(
    *,
    ocular_root: Path,
    coverage: pd.DataFrame,
    task: pd.DataFrame,
    q1: pd.DataFrame,
    q2: pd.DataFrame,
    behavior_links: pd.DataFrame,
    frozen_probe_table: pd.DataFrame,
) -> dict[str, Any]:
    del frozen_probe_table  # reserved for future QC panels; no data-driven feature selection.
    manifest_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    def register(fid: str, purpose: str, source: pd.DataFrame, row: dict[str, Any] | None):
        if row is not None:
            manifest_rows.append(row)
        audit_rows.append({
            "figure_id": fid, "purpose": purpose, "expected": True,
            "generated_png": row is not None, "generated_svg": row is not None,
            "source_row_n": int(len(source)),
            "status": "generated" if row is not None else "not_generated",
            "reason": "" if row is not None else "no_estimable_source_rows",
        })

    row = _coverage_figure(coverage, ocular_root, "qualification", "ocular_frozen_feature_coverage")
    register("ocular_frozen_feature_coverage", "qualification", coverage, row)

    d = task[
        task.get("term", pd.Series(dtype=str)).astype(str).isin(
            ["progression_centered", "block_b2:progression_centered"]
        )
    ].copy() if not task.empty else task
    labels = []
    for r in d.itertuples(index=False):
        base = FROZEN_OCULAR_FEATURES.get(str(r.feature_id), {}).get("label", str(r.feature_id))
        term = "B1 progression" if str(r.term) == "progression_centered" else "B2-B1 slope difference"
        labels.append(f"{base} | {term}")
    row = _forest(
        d, ocular_root, "main", "ocular_task_progression_coefficients", labels,
        xlabel="Coefficient in native Ocular units", source_table="ocular_task_progression.csv",
        question="How do frozen Ocular features change across within-block task progression, and does the slope differ in Block 2?",
        caption_zh="冻结眼部特征随区块内任务进程的变化系数及95%置信区间；交互项表示第二个区块相对第一个区块的进程斜率差异。",
    )
    register("ocular_task_progression_coefficients", "main", d, row)

    d = q1.copy()
    labels = []
    for r in d.itertuples(index=False):
        base = FROZEN_OCULAR_FEATURES.get(str(r.feature_id), {}).get("label", str(r.feature_id))
        comp = "within-person" if str(r.term) == "ocular_within_z" else "between-person"
        labels.append(f"{base} | {comp} | Q1={int(r.contrast_category)} vs 1")
    row = _forest(
        d, ocular_root, "main", "ocular_q1_coefficients", labels,
        xlabel="Log-odds coefficient per 1 SD Ocular component", source_table="ocular_q1_models.csv",
        question="How are within-person and between-person Ocular differences associated with four-category Q1 attention content?",
        caption_zh="五个冻结眼部特征的人内与人际成分对Q1四分类的多项逻辑回归系数及95%置信区间；Q1=1为参照。",
    )
    register("ocular_q1_coefficients", "main", d, row)

    d = q2.copy()
    labels = []
    for r in d.itertuples(index=False):
        base = FROZEN_OCULAR_FEATURES.get(str(r.feature_id), {}).get("label", str(r.feature_id))
        comp = "within-person" if str(r.term) == "ocular_within_z" else "between-person"
        labels.append(f"{base} | {comp}")
    row = _forest(
        d, ocular_root, "main", "ocular_q2_coefficients", labels,
        xlabel="Proportional log-odds coefficient per 1 SD Ocular component", source_table="ocular_q2_models.csv",
        question="How are within-person and between-person Ocular differences associated with ordered Q2 subjective state?",
        caption_zh="五个冻结眼部特征的人内与人际成分对Q2有序主观状态的累积logit系数及95%置信区间。Q2按有序变量建模。",
    )
    register("ocular_q2_coefficients", "main", d, row)

    d = behavior_links.copy()
    labels = []
    for r in d.itertuples(index=False):
        base = FROZEN_OCULAR_FEATURES.get(str(r.feature_id), {}).get("label", str(r.feature_id))
        comp = "within-person" if str(r.term) == "ocular_within_z" else "between-person"
        labels.append(f"{base} | {comp} | {r.outcome}")
    row = _forest(
        d, ocular_root, "main", "ocular_behavior_link_coefficients", labels,
        xlabel="Model coefficient per 1 SD Ocular component", source_table="ocular_behavior_links.csv",
        question="How are frozen Ocular features associated with the nonredundant frozen recent-Behavior indicators?",
        caption_zh="冻结眼部特征与近期行为指标关系的模型系数及95%置信区间。RT均值/中位数因研究者冻结尚未完成，不进入本图。",
    )
    register("ocular_behavior_link_coefficients", "main", d, row)

    row = _coverage_figure(coverage, ocular_root, "qc", "ocular_missingness_qc")
    if row is not None:
        row["scientific_question"] = "Where does model availability differ across frozen Ocular features because of retained missingness?"
        row["caption_zh"] = "冻结眼部特征的有效覆盖情况，用于审计结构性缺失与特征级可估性差异；不将缺失解释为零值。"
    register("ocular_missingness_qc", "qc", coverage, row)

    figure_manifest = pd.DataFrame(manifest_rows, columns=FIGURE_MANIFEST_COLUMNS)
    figure_audit = pd.DataFrame(audit_rows, columns=FIGURE_AUDIT_COLUMNS)
    figure_manifest.to_csv(ocular_root / "manifests/figure_manifest.csv", index=False, encoding="utf-8-sig")
    figure_audit.to_csv(ocular_root / "manifests/figure_audit.csv", index=False, encoding="utf-8-sig")
    all_generated = bool(figure_audit["generated_png"].all() and figure_audit["generated_svg"].all())
    return {
        "status": "complete" if all_generated else "complete_with_unrendered_figures",
        "figure_n": int(len(figure_manifest)), "audit_n": int(len(figure_audit)),
        "unrendered_n": int((~figure_audit["generated_png"]).sum()),
    }
