"""Post-hoc reporting for the authoritative lightweight RGB formal route.

The current RGB runner intentionally defers large scientific inference.  This
reporter therefore visualizes only active-route QC/candidate outputs and makes
deferred analyses explicit instead of manufacturing inferential results.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from attention_pipeline.formal_analysis.publication_style import (
    FONT_FAMILY,
    configure_publication_style,
    finalize_publication_figure,
)

configure_publication_style()
mpl.rcParams.update({"savefig.dpi": 300})


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False) if path.is_file() else pd.DataFrame()


def _save(fig: plt.Figure, stem: Path) -> tuple[str, str]:
    finalize_publication_figure(fig)
    fig.tight_layout()
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png"); svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=300, bbox_inches="tight"); fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return str(png), str(svg)


def _md(frame: pd.DataFrame, cols: list[str]) -> str:
    if frame.empty:
        return "_No rows._\n"
    d = frame[[c for c in cols if c in frame]].fillna("")
    headers = list(d.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in d.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in row) + " |")
    return "\n".join(lines) + "\n"


def build_rgb_report(output_root: Path, analysis_ready_root: Path) -> dict[str, Any]:
    output_root = Path(output_root); analysis_ready_root = Path(analysis_ready_root)
    report_root = output_root / "reporting"; fig_root = report_root / "figures_publication"
    report_root.mkdir(parents=True, exist_ok=True); fig_root.mkdir(parents=True, exist_ok=True)
    session_qc = _read(output_root / "qc" / "session_qc.csv")
    components = _read(output_root / "qc" / "rgb_component_status.csv")
    sources = _read(output_root / "provenance" / "rgb_source_manifest.csv")
    blinks = _read(output_root / "tables" / "rgb_blink_candidate_events.csv")
    manifest_path = output_root / "provenance" / "rgb_formal_manifest.json"
    run = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.is_file() else {}

    figure_rows: list[dict[str, Any]] = []
    if not components.empty and {"component", "status"}.issubset(components.columns):
        counts = components.groupby(["component", "status"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(7.0, 4.3))
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("RGB component"); ax.set_ylabel("Sessions"); ax.legend(title="Status", frameon=False)
        png, svg = _save(fig, fig_root / "rgb_component_status")
        figure_rows.append({"figure":"rgb_component_status","png":Path(png).name,"svg":Path(svg).name,"caption_zh":"各 RGB 正式组件在 governed sessions 中的可生成/不可估计状态计数。","caption_en":"Session counts by RGB component and generation status.","role":"QC/availability","font_family":FONT_FAMILY,"legend_frame":False})

    if not session_qc.empty and "rgb_source_present" in session_qc:
        present = int(pd.to_numeric(session_qc["rgb_source_present"], errors="coerce").fillna(0).eq(1).sum())
        absent = int(len(session_qc) - present)
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.bar(["Source present", "Source missing"], [present, absent])
        ax.set_ylabel("Sessions")
        png, svg = _save(fig, fig_root / "rgb_source_coverage")
        figure_rows.append({"figure":"rgb_source_coverage","png":Path(png).name,"svg":Path(svg).name,"caption_zh":"RGB source 在 governed cohort 中的可用性；缺失模态不改变 cohort membership。","caption_en":"RGB source coverage within the governed cohort.","role":"QC/availability"})

    blink_counts = pd.DataFrame()
    if not blinks.empty and "session_id" in blinks:
        blink_counts = blinks.groupby("session_id").size().rename("blink_candidate_n").reset_index()
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.hist(blink_counts["blink_candidate_n"].to_numpy(float), bins=min(15, max(5, int(np.sqrt(len(blink_counts))))), edgecolor="black")
        ax.set_xlabel("Algorithm-defined blink candidates per session"); ax.set_ylabel("Sessions")
        png, svg = _save(fig, fig_root / "rgb_blink_candidate_count_distribution")
        figure_rows.append({"figure":"rgb_blink_candidate_count_distribution","png":Path(png).name,"svg":Path(svg).name,"caption_zh":"每场算法定义 blink candidate 事件数的分布；该事件尚未进行人工视频验证，不等同于正式 blink/PERCLOS 结局。","caption_en":"Distribution of algorithm-defined blink-candidate event counts per session; candidates are not manually validated blink/PERCLOS endpoints.","role":"candidate QC"})
    blink_counts.to_csv(report_root / "rgb_blink_candidate_counts_by_session.csv", index=False, encoding="utf-8-sig")

    # Inventory active per-session derived files without turning disabled science into inference.
    ready_rows: list[dict[str, Any]] = []
    if analysis_ready_root.is_dir():
        for session_dir in sorted(p for p in analysis_ready_root.iterdir() if p.is_dir()):
            ready_rows.append({
                "session_id": session_dir.name,
                "motion_qc": int(any(session_dir.glob("*_motion_qc.parquet"))),
                "pose_confirmation": int(any(session_dir.glob("*_pose_confirmation.parquet"))),
                "blink_candidate_frames": int(any(session_dir.glob("*_blink_candidate_frames.parquet"))),
                "blink_candidate_events": int(any(session_dir.glob("*_blink_candidate_events.parquet"))),
            })
    ready_inventory = pd.DataFrame(ready_rows)
    ready_inventory.to_csv(report_root / "rgb_analysis_ready_inventory.csv", index=False, encoding="utf-8-sig")
    figure_manifest = pd.DataFrame(figure_rows)
    figure_manifest.to_csv(report_root / "rgb_figure_manifest.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# RGB 正式下游结果说明与图表映射",
        "",
        "当前权威路线是 lightweight Motion + Pose confirmation + algorithm-defined Blink candidates。该路线有意不执行 endpoint freeze 之前的大规模推断、预测或多模态融合，因此本报告只呈现已授权的 QC/候选结果，不把 deferred 项目伪装成缺失结果。",
        "",
        "## 1. 运行状态",
        "",
        f"- governed_session_n: **{run.get('governed_session_n', 'unknown')}**",
        f"- participant_group_n_resolved: **{run.get('participant_group_n_resolved', 'unknown')}**",
        f"- component_exception_rows: **{run.get('component_exception_rows', 'unknown')}**",
        f"- large_figure_suite_run (原 runner): **{run.get('large_figure_suite_run', False)}**",
        "",
        "## 2. Session/component QC",
        "",
        _md(session_qc, ["session_id","participant_group_id","rgb_source_present","motion_status","pose_status","blink_status","blink_event_candidate_n"]),
        "## 3. Component 状态审计",
        "",
        _md(components, ["session_id","participant_group_id","component","status","reason"]),
        "## 4. 当前 publication/QC 图",
        "",
        _md(figure_manifest, ["figure","png","svg","role","caption_zh","caption_en"]),
        "## 5. Analysis-ready 文件覆盖",
        "",
        _md(ready_inventory, ["session_id","motion_qc","pose_confirmation","blink_candidate_frames","blink_candidate_events"]),
        "## 6. 科学解释边界",
        "",
        "- Motion Energy 与 exposure/gray-level change 分开保存，不合成为 outcome-tuned risk score。",
        "- Pose direction 是辅助运动 QC candidate，不是物理位移真值。",
        "- Blink outputs 是算法候选事件；PERCLOS 仍 deferred，不能从候选事件直接宣称正式 PERCLOS。",
        "- 当前 runner 的 inference / prediction / multimodal fusion 均为 deferred；本报告不会生成虚假的显著性检验。",
        "- 图片内部统一英文、无大标题；PNG 300 dpi，同时保留 SVG 矢量版。",
    ]
    report = report_root / "RGB结果说明_完整.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status":"complete","report":str(report),"figure_root":str(fig_root),"figure_n":int(len(figure_manifest)),"analysis_refit":False,"producer_rerun":False}
