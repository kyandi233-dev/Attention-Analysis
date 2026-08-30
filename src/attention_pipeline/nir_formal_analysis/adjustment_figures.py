from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from attention_pipeline.formal_analysis.publication_style import (
    configure_publication_style,
    finalize_publication_figure,
)

from attention_pipeline.config import load_config

configure_publication_style()

# 中文出图字体：SimSun 优先，依次回退 Microsoft YaHei / SimHei；
# 数字与西文使用 Arial。检测块写法与 Behavior/mmWave 定稿基准脚本一致。
def _detect_cjk_font() -> str:
    """检测本机可用的中文字体，返回字体族名。"""
    for name in ("SimSun", "Microsoft YaHei", "SimHei"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "SimSun"


FIGURE_FONT = _detect_cjk_font()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [FIGURE_FONT, "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def _resolve(config, key: str) -> Path:
    raw = config.section("paths").get(key)
    if raw in (None, ""):
        raise KeyError(f"formal pupil config missing paths.{key}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (config.path.parent.parent / path).resolve()
    return path


def _save(fig: plt.Figure, path: Path) -> str:
    finalize_publication_figure(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run_adjustment_figures(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = _resolve(config, "output_root")
    comparison_path = root / "reference_adjusted_models" / "formal_unadjusted_vs_adjusted_comparison.csv"
    figure_root = root / "formal_figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(comparison_path, encoding="utf-8-sig") if comparison_path.is_file() else pd.DataFrame()

    manifest_path = root / "nir_figure_manifest.csv"
    coverage_path = root / "nir_figure_coverage_audit.csv"
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig") if manifest_path.is_file() else pd.DataFrame()
    coverage = pd.read_csv(coverage_path, encoding="utf-8-sig") if coverage_path.is_file() else pd.DataFrame()
    if not coverage.empty and "figure_family" in coverage:
        coverage = coverage[coverage["figure_family"].astype(str).ne("unadjusted_vs_adjusted_effect")].copy()
    if not manifest.empty and "figure_family" in manifest:
        manifest = manifest[manifest["figure_family"].astype(str).ne("unadjusted_vs_adjusted_effect")].copy()

    rows: list[dict[str, Any]] = []
    generated: list[str] = []
    for outcome in ("rt", "omission", "commission"):
        current = comparison[comparison.get("outcome", pd.Series(dtype=str)).astype(str).eq(outcome)].copy() if not comparison.empty else pd.DataFrame()
        if current.empty:
            rows.append({
                "metric": outcome,
                "metric_label_zh": outcome,
                "figure_family": "unadjusted_vs_adjusted_effect",
                "analysis_scale": "trial_model",
                "status": "not_estimable",
                "reason": "audited_comparison_row_missing",
                "filename": "",
                "caption_zh": "",
                "internal_title_allowed": False,
                "caption_is_external": True,
                "participant_group_n": 0,
                "session_n": 0,
                "report_layer": "core_after_endpoint_freeze",
            })
            continue
        current = current[current["comparison_status"].astype(str).eq("estimable_pair")].copy()
        if current.empty:
            rows.append({
                "metric": outcome, "metric_label_zh": outcome,
                "figure_family": "unadjusted_vs_adjusted_effect", "analysis_scale": "trial_model",
                "status": "not_estimable", "reason": "no_estimable_adjustment_pair", "filename": "", "caption_zh": "",
                "internal_title_allowed": False, "caption_is_external": True,
                "participant_group_n": 0, "session_n": 0, "report_layer": "core_after_endpoint_freeze",
            })
            continue
        labels: list[str] = []
        estimates: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for r in current.itertuples(index=False):
            term = "个体内" if str(r.pupil_term) == "pupil_within" else "个体间"
            labels.extend([f"{term}：未调整", f"{term}：调整后"])
            estimates.extend([float(r.unadjusted_estimate), float(r.adjusted_estimate)])
            lows.extend([float(r.unadjusted_ci_low), float(r.adjusted_ci_low)])
            highs.extend([float(r.unadjusted_ci_high), float(r.adjusted_ci_high)])
        est = np.asarray(estimates, dtype=float); lo = np.asarray(lows, dtype=float); hi = np.asarray(highs, dtype=float)
        y = np.arange(len(est))
        fig, ax = plt.subplots(figsize=(7.4, max(4.0, 0.55 * len(est) + 1.4)))
        ax.errorbar(est, y, xerr=np.vstack([est - lo, hi - est]), fmt="o", capsize=3)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.set_xlabel("效应估计与 95% CI"); ax.set_ylabel("Pupil 效应与调整状态")
        filename = f"NIR模型_{outcome}_调整前后审计.png"
        path = _save(fig, figure_root / filename)
        generated.append(path)
        visual_state = ";".join(sorted(set(current["formal_visual_adjustment_status"].astype(str)))) if "formal_visual_adjustment_status" in current else "unknown"
        caption = f"{outcome} 的参与者内/参与者间瞳孔效应在未调整与调整后模型中的比较；视觉调整状态={visual_state}。"
        participant_n = int(pd.to_numeric(current.get("participant_group_n_adjusted"), errors="coerce").max()) if "participant_group_n_adjusted" in current else 0
        session_n = int(pd.to_numeric(current.get("session_n_adjusted"), errors="coerce").max()) if "session_n_adjusted" in current else 0
        rows.append({
            "metric": outcome,
            "metric_label_zh": outcome,
            "figure_family": "unadjusted_vs_adjusted_effect",
            "analysis_scale": "trial_model",
            "status": "generated",
            "reason": "generated_from_formal_adjustment_audit",
            "filename": filename,
            "caption_zh": caption,
            "internal_title_allowed": False,
            "caption_is_external": True,
            "participant_group_n": participant_n,
            "session_n": session_n,
            "report_layer": "core_after_endpoint_freeze",
        })

    add = pd.DataFrame(rows)
    coverage = pd.concat([coverage, add], ignore_index=True, sort=False)
    manifest = pd.concat([manifest, add[add["status"].eq("generated")]], ignore_index=True, sort=False)
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    summary = {
        "status": "complete",
        "n_generated": len(generated),
        "internal_title_allowed": False,
        "source": str(comparison_path),
        "figure_files": generated,
    }
    (root / "nir_adjustment_figure_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
