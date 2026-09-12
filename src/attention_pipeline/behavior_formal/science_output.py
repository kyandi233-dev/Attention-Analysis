"""Compact, question-driven Behavior scientific outputs and feature handoff."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from attention_pipeline.formal_analysis.publication_style import (
    configure_publication_style,
    finalize_publication_figure,
)

SCHEMA = "behavior-science-output-v1"
TIME_EVIDENCE = (
    "build_probe_windows: same session/block; trial_num < anchor_trial; "
    "absolute_onset_time < probe_time; primary interval [probe_time-30s, probe_time); "
    "anchor trial excluded; block crossing forbidden"
)

# feature_id, representation_id, predictor, label, unit, role, qc, estimability,
# redundancy, registry_ready, researcher_freeze_required
SPECS = (
    ("behavior_rt_level", "rt_level_mean", "go_correct_rt_mean_ms", "RT level (mean)", "ms",
     "primary_candidate", "representation_choice_pending", "finite probe-level mean",
     "mutually exclusive with rt_level_median in first-round Behavior reference", True, True),
    ("behavior_rt_level", "rt_level_median", "go_correct_rt_median_ms", "RT level (median)", "ms",
     "primary_candidate", "representation_choice_pending", "finite probe-level median",
     "mutually exclusive with rt_level_mean in first-round Behavior reference", True, True),
    ("behavior_rt_variability", "rt_cv", "go_correct_rt_cv", "RT variability (CV)", "ratio",
     "primary", "method_frozen", "rt_cv_status == estimable; mathematical minimum n=2 only",
     "SD/MAD/IQR are descriptive/sensitivity alternatives", True, False),
    ("behavior_rt_trend", "rt_theilsen_slope", "go_correct_rt_theilsen_slope_ms_per_s", "RT trend (Theil-Sen slope)", "ms/s",
     "primary", "method_frozen", "rt_slope_status == estimable",
     "other slope estimators are sensitivity-only unless separately frozen", True, False),
    ("behavior_go_omission", "raw_go_omission", "raw_go_omission_rate", "Go omission rate", "proportion",
     "primary", "method_frozen", "finite rate with explicit Go-opportunity denominator",
     "clean/timing-ambiguous rates partition raw omission", True, False),
    ("behavior_nogo_commission", "commission_rate", "commission_rate", "No-Go commission rate", "proportion",
     "primary", "method_frozen", "finite rate with explicit No-Go-opportunity denominator",
     "dprime_loglinear is a composite sensitivity representation", True, False),
    ("behavior_rt_variability", "rt_sd", "go_correct_rt_sd_ms", "RT variability (SD)", "ms",
     "sensitivity", "sensitivity_only", "finite sample SD", "alternative to primary CV", False, False),
    ("behavior_rt_variability", "rt_mad", "go_correct_rt_mad_ms", "RT variability (MAD)", "ms",
     "sensitivity", "sensitivity_only", "finite MAD", "alternative to primary CV", False, False),
    ("behavior_rt_variability", "rt_iqr", "go_correct_rt_iqr_ms", "RT variability (IQR)", "ms",
     "sensitivity", "sensitivity_only", "finite IQR", "alternative to primary CV", False, False),
    ("behavior_error_composite", "dprime_loglinear", "dprime_loglinear", "Loglinear d-prime", "index",
     "sensitivity", "sensitivity_only", "finite loglinear d-prime", "composite of omission/hit and commission/false-alarm information", False, False),
    ("behavior_go_omission_partition", "clean_go_omission", "clean_go_omission_rate", "Go omission without detected timing ambiguity", "proportion",
     "qc_sensitivity", "qc_only", "finite partition rate", "component of raw Go omission", False, False),
    ("behavior_go_omission_partition", "timing_ambiguous_go_omission", "timing_ambiguous_go_omission_rate", "Timing-ambiguous Go omission", "proportion",
     "qc_sensitivity", "qc_only", "finite partition rate", "component of raw Go omission", False, False),
)
LABELS = {s[2]: s[3] for s in SPECS}
PRIMARY = tuple(s[2] for s in SPECS if s[9])


def _read(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _nunique(frame: pd.DataFrame, column: str, mask: pd.Series) -> int:
    return int(frame.loc[mask, column].astype("string").nunique()) if column in frame else 0


def build_behavior_feature_handoff(probe: pd.DataFrame) -> pd.DataFrame:
    """Create candidate metadata without mutating the formal feature registry."""
    rows = []
    total = len(probe)
    for feature_id, rep_id, predictor, label, unit, role, qc, estimability, redundancy, ready, freeze in SPECS:
        value = pd.to_numeric(probe[predictor], errors="coerce") if predictor in probe else pd.Series(np.nan, index=probe.index)
        finite = value.notna() & np.isfinite(value)
        coverage = {
            "probe_total_n": int(total), "finite_probe_n": int(finite.sum()),
            "finite_fraction": float(finite.mean()) if total else None,
            "participant_group_n": _nunique(probe, "repeat_participant_id", finite),
            "session_n": _nunique(probe, "session_id", finite),
        }
        rows.append({
            "scientific_feature_id": feature_id,
            "candidate_representation_id": rep_id,
            "predictor_columns": predictor,
            "predictor_column": predictor,
            "display_name": label,
            "unit": unit,
            "scientific_modality": "behavior",
            "source_namespace": "behavior",
            "required_devices": "[]",
            "preprocessing_dependencies": "[]",
            "measurement_qc_status": qc,
            "estimability_rule": estimability,
            "estimability_status": "candidate_available" if finite.any() else "not_available",
            "coverage_summary": json.dumps(coverage, ensure_ascii=False, sort_keys=True),
            "redundancy_relation": redundancy,
            "temporal_anchor": "probe_time_ms",
            "temporal_scope": "pre_probe_only",
            "time_legality_status": "verified_pre_probe_only",
            "time_legality_evidence": TIME_EVIDENCE,
            "report_role": role,
            "registry_ready": ready,
            "researcher_freeze_required": freeze,
            "selection_policy": "Q1/Q2 significance and supervised outer-test performance are not freeze criteria",
        })
    return pd.DataFrame(rows)


def _ecdf(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    x = np.sort(x[np.isfinite(x)])
    return x, np.arange(1, len(x) + 1) / len(x) if len(x) else x


def _save(fig, root: Path, role: str, figure_id: str) -> tuple[str, str]:
    folder = root / "figures" / role
    folder.mkdir(parents=True, exist_ok=True)
    finalize_publication_figure(fig, remove_titles=True)
    fig.tight_layout()
    png, svg = folder / f"{figure_id}.png", folder / f"{figure_id}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return str(png.relative_to(root)), str(svg.relative_to(root))


def _counts(frame: pd.DataFrame, analysis_unit: str) -> tuple[Any, Any, Any]:
    pg = int(frame["repeat_participant_id"].nunique()) if "repeat_participant_id" in frame else pd.NA
    ss = int(frame["session_id"].nunique()) if "session_id" in frame else pd.NA
    if pg is pd.NA and "participant_group_n" in frame:
        x = pd.to_numeric(frame["participant_group_n"], errors="coerce").dropna(); pg = int(x.max()) if len(x) else pd.NA
    if ss is pd.NA and "session_n" in frame:
        x = pd.to_numeric(frame["session_n"], errors="coerce").dropna(); ss = int(x.max()) if len(x) else pd.NA
    probe_n = int(len(frame)) if analysis_unit == "probe" and "repeat_participant_id" in frame else pd.NA
    if analysis_unit == "probe" and "n_rows" in frame:
        x = pd.to_numeric(frame["n_rows"], errors="coerce").dropna(); probe_n = int(x.max()) if len(x) else probe_n
    return pg, ss, probe_n


def _row(fid: str, role: str, question: str, variables: str, units: str, unit: str,
         frame: pd.DataFrame, inference: str, repeated: str, source: str, caption: str,
         paths: tuple[str, str]) -> dict[str, Any]:
    pg, ss, pn = _counts(frame, unit)
    return {
        "figure_id": fid, "purpose": role, "scientific_question": question,
        "variables": variables, "units": units, "analysis_unit": unit,
        "participant_group_n": pg, "session_n": ss, "probe_n": pn,
        "repeated_measure_handling": repeated, "inference_status": inference,
        "source_table": source, "code_contract": SCHEMA, "internal_title": False,
        "image_language": "English", "font_contract": "Times New Roman with serif fallbacks",
        "legend_frame": False, "caption_zh": caption, "png_path": paths[0], "svg_path": paths[1],
    }


def _rt_level_figures(probe: pd.DataFrame, root: Path) -> list[dict[str, Any]]:
    a, b = "go_correct_rt_mean_ms", "go_correct_rt_median_ms"
    if not {a, b}.issubset(probe): return []
    d = probe.dropna(subset=[a, b]).copy()
    if d.empty: return []
    rows = []
    fig, ax = plt.subplots(figsize=(5.6, 4.8)); ax.scatter(d[b], d[a], s=12, alpha=.35)
    lo, hi = float(d[[a,b]].min().min()), float(d[[a,b]].max().max()); ax.plot([lo,hi],[lo,hi], "--", linewidth=1)
    ax.set(xlabel="Median correct-Go RT (ms)", ylabel="Mean correct-Go RT (ms)")
    paths = _save(fig, root, "qualification", "behavior_rt_level_mean_median_agreement")
    rows.append(_row("behavior_rt_level_mean_median_agreement", "qualification",
        "How strongly do mean and median represent the same probe-level RT level, and where do they diverge?",
        f"{a}; {b}", "ms", "probe", d, "descriptive_only; identity line is not an inferential test",
        "probe points are descriptive; no pseudo-independent CI", "probe_primary_30s.csv",
        "探针前30秒正确Go反应时均值与中位数的一致性。虚线为恒等线；该图用于表示冻结，不以Q1/Q2关系选择表示。", paths))
    diff = d[a] - d[b]; x,y = _ecdf(diff)
    fig, ax = plt.subplots(figsize=(5.6,4.3)); ax.plot(x,y); ax.axvline(0, linestyle="--", linewidth=1)
    ax.set(xlabel="Mean RT - median RT (ms)", ylabel="Empirical cumulative probability")
    paths = _save(fig, root, "qualification", "behavior_rt_level_difference_ecdf")
    rows.append(_row("behavior_rt_level_difference_ecdf", "qualification",
        "How large and asymmetric is the mean-minus-median RT discrepancy across probe windows?",
        f"{a} - {b}", "ms", "probe", d, "descriptive_only",
        "probe-level ECDF; no independence-based error band", "probe_primary_30s.csv",
        "探针前30秒正确Go反应时均值减中位数的经验累积分布，用于检查长尾对均值表示的影响。", paths))
    return rows


def _feature_ecdf(probe: pd.DataFrame, root: Path, predictor: str, fid: str, xlabel: str, unit: str, caption: str):
    if predictor not in probe: return None
    x,y = _ecdf(probe[predictor])
    if not len(x): return None
    fig, ax = plt.subplots(figsize=(5.6,4.3)); ax.plot(x,y); ax.set(xlabel=xlabel, ylabel="Empirical cumulative probability")
    paths = _save(fig, root, "qualification", fid)
    mask = pd.to_numeric(probe[predictor], errors="coerce").notna(); d=probe.loc[mask]
    return _row(fid, "qualification", f"What is the probe-level distribution and tail structure of {LABELS[predictor]}?",
        predictor, unit, "probe", d, "descriptive_only", "repeated probes are not treated as independent for inference",
        "probe_primary_30s.csv", caption, paths)


def _coefficient_figure(models: pd.DataFrame, root: Path, outcome: str):
    if models.empty or "predictor" not in models: return None
    d=models[models["predictor"].astype(str).isin(PRIMARY)].copy()
    need=["estimate_per_predictor_sd","ci_low","ci_high"]
    if d.empty or not set(need).issubset(d): return None
    for c in need: d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=need)
    if d.empty: return None
    if outcome=="Q1":
        d["label"]=d.apply(lambda r:f"{LABELS.get(str(r['predictor']),r['predictor'])} | Q1={int(r['contrast_category'])} vs 1",axis=1)
        source="q1_nominal_models.csv"; family="participant-cluster robust MNLogit; 95% CI"
        question="What participant-clustered Q1 associations are estimated for the compact Behavior candidate set?"
        caption="行为候选指标与Q1四分类的标准化多项逻辑回归系数及95%置信区间；Q1=1为参照。该图不用于按显著性筛选特征。"
    else:
        d["label"]=d["predictor"].astype(str).map(lambda x:LABELS.get(x,x))
        source="q2_ordinal_gee_models.csv"; family="participant-cluster Ordinal GEE; 95% CI"
        question="What participant-clustered Q2 associations are estimated for the compact Behavior candidate set?"
        caption="行为候选指标与Q2有序自评的标准化Ordinal GEE系数及95%置信区间。该图用于单模态科学解释，不作为监督学习特征筛选器。"
    y=np.arange(len(d)); est=d[need[0]].to_numpy(float); lo=d[need[1]].to_numpy(float); hi=d[need[2]].to_numpy(float)
    fig,ax=plt.subplots(figsize=(7.8,max(4.5,.34*len(d)+1.6))); ax.errorbar(est,y,xerr=np.vstack([est-lo,hi-est]),fmt="o",capsize=2)
    ax.axvline(0,linestyle="--",linewidth=1); ax.set_yticks(y,d["label"].tolist()); ax.set_xlabel("Coefficient per 1 SD increase in predictor"); ax.set_ylabel(""); ax.invert_yaxis()
    fid=f"behavior_{outcome.lower()}_candidate_coefficients"; paths=_save(fig,root,"main",fid)
    return _row(fid,"main",question,"; ".join(sorted(d["predictor"].astype(str).unique())),"standardized log-odds coefficient","probe",d,family,
        "producer models use participant-group clustering; CIs are reused without re-estimation",source,caption,paths)


def _b1b2_figure(summary: pd.DataFrame, root: Path):
    keep=["go_correct_rt_cv","raw_go_omission_rate","commission_rate"]
    if summary.empty or "metric" not in summary:return None
    d=summary[summary["metric"].astype(str).isin(keep)].copy(); need=["estimate_b2_minus_b1","ci_low","ci_high"]
    if d.empty or not set(need).issubset(d):return None
    for c in need:d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=need)
    if d.empty:return None
    y=np.arange(len(d)); est=d[need[0]].to_numpy(float); lo=d[need[1]].to_numpy(float); hi=d[need[2]].to_numpy(float)
    fig,ax=plt.subplots(figsize=(6.8,3.6)); ax.errorbar(est,y,xerr=np.vstack([est-lo,hi-est]),fmt="o",capsize=2); ax.axvline(0,linestyle="--",linewidth=1)
    ax.set_yticks(y,[LABELS.get(str(x),str(x)) for x in d["metric"]]); ax.set_xlabel("B2 - B1 (ratio/proportion scale)"); ax.set_ylabel(""); ax.invert_yaxis()
    paths=_save(fig,root,"main","behavior_b1_b2_compact_effects")
    return _row("behavior_b1_b2_compact_effects","main","Do core dimensionless Behavior indicators change from B1 to B2 after participant-first clustering?",
        "; ".join(keep),"ratio/proportion difference","participant-group clustered session pair",d,"participant-cluster bootstrap mean difference; 95% percentile CI",
        "session-internal differences collapse to participant-group means before bootstrap","b1_b2_participant_cluster_bootstrap.csv",
        "反应时变异系数、Go遗漏率和No-Go误按率的B2-B1变化。重复场次先在人内聚合，再按参与者组进行聚类bootstrap并给出95%置信区间。",paths)


def _coverage_figure(handoff: pd.DataFrame, root: Path):
    d=handoff[handoff["registry_ready"].eq(True)].copy()
    d["fraction"]=d["coverage_summary"].map(lambda x:json.loads(x).get("finite_fraction")); d=d.dropna(subset=["fraction"]).sort_values("fraction")
    if d.empty:return None
    y=np.arange(len(d)); fig,ax=plt.subplots(figsize=(6.8,4.2)); ax.hlines(y,0,d["fraction"],linewidth=1); ax.scatter(d["fraction"],y,s=28)
    ax.set_yticks(y,d["display_name"].tolist()); ax.set_xlim(0,1.02); ax.set_xlabel("Finite probe-window fraction"); ax.set_ylabel("")
    paths=_save(fig,root,"qc","behavior_candidate_coverage")
    return _row("behavior_candidate_coverage","qc","How much of the governed probe table is estimable for each first-round Behavior candidate?",
        "registry-ready Behavior candidate coverage","fraction","probe",pd.DataFrame(),"QC descriptive only","denominator audit; not an inferential comparison",
        "behavior_feature_handoff.csv","首轮行为候选指标在探针前30秒表中的有限值覆盖率。覆盖率用于资格与缺失结构审计，不作为全样本自动删特征阈值。",paths)


def build_behavior_science_output(formal_root: str|Path, science_root: str|Path, *, authoritative: bool=True,
                                  replace: bool=True, producer_provenance: dict[str,Any]|None=None) -> dict[str,Any]:
    """Build the compact Behavior package; large producer tables remain in place."""
    configure_publication_style(); formal_root=Path(formal_root).expanduser().resolve(); science_root=Path(science_root).expanduser().resolve(); root=science_root/"Behavior"
    if replace and root.exists():shutil.rmtree(root)
    for rel in ("tables","figures/main","figures/qualification","figures/qc","figures/sensitivity","manifests"):(root/rel).mkdir(parents=True,exist_ok=True)
    probe=_read(formal_root,"probe_primary_30s.csv")
    if probe.empty:raise FileNotFoundError(f"Behavior science output requires non-empty {formal_root/'probe_primary_30s.csv'}")
    handoff=build_behavior_feature_handoff(probe); handoff.to_csv(root/"tables"/"behavior_feature_handoff.csv",index=False,encoding="utf-8-sig")
    handoff[["predictor_column","display_name","report_role","coverage_summary"]].to_csv(root/"tables"/"behavior_feature_coverage.csv",index=False,encoding="utf-8-sig")
    source_names=["probe_primary_30s.csv","block_metrics.csv","cycle_metrics.csv","b1_b2_pairs.csv","b1_b2_participant_cluster_bootstrap.csv","q1_nominal_models.csv","q2_ordinal_gee_models.csv","model_failures.csv","run_manifest.json"]
    pd.DataFrame([{"producer_file":n,"producer_path":str((formal_root/n).resolve()),"exists":(formal_root/n).exists(),"copy_policy":"pointer_only_no_large_table_duplication"} for n in source_names]).to_csv(root/"manifests"/"producer_source_pointers.csv",index=False,encoding="utf-8-sig")
    rows=_rt_level_figures(probe,root)
    for item in (
        _feature_ecdf(probe,root,"go_correct_rt_cv","behavior_rt_cv_ecdf","Correct-Go RT CV","ratio","探针前30秒反应时变异系数的经验累积分布。"),
        _feature_ecdf(probe,root,"go_correct_rt_theilsen_slope_ms_per_s","behavior_rt_slope_ecdf","Theil-Sen RT slope (ms/s)","ms/s","探针前30秒Theil–Sen反应时斜率的经验累积分布。"),
        _coefficient_figure(_read(formal_root,"q1_nominal_models.csv"),root,"Q1"),
        _coefficient_figure(_read(formal_root,"q2_ordinal_gee_models.csv"),root,"Q2"),
        _b1b2_figure(_read(formal_root,"b1_b2_participant_cluster_bootstrap.csv"),root),
        _coverage_figure(handoff,root),
    ):
        if item is not None:rows.append(item)
    figures=pd.DataFrame(rows); figures.to_csv(root/"manifests"/"figure_manifest.csv",index=False,encoding="utf-8-sig")
    producer_manifest={}; p=formal_root/"run_manifest.json"
    if p.exists():
        try:producer_manifest=json.loads(p.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError,OSError):producer_manifest={"status":"unreadable"}
    manifest={
        "pipeline":SCHEMA,"scientific_modality":"behavior",
        "status":"candidate_handoff_ready_feature_freeze_pending" if authoritative else "non_authoritative_subset_or_smoke",
        "authoritative":bool(authoritative),"producer_formal_root":str(formal_root),
        "large_data_policy":"producer tables remain in producer output; science layer stores compact tables, figures and pointers",
        "feature_selection_policy":"Q1/Q2 significance and supervised performance are not freeze criteria","formal_registry_mutated":False,
        "registry_ready_candidate_columns":handoff.loc[handoff["registry_ready"].eq(True),"predictor_column"].astype(str).tolist(),
        "researcher_freeze_pending":handoff.loc[handoff["researcher_freeze_required"].eq(True),"candidate_representation_id"].astype(str).tolist(),
        "rt_level_freeze_pending":True,"figure_n":int(len(figures)),"figure_ids":figures.get("figure_id",pd.Series(dtype=str)).astype(str).tolist(),
        "producer_run_manifest_snapshot":producer_manifest,"producer_provenance":producer_provenance,
    }
    (root/"manifests"/"science_output_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    manifest["science_output_root"]=str(root); manifest["figure_files"]=figures.get("png_path",pd.Series(dtype=str)).astype(str).tolist()+figures.get("svg_path",pd.Series(dtype=str)).astype(str).tolist()
    return manifest
