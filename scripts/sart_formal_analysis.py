"""正式 BBB SART 行为分析总控。

用法（主环境，PYTHONPATH=src）：
    python scripts/sart_formal_analysis.py --config configs/sart_formal.yaml --stage extract
    python scripts/sart_formal_analysis.py --config configs/sart_formal.yaml --stage metrics
    python scripts/sart_formal_analysis.py --config configs/sart_formal.yaml --stage stats
    python scripts/sart_formal_analysis.py --config configs/sart_formal.yaml --stage figures
    python scripts/sart_formal_analysis.py --config configs/sart_formal.yaml --stage report
    python scripts/sart_formal_analysis.py --config configs/sart_formal.yaml --stage all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.metadata import run_metadata, source_id
from attention_pipeline.behavior_formal import extract as fex
from attention_pipeline.behavior_formal import metrics as fmet
from attention_pipeline.behavior_formal import stats as fstat


def _out_paths(config):
    root = config.path_value("output_root")
    reports = root / "000-reports"
    behavior = root / "040-behavior"
    manifests = root / "090-manifests"
    for d in (root, reports, behavior, manifests):
        d.mkdir(parents=True, exist_ok=True)
    return root, reports, behavior, manifests


def _primary_subjects(config):
    return [s for s in config.section("subjects")["include"] if s not in ("sub-015",)]


def stage_extract(config) -> dict:
    subjects = config.section("subjects")["include"]
    trials = fex.load_cohort(config, subjects)
    rep = fex.validate_formal(config, trials)
    root, reports, behavior, manifests = _out_paths(config)
    trials.to_csv(behavior / "051-trials.csv", index=False, encoding="utf-8-sig")
    rep.to_csv(behavior / "051-validation.csv", index=False, encoding="utf-8-sig")
    # 逐被试审计表
    audit = (trials.groupby(["subject", "block_num"], as_index=False)
             .agg(trials=("trial_num", "size"),
                  nogo=("is_no_go", "sum"),
                  probes=("is_probe", "sum"),
                  rt_available=("rt", lambda s: int(s.notna().sum())),
                  prestimulus=("prestimulus_press_ms", lambda s: int(s.fillna("").astype(str).str.len().gt(0).sum())),
                  timestamp_qc_fail=("rt_qc_timestamp_inconsistent", "sum")))
    audit.to_csv(behavior / "051-subject_block_audit.csv", index=False, encoding="utf-8-sig")
    sources = [config.path] + [s for sub in subjects for s in fex.formal_block_files(config, sub)]
    manifest = run_metadata(config, sources)
    manifest.update({
        "stage": "extract",
        "subjects": len(subjects),
        "trials": int(len(trials)),
        "nogo_total": int(trials["is_no_go"].sum()),
        "probe_total": int(trials["is_probe"].sum()),
        "rt_rows_deleted": 0,
        "excluded_subjects": config.section("subjects")["exclude"],
    })
    (manifests / "051-extract-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"trials": int(len(trials)), "validation": len(rep), "audit": len(audit)}


def stage_metrics(config, trials: pd.DataFrame) -> dict:
    primary = _primary_subjects(config)
    blocks = fmet.formal_block_metrics(config, trials)
    bins = fmet.cycle_bin_metrics(config, trials)
    rolling = fmet.rolling_evidence_formal(config, trials)
    probe = fmet.probe_evidence_formal(config, trials)
    link = fmet.probe_behaviour_link(trials)
    root, reports, behavior, manifests = _out_paths(config)
    blocks.to_csv(behavior / "051-block_metrics.csv", index=False, encoding="utf-8-sig")
    bins.to_csv(behavior / "051-cycle_bin_metrics.csv", index=False, encoding="utf-8-sig")
    rolling.to_csv(behavior / "051-rolling_evidence.csv", index=False, encoding="utf-8-sig")
    probe.to_csv(behavior / "051-probe_evidence.csv", index=False, encoding="utf-8-sig")
    link.to_csv(behavior / "051-probe_behaviour_link.csv", index=False, encoding="utf-8-sig")
    return {"blocks": len(blocks), "bins": len(bins), "rolling": len(rolling), "probe": len(probe), "link": len(link)}


def stage_stats(config, trials: pd.DataFrame) -> dict:
    primary = _primary_subjects(config)
    root, reports, behavior, manifests = _out_paths(config)
    blocks = fmet.formal_block_metrics(config, trials)
    bins = fmet.cycle_bin_metrics(config, trials)
    link = fmet.probe_behaviour_link(trials)
    # 主效应（主队列 n=19）
    blocks_p = blocks.loc[blocks["subject"].isin(primary)]
    main_df, main_details = fstat.main_effects(config, blocks_p)
    # 敏感性（n=20）
    main_sens, _ = fstat.main_effects(config, blocks)
    main_df["cohort"] = "primary_n19"
    main_sens["cohort"] = "sensitivity_n20"
    main_all = pd.concat([main_df, main_sens], ignore_index=True)
    main_all.to_csv(behavior / "051-main_effects.csv", index=False, encoding="utf-8-sig")
    # 交互
    bins_p = bins.loc[bins["subject"].isin(primary)]
    inter = fstat.interaction_analysis(config, bins_p, trials)
    (manifests / "051-interaction.json").write_text(json.dumps(inter, ensure_ascii=False, indent=2), encoding="utf-8")
    # 回归
    drift = fstat.regression_rt_drift(trials.loc[trials["subject"].isin(primary)])
    drift.to_csv(behavior / "051-rt_drift_mixedlm.csv", index=False, encoding="utf-8-sig")
    events = fstat.pre_nogo_events(trials.loc[trials["subject"].isin(primary)])
    prenogo = fstat.pre_nogo_stats(events)
    prenogo.to_csv(behavior / "051-pre_nogo_stats.csv", index=False, encoding="utf-8-sig")
    gee = fstat.commission_gee(trials.loc[trials["subject"].isin(primary)])
    (manifests / "051-commission_gee.json").write_text(json.dumps(gee, ensure_ascii=False, indent=2), encoding="utf-8")
    # 探针关联
    pa = fstat.probe_association(link.loc[link["subject"].isin(primary)])
    probe_parts = {k: (v.to_csv(behavior / f"051-probe_assoc_{k}.csv", index=False, encoding="utf-8-sig")
                       if isinstance(v, pd.DataFrame) else v) for k, v in pa.items()}
    (manifests / "051-probe_association.json").write_text(
        json.dumps({k: v for k, v in pa.items() if not isinstance(v, pd.DataFrame)}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    # 相关
    corr = fstat.correlation_analysis(config, blocks_p)
    corr["corr_matrix"].to_csv(behavior / "051-correlation_matrix.csv", encoding="utf-8-sig")
    corr["cross_block"].to_csv(behavior / "051-cross_block_consistency.csv", index=False, encoding="utf-8-sig")
    (manifests / "051-correlation.json").write_text(
        json.dumps({k: v for k, v in corr.items() if k != "corr_matrix" and k != "cross_block"},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"main_rows": len(main_all), "drift": len(drift), "prenogo": len(prenogo)}


def main() -> None:
    ap = argparse.ArgumentParser(description="正式 BBB SART 行为分析")
    ap.add_argument("--config", default="configs/sart_formal.yaml")
    ap.add_argument("--stage", choices=["extract", "metrics", "stats", "figures", "report", "all"], default="all")
    args = ap.parse_args()
    config = load_config(args.config)
    stages = ["extract", "metrics", "stats", "figures", "report"] if args.stage == "all" else [args.stage]
    result = {}
    trials = None
    for stage in stages:
        if stage == "extract":
            result["extract"] = stage_extract(config)
            trials = fex.load_cohort(config, config.section("subjects")["include"])
        elif stage == "metrics":
            if trials is None:
                trials = fex.load_cohort(config, config.section("subjects")["include"])
            result["metrics"] = stage_metrics(config, trials)
        elif stage == "stats":
            if trials is None:
                trials = fex.load_cohort(config, config.section("subjects")["include"])
            result["stats"] = stage_stats(config, trials)
        elif stage == "figures":
            from attention_pipeline.behavior_formal import figures as ffig
            if trials is None:
                trials = fex.load_cohort(config, config.section("subjects")["include"])
            result["figures"] = ffig.generate_all(config, trials)
        elif stage == "report":
            from attention_pipeline.behavior_formal import report as frep
            if trials is None:
                trials = fex.load_cohort(config, config.section("subjects")["include"])
            result["report"] = frep.generate_report(config, trials)
        print(f"[{stage}] {json.dumps(result[stage], ensure_ascii=False, default=str)}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
