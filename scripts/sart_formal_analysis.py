"""FocusWave v3.1.3 final formal SART behavior analysis (two B blocks).

Examples:
    python scripts/sart_formal_analysis.py --stage all
    python scripts/sart_formal_analysis.py --config configs/behavior_formal.yaml --stage stats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from attention_pipeline.config import load_config
from attention_pipeline.behavior_formal import extract as fex
from attention_pipeline.behavior_formal import figures as ffig
from attention_pipeline.behavior_formal import metrics as fmet
from attention_pipeline.behavior_formal import report as frep
from attention_pipeline.behavior_formal import stats as fstat


def _output_root(config) -> Path:
    root = config.path_value("output_root")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load(config):
    subjects = fex.discover_subjects(config)
    trials = fex.load_cohort(config, subjects)
    return subjects, trials


def stage_extract(config, subjects, trials) -> dict:
    root = _output_root(config)
    validation = fex.validate_formal(config, trials)
    trials.to_csv(root / "trials.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(root / "validation.csv", index=False, encoding="utf-8-sig")
    manifest = {"focuswave_release": config.section("pipeline").get("focuswave_release"), "config_digest": config.digest, "subjects": subjects, "n_subjects": len(subjects), "n_trials": int(len(trials)), "expected_blocks": [1, 2]}
    (root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"subjects": len(subjects), "trials": len(trials), "validation_rows": len(validation)}


def stage_metrics(config, trials) -> dict:
    root = _output_root(config)
    blocks = fmet.formal_block_metrics(config, trials)
    bins = fmet.cycle_bin_metrics(config, trials)
    probes = fmet.probe_behaviour_link(config, trials)
    blocks.to_csv(root / "block_metrics.csv", index=False, encoding="utf-8-sig")
    bins.to_csv(root / "cycle_bin_metrics.csv", index=False, encoding="utf-8-sig")
    probes.to_csv(root / "probe_behaviour_link.csv", index=False, encoding="utf-8-sig")
    return {"block_rows": len(blocks), "bin_rows": len(bins), "probe_rows": len(probes)}


def stage_stats(config, trials) -> dict:
    root = _output_root(config)
    blocks = fmet.formal_block_metrics(config, trials)
    bins = fmet.cycle_bin_metrics(config, trials)
    probes = fmet.probe_behaviour_link(config, trials)
    effects = fstat.paired_block_effects(config, blocks)
    effects.to_csv(root / "paired_block_effects.csv", index=False, encoding="utf-8-sig")
    pre_stats = fstat.pre_nogo_stats(fstat.pre_nogo_events(trials))
    pre_stats.to_csv(root / "pre_nogo_stats.csv", index=False, encoding="utf-8-sig")
    model_results = {"rt_block_by_bin": fstat.block_bin_anova(bins, "go_rt_median_ms"), "commission_block_by_bin": fstat.block_bin_anova(bins, "commission_rate"), "rt_drift_mixedlm": fstat.rt_drift_mixedlm(trials), "probe_associations": fstat.probe_associations(probes)}
    (root / "model_results.json").write_text(json.dumps(model_results, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"paired_effects": len(effects), "pre_nogo_rows": len(pre_stats)}


def main() -> None:
    parser = argparse.ArgumentParser(description="FocusWave v3.1.3 final BB formal behavior analysis")
    parser.add_argument("--config", default="configs/behavior_formal.yaml")
    parser.add_argument("--stage", choices=["extract", "metrics", "stats", "figures", "report", "all"], default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    subjects, trials = _load(config)
    stages = ["extract", "metrics", "stats", "figures", "report"] if args.stage == "all" else [args.stage]
    result = {}
    for stage in stages:
        if stage == "extract":
            result[stage] = stage_extract(config, subjects, trials)
        elif stage == "metrics":
            result[stage] = stage_metrics(config, trials)
        elif stage == "stats":
            result[stage] = stage_stats(config, trials)
        elif stage == "figures":
            result[stage] = {"files": ffig.generate_all(config, trials)}
        elif stage == "report":
            result[stage] = frep.generate_report(config, trials)
        print(f"[{stage}] {json.dumps(result[stage], ensure_ascii=False, default=str)}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
