"""Path-neutral FocusWave v3.1.3 BB extraction/metric entry for formal v2."""

from __future__ import annotations

import argparse
import json

from attention_pipeline.config import load_config
from attention_pipeline.behavior_formal import extract as fex
from attention_pipeline.behavior_formal import metrics as fmet
from attention_pipeline.formal_analysis.behavior_adapter import (
    assert_behavior_inference_allowed,
    attach_behavior_groups,
    prepare_behavior_runtime_config,
)
from attention_pipeline.formal_analysis.cohort import summarize_cohort


def main() -> int:
    parser = argparse.ArgumentParser(description="FocusWave formal BB v2 portable extraction and metric materialization")
    parser.add_argument("--config", default="configs/behavior_formal_v2.yaml")
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--stage", choices=["preflight", "extract", "metrics", "stats"], default="preflight")
    args = parser.parse_args()

    science = load_config(args.config, paths_config=args.paths_config)
    runtime, cohort = prepare_behavior_runtime_config(science)
    summary = summarize_cohort(cohort)
    if args.stage == "preflight":
        print(json.dumps({
            "science_config_digest": science.digest,
            "paths_config_digest": science.path_registry.digest if science.path_registry else None,
            "sessions": summary.sessions,
            "groups": summary.groups,
            "repeated_groups": summary.repeated_groups,
            "repeated_sessions": summary.repeated_sessions,
            "data_roots": [str(path) for path in runtime.section("data")["roots"]],
            "output_root": str(runtime.path_value("output_root")),
        }, ensure_ascii=False, indent=2))
        return 0

    subjects = fex.discover_subjects(runtime)
    trials = fex.load_cohort(runtime, subjects)
    trials = attach_behavior_groups(trials, cohort, require_all=True)
    output_root = runtime.path_value("output_root")
    output_root.mkdir(parents=True, exist_ok=True)

    if args.stage == "extract":
        validation = fex.validate_formal(runtime, trials)
        trials.to_csv(output_root / "trials_v2.csv", index=False, encoding="utf-8-sig")
        validation.to_csv(output_root / "validation_v2.csv", index=False, encoding="utf-8-sig")
        result = {
            "sessions": int(trials["session_id"].nunique()),
            "groups": int(trials["repeat_participant_id"].nunique()),
            "trials": int(len(trials)),
            "validation_rows": int(len(validation)),
        }
    elif args.stage == "metrics":
        blocks = fmet.formal_block_metrics(runtime, trials)
        bins = fmet.cycle_bin_metrics(runtime, trials)
        probes = fmet.probe_behaviour_link(runtime, trials)
        mapping = trials[["subject", "session_id", "repeat_participant_id"]].drop_duplicates()
        for name, table in (
            ("block_metrics_v2.csv", blocks),
            ("cycle_bin_metrics_v2.csv", bins),
            ("probe_behaviour_link_v2.csv", probes),
        ):
            enriched = table.merge(mapping, on="subject", how="left", validate="many_to_one")
            enriched.to_csv(output_root / name, index=False, encoding="utf-8-sig")
        result = {"block_rows": int(len(blocks)), "cycle_bin_rows": int(len(bins)), "probe_rows": int(len(probes))}
    elif args.stage == "stats":
        assert_behavior_inference_allowed(science, trials)
        raise AssertionError("unreachable: legacy session-level stats must remain blocked")
    else:
        raise AssertionError(args.stage)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
