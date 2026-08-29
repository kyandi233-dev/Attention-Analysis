"""Run the missing scientific-completeness layer for pupil-only NIR tables.

Consumes completed 11_analysis_tables products and the optional stimulus visual
table. It does not read raw RITnet production output, does not rerun extraction,
and does not change repeat-participant mapping.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.nir_pipeline_validation.pupil_validation import (
    attach_visual_with_temporal_gate,
    load_analysis_table_cohort,
)
from attention_pipeline.nir_pipeline_validation.scientific_completion import (
    aggregate_probe_visual_exposure,
    decompose_pupil_within_between,
    dynamic_feature_admission_registry,
    fit_visual_adjustment_models,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nir_pipeline_validation.yaml")
    parser.add_argument(
        "--output-dir",
        help="Default: <validation output_root>/scientific_completion",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    tables_root = config.path_value("analysis_tables_root")
    validation_root = config.path_value("output_root")
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else validation_root / "scientific_completion"
    )
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"scientific completion output exists: {output}; use --force")
    output.mkdir(parents=True, exist_ok=True)

    data = load_analysis_table_cohort(tables_root)
    trial_within_between = decompose_pupil_within_between(data["trial_windows"])
    probe_within_between = decompose_pupil_within_between(data["probe_windows"])
    time_within_between = decompose_pupil_within_between(data["time_on_task"])

    visual_path = config.section("paths").get("stimulus_visual_table")
    visual_status = "not_configured"
    visual_audit = pd.DataFrame()
    probe_exposure = pd.DataFrame()
    adjusted = pd.DataFrame()
    model_failures = pd.DataFrame()

    if visual_path not in (None, ""):
        path = Path(str(visual_path)).expanduser()
        if not path.is_absolute():
            path = (config.path.parent.parent / path).resolve()
        if path.is_file():
            visual = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            visual_trial, visual_audit = attach_visual_with_temporal_gate(
                data["trial_windows"], data["trials"], visual
            )
            probe_exposure = aggregate_probe_visual_exposure(
                data["probe_windows"], data["trials"], visual
            )
            adjusted, model_failures = fit_visual_adjustment_models(
                visual_trial, data["trials"]
            )
            visual_status = "complete"
        else:
            visual_status = f"not_estimable_missing_visual_table:{path}"

    trial_within_between.to_csv(
        output / "nir_trial_windows_within_between.csv", index=False, encoding="utf-8-sig"
    )
    probe_within_between.to_csv(
        output / "nir_probe_windows_within_between.csv", index=False, encoding="utf-8-sig"
    )
    time_within_between.to_csv(
        output / "nir_time_on_task_within_between.csv", index=False, encoding="utf-8-sig"
    )
    dynamic_feature_admission_registry().to_csv(
        output / "nir_dynamic_feature_admission.csv", index=False, encoding="utf-8-sig"
    )
    visual_audit.to_csv(
        output / "nir_visual_temporal_audit.csv", index=False, encoding="utf-8-sig"
    )
    probe_exposure.to_csv(
        output / "nir_probe_visual_exposure.csv", index=False, encoding="utf-8-sig"
    )
    adjusted.to_csv(
        output / "nir_visual_adjustment_models.csv", index=False, encoding="utf-8-sig"
    )
    model_failures.to_csv(
        output / "nir_scientific_model_failures.csv", index=False, encoding="utf-8-sig"
    )

    manifest = {
        "pipeline": "nir-pupil-scientific-completion-v1",
        "source_stage": "11_analysis_tables",
        "direct_raw_production_read": False,
        "participant_mapping_adapter_modified": False,
        "within_between_decomposition": True,
        "probe_visual_exposure_strict_pre_event": True,
        "visual_adjustment_status": visual_status,
        "unadjusted_vs_adjusted_models": bool(not adjusted.empty),
        "model_failure_rows": int(len(model_failures)),
        "dynamic_feature_contract_written": True,
        "recovery_feature_admitted": False,
        "frequency_feature_admitted": False,
        "scientific_inference_authorized_by_code_alone": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
