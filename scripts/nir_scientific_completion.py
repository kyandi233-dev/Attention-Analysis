"""Run the scientific-completeness layer for pupil-only NIR formal tables.

Consumes completed ``11_analysis_tables`` products.  Strict pre-probe behavior
and visual exposure are owned by ``nir_formal_analysis.probe_contract``; this
runner reuses that output rather than reimplementing the exposure join.  It
does not read raw RITnet production output and does not modify participant
mapping.
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
    decompose_pupil_within_between,
    dynamic_feature_admission_registry,
    fit_visual_adjustment_models,
)


def _authoritative_probe_exposure(tables_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    root = tables_root / "probe_contract"
    table_path = root / "probe_visual_exposure.csv"
    manifest_path = root / "probe_contract_manifest.json"
    if not table_path.is_file() or not manifest_path.is_file():
        return pd.DataFrame(), {
            "status": "not_estimable",
            "reason": "authoritative probe_contract output missing; rebuild 11_analysis_tables entrypoint",
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    strict = bool(
        manifest.get("status") == "complete"
        and manifest.get("future_information_allowed") is False
        and not manifest.get("anchor_leak_after_correction", False)
    )
    return pd.read_csv(table_path, encoding="utf-8-sig", low_memory=False), {
        "status": "verified" if strict else "blocked",
        "contract_version": manifest.get("contract_version"),
        "future_information_allowed": manifest.get("future_information_allowed"),
        "anchor_leak_after_correction": manifest.get("anchor_leak_after_correction"),
        "source": str(table_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nir_pipeline_validation.yaml")
    parser.add_argument("--output-dir", help="Default: <validation output_root>/scientific_completion")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    tables_root = config.path_value("analysis_tables_root")
    validation_root = config.path_value("output_root")
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else validation_root / "scientific_completion"
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"scientific completion output exists: {output}; use --force")
    output.mkdir(parents=True, exist_ok=True)

    data = load_analysis_table_cohort(tables_root)
    trial_within_between = decompose_pupil_within_between(data["trial_windows"])
    probe_within_between = decompose_pupil_within_between(data["probe_windows"])
    time_within_between = decompose_pupil_within_between(data["time_on_task"])
    probe_exposure, probe_contract = _authoritative_probe_exposure(tables_root)

    visual_path = config.section("paths").get("stimulus_visual_table")
    visual_status = "not_configured"
    visual_audit = pd.DataFrame()
    adjusted = pd.DataFrame()
    model_failures = pd.DataFrame()
    if visual_path not in (None, ""):
        path = Path(str(visual_path)).expanduser()
        if not path.is_absolute():
            path = (config.path.parent.parent / path).resolve()
        if path.is_file():
            visual = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            try:
                visual_trial, visual_audit = attach_visual_with_temporal_gate(
                    data["trial_windows"], data["trials"], visual
                )
                adjusted, model_failures = fit_visual_adjustment_models(
                    visual_trial, data["trials"]
                )
                visual_status = "complete"
            except Exception as exc:
                visual_status = f"not_estimable:{type(exc).__name__}:{exc}"
                model_failures = pd.DataFrame(
                    [{
                        "analysis_question": "visual_confound_adjustment",
                        "target": "all",
                        "model_stage": "join_or_model",
                        "fold": pd.NA,
                        "model_name": "GEE_Gaussian_exchangeable",
                        "status": "not_estimable",
                        "failure_type": type(exc).__name__,
                        "failure_detail": str(exc),
                    }]
                )
        else:
            visual_status = f"not_estimable_missing_visual_table:{path}"

    outputs = {
        "nir_trial_windows_within_between.csv": trial_within_between,
        "nir_probe_windows_within_between.csv": probe_within_between,
        "nir_time_on_task_within_between.csv": time_within_between,
        "nir_dynamic_feature_admission.csv": dynamic_feature_admission_registry(),
        "nir_visual_temporal_audit.csv": visual_audit,
        "nir_probe_visual_exposure.csv": probe_exposure,
        "nir_visual_adjustment_models.csv": adjusted,
        "nir_scientific_model_failures.csv": model_failures,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False, encoding="utf-8-sig")

    manifest = {
        "pipeline": "nir-pupil-scientific-completion-v2",
        "source_stage": "11_analysis_tables",
        "direct_raw_production_read": False,
        "participant_mapping_adapter_modified": False,
        "within_between_decomposition": True,
        "probe_visual_exposure_source": "nir_formal_analysis.probe_contract",
        "probe_contract": probe_contract,
        "probe_visual_exposure_strict_pre_event_verified": probe_contract.get("status") == "verified",
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
    return 0 if probe_contract.get("status") != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
