from __future__ import annotations

import json

import pandas as pd

from attention_pipeline.supervised_learning.comparison_provenance import (
    build_paired_comparison_specs,
    write_paired_comparison_provenance,
)
from attention_pipeline.supervised_learning.feature_registry import (
    RegisteredFeature,
    build_feature_comparison_plan,
)
from attention_pipeline.supervised_learning.reporting import write_supervised_run
from attention_pipeline.supervised_learning.runner import SupervisedRunResult


def _registry() -> list[RegisteredFeature]:
    return [
        RegisteredFeature(
            feature_id="behavior_signal",
            scientific_feature_id="behavior_signal",
            columns=("behavior_signal",),
            role="behavior",
            modality="behavior",
            raw_source="SART behavior",
            source_namespace="behavior",
            required_devices=(),
            behavior_reference_eligible=True,
        ),
        RegisteredFeature(
            feature_id="pupil_level",
            scientific_feature_id="pupil_level",
            columns=("pupil_level",),
            role="sensor",
            modality="ocular",
            feature_type="pupil",
            raw_source="NIR pupil",
            source_namespace="nir",
            required_devices=("nir", "rgb"),
            preprocessing_dependencies=("RGB blink mask",),
            behavior_increment_eligible=True,
            modality_model_eligible=True,
        ),
        RegisteredFeature(
            feature_id="blink_rate",
            scientific_feature_id="blink_rate",
            columns=("blink_rate",),
            role="sensor",
            modality="ocular",
            feature_type="blink",
            raw_source="RGB blink events",
            source_namespace="rgb",
            required_devices=("rgb",),
            preprocessing_dependencies=("RGB eye landmarks",),
            behavior_increment_eligible=True,
            modality_model_eligible=True,
        ),
    ]


def test_modality_pair_uses_actual_added_features_and_preserves_provenance() -> None:
    plan = build_feature_comparison_plan(_registry())
    specs = build_paired_comparison_specs(
        plan,
        ["behavior_reference", "behavior_plus_modality::ocular"],
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec["comparison_type"] == "behavior_modality_increment"
    assert spec["comparison_unit"] == "modality"
    assert spec["comparison_unit_id"] == "ocular"
    assert spec["feature_id"] == "modality::ocular"
    assert spec["defining_feature_ids"] == ["pupil_level", "blink_rate"]
    assert spec["feature_columns"] == ["pupil_level", "blink_rate"]
    assert spec["scientific_modalities"] == ["ocular"]
    assert spec["source_namespaces"] == ["nir", "rgb"]
    assert spec["required_devices"] == ["nir", "rgb"]
    assert spec["preprocessing_dependencies"] == {
        "pupil_level": ["RGB blink mask"],
        "blink_rate": ["RGB eye landmarks"],
    }
    assert spec["baseline_comparison_role"] == "behavior_reference"
    assert spec["added_comparison_role"] == "behavior_plus_modality"
    assert spec["baseline_includes_behavior_reference"] is True
    assert spec["added_includes_behavior_reference"] is True


def _modality_result(*, absent_for: str | None = None) -> SupervisedRunResult:
    plan = build_feature_comparison_plan(_registry())
    spec = build_paired_comparison_specs(
        plan,
        ["behavior_reference", "behavior_plus_modality::ocular"],
    )[0]
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    probabilities = {
        "behavior_reference": [0.6, 0.4],
        "behavior_plus_modality::ocular": [0.9, 0.1],
    }
    for model_id, model_probabilities in probabilities.items():
        for participant, session, q1, probability in zip(
            ["P01", "P02"], ["S01", "S02"], [1, 2], model_probabilities, strict=True
        ):
            rows.append(
                {
                    "run_id": "run-modality",
                    "analysis_set_id": "set-ocular",
                    "membership_type": "included_complete",
                    "participant_group_id": participant,
                    "model_id": model_id,
                    "outer_fold_group": participant,
                    "session_id": session,
                    "block_id": "B1",
                    "probe_event_id": f"{session}|B1|P1",
                    "q1_nominal_4class": q1,
                    "q1_binary": 1 if q1 == 1 else 0,
                    "feature_set_id": model_id,
                    "selected_c": 1.0,
                    "p_q1_equals_1": probability,
                    "predicted_q1_binary": 1 if probability >= 0.5 else 0,
                    "model_failed": False,
                    "failure_reason": "",
                }
            )
            output_columns = ["behavior_signal"]
            dropped_columns: dict[str, str] = {}
            if model_id == "behavior_plus_modality::ocular":
                output_columns.append("pupil_level")
                if participant == absent_for:
                    dropped_columns["blink_rate"] = "zero_variance_in_training"
                else:
                    output_columns.append("blink_rate")
            audits.append(
                {
                    "run_id": "run-modality",
                    "analysis_set_id": "set-ocular",
                    "membership_type": "included_complete",
                    "model_id": model_id,
                    "outer_fold_group": participant,
                    "failed": False,
                    "reason": "",
                    "final_refit": {
                        "preprocessing": {
                            "output_columns": output_columns,
                            "dropped_columns": dropped_columns,
                        }
                    },
                }
            )
    return SupervisedRunResult(
        predictions=pd.DataFrame(rows),
        fold_audits=audits,
        failures=pd.DataFrame(),
        metadata={
            "run_id": "run-modality",
            "analysis_set_id": "set-ocular",
            "membership_type": "included_complete",
            "task": "q1_equals_1_vs_2_3_4",
            "n_input_rows": 2,
            "n_participant_groups": 2,
            "n_models": 2,
            "paired_comparisons": [spec],
        },
    )


def test_modality_component_absence_excludes_fold_instead_of_assigning_zero(tmp_path) -> None:
    result = _modality_result(absent_for="P02")
    manifest = write_supervised_run(result, output_root=tmp_path)
    run_root = tmp_path / "run-modality"

    summary = pd.read_csv(run_root / "paired_model_increments.csv")
    fold = pd.read_csv(run_root / "paired_fold_estimability.csv")
    participant = pd.read_csv(run_root / "paired_participant_increments.csv")

    assert summary.loc[0, "feature_id"] == "modality::ocular"
    assert summary.loc[0, "status"] == "estimable_partial_fold_coverage"
    assert int(summary.loc[0, "n_expected_outer_folds"]) == 2
    assert int(summary.loc[0, "n_estimable_outer_folds"]) == 1
    assert int(summary.loc[0, "n_feature_absent_outer_folds"]) == 1
    p02 = fold.loc[fold["outer_fold_group"].astype(str).eq("P02")].iloc[0]
    assert p02["status"] == "not_estimable_feature_absent"
    assert "blink_rate:zero_variance_in_training" in str(p02["reason"])
    assert participant["participant_group_id"].astype(str).tolist() == ["P01"]

    manifest = write_paired_comparison_provenance(
        output_root=tmp_path,
        run_id="run-modality",
        analysis_set_id="set-ocular",
        membership_type="included_complete",
        paired_comparisons=result.metadata["paired_comparisons"],
        fold_audits=result.fold_audits,
        manifest=manifest,
    )
    artifact = json.loads((run_root / "paired_comparison_provenance.json").read_text(encoding="utf-8"))
    assert artifact["comparisons"][0]["comparison_unit"] == "modality"
    assert artifact["reporting_contract"]["cross_analysis_set_increment_ranking_allowed"] is False
    assert manifest["outer_evaluation"]["cross_analysis_set_increment_ranking_allowed"] is False


def test_provenance_records_failure_side_for_both_models(tmp_path) -> None:
    plan = build_feature_comparison_plan(_registry())
    spec = build_paired_comparison_specs(
        plan,
        ["behavior_reference", "behavior_plus_modality::ocular"],
    )[0]
    run_root = tmp_path / "run-failure"
    run_root.mkdir()
    manifest = {"run_id": "run-failure", "outer_evaluation": {}}
    (run_root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    updated = write_paired_comparison_provenance(
        output_root=tmp_path,
        run_id="run-failure",
        analysis_set_id="set-ocular",
        membership_type="included_complete",
        paired_comparisons=[spec],
        fold_audits=[
            {
                "model_id": "behavior_reference",
                "outer_fold_group": "P01",
                "failed": True,
                "reason": "baseline failed",
            },
            {
                "model_id": "behavior_plus_modality::ocular",
                "outer_fold_group": "P01",
                "failed": False,
                "reason": "",
            },
        ],
        manifest=manifest,
    )
    artifact = json.loads((run_root / "paired_comparison_provenance.json").read_text(encoding="utf-8"))
    comparison = artifact["comparisons"][0]
    assert comparison["baseline_failed_outer_folds"] == ["P01"]
    assert comparison["baseline_failure_reasons"] == {"P01": "baseline failed"}
    assert comparison["added_failed_outer_folds"] == []
    assert updated["paired_comparison_reporting_contract"]["comparison_specific_same_analysis_set_required"] is True
