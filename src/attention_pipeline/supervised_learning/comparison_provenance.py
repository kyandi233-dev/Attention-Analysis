"""Build and archive scientific provenance for paired supervised comparisons."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .feature_registry import FeatureComparisonPlan, FeatureRegistryContractError


PROVENANCE_SCHEMA_VERSION = "1.16.10-v1"


def paired_comparison_reporting_contract() -> dict[str, object]:
    return {
        "comparison_specific_same_analysis_set_required": True,
        "cross_analysis_set_increment_ranking_allowed": False,
        "defining_predictor_absence_rule": (
            "If any predictor defining the added feature/modality is absent from the added/full model "
            "after outer-train preprocessing, that outer fold is not an estimable increment and is "
            "excluded rather than assigned zero contribution."
        ),
        "model_failure_rule": (
            "A failed baseline or added/full OOF model prevents a valid direct paired estimate for the "
            "affected comparison; model-side failures remain explicit in fold/provenance audit."
        ),
        "device_interpretation_rule": (
            "required_devices and source_namespace are provenance/deployment attributes and do not define "
            "scientific modality membership."
        ),
    }


def _unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _comparison_spec(
    plan: FeatureComparisonPlan,
    *,
    comparison_type: str,
    comparison_unit: str,
    comparison_unit_id: str,
    baseline_model_id: str,
    added_model_id: str,
) -> dict[str, object]:
    model_map = plan.model_map()
    feature_map = plan.feature_map()
    if baseline_model_id not in model_map or added_model_id not in model_map:
        raise FeatureRegistryContractError(
            f"paired comparison references unknown model(s): {baseline_model_id}, {added_model_id}"
        )
    baseline = model_map[baseline_model_id]
    added = model_map[added_model_id]
    baseline_feature_ids = set(baseline.feature_ids)
    if not baseline_feature_ids.issubset(set(added.feature_ids)):
        raise FeatureRegistryContractError(
            f"paired comparison must be nested for defining-feature audit: {baseline_model_id} -> {added_model_id}"
        )
    defining_feature_ids = [
        feature_id for feature_id in added.feature_ids if feature_id not in baseline_feature_ids
    ]
    if not defining_feature_ids:
        raise FeatureRegistryContractError(
            f"paired comparison adds no registered feature: {baseline_model_id} -> {added_model_id}"
        )
    defining_features = [feature_map[feature_id] for feature_id in defining_feature_ids]

    if comparison_unit == "feature":
        if defining_feature_ids != [comparison_unit_id]:
            raise FeatureRegistryContractError(
                f"feature comparison {comparison_unit_id} does not match actual added feature IDs "
                f"{defining_feature_ids}"
            )
    elif comparison_unit == "modality":
        wrong = [
            feature.feature_id
            for feature in defining_features
            if feature.modality != comparison_unit_id
        ]
        if wrong:
            raise FeatureRegistryContractError(
                f"modality comparison {comparison_unit_id} adds features from other modalities: {wrong}"
            )
    else:
        raise FeatureRegistryContractError(
            f"unknown paired comparison unit {comparison_unit!r}"
        )

    defining_columns = _unique(
        [column for feature in defining_features for column in feature.columns]
    )
    required_devices = sorted(
        {device for feature in defining_features for device in feature.required_devices}
    )
    source_namespaces = sorted({feature.source_namespace for feature in defining_features})
    scientific_modalities = sorted({feature.modality for feature in defining_features})
    scientific_feature_ids = _unique(
        [feature.scientific_feature_id for feature in defining_features]
    )
    dependencies = {
        feature.feature_id: list(feature.preprocessing_dependencies)
        for feature in defining_features
    }

    # Legacy reporting fields are intentionally retained so the existing fixed-OOF
    # evaluator can consume modality comparisons without a second statistics path.
    legacy_feature_id = (
        comparison_unit_id
        if comparison_unit == "feature"
        else f"modality::{comparison_unit_id}"
    )
    return {
        "comparison_type": comparison_type,
        "comparison_unit": comparison_unit,
        "comparison_unit_id": comparison_unit_id,
        "baseline_model_id": baseline_model_id,
        "added_model_id": added_model_id,
        "feature_id": legacy_feature_id,
        "feature_columns": defining_columns,
        "defining_feature_ids": defining_feature_ids,
        "defining_scientific_feature_ids": scientific_feature_ids,
        "defining_columns": defining_columns,
        "scientific_modalities": scientific_modalities,
        "source_namespaces": source_namespaces,
        "required_devices": required_devices,
        "preprocessing_dependencies": dependencies,
        "baseline_comparison_role": baseline.comparison_role,
        "added_comparison_role": added.comparison_role,
        "baseline_modalities": list(baseline.modalities),
        "added_modalities": list(added.modalities),
        "baseline_required_devices": list(baseline.required_devices),
        "added_required_devices": list(added.required_devices),
        "baseline_includes_behavior_reference": baseline.includes_behavior_reference,
        "added_includes_behavior_reference": added.includes_behavior_reference,
        "feature_provenance": [feature.audit_dict() for feature in defining_features],
    }


def build_paired_comparison_specs(
    plan: FeatureComparisonPlan,
    selected_model_ids: Sequence[str],
) -> list[dict[str, object]]:
    """Build auditable feature- and modality-level pairs for one analysis set."""
    selected = set(str(model_id) for model_id in selected_model_ids)
    specs: list[dict[str, object]] = []

    for baseline, added, feature_id in plan.behavior_increment_pairs:
        if baseline in selected and added in selected:
            specs.append(
                _comparison_spec(
                    plan,
                    comparison_type="behavior_increment",
                    comparison_unit="feature",
                    comparison_unit_id=feature_id,
                    baseline_model_id=baseline,
                    added_model_id=added,
                )
            )
    for reduced, full, feature_id in plan.full_leave_one_out_pairs:
        if reduced in selected and full in selected:
            specs.append(
                _comparison_spec(
                    plan,
                    comparison_type="full_leave_one_out",
                    comparison_unit="feature",
                    comparison_unit_id=feature_id,
                    baseline_model_id=reduced,
                    added_model_id=full,
                )
            )
    for baseline, added, modality in plan.modality_increment_pairs:
        if baseline in selected and added in selected:
            specs.append(
                _comparison_spec(
                    plan,
                    comparison_type="behavior_modality_increment",
                    comparison_unit="modality",
                    comparison_unit_id=modality,
                    baseline_model_id=baseline,
                    added_model_id=added,
                )
            )
    for reduced, full, modality in plan.full_leave_one_modality_out_pairs:
        if reduced in selected and full in selected:
            specs.append(
                _comparison_spec(
                    plan,
                    comparison_type="full_leave_one_modality_out",
                    comparison_unit="modality",
                    comparison_unit_id=modality,
                    baseline_model_id=reduced,
                    added_model_id=full,
                )
            )
    return specs


def _fold_failure_index(fold_audits: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    failures: dict[tuple[str, str], str] = {}
    for audit in fold_audits:
        if not bool(audit.get("failed", False)):
            continue
        model_id = str(audit.get("model_id", "")).strip()
        outer_group = str(audit.get("outer_fold_group", "")).strip()
        if model_id and outer_group:
            failures[(model_id, outer_group)] = str(
                audit.get("reason", "model failed before paired-comparison audit")
            )
    return failures


def write_paired_comparison_provenance(
    *,
    output_root: str | Path,
    run_id: str,
    analysis_set_id: str,
    membership_type: str,
    paired_comparisons: Sequence[Mapping[str, Any]],
    fold_audits: Sequence[Mapping[str, Any]],
    manifest: dict[str, object],
) -> dict[str, object]:
    """Write an explicit audit artifact and extend the already-written run manifest."""
    run_root = Path(output_root) / str(run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    failures = _fold_failure_index(fold_audits)
    enriched: list[dict[str, object]] = []
    all_groups = sorted(
        {
            str(audit.get("outer_fold_group", "")).strip()
            for audit in fold_audits
            if str(audit.get("outer_fold_group", "")).strip()
        }
    )
    for raw in paired_comparisons:
        spec = dict(raw)
        baseline = str(spec["baseline_model_id"])
        added = str(spec["added_model_id"])
        baseline_failed = [group for group in all_groups if (baseline, group) in failures]
        added_failed = [group for group in all_groups if (added, group) in failures]
        spec["baseline_failed_outer_folds"] = baseline_failed
        spec["added_failed_outer_folds"] = added_failed
        spec["baseline_failure_reasons"] = {
            group: failures[(baseline, group)] for group in baseline_failed
        }
        spec["added_failure_reasons"] = {
            group: failures[(added, group)] for group in added_failed
        }
        enriched.append(spec)

    contract = paired_comparison_reporting_contract()
    artifact_name = "paired_comparison_provenance.json"
    payload = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "analysis_set_id": str(analysis_set_id),
        "membership_type": str(membership_type),
        "reporting_contract": contract,
        "comparisons": enriched,
    }
    (run_root / artifact_name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest["paired_comparison_provenance"] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "artifact": artifact_name,
        "n_comparisons": len(enriched),
    }
    manifest["paired_comparison_reporting_contract"] = contract
    outer = manifest.setdefault("outer_evaluation", {})
    if isinstance(outer, dict):
        outer["paired_modality_absence_rule"] = contract["defining_predictor_absence_rule"]
        outer["paired_model_failure_rule"] = contract["model_failure_rule"]
        outer["cross_analysis_set_increment_ranking_allowed"] = False
        outer["comparison_specific_same_analysis_set_required"] = True

    (run_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
