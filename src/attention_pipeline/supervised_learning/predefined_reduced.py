"""Prespecified nested simplification validation against the frozen full model.

The frozen binary Q1 full model is retained as a separate outer-LOSO benchmark.
Within every outer-training partition, a second procedure chooses one of three
prespecified feature sets (Full, Behavior-core, Behavior-core + Ocular) together
with the regularization value by participant-grouped inner cross-validation.
The held-out participant is never used for candidate or hyperparameter selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from attention_pipeline.config import load_config

from .entrypoint import _require_frozen_runtime_contract
from .feature_registry import (
    RegisteredFeature,
    build_feature_comparison_plan,
    load_registered_features,
)
from .feature_schemes import FeatureScheme, validate_mainline_feature_scheme
from .outcome_scope import validate_task_a_required_outcomes
from .reporting import write_supervised_run
from .runner import SupervisedRunResult, run_nested_loso
from .task import SupervisedLearningContractError
from .time_legality import time_legality_audit


class PredefinedReducedContractError(SupervisedLearningContractError):
    """Raised when the prespecified simplification contract drifts."""


@dataclass(frozen=True)
class PredefinedReducedPlan:
    """Resolved benchmark and nested candidate feature schemes."""

    benchmark_model_id: str
    selected_model_id: str
    benchmark_scheme: FeatureScheme
    candidate_schemes: tuple[FeatureScheme, ...]
    candidate_feature_ids: dict[str, tuple[str, ...]]
    common_core_columns: tuple[str, ...]
    audit: dict[str, object]


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PredefinedReducedContractError(f"configuration must be a mapping: {path}")
    return dict(payload)


def _clean_unique(values: Sequence[object], *, field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise PredefinedReducedContractError(f"{field} must be a non-string sequence")
    cleaned = tuple(str(value).strip() for value in values)
    if not cleaned or any(not value for value in cleaned):
        raise PredefinedReducedContractError(f"{field} must be non-empty and contain no blanks")
    if len(set(cleaned)) != len(cleaned):
        raise PredefinedReducedContractError(f"{field} contains duplicates")
    return cleaned


def _ordered_union(values: Sequence[Sequence[str]]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for sequence in values:
        for value in sequence:
            if value not in seen:
                seen.add(value)
                output.append(value)
    return tuple(output)


def _scheme_from_feature_ids(
    feature_set_id: str,
    feature_ids: Sequence[str],
    *,
    feature_map: Mapping[str, RegisteredFeature],
    description: str,
) -> FeatureScheme:
    ids = _clean_unique(feature_ids, field=f"{feature_set_id}.feature_ids")
    unknown = sorted(set(ids) - set(feature_map))
    if unknown:
        raise PredefinedReducedContractError(
            f"candidate {feature_set_id} references unknown registered feature IDs: {unknown}"
        )
    selected = [feature_map[feature_id] for feature_id in ids]
    ineligible = [feature.feature_id for feature in selected if not feature.full_model_eligible]
    if ineligible:
        raise PredefinedReducedContractError(
            f"candidate {feature_set_id} uses features not eligible for the frozen full comparison: {ineligible}"
        )
    scheme = FeatureScheme(
        feature_set_id=feature_set_id,
        columns=_ordered_union([feature.columns for feature in selected]),
        description=description,
        modalities=_ordered_union([(feature.modality,) for feature in selected]),
        required_devices=_ordered_union([feature.required_devices for feature in selected]),
    )
    validate_mainline_feature_scheme(scheme)
    return scheme


def build_predefined_reduced_plan(
    base_config: Mapping[str, Any],
    design_config: Mapping[str, Any],
) -> PredefinedReducedPlan:
    """Resolve and fail-closed validate the three frozen candidate models."""
    registry_section = base_config.get("feature_registry")
    if not isinstance(registry_section, Mapping):
        raise PredefinedReducedContractError("base config lacks a feature_registry mapping")
    registry = load_registered_features(registry_section)
    plan = build_feature_comparison_plan(registry)
    model_map = plan.model_map()

    input_contract = design_config.get("input_contract")
    model_section = design_config.get("models")
    comparison = design_config.get("comparison")
    if not isinstance(input_contract, Mapping):
        raise PredefinedReducedContractError("input_contract must be a mapping")
    if not isinstance(model_section, Mapping):
        raise PredefinedReducedContractError("models must be a mapping")
    if not isinstance(comparison, Mapping):
        raise PredefinedReducedContractError("comparison must be a mapping")

    frozen_full_model_id = str(input_contract.get("frozen_full_model_id", "")).strip()
    if frozen_full_model_id not in model_map:
        raise PredefinedReducedContractError(
            f"frozen full model {frozen_full_model_id!r} is absent from the base registry plan"
        )
    frozen_full = model_map[frozen_full_model_id]

    benchmark_model_id = str(model_section.get("benchmark_model_id", "")).strip()
    selected_model_id = str(model_section.get("selected_model_id", "")).strip()
    if not benchmark_model_id or not selected_model_id or benchmark_model_id == selected_model_id:
        raise PredefinedReducedContractError(
            "benchmark_model_id and selected_model_id must be distinct nonblank values"
        )

    raw_candidates = model_section.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 3:
        raise PredefinedReducedContractError("exactly three prespecified candidates are required")

    feature_map = {feature.feature_id: feature for feature in registry}
    candidate_schemes: list[FeatureScheme] = []
    candidate_feature_ids: dict[str, tuple[str, ...]] = {}
    labels: list[str] = []
    for raw in raw_candidates:
        if not isinstance(raw, Mapping):
            raise PredefinedReducedContractError("each candidate must be a mapping")
        feature_set_id = str(raw.get("feature_set_id", "")).strip()
        label = str(raw.get("label", "")).strip()
        raw_ids = raw.get("feature_ids")
        if not feature_set_id or not label or not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
            raise PredefinedReducedContractError(
                "each candidate requires feature_set_id, label and a feature_ids sequence"
            )
        ids = _clean_unique(raw_ids, field=f"{feature_set_id}.feature_ids")
        if feature_set_id in candidate_feature_ids:
            raise PredefinedReducedContractError(f"duplicate candidate feature_set_id: {feature_set_id}")
        candidate_feature_ids[feature_set_id] = ids
        labels.append(label)
        candidate_schemes.append(
            _scheme_from_feature_ids(
                feature_set_id,
                ids,
                feature_map=feature_map,
                description=f"Prespecified nested candidate: {label}",
            )
        )
    if len(set(labels)) != len(labels):
        raise PredefinedReducedContractError("candidate labels must be unique")

    full_candidate_ids = candidate_feature_ids[candidate_schemes[0].feature_set_id]
    if tuple(full_candidate_ids) != tuple(frozen_full.feature_ids):
        raise PredefinedReducedContractError(
            "the first candidate must exactly reproduce the current frozen full model feature IDs and order"
        )
    if candidate_schemes[0].columns != frozen_full.columns:
        raise PredefinedReducedContractError(
            "the first candidate columns do not exactly reproduce the frozen full model"
        )

    common_core_ids_raw = comparison.get("common_core_feature_ids")
    if not isinstance(common_core_ids_raw, Sequence) or isinstance(common_core_ids_raw, (str, bytes)):
        raise PredefinedReducedContractError("comparison.common_core_feature_ids must be a sequence")
    common_core_ids = _clean_unique(common_core_ids_raw, field="comparison.common_core_feature_ids")
    for scheme_id, ids in candidate_feature_ids.items():
        missing = sorted(set(common_core_ids) - set(ids))
        if missing:
            raise PredefinedReducedContractError(
                f"candidate {scheme_id} lacks required common-core features: {missing}"
            )
    common_core_columns = _ordered_union([feature_map[feature_id].columns for feature_id in common_core_ids])

    benchmark_scheme = FeatureScheme(
        feature_set_id="frozen_full_benchmark_11",
        columns=frozen_full.columns,
        description="Unchanged confirmatory full-model benchmark",
        modalities=frozen_full.modalities,
        required_devices=frozen_full.required_devices,
    )
    validate_mainline_feature_scheme(benchmark_scheme)

    return PredefinedReducedPlan(
        benchmark_model_id=benchmark_model_id,
        selected_model_id=selected_model_id,
        benchmark_scheme=benchmark_scheme,
        candidate_schemes=tuple(candidate_schemes),
        candidate_feature_ids=candidate_feature_ids,
        common_core_columns=common_core_columns,
        audit={
            "analysis_role": "secondary_nested_simplification_validation",
            "benchmark_unchanged": True,
            "frozen_full_model_id": frozen_full_model_id,
            "benchmark_model_id": benchmark_model_id,
            "selected_model_id": selected_model_id,
            "candidate_order": [scheme.feature_set_id for scheme in candidate_schemes],
            "candidate_feature_ids": {
                key: list(value) for key, value in candidate_feature_ids.items()
            },
            "candidate_columns": {
                scheme.feature_set_id: list(scheme.columns) for scheme in candidate_schemes
            },
            "common_core_columns": list(common_core_columns),
            "selection_boundary": "outer_training_participants_only",
        },
    )


def _read_probe_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise PredefinedReducedContractError(
        f"unsupported supervised input format {path.suffix!r}; use CSV or Parquet"
    )


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_text_value(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns or frame[column].isna().any():
        raise PredefinedReducedContractError(f"input must contain nonmissing {column}")
    values = frame[column].astype(str).str.strip().drop_duplicates().tolist()
    if len(values) != 1 or not values[0]:
        raise PredefinedReducedContractError(
            f"input requires exactly one nonblank {column}; got {values}"
        )
    return values[0]


def _single_json_value(frame: pd.DataFrame, column: str) -> object:
    raw = _single_text_value(frame, column)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PredefinedReducedContractError(f"input {column} is not valid JSON: {exc}") from exc


def validate_input_contract(
    frame: pd.DataFrame,
    input_path: Path,
    design_config: Mapping[str, Any],
    plan: PredefinedReducedPlan,
) -> dict[str, object]:
    """Verify exact reuse of the frozen AS.full probe cohort and feature union."""
    contract = design_config.get("input_contract")
    if not isinstance(contract, Mapping):
        raise PredefinedReducedContractError("input_contract must be a mapping")
    actual_hash = _sha256_path(input_path)
    expected_hash = str(contract.get("expected_sha256", "")).strip().lower()
    if actual_hash.lower() != expected_hash:
        raise PredefinedReducedContractError(
            f"input SHA-256 drift: expected {expected_hash}, got {actual_hash}"
        )
    expected_analysis_set = str(contract.get("analysis_set_id", "")).strip()
    actual_analysis_set = _single_text_value(frame, "analysis_set_id")
    if actual_analysis_set != expected_analysis_set:
        raise PredefinedReducedContractError(
            f"analysis_set_id drift: expected {expected_analysis_set}, got {actual_analysis_set}"
        )
    expected_membership = str(contract.get("membership_type", "")).strip()
    actual_membership = _single_text_value(frame, "membership_type")
    if actual_membership != expected_membership:
        raise PredefinedReducedContractError(
            f"membership_type drift: expected {expected_membership}, got {actual_membership}"
        )
    expected_rows = int(contract.get("expected_rows", -1))
    if len(frame) != expected_rows:
        raise PredefinedReducedContractError(
            f"input row count drift: expected {expected_rows}, got {len(frame)}"
        )
    if "participant_group_id" not in frame.columns or frame["participant_group_id"].isna().any():
        raise PredefinedReducedContractError("input lacks complete participant_group_id")
    n_groups = int(frame["participant_group_id"].astype(str).nunique())
    expected_groups = int(contract.get("expected_participant_groups", -1))
    if n_groups != expected_groups:
        raise PredefinedReducedContractError(
            f"participant count drift: expected {expected_groups}, got {n_groups}"
        )

    declared_models = _single_json_value(frame, "comparison_models")
    if not isinstance(declared_models, list) or str(contract.get("frozen_full_model_id")) not in declared_models:
        raise PredefinedReducedContractError(
            "input comparison_models does not declare the frozen full model"
        )
    required_features = _single_json_value(frame, "required_features")
    if not isinstance(required_features, Mapping):
        raise PredefinedReducedContractError("input required_features must be a JSON mapping")
    required_columns = {
        str(column)
        for columns in required_features.values()
        if isinstance(columns, list)
        for column in columns
    }
    expected_columns = set(plan.benchmark_scheme.columns)
    if required_columns != expected_columns:
        raise PredefinedReducedContractError(
            "input required_features does not equal the frozen full-model predictor union; "
            f"extra={sorted(required_columns - expected_columns)}, "
            f"missing={sorted(expected_columns - required_columns)}"
        )
    validate_task_a_required_outcomes(frame)
    return {
        "input_path": str(input_path),
        "input_sha256": actual_hash,
        "analysis_set_id": actual_analysis_set,
        "membership_type": actual_membership,
        "n_rows": int(len(frame)),
        "n_participant_groups": n_groups,
        "same_frozen_analysis_set_verified": True,
    }


def _selection_tables(result: SupervisedRunResult, plan: PredefinedReducedPlan) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_ids = [scheme.feature_set_id for scheme in plan.candidate_schemes]
    rows: list[dict[str, object]] = []
    for audit in result.fold_audits:
        if str(audit.get("model_id")) != plan.selected_model_id:
            continue
        selection = audit.get("selection")
        if not isinstance(selection, Mapping):
            rows.append(
                {
                    "outer_fold_group": str(audit.get("outer_fold_group", "")),
                    "status": "failed",
                    "selected_feature_set_id": "",
                    "selected_c": None,
                    "reason": str(audit.get("reason", "")),
                }
            )
            continue
        scores = selection.get("candidate_participant_macro_log_loss", {})
        if not isinstance(scores, Mapping):
            raise PredefinedReducedContractError("selection audit lacks candidate score mapping")
        record: dict[str, object] = {
            "outer_fold_group": str(audit.get("outer_fold_group", "")),
            "status": "estimable",
            "selected_feature_set_id": str(selection.get("feature_scheme", {}).get("feature_set_id", ""))
            if isinstance(selection.get("feature_scheme"), Mapping)
            else "",
            "selected_c": selection.get("selected_c"),
            "reason": "",
        }
        for candidate_id in candidate_ids:
            candidate_values = [
                float(value)
                for key, value in scores.items()
                if str(key).startswith(candidate_id + "|C=")
            ]
            record[f"best_inner_log_loss__{candidate_id}"] = (
                min(candidate_values) if candidate_values else None
            )
        rows.append(record)
    fold_table = pd.DataFrame(rows).sort_values("outer_fold_group").reset_index(drop=True)
    successful = fold_table.loc[fold_table["status"].eq("estimable")]
    counts = successful["selected_feature_set_id"].value_counts()
    frequency = pd.DataFrame(
        {
            "feature_set_id": candidate_ids,
            "selected_outer_folds": [int(counts.get(candidate_id, 0)) for candidate_id in candidate_ids],
            "selection_fraction": [
                float(counts.get(candidate_id, 0) / len(successful)) if len(successful) else None
                for candidate_id in candidate_ids
            ],
            "total_estimable_outer_folds": int(len(successful)),
        }
    )
    return fold_table, frequency


def _augment_outputs(
    output_root: Path,
    run_id: str,
    result: SupervisedRunResult,
    plan: PredefinedReducedPlan,
    design_config: Mapping[str, Any],
) -> dict[str, object]:
    run_root = output_root / run_id
    fold_table, frequency = _selection_tables(result, plan)
    fold_path = run_root / "outer_fold_candidate_selection.csv"
    frequency_path = run_root / "candidate_selection_frequency.csv"
    fold_table.to_csv(fold_path, index=False, encoding="utf-8-sig")
    frequency.to_csv(frequency_path, index=False, encoding="utf-8-sig")

    model_scores = pd.read_csv(run_root / "model_evaluation.csv")
    paired_scores = pd.read_csv(run_root / "paired_model_increments.csv")
    paired_bootstrap = json.loads(
        (run_root / "paired_increment_bootstrap.json").read_text(encoding="utf-8")
    )
    benchmark_row = model_scores.loc[model_scores["model_id"].eq(plan.benchmark_model_id)].iloc[0]
    selected_row = model_scores.loc[model_scores["model_id"].eq(plan.selected_model_id)].iloc[0]
    comparison_row = paired_scores.iloc[0]
    bootstrap_row = paired_bootstrap[0]

    contract = design_config["input_contract"]
    expected_benchmark = float(contract["frozen_full_participant_macro_log_loss"])
    tolerance = float(contract["reproduction_absolute_tolerance"])
    actual_benchmark = float(benchmark_row["participant_equal_log_loss"])
    reproduction_difference = actual_benchmark - expected_benchmark
    benchmark_reproduced = abs(reproduction_difference) <= tolerance

    summary = {
        "status": "PASS" if benchmark_reproduced else "PARTIAL",
        "analysis_role": "secondary_nested_simplification_validation",
        "confirmatory_full_model_changed": False,
        "benchmark_model_id": plan.benchmark_model_id,
        "nested_selected_model_id": plan.selected_model_id,
        "n_participants": int(benchmark_row["n_participants"]),
        "n_probes": int(benchmark_row["n_probes"]),
        "frozen_full_log_loss_expected": expected_benchmark,
        "frozen_full_log_loss_reproduced": actual_benchmark,
        "frozen_full_absolute_difference": abs(reproduction_difference),
        "frozen_full_reproduction_tolerance": tolerance,
        "frozen_full_reproduction_pass": benchmark_reproduced,
        "nested_selected_log_loss": float(selected_row["participant_equal_log_loss"]),
        "delta_log_loss_full_minus_nested": float(comparison_row["overall_log_loss_increment"]),
        "delta_log_loss_ci_lower": float(bootstrap_row["ci_lower"]),
        "delta_log_loss_ci_upper": float(bootstrap_row["ci_upper"]),
        "increment_definition": "frozen_full_log_loss_minus_nested_selected_log_loss",
        "positive_increment_interpretation": "nested_selected_model_has_lower_loss",
        "n_estimable_outer_folds": int(comparison_row["n_estimable_outer_folds"]),
        "candidate_selection_frequency": frequency.to_dict(orient="records"),
        "candidate_feature_ids": {
            key: list(value) for key, value in plan.candidate_feature_ids.items()
        },
    }
    summary_path = run_root / "predefined_reduced_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["predefined_reduced_summary"] = summary
    manifest["outputs"]["outer_fold_candidate_selection"] = fold_path.name
    manifest["outputs"]["candidate_selection_frequency"] = frequency_path.name
    manifest["outputs"]["predefined_reduced_summary"] = summary_path.name
    manifest["predefined_reduced_output_sha256"] = {
        fold_path.name: _sha256_path(fold_path),
        frequency_path.name: _sha256_path(frequency_path),
        summary_path.name: _sha256_path(summary_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_predefined_reduced_from_config(
    design_config_path: str | Path,
    *,
    input_table: str | Path,
    output_root: str | Path,
    run_id: str,
    base_config_path: str | Path | None = None,
) -> dict[str, object]:
    """Execute the frozen two-model outer LOSO comparison and persist its audit."""
    design_path = Path(design_config_path).expanduser().resolve()
    design = _load_yaml_mapping(design_path)
    pipeline = design.get("pipeline")
    if not isinstance(pipeline, Mapping):
        raise PredefinedReducedContractError("pipeline must be a mapping")
    if base_config_path is None:
        base_raw = str(pipeline.get("base_supervised_config", "")).strip()
        if not base_raw:
            raise PredefinedReducedContractError("pipeline.base_supervised_config is required")
        base_path = Path(base_raw)
        if not base_path.is_absolute():
            base_path = Path.cwd() / base_path
    else:
        base_path = Path(base_config_path)
    base_path = base_path.expanduser().resolve()
    base_config = load_config(base_path)
    _require_frozen_runtime_contract(base_config.data)
    feature_time_legality = time_legality_audit(base_config.data["feature_registry"])
    plan = build_predefined_reduced_plan(base_config.data, design)

    input_path = Path(input_table).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"supervised input probe table not found: {input_path}")
    frame = _read_probe_table(input_path)
    input_audit = validate_input_contract(frame, input_path, design, plan)

    validation = base_config.section("validation")
    inner = validation.get("inner", {})
    primary_model = base_config.section("models").get("primary", {})
    result = run_nested_loso(
        frame,
        model_feature_schemes={
            plan.benchmark_model_id: [plan.benchmark_scheme],
            plan.selected_model_id: list(plan.candidate_schemes),
        },
        group_col=str(validation.get("outer", {}).get("group_column", "participant_group_id")),
        c_candidates=primary_model.get("C_candidates", (0.01, 0.1, 1.0, 10.0)),
        inner_splits=int(inner.get("n_splits", 5)),
        max_iter=int(primary_model.get("max_iter", 2000)),
        seed=int(pipeline.get("random_seed", 20260910)),
        run_id=str(run_id),
        analysis_set_id=str(input_audit["analysis_set_id"]),
        membership_type=str(input_audit["membership_type"]),
    )
    comparison = design["comparison"]
    result.metadata["analysis_set_required_outcomes"] = list(
        validate_task_a_required_outcomes(frame)
    )
    result.metadata["analysis_set_outcome_scope_verified"] = True
    result.metadata["feature_time_legality"] = feature_time_legality
    result.metadata["time_legality_runtime_verified"] = True
    result.metadata["predefined_reduced_plan"] = plan.audit
    result.metadata["input_contract_audit"] = input_audit
    result.metadata["paired_comparisons"] = [
        {
            "comparison_type": str(comparison["comparison_type"]),
            "feature_id": str(comparison["feature_id"]),
            "feature_columns": list(plan.common_core_columns),
            "baseline_model_id": plan.benchmark_model_id,
            "added_model_id": plan.selected_model_id,
        }
    ]
    result.metadata["paired_comparison_reporting_contract"] = {
        "same_analysis_set_required": True,
        "same_probe_required": True,
        "increment_definition": str(comparison["increment_definition"]),
        "positive_increment_interpretation": str(
            comparison["positive_increment_interpretation"]
        ),
        "selection_boundary": "outer_training_participants_only",
    }

    write_supervised_run(
        result,
        output_root=output_path,
        provenance={
            "input_table": str(input_path),
            "input_sha256": str(input_audit["input_sha256"]),
            "base_config_path": str(base_path),
            "base_config_digest": base_config.digest,
            "design_config_path": str(design_path),
            "design_config_sha256": _sha256_path(design_path),
        },
    )
    return _augment_outputs(output_path, str(run_id), result, plan, design)


__all__ = [
    "PredefinedReducedContractError",
    "PredefinedReducedPlan",
    "build_predefined_reduced_plan",
    "run_predefined_reduced_from_config",
    "validate_input_contract",
]
