"""Configuration-to-run entrypoint for the Task A supervised-learning core."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from attention_pipeline.config import load_config

from .evaluation import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
)
from .feature_registry import FeatureComparisonPlan, build_feature_comparison_plan, load_registered_features
from .feature_schemes import FeatureScheme, load_feature_schemes
from .models import SELECTION_METRIC
from .reporting import write_supervised_run
from .runner import run_nested_loso
from .task import Q1_BINARY_SPEC, SupervisedLearningContractError


FORMAL_PARTICIPANT_GROUP_COLUMN = "participant_group_id"
FORMAL_ANALYSIS_SET_COLUMN = "analysis_set_id"
FORMAL_COMPARISON_MODELS_COLUMN = "comparison_models"
FORMAL_INNER_SPLITS = 5


def _require_frozen_runtime_contract(config_data: Mapping[str, Any]) -> None:
    task = config_data.get("task", {})
    expected_task = {
        "name": Q1_BINARY_SPEC.name,
        "analysis_unit": "probe_preceding_window",
        "source_column": Q1_BINARY_SPEC.source_column,
        "positive_values": [1],
        "negative_values": [2, 3, 4],
        "positive_label": 1,
        "negative_label": 0,
        "positive_probability_name": Q1_BINARY_SPEC.positive_probability_name,
        "primary_window_seconds": 30,
    }
    for key, expected in expected_task.items():
        if task.get(key) != expected:
            raise SupervisedLearningContractError(
                f"task.{key}={task.get(key)!r} conflicts with frozen Task A value {expected!r}"
            )

    validation = config_data.get("validation", {})
    outer = validation.get("outer", {})
    inner = validation.get("inner", {})
    if outer.get("method") != "leave_one_participant_out" or outer.get("participant_disjoint") is not True:
        raise SupervisedLearningContractError("Task A outer validation must remain participant-disjoint LOSO")
    if inner.get("method") != "grouped_k_fold" or inner.get("refit_preprocessing_per_split") is not True:
        raise SupervisedLearningContractError("Task A inner validation must refit preprocessing within each grouped split")
    if int(inner.get("n_splits", -1)) != FORMAL_INNER_SPLITS:
        raise SupervisedLearningContractError(
            f"formal Task A inner validation must use exactly {FORMAL_INNER_SPLITS} participant-grouped folds"
        )
    outer_group = str(outer.get("group_column", ""))
    inner_group = str(inner.get("group_column", ""))
    if outer_group != FORMAL_PARTICIPANT_GROUP_COLUMN or inner_group != outer_group:
        raise SupervisedLearningContractError(
            "Task A inner and outer validation must use the same participant grouping column: participant_group_id"
        )
    if validation.get("zero_individual_calibration") is not True:
        raise SupervisedLearningContractError("Task A mainline requires zero individual calibration")
    if validation.get("forbid_test_participant_sequence_statistics") is not True:
        raise SupervisedLearningContractError("test-participant sequence statistics must remain forbidden")
    if validation.get("forbid_test_participant_future_information") is not True:
        raise SupervisedLearningContractError("test-participant future information must remain forbidden")
    if validation.get("require_analysis_set_id") is not True:
        raise SupervisedLearningContractError("formal Task A runs must require analysis_set_id")

    legacy_prediction_folds = (
        config_data.get("prediction_folds"),
        validation.get("prediction_folds"),
        config_data.get("science", {}).get("prediction_folds")
        if isinstance(config_data.get("science", {}), Mapping)
        else None,
    )
    if any(value is not None for value in legacy_prediction_folds):
        raise SupervisedLearningContractError(
            "standalone prediction_folds is deprecated; formal Task A uses outer LOSO plus inner GroupKFold only"
        )

    preprocessing = config_data.get("preprocessing", {})
    required_training_only = (
        "participant_equal_weighted_median_imputation_fit_on_training_only",
        "participant_equal_standardization_fit_on_training_only",
        "data_dependent_column_handling_fit_on_training_only",
        "participant_equal_training_weights_normalized_to_mean_one",
    )
    for key in required_training_only:
        if preprocessing.get(key) is not True:
            raise SupervisedLearningContractError(f"preprocessing.{key} must remain true for Task A")
    if preprocessing.get("unified_global_coverage_cutoff") is not None:
        raise SupervisedLearningContractError("Task A forbids a unified global coverage cutoff")
    if preprocessing.get("participant_specific_within_between_mainline") is not False:
        raise SupervisedLearningContractError(
            "participant-specific within/between decomposition is disabled in the zero-calibration mainline"
        )

    models = config_data.get("models", {})
    primary = models.get("primary", {})
    if primary.get("kind") != "logistic_l2":
        raise SupervisedLearningContractError("Task A primary model must remain L2 logistic regression")
    if models.get("selection_metric") != SELECTION_METRIC:
        raise SupervisedLearningContractError(
            f"Task A candidate selection metric must be {SELECTION_METRIC}"
        )

    uncertainty = config_data.get("uncertainty", {})
    participant_bootstrap = uncertainty.get("participant_cluster_bootstrap", {})
    expected_bootstrap = {
        "method": "fixed_oof_participant_cluster_percentile",
        "replicates": DEFAULT_BOOTSTRAP_REPLICATES,
        "seed": DEFAULT_BOOTSTRAP_SEED,
        "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
        "paired_model_resampling": True,
        "retrain_within_bootstrap": False,
    }
    for key, expected in expected_bootstrap.items():
        if participant_bootstrap.get(key) != expected:
            raise SupervisedLearningContractError(
                f"uncertainty.participant_cluster_bootstrap.{key}={participant_bootstrap.get(key)!r} "
                f"conflicts with frozen D10 value {expected!r}"
            )


def _load_feature_families(section: Mapping[str, Any]) -> dict[str, list[FeatureScheme]]:
    """Compatibility loader for manually declared model families."""
    raw_families = section.get("model_families")
    if raw_families is None:
        schemes = load_feature_schemes(section)
        return {"primary": schemes} if schemes else {}
    if not isinstance(raw_families, Mapping):
        raise SupervisedLearningContractError("feature_schemes.model_families must be a mapping")

    families: dict[str, list[FeatureScheme]] = {}
    seen_feature_ids: set[str] = set()
    for raw_name, family_section in raw_families.items():
        name = str(raw_name).strip()
        if not name:
            raise SupervisedLearningContractError("model family name must be non-empty")
        if not isinstance(family_section, Mapping):
            raise SupervisedLearningContractError(f"model family {name} must be a mapping")
        schemes = load_feature_schemes(family_section)
        if not schemes:
            raise SupervisedLearningContractError(f"model family {name} has no candidate feature schemes")
        for scheme in schemes:
            if not scheme.modality_blocks:
                raise SupervisedLearningContractError(
                    f"model family {name} candidate {scheme.feature_set_id} must declare modality_blocks"
                )
        expected_blocks = frozenset(schemes[0].modality_blocks)
        for scheme in schemes[1:]:
            if frozenset(scheme.modality_blocks) != expected_blocks:
                raise SupervisedLearningContractError(
                    f"all candidates within model family {name} must use the same modality_blocks"
                )
        ids = {scheme.feature_set_id for scheme in schemes}
        overlap = seen_feature_ids & ids
        if overlap:
            raise SupervisedLearningContractError(
                f"feature_set_id must be unique across model families: {sorted(overlap)}"
            )
        seen_feature_ids.update(ids)
        families[name] = schemes
    return families


def _resolve_model_plan(config_data: Mapping[str, Any]) -> tuple[dict[str, list[FeatureScheme]], FeatureComparisonPlan | None]:
    """Prefer the frozen provenance registry; retain manual families for compatibility/tests."""
    registry_section = config_data.get("feature_registry", {})
    if registry_section is None:
        registry_section = {}
    if not isinstance(registry_section, Mapping):
        raise SupervisedLearningContractError("feature_registry must be a mapping")
    registry_entries = registry_section.get("features", [])
    if registry_entries:
        registry = load_registered_features(registry_section)
        plan = build_feature_comparison_plan(registry)
        return plan.to_runner_feature_schemes(), plan

    feature_section = config_data.get("feature_schemes", {})
    if not isinstance(feature_section, Mapping):
        raise SupervisedLearningContractError("feature_schemes must be a mapping")
    return _load_feature_families(feature_section), None


def _read_probe_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise SupervisedLearningContractError(
        f"unsupported supervised input table format {suffix!r}; use CSV or Parquet"
    )


def _require_single_analysis_set_id(frame: pd.DataFrame) -> str:
    if FORMAL_ANALYSIS_SET_COLUMN not in frame.columns:
        raise SupervisedLearningContractError(
            "formal supervised input must contain analysis_set_id supplied by the B-layer analysis-set contract"
        )
    raw = frame[FORMAL_ANALYSIS_SET_COLUMN]
    if raw.isna().any():
        raise SupervisedLearningContractError("analysis_set_id contains missing values")
    normalized = raw.astype(str).str.strip()
    if normalized.eq("").any():
        raise SupervisedLearningContractError("analysis_set_id contains blank values")
    values = normalized.drop_duplicates().tolist()
    if len(values) != 1:
        raise SupervisedLearningContractError(
            f"formal Task A run requires exactly one analysis_set_id; got {values}"
        )
    return values[0]


def _parse_comparison_models(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise SupervisedLearningContractError("comparison_models contains a blank value")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SupervisedLearningContractError(
                f"comparison_models must be a JSON list of model IDs: {exc}"
            ) from exc
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parsed = list(value)
    else:
        raise SupervisedLearningContractError("comparison_models must be a JSON/list sequence of model IDs")
    if not isinstance(parsed, list) or not parsed:
        raise SupervisedLearningContractError("comparison_models must contain at least one model ID")
    models = tuple(str(item).strip() for item in parsed)
    if any(not item for item in models):
        raise SupervisedLearningContractError("comparison_models contains blank model IDs")
    if len(set(models)) != len(models):
        raise SupervisedLearningContractError("comparison_models contains duplicate model IDs")
    return models


def _require_comparison_models(frame: pd.DataFrame) -> tuple[str, ...]:
    """Read the one comparison-specific model list supplied by Task B.

    Task B builds one analysis_set_id per requested comparison.  A registry-backed
    Task A run therefore consumes only the models declared for that set instead of
    forcing every registry model onto one global common sample.
    """
    if FORMAL_COMPARISON_MODELS_COLUMN not in frame.columns:
        raise SupervisedLearningContractError(
            "registry-backed formal input must contain comparison_models supplied by the B-layer analysis-set contract"
        )
    if frame[FORMAL_COMPARISON_MODELS_COLUMN].isna().any():
        raise SupervisedLearningContractError("comparison_models contains missing values")
    parsed_rows = [_parse_comparison_models(value) for value in frame[FORMAL_COMPARISON_MODELS_COLUMN].tolist()]
    unique = set(parsed_rows)
    if len(unique) != 1:
        raise SupervisedLearningContractError(
            f"one analysis_set_id must declare one comparison_models list; got {sorted(unique)}"
        )
    return parsed_rows[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
    except Exception:
        pass
    return "unknown"


def run_supervised_from_config(
    config_path: str | Path = "configs/supervised_learning_v1.yaml",
    *,
    paths_config: str | Path | None = None,
    input_table: str | Path | None = None,
    output_root: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    """Execute Task A from one upstream comparison-specific admitted probe table."""
    config = load_config(config_path, paths_config=paths_config)
    _require_frozen_runtime_contract(config.data)

    input_path = Path(input_table).resolve() if input_table is not None else config.path_value("input_table")
    output_path = Path(output_root).resolve() if output_root is not None else config.path_value("output_root")
    if not input_path.is_file():
        raise FileNotFoundError(f"supervised input probe table not found: {input_path}")

    all_families, comparison_plan = _resolve_model_plan(config.data)
    if not all_families:
        raise SupervisedLearningContractError(
            "no supervised feature families are configured; Task C/D must freeze a feature registry or candidate schemes before a formal real-data run"
        )

    frame = _read_probe_table(input_path)
    analysis_set_id = _require_single_analysis_set_id(frame)
    families = all_families
    declared_models: tuple[str, ...] | None = None
    if comparison_plan is not None:
        declared_models = _require_comparison_models(frame)
        unknown = sorted(set(declared_models) - set(all_families))
        if unknown:
            raise SupervisedLearningContractError(
                f"analysis_set_id={analysis_set_id} declares models absent from frozen feature registry: {unknown}"
            )
        families = {model_id: all_families[model_id] for model_id in declared_models}

    validation = config.section("validation")
    inner = validation.get("inner", {})
    primary_model = config.section("models").get("primary", {})
    pipeline = config.section("pipeline")
    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    result = run_nested_loso(
        frame,
        model_feature_schemes=families,
        group_col=str(validation.get("outer", {}).get("group_column", FORMAL_PARTICIPANT_GROUP_COLUMN)),
        c_candidates=primary_model.get("C_candidates", (0.01, 0.1, 1.0, 10.0)),
        inner_splits=int(inner.get("n_splits", FORMAL_INNER_SPLITS)),
        max_iter=int(primary_model.get("max_iter", 2000)),
        seed=int(pipeline.get("random_seed", 20260910)),
        run_id=str(resolved_run_id),
        analysis_set_id=analysis_set_id,
    )
    if comparison_plan is not None:
        selected = set(families)
        result.metadata["feature_comparison_plan"] = comparison_plan.audit_dict()
        result.metadata["analysis_set_declared_models"] = list(declared_models or ())
        result.metadata["paired_comparisons"] = [
            {
                "comparison_type": "behavior_increment",
                "baseline_model_id": baseline,
                "added_model_id": added,
                "feature_id": feature_id,
            }
            for baseline, added, feature_id in comparison_plan.behavior_increment_pairs
            if baseline in selected and added in selected
        ] + [
            {
                "comparison_type": "full_leave_one_out",
                "baseline_model_id": reduced,
                "added_model_id": full,
                "feature_id": feature_id,
            }
            for reduced, full, feature_id in comparison_plan.full_leave_one_out_pairs
            if reduced in selected and full in selected
        ]

    repo_root = Path(__file__).resolve().parents[3]
    return write_supervised_run(
        result,
        output_root=output_path,
        provenance={
            "input_table": str(input_path),
            "input_sha256": _sha256(input_path),
            "config_path": str(Path(config_path).resolve()),
            "config_digest": config.digest,
            "code_sha": _git_sha(repo_root),
        },
    )
