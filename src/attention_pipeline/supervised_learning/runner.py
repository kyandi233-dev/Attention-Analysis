"""Participant-disjoint outer LOSO orchestration for Task A.

The runner consumes an already aligned/admitted probe-level frame. It does not
construct analysis sets, infer modality availability, or apply producer QC; those
belong upstream. Its responsibility begins at an explicit analysis frame and
ends with fold audits plus one prediction row per held-out probe and model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .feature_schemes import FeatureScheme, require_scheme_columns, validate_mainline_feature_scheme
from .models import ModelSelectionError, refit_logistic_and_predict, select_logistic_model
from .preprocessing import PreprocessingContractError
from .task import Q1_BINARY_SPEC, SupervisedLearningContractError, encode_q1_binary


DEFAULT_GROUP_COLUMN = "participant_group_id"
MEMBERSHIP_COLUMN = "membership_type"
ALLOWED_MEMBERSHIP_TYPES = frozenset({"included_complete", "included_missing_aware"})
REQUIRED_PROBE_LOCATORS = ("session_id", "block_id", "probe_event_id")
OPTIONAL_PROBE_LOCATORS = (
    "probe_id",
    "probe_order_in_block",
    "probe_index_in_block",
    "probe_index_global",
    "probe_time_ms",
    "probe_onset_unix_ms",
    "window_name",
    "window_start_unix_ms",
    "window_effective_start_unix_ms",
    "window_end_unix_ms",
)
_EXPECTED_FOLD_FAILURES = (
    ModelSelectionError,
    PreprocessingContractError,
    SupervisedLearningContractError,
    ValueError,
    FloatingPointError,
    np.linalg.LinAlgError,
)


@dataclass
class SupervisedRunResult:
    predictions: pd.DataFrame
    fold_audits: list[dict[str, object]] = field(default_factory=list)
    failures: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, object] = field(default_factory=dict)


def _resolve_analysis_set_id(frame: pd.DataFrame, explicit: str | None) -> str | None:
    if "analysis_set_id" not in frame.columns:
        return explicit
    values = frame["analysis_set_id"].dropna().astype(str).str.strip().drop_duplicates().tolist()
    if len(values) > 1:
        raise SupervisedLearningContractError(f"Task A expects one analysis_set_id per run; got {values}")
    from_frame = values[0] if values else None
    if explicit is not None and from_frame is not None and str(explicit).strip() != from_frame:
        raise SupervisedLearningContractError(
            f"explicit analysis_set_id={explicit} conflicts with frame value={from_frame}"
        )
    resolved = from_frame if from_frame is not None else explicit
    if resolved is None or not str(resolved).strip():
        raise SupervisedLearningContractError("Task A requires a non-empty analysis_set_id")
    return str(resolved).strip()


def _resolve_membership_type(frame: pd.DataFrame, explicit: str | None) -> str:
    if MEMBERSHIP_COLUMN in frame.columns:
        raw = frame[MEMBERSHIP_COLUMN]
        if raw.isna().any():
            raise SupervisedLearningContractError(f"{MEMBERSHIP_COLUMN} contains missing values")
        values = raw.astype(str).str.strip()
        if values.eq("").any():
            raise SupervisedLearningContractError(f"{MEMBERSHIP_COLUMN} contains blank values")
        unique = values.drop_duplicates().tolist()
        if len(unique) != 1:
            raise SupervisedLearningContractError(
                f"Task A expects one {MEMBERSHIP_COLUMN} per run; got {unique}"
            )
        from_frame = unique[0]
    else:
        from_frame = None

    if explicit is not None:
        explicit_value = str(explicit).strip()
        if not explicit_value:
            raise SupervisedLearningContractError(f"explicit {MEMBERSHIP_COLUMN} is blank")
        if from_frame is not None and explicit_value != from_frame:
            raise SupervisedLearningContractError(
                f"explicit {MEMBERSHIP_COLUMN}={explicit_value} conflicts with frame value={from_frame}"
            )
        resolved = from_frame if from_frame is not None else explicit_value
    else:
        resolved = from_frame

    if resolved is None:
        raise SupervisedLearningContractError(
            f"Task A requires {MEMBERSHIP_COLUMN} so complete and missing-aware memberships cannot be mixed"
        )
    if resolved not in ALLOWED_MEMBERSHIP_TYPES:
        raise SupervisedLearningContractError(
            f"unsupported {MEMBERSHIP_COLUMN}={resolved!r}; expected one of {sorted(ALLOWED_MEMBERSHIP_TYPES)}"
        )
    return resolved


def _validate_frame(
    frame: pd.DataFrame,
    model_feature_schemes: Mapping[str, Sequence[FeatureScheme]],
    *,
    group_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    required = {group_col, Q1_BINARY_SPEC.source_column, *REQUIRED_PROBE_LOCATORS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SupervisedLearningContractError(f"supervised analysis frame missing required columns: {missing}")
    if frame.empty:
        raise SupervisedLearningContractError("supervised analysis frame is empty")
    if frame[group_col].isna().any():
        raise SupervisedLearningContractError(f"{group_col} contains missing participant IDs")
    if frame.duplicated(list(REQUIRED_PROBE_LOCATORS)).any():
        raise SupervisedLearningContractError(
            f"duplicate probe locator rows for {list(REQUIRED_PROBE_LOCATORS)}"
        )
    if not model_feature_schemes:
        raise SupervisedLearningContractError("at least one model/feature-scheme family is required")

    for model_id, schemes in model_feature_schemes.items():
        name = str(model_id).strip()
        if not name:
            raise SupervisedLearningContractError("model_id must be non-empty")
        if not schemes:
            raise SupervisedLearningContractError(f"model {name} has no feature schemes")
        for scheme in schemes:
            validate_mainline_feature_scheme(scheme)
            require_scheme_columns(frame, scheme)

    encoded = encode_q1_binary(frame[Q1_BINARY_SPEC.source_column])
    if encoded.isna().any():
        raise SupervisedLearningContractError(
            "analysis frame contains probes with missing Q1; B-layer analysis set must resolve this before Task A"
        )
    out = frame.copy().reset_index(drop=True)
    out["q1_binary"] = encoded.reset_index(drop=True).astype(int)
    locators = list(REQUIRED_PROBE_LOCATORS) + [c for c in OPTIONAL_PROBE_LOCATORS if c in out.columns]
    return out, locators


def run_nested_loso(
    frame: pd.DataFrame,
    *,
    model_feature_schemes: Mapping[str, Sequence[FeatureScheme]],
    group_col: str = DEFAULT_GROUP_COLUMN,
    c_candidates: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    inner_splits: int = 5,
    max_iter: int = 2000,
    seed: int = 20260910,
    run_id: str = "task-a-in-memory",
    analysis_set_id: str | None = None,
    membership_type: str | None = None,
) -> SupervisedRunResult:
    """Run one full participant-disjoint LOSO analysis on one explicit membership."""
    data, locator_columns = _validate_frame(frame, model_feature_schemes, group_col=group_col)
    resolved_analysis_set = _resolve_analysis_set_id(data, analysis_set_id)
    resolved_membership = _resolve_membership_type(data, membership_type)
    groups = sorted(data[group_col].astype(str).unique().tolist())
    if len(groups) < 2:
        raise SupervisedLearningContractError("outer LOSO requires at least two participant groups")
    if len(groups) - 1 < int(inner_splits):
        raise SupervisedLearningContractError(
            f"after holding out one participant, inner CV needs {inner_splits} training groups; only {len(groups) - 1} remain"
        )

    prediction_frames: list[pd.DataFrame] = []
    fold_audits: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    model_items = [(str(name), list(schemes)) for name, schemes in model_feature_schemes.items()]

    for outer_index, held_out_group in enumerate(groups):
        test_mask = data[group_col].astype(str).eq(held_out_group)
        outer_train = data.loc[~test_mask].copy()
        outer_test = data.loc[test_mask].copy()
        train_groups = sorted(outer_train[group_col].astype(str).unique().tolist())
        test_groups = sorted(outer_test[group_col].astype(str).unique().tolist())
        if set(train_groups) & set(test_groups):
            raise SupervisedLearningContractError("outer train/test participant overlap detected")
        if test_groups != [held_out_group]:
            raise SupervisedLearningContractError("outer test fold does not contain exactly the held-out participant")

        y_train = outer_train["q1_binary"].to_numpy(dtype=int)

        for model_index, (model_id, schemes) in enumerate(model_items):
            fold_seed = int(seed) + outer_index * 10000 + model_index * 1000
            base_prediction = outer_test[locator_columns + [group_col, Q1_BINARY_SPEC.source_column, "q1_binary"]].copy()
            base_prediction["run_id"] = str(run_id)
            base_prediction["analysis_set_id"] = resolved_analysis_set
            base_prediction[MEMBERSHIP_COLUMN] = resolved_membership
            base_prediction["model_id"] = model_id
            base_prediction["outer_fold_group"] = held_out_group

            try:
                selection = select_logistic_model(
                    outer_train,
                    y_train,
                    feature_schemes=schemes,
                    group_col=group_col,
                    c_candidates=c_candidates,
                    n_splits=int(inner_splits),
                    max_iter=int(max_iter),
                    seed=fold_seed,
                )
                model_test_columns = list(dict.fromkeys([group_col, *selection.feature_scheme.columns]))
                outer_test_features = outer_test[model_test_columns].copy()
                fitted = refit_logistic_and_predict(
                    outer_train,
                    y_train,
                    outer_test_features,
                    feature_scheme=selection.feature_scheme,
                    selected_c=selection.selected_c,
                    group_col=group_col,
                    max_iter=int(max_iter),
                    seed=fold_seed + 777,
                )
                if set(fitted["train_group_ids"]) & set(fitted["test_group_ids"]):
                    raise SupervisedLearningContractError("final refit participant overlap detected")

                base_prediction["feature_set_id"] = selection.feature_scheme.feature_set_id
                base_prediction["selected_c"] = float(selection.selected_c)
                base_prediction[Q1_BINARY_SPEC.positive_probability_name] = np.asarray(fitted["p_positive"], dtype=float)
                base_prediction["predicted_q1_binary"] = np.asarray(fitted["predicted_label"], dtype=int)
                base_prediction["model_failed"] = False
                base_prediction["failure_reason"] = ""
                prediction_frames.append(base_prediction)

                final_audit = dict(fitted)
                final_audit.pop("p_positive", None)
                final_audit.pop("predicted_label", None)
                fold_audits.append(
                    {
                        "run_id": str(run_id),
                        "analysis_set_id": resolved_analysis_set,
                        MEMBERSHIP_COLUMN: resolved_membership,
                        "model_id": model_id,
                        "outer_fold_group": held_out_group,
                        "outer_train_group_ids": train_groups,
                        "outer_test_group_ids": test_groups,
                        "n_outer_train_rows": int(len(outer_train)),
                        "n_outer_test_rows": int(len(outer_test)),
                        "selection": selection.audit_dict(),
                        "final_refit": final_audit,
                        "failed": False,
                        "reason": "",
                    }
                )
            except _EXPECTED_FOLD_FAILURES as exc:
                reason = f"{type(exc).__name__}: {exc}"
                base_prediction["feature_set_id"] = None
                base_prediction["selected_c"] = np.nan
                base_prediction[Q1_BINARY_SPEC.positive_probability_name] = np.nan
                base_prediction["predicted_q1_binary"] = pd.NA
                base_prediction["model_failed"] = True
                base_prediction["failure_reason"] = reason
                prediction_frames.append(base_prediction)
                failure_rows.append(
                    {
                        "run_id": str(run_id),
                        "analysis_set_id": resolved_analysis_set,
                        MEMBERSHIP_COLUMN: resolved_membership,
                        "model_id": model_id,
                        "outer_fold_group": held_out_group,
                        "n_outer_train_rows": int(len(outer_train)),
                        "n_outer_test_rows": int(len(outer_test)),
                        "reason": reason,
                    }
                )
                fold_audits.append(
                    {
                        "run_id": str(run_id),
                        "analysis_set_id": resolved_analysis_set,
                        MEMBERSHIP_COLUMN: resolved_membership,
                        "model_id": model_id,
                        "outer_fold_group": held_out_group,
                        "outer_train_group_ids": train_groups,
                        "outer_test_group_ids": test_groups,
                        "n_outer_train_rows": int(len(outer_train)),
                        "n_outer_test_rows": int(len(outer_test)),
                        "selection": None,
                        "final_refit": None,
                        "failed": True,
                        "reason": reason,
                    }
                )

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    failures = pd.DataFrame(failure_rows)
    metadata = {
        "run_id": str(run_id),
        "analysis_set_id": resolved_analysis_set,
        MEMBERSHIP_COLUMN: resolved_membership,
        "task": Q1_BINARY_SPEC.name,
        "positive_probability_name": Q1_BINARY_SPEC.positive_probability_name,
        "n_input_rows": int(len(data)),
        "n_participant_groups": int(len(groups)),
        "participant_groups": groups,
        "n_models": int(len(model_items)),
        "model_ids": [name for name, _ in model_items],
        "inner_splits": int(inner_splits),
        "c_candidates": [float(v) for v in c_candidates],
        "outer_method": "leave_one_participant_out",
        "zero_individual_calibration": True,
        "outer_test_outcomes_passed_to_model": False,
        "upstream_analysis_set_generation_in_task_a": False,
    }
    return SupervisedRunResult(predictions=predictions, fold_audits=fold_audits, failures=failures, metadata=metadata)
