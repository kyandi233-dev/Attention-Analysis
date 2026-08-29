from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss
    from sklearn.model_selection import GroupKFold
except Exception:  # pragma: no cover - optional at import time
    average_precision_score = balanced_accuracy_score = brier_score_loss = None
    GroupKFold = None

SCHEMA_VERSION = 2
PRIMARY_SIGNAL = "pupil_geom_mean_diameter"
FORBIDDEN_FORMAL_SIGNALS = {
    "pir",
    "pupil_to_iris_ratio",
    "fullclass_pupil_to_iris_diameter_ratio",
    "iris_outer_geom_mean_diameter",
}
PRIMARY_PROBE_UNIT = ("participant_id", "session_id", "block_id", "probe_id")
REQUIRED_PUPIL_COLUMNS = {
    "subject",
    "session_id",
    "phase",
    "phase_segment",
    "frame_idx",
    "eye",
    "unix_ms",
    PRIMARY_SIGNAL,
    "quality_track",
}
FORMAL_METRICS = ("pr_auc", "balanced_accuracy", "brier_score")


class NIRContractError(ValueError):
    pass


@dataclass(frozen=True)
class ModelFailure:
    model_name: str
    endpoint: str
    stage: str
    error_type: str
    error_message: str
    input_unit: str
    n_rows: int
    n_participants: int
    admission_status: str = "blocked"

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def validate_pupil_only_rows(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_PUPIL_COLUMNS - set(frame.columns))
    if missing:
        raise NIRContractError(f"pupil-only input missing columns: {missing}")
    lowered = {str(column).lower() for column in frame.columns}
    leaked = sorted(
        name for name in FORBIDDEN_FORMAL_SIGNALS if any(name in column for column in lowered)
    )
    if leaked:
        raise NIRContractError(
            "legacy iris/PIR semantics are forbidden in formal pupil-only input: "
            f"{leaked}"
        )
    key = ["subject", "session_id", "phase_segment", "frame_idx", "eye"]
    if frame[key].isna().any().any():
        raise NIRContractError("pupil-only primary key contains missing values")
    if frame.duplicated(key).any():
        raise NIRContractError("duplicate pupil-only frame/eye key")
    times = pd.to_numeric(frame["unix_ms"], errors="coerce")
    if times.isna().any():
        raise NIRContractError("unix_ms must be numeric epoch milliseconds")


def build_analysis_ready_timepoints(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert canonical pupil-only eye rows to one timepoint row without restoring PIR."""
    validate_pupil_only_rows(frame)
    work = frame.copy()
    work[PRIMARY_SIGNAL] = pd.to_numeric(work[PRIMARY_SIGNAL], errors="coerce")
    work["pupil_valid"] = work[PRIMARY_SIGNAL].gt(0) & work["quality_track"].eq("observed")
    keys = ["subject", "session_id", "phase", "phase_segment", "frame_idx", "unix_ms"]
    if "block_id" in work:
        keys.insert(2, "block_id")
    if "participant_id" in work:
        keys.insert(1, "participant_id")

    rows: list[dict[str, Any]] = []
    for key, group in work.groupby(keys, sort=False, dropna=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        by_eye = {str(item.eye): item for item in group.itertuples(index=False)}
        values: list[float] = []
        for eye in ("left", "right"):
            item = by_eye.get(eye)
            value = (
                float(getattr(item, PRIMARY_SIGNAL))
                if item is not None and bool(getattr(item, "pupil_valid"))
                else np.nan
            )
            row[f"{eye}_pupil_diameter_px"] = value
            row[f"{eye}_pupil_valid"] = bool(np.isfinite(value))
            if np.isfinite(value):
                values.append(value)
        row["binocular_pupil_diameter_px"] = float(np.mean(values)) if values else np.nan
        row["binocular_pupil_valid"] = bool(values)
        row["binocular_source_mode"] = (
            "binocular"
            if len(values) == 2
            else "left_only"
            if row["left_pupil_valid"]
            else "right_only"
            if row["right_pupil_valid"]
            else "missing"
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    return out if out.empty else add_session_standardization(out)


def _robust_center_scale(series: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return np.nan, np.nan
    median = float(numeric.median())
    mad = float(np.median(np.abs(numeric.to_numpy(dtype=float) - median)))
    scale = 1.4826 * mad
    return median, scale if np.isfinite(scale) and scale > 0 else np.nan


def add_session_standardization(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    grouping = [column for column in ("participant_id", "session_id") if column in out]
    if not grouping:
        grouping = ["subject"]
    signals = (
        "left_pupil_diameter_px",
        "right_pupil_diameter_px",
        "binocular_pupil_diameter_px",
    )
    for signal in signals:
        centered = pd.Series(np.nan, index=out.index, dtype=float)
        robust_z = pd.Series(np.nan, index=out.index, dtype=float)
        for _, index in out.groupby(grouping, sort=False, dropna=False).groups.items():
            median, scale = _robust_center_scale(out.loc[index, signal])
            values = pd.to_numeric(out.loc[index, signal], errors="coerce")
            centered.loc[index] = values - median
            if np.isfinite(scale):
                robust_z.loc[index] = (values - median) / scale
        out[signal.replace("_px", "_session_centered_px")] = centered
        out[signal.replace("_px", "_session_robust_z")] = robust_z
    return out


def derive_behavior_endpoints(trials: pd.DataFrame) -> pd.DataFrame:
    required = {"trial_id", "is_go", "responded"}
    missing = sorted(required - set(trials.columns))
    if missing:
        raise NIRContractError(f"behavior endpoint input missing columns: {missing}")
    out = trials.copy()
    is_go = out["is_go"].astype(bool)
    responded = out["responded"].astype(bool)
    out["endpoint_name"] = np.where(is_go, "go_omission", "nogo_commission")
    out["endpoint_value"] = np.where(is_go, ~responded, responded).astype(int)
    out["endpoint_eligible"] = True
    out["label_definition"] = np.where(
        is_go,
        "Go trial: 1 = omitted response, 0 = responded",
        "NoGo trial: 1 = commission response, 0 = withheld",
    )
    return out


def assert_primary_probe_unit(probes: pd.DataFrame) -> None:
    missing = sorted(set(PRIMARY_PROBE_UNIT) - set(probes.columns))
    if missing:
        raise NIRContractError(f"primary probe table missing hierarchy columns: {missing}")
    if probes[list(PRIMARY_PROBE_UNIT)].isna().any().any():
        raise NIRContractError("primary probe unit contains missing hierarchy keys")
    if probes.duplicated(list(PRIMARY_PROBE_UNIT), keep=False).any():
        raise NIRContractError(
            "primary analysis requires exactly one row per probe; duplicate probe units detected"
        )


def validate_window_registry(registry: pd.DataFrame) -> None:
    required = {"window_name", "unit", "role", "start_offset_ms", "end_offset_ms"}
    missing = sorted(required - set(registry.columns))
    if missing:
        raise NIRContractError(f"window registry missing columns: {missing}")
    primary = registry[(registry["unit"] == "probe") & (registry["role"] == "primary")]
    if len(primary) != 1:
        raise NIRContractError("exactly one fixed primary probe window is required")
    non_primary = registry.loc[registry.index.difference(primary.index)]
    if (non_primary["role"] == "primary").any():
        raise NIRContractError(
            "trial/event windows and alternative probe windows must be sensitivity or descriptive"
        )


def audit_brightness_direction(columns: Iterable[str], *, analysis_kind: str) -> None:
    names = {str(column) for column in columns}
    current = {
        column
        for column in names
        if column.startswith("current_") and ("lum" in column or "brightness" in column)
    }
    history = {
        column
        for column in names
        if column.startswith("history_") and ("lum" in column or "brightness" in column)
    }
    local_baseline = {column for column in names if column.startswith("local_baseline_")}
    if analysis_kind == "pre_event_tonic":
        if current:
            raise NIRContractError(
                "pre-event tonic analysis must not include current-stimulus brightness"
            )
        if not history:
            raise NIRContractError(
                "pre-event tonic analysis requires explicitly historical brightness covariates"
            )
    elif analysis_kind == "post_event_phasic":
        if not current:
            raise NIRContractError(
                "post-event phasic analysis requires current-stimulus covariates"
            )
        if not local_baseline:
            raise NIRContractError(
                "post-event phasic analysis requires an event-pre local baseline"
            )
    else:
        raise NIRContractError(f"unknown brightness audit analysis_kind: {analysis_kind}")


def stimulus_history_features(
    events: pd.DataFrame, target_ms: float, start_ms: float, end_ms: float
) -> dict[str, Any]:
    if start_ms >= end_ms or end_ms > target_ms:
        raise NIRContractError("historical stimulus window must end at or before target time")
    required = {"onset_ms", "relative_luminance"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise NIRContractError(f"stimulus history missing columns: {missing}")
    onset = pd.to_numeric(events["onset_ms"], errors="coerce")
    selected = events.loc[onset.ge(start_ms) & onset.lt(end_ms)].sort_values("onset_ms")
    luminance = pd.to_numeric(selected["relative_luminance"], errors="coerce")
    return {
        "history_event_count": int(len(selected)),
        "history_luminance_mean": (
            float(luminance.mean()) if luminance.notna().any() else np.nan
        ),
        "history_luminance_last": (
            float(luminance.iloc[-1])
            if len(luminance) and pd.notna(luminance.iloc[-1])
            else np.nan
        ),
        "history_sequence_onset_ms": tuple(
            pd.to_numeric(selected["onset_ms"], errors="coerce").tolist()
        ),
    }


def participant_exclusive_folds(
    frame: pd.DataFrame, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    if GroupKFold is None:
        raise RuntimeError("scikit-learn is required for participant-exclusive folds")
    if "participant_id" not in frame:
        raise NIRContractError("participant_id is required for outer-fold construction")
    groups = frame["participant_id"].astype(str).to_numpy()
    if len(np.unique(groups)) < n_splits:
        raise NIRContractError(
            f"need at least {n_splits} participants for {n_splits}-fold outer CV"
        )
    splitter = GroupKFold(n_splits=n_splits)
    folds = []
    for train, test in splitter.split(np.zeros(len(frame)), groups=groups):
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("participant leakage across outer fold")
        folds.append((train, test))
    return folds


def binary_metrics(
    y_true: Sequence[int], score: Sequence[float], label: Sequence[int] | None = None
) -> dict[str, float]:
    if average_precision_score is None:
        raise RuntimeError("scikit-learn is required for binary metrics")
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(score, dtype=float)
    predicted = np.asarray(label if label is not None else (scores >= 0.5), dtype=int)
    if len(np.unique(y)) < 2:
        raise NIRContractError("binary metrics require both outcome classes")
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "brier_score": float(brier_score_loss(y, scores)),
    }


def majority_baseline(y_true: Sequence[int]) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    if len(y) == 0:
        raise NIRContractError("empty endpoint")
    prevalence = float(y.mean())
    majority = int(prevalence >= 0.5)
    return {
        "model_role": "majority_baseline",
        "prevalence": prevalence,
        **binary_metrics(
            y,
            np.full(len(y), prevalence, dtype=float),
            np.full(len(y), majority, dtype=int),
        ),
    }


def incremental_comparison(
    y_true: Sequence[int], behavior_score: Sequence[float], nir_score: Sequence[float]
) -> pd.DataFrame:
    baseline = binary_metrics(y_true, behavior_score)
    full = binary_metrics(y_true, nir_score)
    return pd.DataFrame(
        [
            {"model_role": "behavior_design_baseline", **baseline},
            {
                "model_role": "behavior_plus_nir",
                **full,
                **{f"delta_{metric}": full[metric] - baseline[metric] for metric in FORMAL_METRICS},
            },
        ]
    )


def safe_model_fit(
    fit: Callable[[], Any],
    *,
    model_name: str,
    endpoint: str,
    stage: str,
    input_unit: str,
    frame: pd.DataFrame,
) -> tuple[Any | None, pd.DataFrame]:
    try:
        return fit(), pd.DataFrame(columns=list(ModelFailure.__dataclass_fields__))
    except Exception as exc:
        participants = (
            int(frame["participant_id"].nunique()) if "participant_id" in frame else 0
        )
        failure = ModelFailure(
            model_name=model_name,
            endpoint=endpoint,
            stage=stage,
            error_type=type(exc).__name__,
            error_message=str(exc),
            input_unit=input_unit,
            n_rows=int(len(frame)),
            n_participants=participants,
        )
        return None, pd.DataFrame([failure.as_dict()])


def repeat_estimability(
    pairs: pd.DataFrame, *, min_pairs: int = 3
) -> dict[str, Any]:
    required = {"participant_id", "session_order", "value"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise NIRContractError(f"repeat table missing columns: {missing}")
    counts = pairs.groupby("participant_id")["session_order"].nunique()
    paired = counts[counts >= 2]
    if len(paired) < min_pairs:
        return {
            "estimable": False,
            "n_repeat_participants": int(len(paired)),
            "minimum_repeat_participants": int(min_pairs),
            "reason": (
                "insufficient independent repeat participants; two block values within "
                "one session are not a stability estimate"
            ),
        }
    return {
        "estimable": True,
        "n_repeat_participants": int(len(paired)),
        "minimum_repeat_participants": int(min_pairs),
        "reason": None,
    }


def figure_denominator_contract(rows: pd.DataFrame) -> None:
    required = {"figure_id", "panel_id", "count_unit", "denominator", "n"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise NIRContractError(f"figure denominator table missing columns: {missing}")
    if (pd.to_numeric(rows["denominator"], errors="coerce") <= 0).any():
        raise NIRContractError("figure denominator must be explicit and positive")
    for _, group in rows.groupby(["figure_id", "panel_id"], dropna=False):
        if group["count_unit"].nunique() > 1:
            raise NIRContractError(
                "one comparable plot panel must not mix frame/session/block/probe/participant count units"
            )
