"""Time-legality provenance contract for frozen supervised predictors.

This module validates producer-side temporal provenance before a feature is allowed
into any formal Q1 prediction model.  It is deliberately separate from fold-local
imputation/scaling/hyper-parameter fitting, which are already governed by Task A.

Q1-blind feature construction is not sufficient evidence of leakage safety: an
upstream producer may still use post-probe frames, future probes, whole-session
statistics, or complete test-participant sequences.  Formal predictor eligibility is
therefore fail-closed until the upstream representation is verified as pre-probe only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .task import SupervisedLearningContractError


TIME_LEGALITY_SCHEMA_VERSION = "1.16.10-time-legality-v1"
VERIFIED_PRE_PROBE_ONLY = "verified_pre_probe_only"
PENDING_UPSTREAM_FREEZE = "pending_upstream_freeze"
BLOCKED_FUTURE_INFORMATION = "blocked_future_information"
BLOCKED_TEMPORAL_SCOPE_UNKNOWN = "blocked_temporal_scope_unknown"
ALLOWED_TIME_LEGALITY_STATUSES = frozenset(
    {
        VERIFIED_PRE_PROBE_ONLY,
        PENDING_UPSTREAM_FREEZE,
        BLOCKED_FUTURE_INFORMATION,
        BLOCKED_TEMPORAL_SCOPE_UNKNOWN,
    }
)

PREDICTION_ELIGIBILITY_FIELDS = (
    "standalone_eligible",
    "behavior_increment_eligible",
    "behavior_reference_eligible",
    "modality_model_eligible",
    "full_model_eligible",
    "full_leave_one_out_eligible",
)
_ELIGIBILITY_DEFAULTS = {
    "standalone_eligible": True,
    "behavior_increment_eligible": False,
    "behavior_reference_eligible": False,
    "modality_model_eligible": False,
    "full_model_eligible": True,
    "full_leave_one_out_eligible": True,
}


class TimeLegalityContractError(SupervisedLearningContractError):
    """Raised when a frozen feature cannot prove its temporal legality."""


@dataclass(frozen=True)
class FeatureTimeLegalityRecord:
    feature_id: str
    temporal_anchor: str
    temporal_scope: str
    time_legality_status: str
    time_legality_evidence: str
    preprocessing_dependencies: tuple[str, ...]
    requests_prediction: bool
    prediction_eligibility_fields_true: tuple[str, ...]
    allowed_device_packages: tuple[str, ...]

    def audit_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["preprocessing_dependencies"] = list(self.preprocessing_dependencies)
        value["prediction_eligibility_fields_true"] = list(
            self.prediction_eligibility_fields_true
        )
        value["allowed_device_packages"] = list(self.allowed_device_packages)
        return value


def _strict_bool(value: object, *, field: str, feature_id: str, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, float) and value in {0.0, 1.0}:
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes"}:
        return True
    if text in {"false", "0", "0.0", "no"}:
        return False
    raise TimeLegalityContractError(
        f"{feature_id}: {field} must be a boolean; got {value!r}"
    )


def _string_sequence(value: object, *, field: str, feature_id: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TimeLegalityContractError(
            f"{feature_id}: {field} must be a sequence"
        )
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise TimeLegalityContractError(
            f"{feature_id}: {field} contains blank values"
        )
    if len(set(result)) != len(result):
        raise TimeLegalityContractError(
            f"{feature_id}: {field} contains duplicate values"
        )
    return result


def prediction_eligibility_fields_true(raw: Mapping[str, Any]) -> tuple[str, ...]:
    feature_id = str(raw.get("feature_id", "<unknown>"))
    enabled: list[str] = []
    for field in PREDICTION_ELIGIBILITY_FIELDS:
        if _strict_bool(
            raw.get(field),
            field=field,
            feature_id=feature_id,
            default=_ELIGIBILITY_DEFAULTS[field],
        ):
            enabled.append(field)
    return tuple(enabled)


def normalize_feature_time_legality(raw: Mapping[str, Any]) -> FeatureTimeLegalityRecord:
    feature_id = str(raw.get("feature_id", "")).strip()
    if not feature_id:
        raise TimeLegalityContractError("feature time-legality record requires feature_id")

    required = (
        "temporal_anchor",
        "temporal_scope",
        "time_legality_status",
        "time_legality_evidence",
    )
    missing = [field for field in required if field not in raw]
    if missing:
        raise TimeLegalityContractError(
            f"{feature_id}: missing time-legality fields {missing}"
        )

    temporal_anchor = str(raw.get("temporal_anchor", "")).strip()
    temporal_scope = str(raw.get("temporal_scope", "")).strip()
    status = str(raw.get("time_legality_status", "")).strip()
    evidence = str(raw.get("time_legality_evidence", "")).strip()
    dependencies = _string_sequence(
        raw.get("preprocessing_dependencies", ()),
        field="preprocessing_dependencies",
        feature_id=feature_id,
    )
    packages = _string_sequence(
        raw.get("allowed_device_packages", ()),
        field="allowed_device_packages",
        feature_id=feature_id,
    )
    eligibility_true = prediction_eligibility_fields_true(raw)
    requests_prediction = bool(eligibility_true or packages)

    if status not in ALLOWED_TIME_LEGALITY_STATUSES:
        raise TimeLegalityContractError(
            f"{feature_id}: unknown time_legality_status={status!r}; expected one of "
            f"{sorted(ALLOWED_TIME_LEGALITY_STATUSES)}"
        )

    if status == VERIFIED_PRE_PROBE_ONLY:
        if temporal_anchor != "probe_time_ms":
            raise TimeLegalityContractError(
                f"{feature_id}: verified predictor must use temporal_anchor='probe_time_ms'"
            )
        if temporal_scope != "pre_probe_only":
            raise TimeLegalityContractError(
                f"{feature_id}: verified predictor must use temporal_scope='pre_probe_only'"
            )
        if not evidence:
            raise TimeLegalityContractError(
                f"{feature_id}: verified predictor requires nonblank time_legality_evidence"
            )
    elif requests_prediction:
        raise TimeLegalityContractError(
            f"{feature_id}: feature requests formal prediction eligibility via "
            f"{list(eligibility_true) or ['allowed_device_packages']} but "
            f"time_legality_status={status!r}; formal prediction is fail-closed until "
            f"status='{VERIFIED_PRE_PROBE_ONLY}'"
        )

    return FeatureTimeLegalityRecord(
        feature_id=feature_id,
        temporal_anchor=temporal_anchor,
        temporal_scope=temporal_scope,
        time_legality_status=status,
        time_legality_evidence=evidence,
        preprocessing_dependencies=dependencies,
        requests_prediction=requests_prediction,
        prediction_eligibility_fields_true=eligibility_true,
        allowed_device_packages=packages,
    )


def validate_feature_registry_time_legality(
    feature_registry_section: Mapping[str, Any],
) -> tuple[FeatureTimeLegalityRecord, ...]:
    entries = feature_registry_section.get("features", [])
    if not isinstance(entries, list):
        raise TimeLegalityContractError("feature_registry.features must be a list")
    if not entries:
        return ()

    records: list[FeatureTimeLegalityRecord] = []
    seen: set[str] = set()
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise TimeLegalityContractError(
                "feature_registry.features entries must be mappings"
            )
        record = normalize_feature_time_legality(raw)
        if record.feature_id in seen:
            raise TimeLegalityContractError(
                f"duplicate feature_id in time-legality registry: {record.feature_id}"
            )
        seen.add(record.feature_id)
        records.append(record)
    return tuple(records)


def time_legality_audit(feature_registry_section: Mapping[str, Any]) -> dict[str, object]:
    records = validate_feature_registry_time_legality(feature_registry_section)
    verified = [
        record.feature_id
        for record in records
        if record.time_legality_status == VERIFIED_PRE_PROBE_ONLY
    ]
    pending_or_blocked = [
        record.feature_id
        for record in records
        if record.time_legality_status != VERIFIED_PRE_PROBE_ONLY
    ]
    return {
        "schema_version": TIME_LEGALITY_SCHEMA_VERSION,
        "n_features": len(records),
        "n_verified_pre_probe_only": len(verified),
        "verified_pre_probe_only_feature_ids": verified,
        "pending_or_blocked_feature_ids": pending_or_blocked,
        "features": [record.audit_dict() for record in records],
        "contract": {
            "prediction_requires_verified_pre_probe_only": True,
            "q1_blind_alone_proves_leakage_safe": False,
            "training_fold_preprocessing_governed_separately": True,
        },
    }
