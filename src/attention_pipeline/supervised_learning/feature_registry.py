"""Frozen feature registry and comparison-plan generation for Q1 supervision.

Scientific modality, producer namespace, and hardware provenance are deliberately
separate. Upstream measurement work freezes concrete predictor representations and
their eligibility; this module records those facts and generates feature-, modality-,
and device-level comparison plans without inferring one layer from another.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .feature_schemes import (
    ALLOWED_SCIENTIFIC_MODALITIES,
    ALLOWED_SENSOR_DEVICES,
    FeatureScheme,
    validate_mainline_feature_scheme,
)
from .task import SupervisedLearningContractError


ALLOWED_DEVICE_KEYS = ALLOWED_SENSOR_DEVICES
ALLOWED_SOURCE_NAMESPACES = frozenset({"behavior", "nir", "rgb", "mmwave"})
DEVICE_PACKAGES: dict[str, frozenset[str]] = {
    "M0": frozenset(),
    "M1": frozenset({"nir"}),
    "M2": frozenset({"mmwave"}),
    "M3": frozenset({"rgb"}),
    "M4": frozenset({"nir", "mmwave"}),
    "M5": frozenset({"nir", "rgb"}),
    "M6": frozenset({"mmwave", "rgb"}),
    "M7": frozenset({"nir", "mmwave", "rgb"}),
}
SENSOR_MODALITIES = ("ocular", "movement", "cardiopulmonary")


class FeatureRegistryContractError(SupervisedLearningContractError):
    """Raised when a frozen feature/provenance registry is internally inconsistent."""


@dataclass(frozen=True)
class RegisteredFeature:
    """One exact frozen predictor representation and its scientific/provenance metadata."""

    feature_id: str
    scientific_feature_id: str
    columns: tuple[str, ...]
    role: str  # behavior | sensor; retained as a coarse compatibility/audit role
    modality: str  # behavior | ocular | movement | cardiopulmonary
    raw_source: str
    source_namespace: str  # Task-B producer table namespace: behavior | nir | rgb | mmwave
    required_devices: tuple[str, ...]
    feature_type: str = ""
    preprocessing_dependencies: tuple[str, ...] = ()
    standalone_eligible: bool = True
    behavior_increment_eligible: bool = False
    behavior_reference_eligible: bool = False
    modality_model_eligible: bool = False
    full_model_eligible: bool = True
    full_leave_one_out_eligible: bool = True
    allowed_device_packages: tuple[str, ...] = ()
    description: str = ""

    def audit_dict(self) -> dict[str, object]:
        return {
            "feature_id": self.feature_id,
            "scientific_feature_id": self.scientific_feature_id,
            "columns": list(self.columns),
            "role": self.role,
            "modality": self.modality,
            "feature_type": self.feature_type,
            "raw_source": self.raw_source,
            "source_namespace": self.source_namespace,
            "required_devices": list(self.required_devices),
            "preprocessing_dependencies": list(self.preprocessing_dependencies),
            "standalone_eligible": self.standalone_eligible,
            "behavior_increment_eligible": self.behavior_increment_eligible,
            "behavior_reference_eligible": self.behavior_reference_eligible,
            "modality_model_eligible": self.modality_model_eligible,
            "full_model_eligible": self.full_model_eligible,
            "full_leave_one_out_eligible": self.full_leave_one_out_eligible,
            "allowed_device_packages": list(self.allowed_device_packages),
            "description": self.description,
        }


@dataclass(frozen=True)
class PlannedModel:
    """One model produced from a frozen feature registry."""

    model_id: str
    comparison_role: str
    feature_ids: tuple[str, ...]
    columns: tuple[str, ...]
    modalities: tuple[str, ...]
    required_devices: tuple[str, ...]
    includes_behavior_reference: bool = False
    description: str = ""

    def as_feature_scheme(self) -> FeatureScheme:
        scheme = FeatureScheme(
            feature_set_id=self.model_id,
            columns=self.columns,
            description=self.description,
            modalities=self.modalities,
            required_devices=self.required_devices,
        )
        validate_mainline_feature_scheme(scheme)
        return scheme

    def audit_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "comparison_role": self.comparison_role,
            "feature_ids": list(self.feature_ids),
            "columns": list(self.columns),
            "modalities": list(self.modalities),
            "required_devices": list(self.required_devices),
            "includes_behavior_reference": self.includes_behavior_reference,
            "description": self.description,
        }


@dataclass
class FeatureComparisonPlan:
    """Generated model plan plus explicit paired-comparison metadata."""

    registry: tuple[RegisteredFeature, ...]
    models: tuple[PlannedModel, ...]
    behavior_increment_pairs: tuple[tuple[str, str, str], ...] = ()
    full_leave_one_out_pairs: tuple[tuple[str, str, str], ...] = ()
    modality_model_ids: dict[str, str] = field(default_factory=dict)
    modality_increment_pairs: tuple[tuple[str, str, str], ...] = ()
    full_leave_one_modality_out_pairs: tuple[tuple[str, str, str], ...] = ()
    unavailable_modalities: dict[str, str] = field(default_factory=dict)
    device_package_model_ids: dict[str, str] = field(default_factory=dict)
    unavailable_device_packages: dict[str, str] = field(default_factory=dict)

    def model_map(self) -> dict[str, PlannedModel]:
        return {model.model_id: model for model in self.models}

    def feature_map(self) -> dict[str, RegisteredFeature]:
        return {feature.feature_id: feature for feature in self.registry}

    def to_runner_feature_schemes(self) -> dict[str, list[FeatureScheme]]:
        """Convert every planned model to a one-candidate Task A model family."""
        return {model.model_id: [model.as_feature_scheme()] for model in self.models}

    def task_b_comparison_spec(
        self,
        model_ids: Sequence[str],
        *,
        required_outcomes: Sequence[str],
    ) -> dict[str, object]:
        """Generate the exact Task-B sample/identity contract for compared models.

        ``required_features`` remains the scientific-modality -> predictor-column
        contract checked by Task A. ``required_feature_records`` is the separate
        producer lookup map consumed by Task B. The producer namespace is taken
        directly from each frozen registry entry and is never inferred from devices.
        """
        normalized_models = tuple(str(model_id).strip() for model_id in model_ids)
        if not normalized_models or any(not model_id for model_id in normalized_models):
            raise FeatureRegistryContractError("Task-B comparison requires nonblank model_ids")
        if len(set(normalized_models)) != len(normalized_models):
            raise FeatureRegistryContractError("Task-B comparison model_ids contain duplicates")

        model_map = self.model_map()
        unknown_models = sorted(set(normalized_models) - set(model_map))
        if unknown_models:
            raise FeatureRegistryContractError(
                f"Task-B comparison references unknown planned models: {unknown_models}"
            )

        outcomes = tuple(str(outcome).strip() for outcome in required_outcomes)
        if not outcomes or any(not outcome for outcome in outcomes):
            raise FeatureRegistryContractError("Task-B comparison requires nonblank required_outcomes")
        if len(set(outcomes)) != len(outcomes):
            raise FeatureRegistryContractError("Task-B required_outcomes contain duplicates")

        required_feature_ids: set[str] = set()
        for model_id in normalized_models:
            required_feature_ids.update(model_map[model_id].feature_ids)

        selected = tuple(
            feature for feature in self.registry if feature.feature_id in required_feature_ids
        )
        selected_ids = {feature.feature_id for feature in selected}
        if selected_ids != required_feature_ids:
            missing = sorted(required_feature_ids - selected_ids)
            raise FeatureRegistryContractError(
                f"planned model references feature IDs absent from registry: {missing}"
            )

        required_features: dict[str, list[str]] = {}
        required_feature_records: list[dict[str, str]] = []
        for feature in selected:
            modality_columns = required_features.setdefault(feature.modality, [])
            for column in feature.columns:
                if column in modality_columns:
                    raise FeatureRegistryContractError(
                        f"duplicate predictor column within modality {feature.modality}: {column}"
                    )
                modality_columns.append(column)
                required_feature_records.append(
                    {
                        "feature_id": feature.feature_id,
                        "scientific_modality": feature.modality,
                        "source_namespace": feature.source_namespace,
                        "predictor_column": column,
                    }
                )

        return {
            "models": list(normalized_models),
            "required_features": required_features,
            "required_feature_records": required_feature_records,
            "required_outcomes": list(outcomes),
        }

    def audit_dict(self) -> dict[str, object]:
        return {
            "registry": [feature.audit_dict() for feature in self.registry],
            "models": [model.audit_dict() for model in self.models],
            "behavior_increment_pairs": [
                {"baseline_model_id": baseline, "added_model_id": added, "feature_id": feature_id}
                for baseline, added, feature_id in self.behavior_increment_pairs
            ],
            "full_leave_one_out_pairs": [
                {"reduced_model_id": reduced, "full_model_id": full, "feature_id": feature_id}
                for reduced, full, feature_id in self.full_leave_one_out_pairs
            ],
            "modality_model_ids": dict(self.modality_model_ids),
            "modality_increment_pairs": [
                {"baseline_model_id": baseline, "added_model_id": added, "modality": modality}
                for baseline, added, modality in self.modality_increment_pairs
            ],
            "full_leave_one_modality_out_pairs": [
                {"reduced_model_id": reduced, "full_model_id": full, "modality": modality}
                for reduced, full, modality in self.full_leave_one_modality_out_pairs
            ],
            "unavailable_modalities": dict(self.unavailable_modalities),
            "device_package_model_ids": dict(self.device_package_model_ids),
            "unavailable_device_packages": dict(self.unavailable_device_packages),
        }


def _clean_tuple(
    values: Sequence[object], *, field_name: str, feature_id: str
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise FeatureRegistryContractError(f"{feature_id}: {field_name} must be a sequence")
    cleaned = tuple(str(value).strip() for value in values)
    if any(not value for value in cleaned):
        raise FeatureRegistryContractError(f"{feature_id}: {field_name} contains blank values")
    if len(set(cleaned)) != len(cleaned):
        raise FeatureRegistryContractError(f"{feature_id}: {field_name} contains duplicates")
    return cleaned


def _strict_bool(value: object, *, field_name: str, feature_id: str, default: bool) -> bool:
    """Parse registry flags without treating serialized non-empty strings as true."""
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
    raise FeatureRegistryContractError(
        f"{feature_id}: {field_name} must be a boolean; got {value!r}"
    )


def validate_registered_features(
    features: Sequence[RegisteredFeature],
) -> tuple[RegisteredFeature, ...]:
    frozen = tuple(features)
    if not frozen:
        raise FeatureRegistryContractError("feature registry is empty")

    feature_ids: set[str] = set()
    column_owner: dict[str, str] = {}
    full_scientific_owner: dict[str, str] = {}
    modality_scientific_owner: dict[tuple[str, str], str] = {}
    package_scientific_owner: dict[tuple[str, str], str] = {}

    for feature in frozen:
        feature_id = feature.feature_id.strip()
        scientific_id = feature.scientific_feature_id.strip()
        raw_source = feature.raw_source.strip()
        source_namespace = feature.source_namespace.strip()
        modality = feature.modality.strip()
        if not feature_id or not scientific_id or not raw_source or not source_namespace or not modality:
            raise FeatureRegistryContractError(
                "feature_id, scientific_feature_id, modality, raw_source and source_namespace must be non-empty"
            )
        if feature_id in feature_ids:
            raise FeatureRegistryContractError(f"duplicate feature_id: {feature_id}")
        feature_ids.add(feature_id)

        if feature.role not in {"behavior", "sensor"}:
            raise FeatureRegistryContractError(f"{feature_id}: role must be behavior or sensor")
        if modality not in ALLOWED_SCIENTIFIC_MODALITIES:
            raise FeatureRegistryContractError(
                f"{feature_id}: unknown scientific modality {modality!r}"
            )
        if source_namespace not in ALLOWED_SOURCE_NAMESPACES:
            raise FeatureRegistryContractError(
                f"{feature_id}: unknown source_namespace {source_namespace!r}"
            )
        if modality == "behavior" and feature.role != "behavior":
            raise FeatureRegistryContractError(
                f"{feature_id}: behavior modality must use role='behavior'"
            )
        if modality != "behavior" and feature.role != "sensor":
            raise FeatureRegistryContractError(
                f"{feature_id}: non-behavior scientific modality must use role='sensor'"
            )
        if feature.role == "behavior" and source_namespace != "behavior":
            raise FeatureRegistryContractError(
                f"{feature_id}: Behavior features must use source_namespace='behavior'"
            )
        if feature.role == "sensor" and source_namespace == "behavior":
            raise FeatureRegistryContractError(
                f"{feature_id}: sensor features cannot use the Behavior source namespace"
            )

        if not feature.columns:
            raise FeatureRegistryContractError(f"{feature_id}: columns must be non-empty")
        columns = _clean_tuple(feature.columns, field_name="columns", feature_id=feature_id)
        devices = frozenset(
            _clean_tuple(
                feature.required_devices,
                field_name="required_devices",
                feature_id=feature_id,
            )
        ) if feature.required_devices else frozenset()
        if feature.preprocessing_dependencies:
            _clean_tuple(
                feature.preprocessing_dependencies,
                field_name="preprocessing_dependencies",
                feature_id=feature_id,
            )
        unknown_devices = sorted(devices - ALLOWED_DEVICE_KEYS)
        if unknown_devices:
            raise FeatureRegistryContractError(
                f"{feature_id}: unknown required_devices {unknown_devices}"
            )
        if feature.role == "behavior":
            if devices:
                raise FeatureRegistryContractError(
                    f"{feature_id}: Behavior comes from task/probe records and must not declare sensor devices"
                )
            if feature.behavior_increment_eligible:
                raise FeatureRegistryContractError(
                    f"{feature_id}: behavior feature cannot request Behavior -> Behavior+x sensor comparison"
                )
        else:
            if feature.behavior_reference_eligible:
                raise FeatureRegistryContractError(
                    f"{feature_id}: sensor feature cannot be part of the Behavior-only reference"
                )

        for column in columns:
            if column in column_owner:
                raise FeatureRegistryContractError(
                    f"column {column} is assigned to multiple registered features: "
                    f"{column_owner[column]} and {feature_id}"
                )
            column_owner[column] = feature_id

        if feature.full_model_eligible:
            previous = full_scientific_owner.get(scientific_id)
            if previous is not None:
                raise FeatureRegistryContractError(
                    f"scientific feature {scientific_id} has multiple full-model representations: "
                    f"{previous}, {feature_id}"
                )
            full_scientific_owner[scientific_id] = feature_id
        if feature.modality_model_eligible:
            key = (modality, scientific_id)
            previous = modality_scientific_owner.get(key)
            if previous is not None:
                raise FeatureRegistryContractError(
                    f"scientific feature {scientific_id} has multiple {modality} modality-model representations: "
                    f"{previous}, {feature_id}"
                )
            modality_scientific_owner[key] = feature_id
        if feature.full_leave_one_out_eligible and not feature.full_model_eligible:
            raise FeatureRegistryContractError(
                f"{feature_id}: full_leave_one_out_eligible requires full_model_eligible"
            )
        if feature.behavior_reference_eligible and feature.role != "behavior":
            raise FeatureRegistryContractError(
                f"{feature_id}: only behavior features may enter the Behavior-only reference"
            )
        if feature.behavior_increment_eligible and feature.role != "sensor":
            raise FeatureRegistryContractError(
                f"{feature_id}: only sensor features may request Behavior-conditioned increments"
            )

        packages = (
            _clean_tuple(
                feature.allowed_device_packages,
                field_name="allowed_device_packages",
                feature_id=feature_id,
            )
            if feature.allowed_device_packages
            else ()
        )
        if feature.role == "behavior" and packages:
            raise FeatureRegistryContractError(
                f"{feature_id}: Behavior is added to device packages through the explicit Behavior reference, "
                "not allowed_device_packages"
            )
        for package_id in packages:
            if package_id not in DEVICE_PACKAGES:
                raise FeatureRegistryContractError(
                    f"{feature_id}: unknown device package {package_id}"
                )
            package_devices = DEVICE_PACKAGES[package_id]
            if not devices.issubset(package_devices):
                raise FeatureRegistryContractError(
                    f"{feature_id}: package {package_id} lacks required devices "
                    f"{sorted(devices - package_devices)}"
                )
            key = (package_id, scientific_id)
            previous = package_scientific_owner.get(key)
            if previous is not None:
                raise FeatureRegistryContractError(
                    f"package {package_id} has multiple representations of scientific feature "
                    f"{scientific_id}: {previous}, {feature_id}"
                )
            package_scientific_owner[key] = feature_id

        validate_mainline_feature_scheme(
            FeatureScheme(
                feature_set_id=f"registry::{feature_id}",
                columns=columns,
                modalities=(modality,),
                required_devices=tuple(sorted(devices)),
            )
        )

    if not any(feature.behavior_reference_eligible for feature in frozen):
        raise FeatureRegistryContractError(
            "registry must contain at least one Behavior-reference feature"
        )
    if not any(feature.full_model_eligible for feature in frozen):
        raise FeatureRegistryContractError("registry must contain at least one full-model feature")
    return frozen


def registered_feature_from_mapping(raw: Mapping[str, Any]) -> RegisteredFeature:
    required = {
        "feature_id",
        "scientific_feature_id",
        "columns",
        "role",
        "modality",
        "raw_source",
        "source_namespace",
        "required_devices",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise FeatureRegistryContractError(
            f"feature registry entry missing required fields: {missing}"
        )
    columns = raw["columns"]
    devices = raw["required_devices"]
    dependencies = raw.get("preprocessing_dependencies", ())
    packages = raw.get("allowed_device_packages", ())
    for name, value in (
        ("columns", columns),
        ("required_devices", devices),
        ("preprocessing_dependencies", dependencies),
        ("allowed_device_packages", packages),
    ):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise FeatureRegistryContractError(f"feature registry {name} must be a sequence")
    feature_id = str(raw["feature_id"])
    return RegisteredFeature(
        feature_id=feature_id,
        scientific_feature_id=str(raw["scientific_feature_id"]),
        columns=tuple(str(v) for v in columns),
        role=str(raw["role"]),
        modality=str(raw["modality"]),
        feature_type=str(raw.get("feature_type", "")),
        raw_source=str(raw["raw_source"]),
        source_namespace=str(raw["source_namespace"]),
        required_devices=tuple(str(v) for v in devices),
        preprocessing_dependencies=tuple(str(v) for v in dependencies),
        standalone_eligible=_strict_bool(
            raw.get("standalone_eligible"),
            field_name="standalone_eligible",
            feature_id=feature_id,
            default=True,
        ),
        behavior_increment_eligible=_strict_bool(
            raw.get("behavior_increment_eligible"),
            field_name="behavior_increment_eligible",
            feature_id=feature_id,
            default=False,
        ),
        behavior_reference_eligible=_strict_bool(
            raw.get("behavior_reference_eligible"),
            field_name="behavior_reference_eligible",
            feature_id=feature_id,
            default=False,
        ),
        modality_model_eligible=_strict_bool(
            raw.get("modality_model_eligible"),
            field_name="modality_model_eligible",
            feature_id=feature_id,
            default=False,
        ),
        full_model_eligible=_strict_bool(
            raw.get("full_model_eligible"),
            field_name="full_model_eligible",
            feature_id=feature_id,
            default=True,
        ),
        full_leave_one_out_eligible=_strict_bool(
            raw.get("full_leave_one_out_eligible"),
            field_name="full_leave_one_out_eligible",
            feature_id=feature_id,
            default=True,
        ),
        allowed_device_packages=tuple(str(v) for v in packages),
        description=str(raw.get("description", "")),
    )


def load_registered_features(section: Mapping[str, Any]) -> tuple[RegisteredFeature, ...]:
    entries = section.get("features", [])
    if not isinstance(entries, list):
        raise FeatureRegistryContractError("feature_registry.features must be a list")
    return validate_registered_features(
        [registered_feature_from_mapping(raw) for raw in entries]
    )


def _unique_columns(features: Sequence[RegisteredFeature]) -> tuple[str, ...]:
    columns: list[str] = []
    seen: set[str] = set()
    for feature in features:
        for column in feature.columns:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return tuple(columns)


def _modalities(features: Sequence[RegisteredFeature]) -> tuple[str, ...]:
    return tuple(sorted({feature.modality for feature in features}))


def _required_devices(features: Sequence[RegisteredFeature]) -> tuple[str, ...]:
    devices: set[str] = set()
    for feature in features:
        devices.update(feature.required_devices)
    return tuple(sorted(devices))


def _planned_model(
    model_id: str,
    comparison_role: str,
    features: Sequence[RegisteredFeature],
    *,
    includes_behavior_reference: bool,
    description: str,
) -> PlannedModel:
    chosen = tuple(features)
    if not chosen:
        raise FeatureRegistryContractError(f"planned model {model_id} has no features")
    return PlannedModel(
        model_id=model_id,
        comparison_role=comparison_role,
        feature_ids=tuple(feature.feature_id for feature in chosen),
        columns=_unique_columns(chosen),
        modalities=_modalities(chosen),
        required_devices=_required_devices(chosen),
        includes_behavior_reference=includes_behavior_reference,
        description=description,
    )


def build_feature_comparison_plan(
    features: Sequence[RegisteredFeature],
) -> FeatureComparisonPlan:
    """Generate feature, modality, and executable device-package models."""
    registry = validate_registered_features(features)
    behavior = tuple(feature for feature in registry if feature.behavior_reference_eligible)
    behavior_ids = {feature.feature_id for feature in behavior}
    full = tuple(feature for feature in registry if feature.full_model_eligible)

    models: list[PlannedModel] = []
    model_ids: set[str] = set()

    def add(model: PlannedModel) -> None:
        if model.model_id in model_ids:
            raise FeatureRegistryContractError(
                f"duplicate generated model_id: {model.model_id}"
            )
        model.as_feature_scheme()
        model_ids.add(model.model_id)
        models.append(model)

    behavior_model_id = "behavior_reference"
    add(
        _planned_model(
            behavior_model_id,
            "behavior_reference",
            behavior,
            includes_behavior_reference=True,
            description=(
                "Frozen Behavior-only reference; task performance information, not attention ground truth."
            ),
        )
    )

    for feature in registry:
        if feature.standalone_eligible:
            add(
                _planned_model(
                    f"standalone::{feature.feature_id}",
                    "standalone_feature",
                    [feature],
                    includes_behavior_reference=False,
                    description=(
                        f"Standalone Q1-report prediction using {feature.scientific_feature_id}."
                    ),
                )
            )

    behavior_pairs: list[tuple[str, str, str]] = []
    for feature in registry:
        if not feature.behavior_increment_eligible:
            continue
        model_id = f"behavior_plus::{feature.feature_id}"
        add(
            _planned_model(
                model_id,
                "behavior_plus_feature",
                [*behavior, feature],
                includes_behavior_reference=True,
                description=(
                    f"Behavior reference plus {feature.scientific_feature_id}; estimates conditional extra Q1-report information."
                ),
            )
        )
        behavior_pairs.append((behavior_model_id, model_id, feature.feature_id))

    modality_model_ids: dict[str, str] = {"behavior": behavior_model_id}
    modality_pairs: list[tuple[str, str, str]] = []
    unavailable_modalities: dict[str, str] = {}
    for modality in SENSOR_MODALITIES:
        selected = tuple(
            feature
            for feature in registry
            if feature.modality == modality and feature.modality_model_eligible
        )
        if not selected:
            unavailable_modalities[modality] = (
                "no frozen registered feature is eligible for the scientific modality model"
            )
            continue
        standalone_id = f"modality::{modality}"
        add(
            _planned_model(
                standalone_id,
                "standalone_modality",
                selected,
                includes_behavior_reference=False,
                description=f"Scientific {modality}-only Q1-report model.",
            )
        )
        modality_model_ids[modality] = standalone_id

        added_id = f"behavior_plus_modality::{modality}"
        add(
            _planned_model(
                added_id,
                "behavior_plus_modality",
                [*behavior, *selected],
                includes_behavior_reference=True,
                description=(
                    f"Behavior reference plus the frozen {modality} scientific modality model."
                ),
            )
        )
        modality_pairs.append((behavior_model_id, added_id, modality))

    full_model_id = "full"
    add(
        _planned_model(
            full_model_id,
            "full_model",
            full,
            includes_behavior_reference=behavior_ids.issubset(
                {feature.feature_id for feature in full}
            ),
            description="All registry features explicitly frozen as eligible for the full Q1-report model.",
        )
    )

    full_pairs: list[tuple[str, str, str]] = []
    for feature in full:
        if not feature.full_leave_one_out_eligible:
            continue
        reduced = tuple(
            candidate for candidate in full if candidate.feature_id != feature.feature_id
        )
        if not reduced:
            continue
        model_id = f"full_minus::{feature.feature_id}"
        add(
            _planned_model(
                model_id,
                "full_minus_feature",
                reduced,
                includes_behavior_reference=behavior_ids.issubset(
                    {candidate.feature_id for candidate in reduced}
                ),
                description=f"Full model with {feature.scientific_feature_id} removed.",
            )
        )
        full_pairs.append((model_id, full_model_id, feature.feature_id))

    full_modality_pairs: list[tuple[str, str, str]] = []
    for modality in sorted({feature.modality for feature in full}):
        reduced = tuple(feature for feature in full if feature.modality != modality)
        if not reduced:
            continue
        model_id = f"full_minus_modality::{modality}"
        add(
            _planned_model(
                model_id,
                "full_minus_modality",
                reduced,
                includes_behavior_reference=behavior_ids.issubset(
                    {candidate.feature_id for candidate in reduced}
                ),
                description=(
                    f"Full model with all frozen {modality} predictors removed; production dependencies of remaining features are retained."
                ),
            )
        )
        full_modality_pairs.append((model_id, full_model_id, modality))

    package_models: dict[str, str] = {}
    unavailable_packages: dict[str, str] = {}
    for package_id, package_devices in DEVICE_PACKAGES.items():
        selected: list[RegisteredFeature] = list(behavior)
        seen_scientific: set[str] = {
            feature.scientific_feature_id for feature in behavior
        }
        for feature in registry:
            if feature.role == "behavior" or package_id not in feature.allowed_device_packages:
                continue
            required = set(feature.required_devices)
            if not required.issubset(package_devices):
                raise FeatureRegistryContractError(
                    f"{feature.feature_id}: generated package {package_id} lacks required devices"
                )
            if feature.scientific_feature_id in seen_scientific:
                raise FeatureRegistryContractError(
                    f"package {package_id} would include duplicate representations of "
                    f"{feature.scientific_feature_id}"
                )
            seen_scientific.add(feature.scientific_feature_id)
            selected.append(feature)

        covered_devices: set[str] = set()
        for feature in selected:
            covered_devices.update(feature.required_devices)
        missing_sensor_information = sorted(package_devices - covered_devices)
        if missing_sensor_information:
            unavailable_packages[package_id] = (
                "no frozen package-eligible scientific feature currently uses sensor device(s): "
                + ",".join(missing_sensor_information)
            )
            continue

        model_id = package_id
        add(
            _planned_model(
                model_id,
                "device_package",
                selected,
                includes_behavior_reference=True,
                description=(
                    f"Device package {package_id}: sensor devices={sorted(package_devices)}; "
                    "Behavior reference is included separately from hardware."
                ),
            )
        )
        package_models[package_id] = model_id

    return FeatureComparisonPlan(
        registry=registry,
        models=tuple(models),
        behavior_increment_pairs=tuple(behavior_pairs),
        full_leave_one_out_pairs=tuple(full_pairs),
        modality_model_ids=modality_model_ids,
        modality_increment_pairs=tuple(modality_pairs),
        full_leave_one_modality_out_pairs=tuple(full_modality_pairs),
        unavailable_modalities=unavailable_modalities,
        device_package_model_ids=package_models,
        unavailable_device_packages=unavailable_packages,
    )
