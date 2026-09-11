"""Frozen feature registry and comparison-plan generation for Q1 supervision.

This module deliberately does *not* decide which real FocusWave features are
scientifically qualified.  Upstream Task C/D work must freeze those choices.
Once frozen, the registry records what each concrete feature measures and which
devices are actually required to produce that representation, then generates a
single consistent model plan for standalone, Behavior-conditioned, full-model
leave-one-feature-out, and M0-M7 device-package analyses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .feature_schemes import FeatureScheme, validate_mainline_feature_scheme
from .task import SupervisedLearningContractError


ALLOWED_DEVICE_KEYS = frozenset({"behavior", "nir", "rgb", "mmwave"})
DEVICE_PACKAGES: dict[str, frozenset[str]] = {
    "M0": frozenset({"behavior"}),
    "M1": frozenset({"behavior", "nir"}),
    "M2": frozenset({"behavior", "mmwave"}),
    "M3": frozenset({"behavior", "rgb"}),
    "M4": frozenset({"behavior", "nir", "mmwave"}),
    "M5": frozenset({"behavior", "nir", "rgb"}),
    "M6": frozenset({"behavior", "mmwave", "rgb"}),
    "M7": frozenset({"behavior", "nir", "mmwave", "rgb"}),
}


class FeatureRegistryContractError(SupervisedLearningContractError):
    """Raised when a frozen feature/provenance registry is internally inconsistent."""


@dataclass(frozen=True)
class RegisteredFeature:
    """One exact frozen predictor representation and its provenance."""

    feature_id: str
    scientific_feature_id: str
    columns: tuple[str, ...]
    role: str  # behavior | sensor
    raw_source: str
    required_devices: tuple[str, ...]
    preprocessing_dependencies: tuple[str, ...] = ()
    standalone_eligible: bool = True
    behavior_increment_eligible: bool = False
    behavior_reference_eligible: bool = False
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
            "raw_source": self.raw_source,
            "required_devices": list(self.required_devices),
            "preprocessing_dependencies": list(self.preprocessing_dependencies),
            "standalone_eligible": self.standalone_eligible,
            "behavior_increment_eligible": self.behavior_increment_eligible,
            "behavior_reference_eligible": self.behavior_reference_eligible,
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
    required_devices: tuple[str, ...]
    description: str = ""

    def as_feature_scheme(self) -> FeatureScheme:
        scheme = FeatureScheme(
            feature_set_id=self.model_id,
            columns=self.columns,
            description=self.description,
            modality_blocks=self.required_devices,
        )
        validate_mainline_feature_scheme(scheme)
        return scheme

    def audit_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "comparison_role": self.comparison_role,
            "feature_ids": list(self.feature_ids),
            "columns": list(self.columns),
            "required_devices": list(self.required_devices),
            "description": self.description,
        }


@dataclass
class FeatureComparisonPlan:
    """Generated model plan plus explicit paired comparison metadata."""

    registry: tuple[RegisteredFeature, ...]
    models: tuple[PlannedModel, ...]
    behavior_increment_pairs: tuple[tuple[str, str, str], ...] = ()
    full_leave_one_out_pairs: tuple[tuple[str, str, str], ...] = ()
    device_package_model_ids: dict[str, str] = field(default_factory=dict)

    def model_map(self) -> dict[str, PlannedModel]:
        return {model.model_id: model for model in self.models}

    def to_runner_feature_schemes(self) -> dict[str, list[FeatureScheme]]:
        """Convert every planned model to a one-candidate Task A model family."""
        return {model.model_id: [model.as_feature_scheme()] for model in self.models}

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
            "device_package_model_ids": dict(self.device_package_model_ids),
        }


def _clean_tuple(values: Sequence[object], *, field_name: str, feature_id: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise FeatureRegistryContractError(f"{feature_id}: {field_name} must be a sequence")
    cleaned = tuple(str(value).strip() for value in values)
    if any(not value for value in cleaned):
        raise FeatureRegistryContractError(f"{feature_id}: {field_name} contains blank values")
    if len(set(cleaned)) != len(cleaned):
        raise FeatureRegistryContractError(f"{feature_id}: {field_name} contains duplicates")
    return cleaned


def validate_registered_features(features: Sequence[RegisteredFeature]) -> tuple[RegisteredFeature, ...]:
    frozen = tuple(features)
    if not frozen:
        raise FeatureRegistryContractError("feature registry is empty")

    feature_ids: set[str] = set()
    column_owner: dict[str, str] = {}
    full_scientific_owner: dict[str, str] = {}
    package_scientific_owner: dict[tuple[str, str], str] = {}

    for feature in frozen:
        feature_id = feature.feature_id.strip()
        scientific_id = feature.scientific_feature_id.strip()
        if not feature_id or not scientific_id:
            raise FeatureRegistryContractError("feature_id and scientific_feature_id must be non-empty")
        if feature_id in feature_ids:
            raise FeatureRegistryContractError(f"duplicate feature_id: {feature_id}")
        feature_ids.add(feature_id)
        if feature.role not in {"behavior", "sensor"}:
            raise FeatureRegistryContractError(f"{feature_id}: role must be behavior or sensor")
        if not feature.columns:
            raise FeatureRegistryContractError(f"{feature_id}: columns must be non-empty")
        columns = _clean_tuple(feature.columns, field_name="columns", feature_id=feature_id)
        devices = frozenset(_clean_tuple(feature.required_devices, field_name="required_devices", feature_id=feature_id))
        if not devices:
            raise FeatureRegistryContractError(f"{feature_id}: required_devices must be non-empty")
        unknown_devices = sorted(devices - ALLOWED_DEVICE_KEYS)
        if unknown_devices:
            raise FeatureRegistryContractError(f"{feature_id}: unknown required_devices {unknown_devices}")
        if feature.role == "behavior":
            if devices != {"behavior"}:
                raise FeatureRegistryContractError(
                    f"{feature_id}: behavior features must require exactly the behavior information source"
                )
            if feature.behavior_increment_eligible:
                raise FeatureRegistryContractError(
                    f"{feature_id}: behavior feature cannot request Behavior -> Behavior+x sensor comparison"
                )
        else:
            if "behavior" in devices:
                raise FeatureRegistryContractError(
                    f"{feature_id}: sensor feature required_devices must describe sensor production dependencies, not Behavior reference input"
                )
            if feature.behavior_reference_eligible:
                raise FeatureRegistryContractError(
                    f"{feature_id}: sensor feature cannot be part of the Behavior-only reference"
                )

        for column in columns:
            if column in column_owner:
                raise FeatureRegistryContractError(
                    f"column {column} is assigned to multiple registered features: {column_owner[column]} and {feature_id}"
                )
            column_owner[column] = feature_id

        if feature.full_model_eligible:
            previous = full_scientific_owner.get(scientific_id)
            if previous is not None:
                raise FeatureRegistryContractError(
                    f"scientific feature {scientific_id} has multiple full-model representations: {previous}, {feature_id}"
                )
            full_scientific_owner[scientific_id] = feature_id
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

        packages = _clean_tuple(
            feature.allowed_device_packages,
            field_name="allowed_device_packages",
            feature_id=feature_id,
        ) if feature.allowed_device_packages else ()
        if feature.role == "behavior" and not packages:
            packages = tuple(DEVICE_PACKAGES)
        for package_id in packages:
            if package_id not in DEVICE_PACKAGES:
                raise FeatureRegistryContractError(f"{feature_id}: unknown device package {package_id}")
            package_devices = DEVICE_PACKAGES[package_id]
            if not devices.issubset(package_devices):
                raise FeatureRegistryContractError(
                    f"{feature_id}: package {package_id} lacks required devices {sorted(devices - package_devices)}"
                )
            key = (package_id, scientific_id)
            previous = package_scientific_owner.get(key)
            if previous is not None:
                raise FeatureRegistryContractError(
                    f"package {package_id} has multiple representations of scientific feature {scientific_id}: "
                    f"{previous}, {feature_id}"
                )
            package_scientific_owner[key] = feature_id

        # Also validate that this exact predictor representation does not violate
        # the mainline forbidden-column contract.
        validate_mainline_feature_scheme(
            FeatureScheme(
                feature_set_id=f"registry::{feature_id}",
                columns=columns,
                modality_blocks=tuple(sorted(devices)),
            )
        )

    if not any(feature.behavior_reference_eligible for feature in frozen):
        raise FeatureRegistryContractError("registry must contain at least one Behavior-reference feature")
    if not any(feature.full_model_eligible for feature in frozen):
        raise FeatureRegistryContractError("registry must contain at least one full-model feature")
    return frozen


def registered_feature_from_mapping(raw: Mapping[str, Any]) -> RegisteredFeature:
    required = {"feature_id", "scientific_feature_id", "columns", "role", "raw_source", "required_devices"}
    missing = sorted(required - set(raw))
    if missing:
        raise FeatureRegistryContractError(f"feature registry entry missing required fields: {missing}")
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
    return RegisteredFeature(
        feature_id=str(raw["feature_id"]),
        scientific_feature_id=str(raw["scientific_feature_id"]),
        columns=tuple(str(v) for v in columns),
        role=str(raw["role"]),
        raw_source=str(raw["raw_source"]),
        required_devices=tuple(str(v) for v in devices),
        preprocessing_dependencies=tuple(str(v) for v in dependencies),
        standalone_eligible=bool(raw.get("standalone_eligible", True)),
        behavior_increment_eligible=bool(raw.get("behavior_increment_eligible", False)),
        behavior_reference_eligible=bool(raw.get("behavior_reference_eligible", False)),
        full_model_eligible=bool(raw.get("full_model_eligible", True)),
        full_leave_one_out_eligible=bool(raw.get("full_leave_one_out_eligible", True)),
        allowed_device_packages=tuple(str(v) for v in packages),
        description=str(raw.get("description", "")),
    )


def load_registered_features(section: Mapping[str, Any]) -> tuple[RegisteredFeature, ...]:
    entries = section.get("features", [])
    if not isinstance(entries, list):
        raise FeatureRegistryContractError("feature_registry.features must be a list")
    return validate_registered_features([registered_feature_from_mapping(raw) for raw in entries])


def _unique_columns(features: Sequence[RegisteredFeature]) -> tuple[str, ...]:
    columns: list[str] = []
    seen: set[str] = set()
    for feature in features:
        for column in feature.columns:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return tuple(columns)


def _required_devices(features: Sequence[RegisteredFeature], *, include_behavior_reference: bool = False) -> tuple[str, ...]:
    devices: set[str] = set()
    for feature in features:
        devices.update(feature.required_devices)
    if include_behavior_reference:
        devices.add("behavior")
    return tuple(sorted(devices))


def _planned_model(
    model_id: str,
    comparison_role: str,
    features: Sequence[RegisteredFeature],
    *,
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
        required_devices=_required_devices(chosen),
        description=description,
    )


def build_feature_comparison_plan(features: Sequence[RegisteredFeature]) -> FeatureComparisonPlan:
    """Generate the frozen concrete-feature and M0-M7 model plan."""
    registry = validate_registered_features(features)
    behavior = tuple(feature for feature in registry if feature.behavior_reference_eligible)
    full = tuple(feature for feature in registry if feature.full_model_eligible)

    models: list[PlannedModel] = []
    model_ids: set[str] = set()

    def add(model: PlannedModel) -> None:
        if model.model_id in model_ids:
            raise FeatureRegistryContractError(f"duplicate generated model_id: {model.model_id}")
        model.as_feature_scheme()
        model_ids.add(model.model_id)
        models.append(model)

    behavior_model_id = "behavior_reference"
    add(
        _planned_model(
            behavior_model_id,
            "behavior_reference",
            behavior,
            description="Frozen Behavior-only reference; task performance information, not attention ground truth.",
        )
    )

    for feature in registry:
        if feature.standalone_eligible:
            add(
                _planned_model(
                    f"standalone::{feature.feature_id}",
                    "standalone_feature",
                    [feature],
                    description=f"Standalone Q1-report prediction using {feature.scientific_feature_id}.",
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
                description=(
                    f"Behavior reference plus {feature.scientific_feature_id}; estimates conditional extra Q1-report information."
                ),
            )
        )
        behavior_pairs.append((behavior_model_id, model_id, feature.feature_id))

    full_model_id = "full"
    add(
        _planned_model(
            full_model_id,
            "full_model",
            full,
            description="All registry features explicitly frozen as eligible for the full Q1-report model.",
        )
    )

    full_pairs: list[tuple[str, str, str]] = []
    for feature in full:
        if not feature.full_leave_one_out_eligible:
            continue
        reduced = tuple(candidate for candidate in full if candidate.feature_id != feature.feature_id)
        if not reduced:
            continue
        model_id = f"full_minus::{feature.feature_id}"
        add(
            _planned_model(
                model_id,
                "full_minus_feature",
                reduced,
                description=f"Full model with {feature.scientific_feature_id} removed.",
            )
        )
        full_pairs.append((model_id, full_model_id, feature.feature_id))

    package_models: dict[str, str] = {}
    for package_id, package_devices in DEVICE_PACKAGES.items():
        selected: list[RegisteredFeature] = []
        seen_scientific: set[str] = set()
        for feature in registry:
            packages = feature.allowed_device_packages
            if feature.role == "behavior" and not packages:
                packages = tuple(DEVICE_PACKAGES)
            if package_id not in packages:
                continue
            required = set(feature.required_devices)
            if not required.issubset(package_devices):
                raise FeatureRegistryContractError(
                    f"{feature.feature_id}: generated package {package_id} lacks required devices"
                )
            if feature.scientific_feature_id in seen_scientific:
                raise FeatureRegistryContractError(
                    f"package {package_id} would include duplicate representations of {feature.scientific_feature_id}"
                )
            seen_scientific.add(feature.scientific_feature_id)
            selected.append(feature)
        if not selected:
            raise FeatureRegistryContractError(f"device package {package_id} has no registered features")
        model_id = package_id
        add(
            _planned_model(
                model_id,
                "device_package",
                selected,
                description=f"Device/information package {package_id}: {sorted(package_devices)}.",
            )
        )
        package_models[package_id] = model_id

    return FeatureComparisonPlan(
        registry=registry,
        models=tuple(models),
        behavior_increment_pairs=tuple(behavior_pairs),
        full_leave_one_out_pairs=tuple(full_pairs),
        device_package_model_ids=package_models,
    )
