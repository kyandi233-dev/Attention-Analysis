"""Build comparison-specific Task-B analysis sets for the frozen scientific registry.

This is the missing Task-B producer for the behaviour/ocular/movement comparison
programme. It generalises the pattern already proven by
``mmwave_cardiopulmonary_ingest.py`` (``audit_quality`` -> ``build_analysis_sets`` ->
``write_quality_audit`` + ``write_analysis_sets``) rather than inventing a second
mechanism, so the mmWave ingest and the multi-modality comparison sets stay on one
contract.

Layering (see ``docs/supervised_learning/1.16.10-feature-and-file-interface.md``):

* the single-modality producers decide what a measurement IS, how it is cleaned and
  where it is not computable;
* this layer decides which probes a given comparison can actually use; and
* Task A only trains and evaluates on the already-frozen predictors and the
  comparison-specific sample.

Hard rules enforced here:

1. **Key normalisation only.** Producer tables are key-normalised and joined on the
   canonical probe key. No value is transformed, imputed, zero-filled or dropped.
2. **Fail-closed on non-finite frozen values.** A frozen representation is computable
   exactly when the producer emitted a finite value. A non-finite frozen value is NOT a
   residual single-feature gap that may be imputed later: the frozen estimability rule
   (for example the pupil 20 s minimum temporal span, or an insufficient number of valid
   frames) simply did not yield that representation. Such a probe is therefore made
   non-eligible for the training-fold missing strategy, so ``included_missing_aware``
   never claims imputability for an upstream not-estimable state. This is deliberately
   conservative and is reported explicitly.
3. **Never infer one layer from another.** Scientific modality, producer source
   namespace and required device are registered independently and are only read here.
4. **One analysis set = models that share one predictor union**, so every directly
   compared pair is trained and evaluated on the same participants, probes and folds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .analysis_sets import build_analysis_sets
from .quality_admission import audit_quality

# Behaviour science-v3 emits ``probe_order_in_block``; the canonical Task-B/multimodal
# key is ``probe_index_in_block``. The alias is explicit and fails closed when both
# columns exist and disagree.
PROBE_INDEX_ALIASES = ("probe_index_in_block", "probe_order_in_block")
GROUP = "participant_group_id"
ALLOWED_MEMBERSHIP_TYPES = ("included_complete", "included_missing_aware")
DEFAULT_REQUIRED_OUTCOMES = ("q1_nominal_4class",)


class SupervisedComparisonSetError(ValueError):
    """Raised when the frozen registry or the producer tables cannot form valid sets."""


@dataclass(frozen=True)
class SupervisedComparisonSetResult:
    quality: dict[str, pd.DataFrame]
    analysis_sets: pd.DataFrame
    analysis_set_summary: pd.DataFrame
    manifest: dict[str, Any]


def _resolve_probe_index(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return ``frame`` carrying a validated canonical probe-index column.

    Raises when neither alias exists, or when both exist but disagree row-for-row.
    Silent precedence is never allowed because it would change probe identity.
    """
    canonical, alias = PROBE_INDEX_ALIASES
    present = [column for column in PROBE_INDEX_ALIASES if column in frame.columns]
    if not present:
        raise SupervisedComparisonSetError(
            f"{label}: no probe-index column; expected one of {list(PROBE_INDEX_ALIASES)}"
        )
    if canonical in frame.columns and alias in frame.columns:
        canonical_values = pd.to_numeric(frame[canonical], errors="coerce")
        alias_values = pd.to_numeric(frame[alias], errors="coerce")
        mismatch = (
            canonical_values.isna() | alias_values.isna() | canonical_values.ne(alias_values)
        )
        if bool(mismatch.any()):
            sample = frame.loc[mismatch, [canonical, alias]].head(5)
            raise SupervisedComparisonSetError(
                f"{label}: probe-index aliases disagree; refusing silent precedence; "
                f"sample={sample.to_dict(orient='records')}"
            )
        return frame
    if canonical in frame.columns:
        return frame
    resolved = frame.copy()
    resolved[canonical] = resolved[alias]
    return resolved


def _normalize_identity(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Normalise block/probe/participant identity without touching measurement values."""
    out = _resolve_probe_index(frame, label)
    required = ["session_id", "block_id", "probe_index_in_block", GROUP]
    missing = sorted(set(required) - set(out.columns))
    if missing:
        raise SupervisedComparisonSetError(f"{label}: missing identity columns {missing}")

    out = out.copy()
    out["session_id"] = out["session_id"].astype("string").str.strip()
    out[GROUP] = out[GROUP].astype("string").str.strip()
    blocks = out["block_id"].astype("string").str.strip().str.lower()
    blocks = blocks.str.replace("block", "", regex=False).str.replace("-", "", regex=False)
    blocks = blocks.replace({"1": "b1", "2": "b2"})
    out["block_id"] = blocks.where(blocks.str.startswith("b"), "b" + blocks)

    probe = pd.to_numeric(out["probe_index_in_block"], errors="coerce")
    if probe.isna().any() or not np.isclose(probe, np.round(probe), atol=0.0, rtol=0.0).all():
        raise SupervisedComparisonSetError(
            f"{label}: probe_index_in_block must be finite integer-valued"
        )
    out["probe_index_in_block"] = probe.astype(int)

    if out[required].isna().any().any():
        raise SupervisedComparisonSetError(f"{label}: incomplete canonical identity")
    if out.duplicated(["session_id", "block_id", "probe_index_in_block"]).any():
        raise SupervisedComparisonSetError(f"{label}: duplicate canonical probe keys")
    if out.groupby("session_id")[GROUP].nunique().gt(1).any():
        raise SupervisedComparisonSetError(f"{label}: session maps to multiple participant groups")
    return out


def _namespace_tables(
    behavior: pd.DataFrame,
    ocular: pd.DataFrame,
    movement: pd.DataFrame,
    cardiopulmonary: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Assemble one producer table per source namespace.

    ``rgb`` must carry both the blink column (produced in the ocular wide table) and the
    body-motion column (produced in the movement table). They are joined on the canonical
    probe key with a fail-closed check, never concatenated by position.

    ``mmwave`` is the cardiopulmonary source namespace. It is supplied only when the frozen
    registry actually registers cardiopulmonary features; the caller decides that, and the
    feature-block step fails closed if a registered feature's namespace has no table.
    """
    behavior_n = _normalize_identity(behavior, "behavior")
    ocular_n = _normalize_identity(ocular, "ocular")
    movement_n = _normalize_identity(movement, "movement")

    key = ["session_id", "block_id", "probe_index_in_block"]
    if not ocular_n[key].merge(movement_n[key], on=key, how="inner").shape[0] == len(ocular_n):
        # A missing key here would silently shrink the rgb namespace; refuse instead.
        left = set(map(tuple, ocular_n[key].to_numpy()))
        right = set(map(tuple, movement_n[key].to_numpy()))
        if left - right:
            raise SupervisedComparisonSetError(
                "ocular and movement probe keys disagree; refusing to build the rgb namespace"
            )
    rgb = ocular_n.merge(movement_n, on=key + [GROUP], how="inner", validate="one_to_one")

    # Each namespace declares explicit source evidence. Finiteness of a value must never
    # be used to prove that a source was present.
    behavior_n["source_present"] = True
    behavior_n["source_readable"] = True
    behavior_n["window_seconds_nominal"] = 30
    for frame in (ocular_n, rgb):
        frame["source_present"] = True
        frame["source_readable"] = True
        frame["window_seconds_nominal"] = 30
    tables = {"behavior": behavior_n, "nir": ocular_n, "rgb": rgb}
    if cardiopulmonary is not None:
        mmwave_n = _normalize_identity(cardiopulmonary, "cardiopulmonary")
        mmwave_n["source_present"] = True
        mmwave_n["source_readable"] = True
        mmwave_n["window_seconds_nominal"] = 30
        tables["mmwave"] = mmwave_n
    return tables


def _feature_blocks(
    registry_features: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Group predictor columns by the source namespace the registry declares.

    Every registered feature additionally gets a per-feature QC gate column derived from
    finiteness, which is how the fail-closed non-imputability rule above is expressed
    inside the existing ``quality_admission`` contract.
    """
    blocks: dict[str, list[str]] = {}
    for feature in registry_features:
        namespace = str(feature["source_namespace"])
        columns = [str(column) for column in feature["columns"]]
        blocks.setdefault(namespace, []).extend(columns)
    return blocks


def add_finiteness_qc_gates(
    tables: Mapping[str, pd.DataFrame],
    registry_features: Sequence[Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    """Add ``<column>_qc_valid = isfinite(value)`` for every registered predictor.

    ``quality_admission._native_qc_state`` already consults ``f"{feature}_qc_valid"``, so
    this makes every non-finite frozen value an explicit upstream QC failure rather than a
    residual single-feature gap eligible for training-fold imputation.
    """
    out: dict[str, pd.DataFrame] = {}
    for namespace, frame in tables.items():
        frame = frame.copy()
        for feature in registry_features:
            if str(feature["source_namespace"]) != namespace:
                continue
            for column in feature["columns"]:
                column = str(column)
                if column not in frame.columns:
                    raise SupervisedComparisonSetError(
                        f"{namespace}: registered predictor column missing from the producer "
                        f"table: {column}"
                    )
                numeric = pd.to_numeric(frame[column], errors="coerce")
                frame[f"{column}_qc_valid"] = np.isfinite(numeric).to_numpy()
        out[namespace] = frame
    return out


def analysis_set_groups(plan: Any) -> dict[str, tuple[str, ...]]:
    """Group planned models into analysis sets that share one predictor union.

    Every group is a set of models that can be directly compared because they are trained
    and evaluated on exactly the same sample, participants and folds.
    """
    models = {model.model_id: model for model in plan.models}
    groups: dict[str, tuple[str, ...]] = {}

    def keep(predicate) -> tuple[str, ...]:
        return tuple(model_id for model_id, model in models.items() if predicate(model))

    groups["AS.behavior_reference"] = ("behavior_reference",)

    ocular_feature_ids = tuple(
        model.feature_ids[0]
        for model_id, model in models.items()
        if model_id.startswith("standalone::")
        and model.modalities == ("ocular",)
    )
    movement_feature_ids = tuple(
        model.feature_ids[0]
        for model_id, model in models.items()
        if model_id.startswith("standalone::")
        and model.modalities == ("movement",)
    )
    # 心肺是本注册表的第四个科学模态（来源命名空间 mmwave）。它与眼部/动作走完全相同的
    # 分组模式；缺了这一组，注册表里 modality::cardiopulmonary、
    # behavior_plus_modality::cardiopulmonary 与逐特征 behavior_plus::cardiopulmonary.*
    # 就没有任何分析集合覆盖，构造器会 fail-closed 报"planned models are not covered"。
    cardiopulmonary_feature_ids = tuple(
        model.feature_ids[0]
        for model_id, model in models.items()
        if model_id.startswith("standalone::")
        and model.modalities == ("cardiopulmonary",)
    )

    groups["AS.behavior_plus_ocular"] = keep(
        lambda m: m.model_id
        in {
            "behavior_reference",
            "modality::ocular",
            "behavior_plus_modality::ocular",
            *(f"behavior_plus::{feature_id}" for feature_id in ocular_feature_ids),
        }
    )
    groups["AS.behavior_plus_movement"] = keep(
        lambda m: m.model_id
        in {
            "behavior_reference",
            "modality::movement",
            "behavior_plus_modality::movement",
            *(f"behavior_plus::{feature_id}" for feature_id in movement_feature_ids),
        }
    )
    groups["AS.behavior_plus_cardiopulmonary"] = keep(
        lambda m: m.model_id
        in {
            "behavior_reference",
            "modality::cardiopulmonary",
            "behavior_plus_modality::cardiopulmonary",
            *(f"behavior_plus::{feature_id}" for feature_id in cardiopulmonary_feature_ids),
        }
    )
    if plan.sensor_joint_model_id:
        groups["AS.sensor_only_joint"] = (plan.sensor_joint_model_id,)

    groups["AS.full"] = keep(
        lambda m: m.model_id == "full" or m.model_id.startswith("full_minus")
    )

    for model_id in models:
        if model_id.startswith("standalone::"):
            groups[f"AS.standalone::{model_id.split('::', 1)[1]}"] = (model_id,)

    for model_id in models:
        if models[model_id].comparison_role == "device_package":
            groups[f"AS.device::{model_id}"] = (model_id,)

    for set_id, model_ids in groups.items():
        if not model_ids:
            raise SupervisedComparisonSetError(
                f"analysis set {set_id} resolved to no models; the frozen plan changed "
                "in a way this grouping does not cover"
            )
    also_planned = set(models) - {mid for ids in groups.values() for mid in ids}
    if also_planned:
        raise SupervisedComparisonSetError(
            f"planned models are not covered by any analysis set: {sorted(also_planned)}"
        )
    return groups


def comparison_specs(
    plan: Any,
    groups: Mapping[str, Sequence[str]],
    *,
    required_outcomes: Sequence[str] = DEFAULT_REQUIRED_OUTCOMES,
) -> dict[str, dict[str, Any]]:
    """Generate the Task-B comparison spec for every analysis set."""
    specs: dict[str, dict[str, Any]] = {}
    for set_id, model_ids in groups.items():
        spec = plan.task_b_comparison_spec(
            list(model_ids), required_outcomes=list(required_outcomes)
        )
        specs[set_id] = {
            "models": spec["models"],
            "required_features": spec["required_features"],
            "required_feature_records": spec["required_feature_records"],
            "required_outcomes": spec["required_outcomes"],
        }
    return specs


def build_supervised_comparison_sets(
    *,
    plan: Any,
    registry_features: Sequence[Mapping[str, Any]],
    behavior_probes: pd.DataFrame,
    ocular_probes: pd.DataFrame,
    movement_probes: pd.DataFrame,
    cardiopulmonary_probes: pd.DataFrame | None = None,
    required_outcomes: Sequence[str] = DEFAULT_REQUIRED_OUTCOMES,
) -> SupervisedComparisonSetResult:
    """Audit availability and build every comparison-specific analysis set.

    ``cardiopulmonary_probes`` is required exactly when the registry declares features in the
    ``mmwave`` source namespace. Supplying it for the three-modality registry would silently add
    a namespace, and omitting it while cardiopulmonary features are registered would silently
    drop those predictors, so both directions fail closed.
    """
    declared_namespaces = {
        str(feature.get("source_namespace", "")).strip()
        for feature in registry_features
        if str(feature.get("source_namespace", "")).strip()
    }
    needs_mmwave = "mmwave" in declared_namespaces
    if needs_mmwave and cardiopulmonary_probes is None:
        raise SupervisedComparisonSetError(
            "the frozen registry registers mmwave-namespace features but no cardiopulmonary "
            "probe table was supplied; refusing to build analysis sets that would silently "
            "drop those predictors"
        )
    if not needs_mmwave and cardiopulmonary_probes is not None:
        raise SupervisedComparisonSetError(
            "a cardiopulmonary probe table was supplied but the frozen registry declares no "
            "mmwave-namespace feature; refusing to add an undeclared namespace"
        )

    tables = add_finiteness_qc_gates(
        _namespace_tables(behavior_probes, ocular_probes, movement_probes, cardiopulmonary_probes),
        registry_features,
    )
    quality = audit_quality(
        tables=tables,
        feature_blocks=_feature_blocks(registry_features),
        rules={},
    )

    groups = analysis_set_groups(plan)
    specs = comparison_specs(plan, groups, required_outcomes=required_outcomes)
    analysis_sets, summary = build_analysis_sets(
        quality["formal_probe_identity"], quality["probe_feature_status"], specs
    )

    status_rows: list[dict[str, Any]] = []
    for set_id, model_ids in groups.items():
        for membership in ALLOWED_MEMBERSHIP_TYPES:
            chosen = analysis_sets[
                analysis_sets["analysis_set_id"].astype(str).eq(set_id)
                & analysis_sets[membership].astype(bool)
            ]
            status_rows.append(
                {
                    "analysis_set_id": set_id,
                    "membership": membership,
                    "model_n": len(model_ids),
                    "probe_n": int(len(chosen)),
                    "session_n": int(chosen["session_id"].nunique()),
                    "participant_group_n": int(chosen[GROUP].nunique()),
                }
            )
    status = pd.DataFrame(status_rows)

    non_finite_gated = 0
    for namespace, frame in tables.items():
        for feature in registry_features:
            if str(feature["source_namespace"]) != namespace:
                continue
            for column in feature["columns"]:
                numeric = pd.to_numeric(frame[str(column)], errors="coerce")
                non_finite_gated += int((~np.isfinite(numeric)).sum())

    manifest = {
        "status": "complete",
        "schema_version": "supervised-comparison-sets-v1",
        "scientific_modality_source_namespace_devices_are_independent": True,
        "required_outcomes": list(required_outcomes),
        "n_registered_features": len(registry_features),
        "n_planned_models": len(plan.models),
        "n_analysis_sets": len(groups),
        "analysis_set_ids": sorted(groups),
        "n_probes_governed": int(len(quality["formal_probe_identity"])),
        "n_participant_groups": int(quality["formal_probe_identity"][GROUP].nunique()),
        "n_sessions": int(quality["formal_probe_identity"]["session_id"].nunique()),
        "non_finite_frozen_values_made_non_imputable": non_finite_gated,
        "missing_policy": (
            "no zero fill, no imputation, no silent row deletion; a non-finite frozen "
            "value is recorded as an upstream QC failure and is never eligible for the "
            "training-fold missing strategy"
        ),
        "analysis_set_status": status.to_dict(orient="records"),
        "imputation_exercised": False,
    }
    return SupervisedComparisonSetResult(
        quality=quality,
        analysis_sets=analysis_sets,
        analysis_set_summary=summary,
        manifest=manifest,
    )


def write_supervised_comparison_sets(
    output_dir: Any,
    result: SupervisedComparisonSetResult,
) -> dict[str, str]:
    """Write Task-B audit tables plus the analysis sets, refusing to overwrite."""
    from pathlib import Path

    from .analysis_sets import write_analysis_sets
    from .quality_admission import write_quality_audit

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing comparison-set output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    paths = write_quality_audit(output, result.quality)
    paths.update(write_analysis_sets(output, result.analysis_sets, result.analysis_set_summary))
    manifest_path = output / "comparison_sets_manifest.json"
    manifest_path.write_text(
        json.dumps(result.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["comparison_sets_manifest"] = str(manifest_path)
    return paths
