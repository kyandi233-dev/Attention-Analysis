"""Q1-blind single-modality measurement review and explicit feature decisions.

Descriptive finite-value coverage never supplies scientific eligibility. Only an
upstream, evidence-backed decision chooses a primary representation. No labels,
fitting, imputation, cohort intersection, or automatic redundancy pruning occurs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

SCIENTIFIC_MODALITIES = ("behavior", "ocular", "movement", "cardiopulmonary")
SENSOR_DEVICES = frozenset({"nir", "rgb", "mmwave"})
REVIEW_VERSION = "single-modality-feature-review-v1"
IDENTITY_COLUMNS = ("participant_group_id", "session_id", "block_id", "probe_event_id")


class FeatureQualificationError(ValueError):
    """A catalog, decision, or source is inconsistent with measurement review."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strings(value: Any, name: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise FeatureQualificationError(f"{name} must be a list of nonblank strings")
    if len(set(value)) != len(value) or (not value and not allow_empty):
        raise FeatureQualificationError(f"{name} must be nonempty and unique")
    return value


def _check_catalog(catalog: Mapping[str, Any], modality: str) -> list[dict[str, Any]]:
    if modality not in SCIENTIFIC_MODALITIES:
        raise FeatureQualificationError(f"unknown scientific modality: {modality}")
    if catalog.get("schema_version") != 1 or not catalog.get("method_authority"):
        raise FeatureQualificationError("catalog requires schema_version=1 and method_authority")
    dimensions = catalog.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise FeatureQualificationError("catalog dimensions must be a nonempty list")
    ids: set[str] = set()
    for row in dimensions:
        if not isinstance(row, dict):
            raise FeatureQualificationError("dimension must be a mapping")
        name = row.get("scientific_feature_id")
        if not isinstance(name, str) or not name.strip() or name in ids:
            raise FeatureQualificationError("scientific_feature_id must be nonblank and unique")
        ids.add(name)
        if row.get("modality") not in SCIENTIFIC_MODALITIES:
            raise FeatureQualificationError(f"{name}: devices cannot be used as scientific modality")
        primary = _strings(row.get("primary_candidates"), f"{name}.primary_candidates", allow_empty=True)
        descriptive = _strings(row.get("descriptive_candidates", []), f"{name}.descriptive_candidates", allow_empty=True)
        if set(primary) & set(descriptive):
            raise FeatureQualificationError(f"{name}: primary and descriptive roles overlap")
        pairs = row.get("comparison_pairs", [])
        if not isinstance(pairs, list):
            raise FeatureQualificationError(f"{name}: comparison_pairs must be a list")
        seen_pairs: set[frozenset[str]] = set()
        for pair in pairs:
            _strings(pair, f"{name}.comparison_pair")
            if len(pair) != 2 or not set(pair) <= set(primary + descriptive) or frozenset(pair) in seen_pairs:
                raise FeatureQualificationError(f"{name}: invalid or duplicate representation comparison")
            seen_pairs.add(frozenset(pair))
        if not isinstance(row.get("required_for_modality"), bool):
            raise FeatureQualificationError(f"{name}: required_for_modality must be a boolean")
        devices = row.get("required_devices")
        if devices is not None:
            if set(_strings(devices, f"{name}.required_devices", allow_empty=True)) - SENSOR_DEVICES:
                raise FeatureQualificationError(f"{name}: unknown sensor device")
            if row["modality"] != "behavior" and not devices:
                raise FeatureQualificationError(f"{name}: sensor feature needs real devices")
        if not isinstance(row.get("raw_source"), str) or not row["raw_source"].strip():
            raise FeatureQualificationError(f"{name}: raw_source is required")
        _strings(row.get("preprocessing_dependencies", []), f"{name}.dependencies", allow_empty=True)
    selected = [row for row in dimensions if row["modality"] == modality]
    if not selected:
        raise FeatureQualificationError(f"catalog has no dimensions for {modality}")
    columns = [col for row in selected for col in row["primary_candidates"] + row.get("descriptive_candidates", [])]
    if len(columns) != len(set(columns)):
        raise FeatureQualificationError("one measured column cannot represent multiple scientific dimensions")
    forbidden = [col for col in columns if col.startswith(("q1", "q2", "p_q1", "predicted_")) or col in IDENTITY_COLUMNS]
    if forbidden:
        raise FeatureQualificationError(f"outcome/identity columns are not measurement candidates: {forbidden}")
    return selected


def _check_source(frame: pd.DataFrame) -> None:
    missing = set(IDENTITY_COLUMNS) - set(frame.columns)
    if frame.empty or missing:
        raise FeatureQualificationError(f"nonempty probe table with identity columns required; missing={sorted(missing)}")
    if frame.columns.duplicated().any():
        raise FeatureQualificationError("duplicate source columns")
    for column in IDENTITY_COLUMNS:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise FeatureQualificationError(f"blank identity: {column}")
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise FeatureQualificationError("duplicate probe identity; do not concatenate measurement tracks as probes")
    if frame.groupby("session_id")["participant_group_id"].nunique().gt(1).any():
        raise FeatureQualificationError("session assigned to multiple participant groups")


def _numeric(frame: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    raw = frame[column]
    missing = raw.isna() | raw.astype(str).str.strip().eq("")
    numeric = pd.to_numeric(raw, errors="coerce")
    malformed = ~missing & numeric.isna()
    finite = pd.Series(np.isfinite(numeric.to_numpy(dtype=float)), index=frame.index)
    return numeric.where(finite), missing, malformed


def _producer_checks(frame: pd.DataFrame, modality: str) -> list[dict[str, str]]:
    if modality != "behavior":
        return []
    # Reuse the producer handoff contract. Do not recompute masked measurements
    # or treat an old n>=20 table as evidence for the current n>=2 interface.
    from attention_pipeline.behavior_formal.behavior_supervised_interface import (
        BehaviorSupervisedInterfaceError, REQUIRED_OMISSION_PARTITION_COLUMNS,
        _validate_omission_handoff, _validate_rt_cv_handoff,
    )
    checks = []
    for name, required, validate in (
        ("behavior_rt_variability", ("rt_cv_min_n", "rt_cv_status", "correct_go_rt_opportunities", "go_correct_rt_cv"), _validate_rt_cv_handoff),
        ("behavior_go_omission", REQUIRED_OMISSION_PARTITION_COLUMNS, _validate_omission_handoff),
    ):
        missing = sorted(set(required) - set(frame.columns))
        status, reason = "passed", ""
        if missing:
            status, reason = "not_checked_missing_audit_columns", ",".join(missing)
        else:
            try:
                validate(frame)
            except BehaviorSupervisedInterfaceError as exc:
                status, reason = "failed", str(exc)
        checks.append({"scientific_feature_id": name, "status": status, "reason": reason})
    return checks


def review_single_modality(
    frame: pd.DataFrame,
    catalog: Mapping[str, Any],
    modality: str,
    *,
    decisions: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return aggregate review, representation contrasts and explicit release state.

    The API intentionally accepts no target/score arguments. Extra label columns
    in a source are neither read nor emitted. Source availability is reported,
    not used to change a scientific decision or delete a participant.
    """
    dimensions = _check_catalog(catalog, modality)
    _check_source(frame)
    producer_checks = _producer_checks(frame, modality)
    failed_producer_dimensions = {row["scientific_feature_id"] for row in producer_checks if row["status"] == "failed"}
    decisions = {} if decisions is None else decisions
    if not isinstance(decisions, Mapping):
        raise FeatureQualificationError("decisions must be a mapping")
    unknown = set(decisions) - {row["scientific_feature_id"] for row in dimensions}
    if unknown:
        raise FeatureQualificationError(f"decisions outside requested modality: {sorted(unknown)}")
    rows: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    dimension_states: list[dict[str, Any]] = []
    qualified: list[dict[str, Any]] = []
    for dimension in dimensions:
        name = dimension["scientific_feature_id"]
        primary = dimension["primary_candidates"]
        representation_state = "prespecified_pending_quality_confirmation" if len(primary) == 1 else "pending_representation_choice" if primary else "pending_producer_schema_freeze"
        candidates = [*primary, *dimension.get("descriptive_candidates", [])]
        decision = decisions.get(name, {})
        if not isinstance(decision, Mapping):
            raise FeatureQualificationError(f"{name}: decision must be a mapping")
        allowed = {"measurement_status", "selected_column", "evidence_refs", "required_devices", "preprocessing_dependencies"}
        if set(decision) - allowed:
            raise FeatureQualificationError(f"{name}: unknown decision fields {sorted(set(decision) - allowed)}")
        status = decision.get("measurement_status", "pending")
        if status not in {"pending", "qualified", "rejected"}:
            raise FeatureQualificationError(f"{name}: invalid measurement_status")
        chosen = decision.get("selected_column")
        if chosen is not None and chosen not in primary:
            raise FeatureQualificationError(f"{name}: selected column must be one allowed primary representation")
        evidence = decision.get("evidence_refs", [])
        _strings(evidence, f"{name}.evidence_refs", allow_empty=status == "pending")
        devices = decision.get("required_devices", dimension.get("required_devices"))
        dependencies = decision.get("preprocessing_dependencies", dimension.get("preprocessing_dependencies", []))
        _strings(dependencies, f"{name}.dependencies", allow_empty=True)
        if devices is not None:
            _strings(devices, f"{name}.required_devices", allow_empty=modality == "behavior")
            if set(devices) - SENSOR_DEVICES:
                raise FeatureQualificationError(f"{name}: invalid required_devices")
            expected = dimension.get("required_devices")
            if expected is not None and set(devices) != set(expected):
                raise FeatureQualificationError(f"{name}: decision cannot override known device dependencies")
        if not set(dimension.get("preprocessing_dependencies", [])) <= set(dependencies):
            raise FeatureQualificationError(f"{name}: decision cannot remove source preprocessing dependencies")
        if status == "qualified" and (chosen is None or devices is None):
            raise FeatureQualificationError(f"{name}: qualified needs a chosen representation and frozen devices")
        execution_state = "pending_measurement_freeze" if status == "pending" else "scientifically_rejected"
        values_by_column: dict[str, pd.Series] = {}
        for column in candidates or [None]:
            present = column is not None and column in frame.columns
            if present:
                values, missing, malformed = _numeric(frame, column)
                values_by_column[column] = values
                finite = values.notna()
                nonfinite_n = int((~missing & ~malformed & ~finite).sum())
                participant_coverage = finite.groupby(frame["participant_group_id"]).mean()
            else:
                values = pd.Series(np.nan, index=frame.index)
                missing = pd.Series(True, index=frame.index)
                malformed = pd.Series(False, index=frame.index)
                finite = values.notna()
                nonfinite_n = 0
                participant_coverage = finite.groupby(frame["participant_group_id"]).mean()
            observed = values.dropna()
            rows.append({
                "scientific_feature_id": name, "modality": modality,
                "report_name": dimension.get("report_name", name), "column": column,
                "role": "unresolved_representation" if column is None else "primary_candidate" if column in primary else "descriptive_sensitivity_qc_only",
                "measurement_status": status, "selected_primary": chosen == column and status == "qualified",
                "representation_state": representation_state,
                "source_column_present": present, "probe_n": len(frame),
                "participant_n": frame["participant_group_id"].nunique(),
                "session_n": frame["session_id"].nunique(),
                "finite_n": int(finite.sum()), "missing_n": int(missing.sum()),
                "malformed_n": int(malformed.sum()), "nonfinite_n": nonfinite_n,
                "finite_fraction": float(finite.mean()),
                "participant_macro_finite_fraction": float(participant_coverage.mean()),
                "pooled_probe_mean": float(observed.mean()) if len(observed) else np.nan,
                "pooled_probe_median": float(observed.median()) if len(observed) else np.nan,
                "pooled_probe_sd": float(observed.std()) if len(observed) > 1 else np.nan,
                "pooled_probe_q05": float(observed.quantile(.05)) if len(observed) else np.nan,
                "pooled_probe_q95": float(observed.quantile(.95)) if len(observed) else np.nan,
                "unique_finite_n": int(observed.nunique()),
                "automatic_selection_applied": False,
                "required_devices": json.dumps(devices),
                "preprocessing_dependencies": json.dumps(dependencies),
            })
            if status == "qualified" and chosen == column:
                execution_state = (
                    "source_column_missing" if not present else
                    "malformed_or_nonfinite_source" if malformed.any() or nonfinite_n else
                    "no_observed_value" if not finite.any() else "qualified_pending_task_b_probe_qc"
                )
        # Same dimension does not imply comparable units (e.g. CV vs SD, or
        # blink frequency vs duration). Only prespecified comparable pairs.
        for left, right in dimension.get("comparison_pairs", []):
            if left not in values_by_column or right not in values_by_column:
                continue
            pair = pd.concat([values_by_column[left], values_by_column[right]], axis=1)
            pair.columns = ["left", "right"]
            complete = pair.notna().all(axis=1)
            delta = pair["left"] - pair["right"]
            contrasts.append({
                "scientific_feature_id": name, "left_column": left, "right_column": right,
                "paired_probe_n": int(complete.sum()),
                "paired_participant_n": int(frame.loc[complete, "participant_group_id"].nunique()),
                "median_difference": float(delta.dropna().median()) if complete.any() else np.nan,
                "median_absolute_difference": float(delta.dropna().abs().median()) if complete.any() else np.nan,
                "participant_macro_mean_absolute_difference": float(delta.abs().groupby(frame["participant_group_id"]).mean().mean()),
                "automatic_selection_applied": False,
            })
        if status == "qualified" and name in failed_producer_dimensions:
            execution_state = "producer_contract_failed"
        dimension_states.append({
            "scientific_feature_id": name, "measurement_status": status,
            "representation_state": representation_state,
            "execution_state": execution_state, "selected_column": chosen,
            "required_for_modality": dimension["required_for_modality"], "evidence_refs": evidence,
        })
        if execution_state == "qualified_pending_task_b_probe_qc":
            qualified.append({
                "feature_id": f"{name}::{chosen}", "scientific_feature_id": name,
                "modality": modality, "feature_type": dimension.get("feature_type"),
                "columns": [chosen], "raw_source": dimension["raw_source"],
                "required_devices": devices, "preprocessing_dependencies": dependencies,
                "freeze_evidence": evidence,
            })
    required = [row for row in dimension_states if row["required_for_modality"]]
    ready = bool(required) and all(row["execution_state"] == "qualified_pending_task_b_probe_qc" for row in required)
    release = {
        "schema_version": REVIEW_VERSION, "modality": modality,
        "method_authority": catalog["method_authority"],
        "measurement_review_available": True,
        "modality_primary_representations_frozen": ready,
        "supervised_execution_authorized": False,
        "next_gate": "task_b_probe_qc_and_real_schema_smoke" if ready else "upstream_measurement_freeze",
        "dimensions": dimension_states, "qualified_primary_features": qualified,
        "producer_contract_checks": producer_checks,
        "label_values_read": False, "automatic_feature_selection_applied": False,
        "row_filter_applied": False,
        "coverage_interpretation": "finite values only; does not certify source, synchronization, QC or estimability",
    }
    return pd.DataFrame(rows), pd.DataFrame(contrasts, columns=[
        "scientific_feature_id", "left_column", "right_column", "paired_probe_n", "paired_participant_n",
        "median_difference", "median_absolute_difference", "participant_macro_mean_absolute_difference",
        "automatic_selection_applied",
    ]), release


def write_single_modality_review(
    input_path: str | Path, catalog_path: str | Path, modality: str,
    output_root: str | Path, *, decisions_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read a real probe table and write a new immutable aggregate review directory."""
    source, catalog_file, output = Path(input_path), Path(catalog_path), Path(output_root)
    if output.exists():
        raise FileExistsError(f"use a new review directory: {output}")
    catalog = yaml.safe_load(catalog_file.read_text(encoding="utf-8-sig"))
    if not isinstance(catalog, dict):
        raise FeatureQualificationError("catalog root must be a mapping")
    source_hash = file_sha256(source)
    catalog_hash = file_sha256(catalog_file)
    decisions: dict[str, Any] = {}
    if decisions_path is not None:
        payload = json.loads(Path(decisions_path).read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict) or set(payload) != {"source_sha256", "catalog_sha256", "modality", "decisions"}:
            raise FeatureQualificationError("decision file needs source_sha256, catalog_sha256, modality and decisions")
        if payload["source_sha256"] != source_hash or payload["catalog_sha256"] != catalog_hash or payload["modality"] != modality:
            raise FeatureQualificationError("decision provenance does not match source/catalog/modality")
        decisions = payload["decisions"]
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)
    elif source.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    else:
        raise FeatureQualificationError("input must be CSV or Parquet")
    review, contrasts, release = review_single_modality(frame, catalog, modality, decisions=decisions)
    release.update({"source_sha256": source_hash, "catalog_sha256": catalog_hash,
                    "decision_sha256": file_sha256(Path(decisions_path)) if decisions_path else None})
    output.mkdir(parents=True, exist_ok=False)
    review.to_csv(output / "feature_review.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(output / "representation_contrasts.csv", index=False, encoding="utf-8-sig")
    release["output_sha256"] = {p.name: file_sha256(p) for p in sorted(output.glob("*.csv"))}
    (output / "modality_release.json").write_text(json.dumps(release, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return release
