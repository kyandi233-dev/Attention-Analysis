"""Authoritative formal RGB downstream orchestrator.

Consumes preserved RGB producer outputs only. The governed cohort defines session
membership; RGB availability is a modality-coverage property and never redefines
the cohort. Questionnaire/registry identity is overlaid through the shared formal
identity contract. No expensive face/pose model is invoked here.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.formal_analysis.cohort import canonical_session_id, included_cohort, load_cohort_manifest
from attention_pipeline.formal_analysis.identity_questionnaire import (
    build_identity_audit,
    load_questionnaire_data,
    load_repeat_registry,
    reconcile_cohort_identity,
    validate_questionnaire_registry_consistency,
)
from .figures import generate_rgb_figure_pack
from .pipeline import (
    _find_subject_file,
    _load_optional,
    _subject_dir,
    attach_behavior_context,
    build_multiscale,
    candidate_validation,
    derive_face_features,
    derive_motion_features,
    derive_pose_features,
)
from .science import (
    RGBScienceConfig,
    build_repeat_visit_sensitivity,
    build_time_on_task,
    build_within_between,
    model_contract_tables,
    participant_cluster_bootstrap,
    participant_exclusive_folds,
)

RGB_FORMAL_RUNNER_VERSION = "rgb-formal-downstream-v2"


def _mkdirs(root: Path) -> dict[str, Path]:
    result = {name: root / name for name in ("tables", "validation", "qc", "statistics", "prediction", "figures", "provenance")}
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _sha256_if_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _governed_identity(config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    identity_cfg = config.section("identity")
    cohort = load_cohort_manifest(
        config,
        path_key=str(identity_cfg.get("cohort_manifest_path_key", "cohort_manifest")),
    )
    included = included_cohort(cohort, require_groups=True)
    registry = load_repeat_registry(
        config,
        path_key=str(identity_cfg.get("repeat_registry_path_key", "repeat_registry")),
    )
    questionnaire = load_questionnaire_data(
        config,
        path_key=str(identity_cfg.get("questionnaire_path_key", "questionnaire_derived_data")),
    )
    consistency = validate_questionnaire_registry_consistency(questionnaire, registry)
    identity = reconcile_cohort_identity(included, registry)
    if identity["participant_group_id"].isna().any():
        missing = identity.loc[identity["participant_group_id"].isna(), "session_id"].tolist()
        raise ValueError("RGB formal participant grouping unresolved: " + ", ".join(missing))
    return included, identity, questionnaire, consistency


def _select_sessions(included: pd.DataFrame, subjects: Iterable[str] | None) -> list[str]:
    governed = included["session_id"].astype(str).tolist()
    governed_set = set(governed)
    if subjects is None:
        return governed
    requested = [canonical_session_id(value) for value in subjects]
    outside = sorted(set(requested) - governed_set)
    if outside:
        raise ValueError("requested RGB sessions are outside governed cohort: " + ", ".join(outside))
    return requested


def _identity_row(identity: pd.DataFrame, session: str) -> pd.Series:
    current = identity[identity["session_id"].eq(session)]
    if len(current) != 1:
        raise ValueError(f"RGB identity row count for {session}: {len(current)}")
    return current.iloc[0]


def _attach_identity(frame: pd.DataFrame, row: pd.Series, session: str) -> pd.DataFrame:
    out = frame.copy()
    out["session_id"] = session
    for column in (
        "participant_group_id", "participant_key", "participant_identity_source", "visit_order",
        "prior_visit_count", "total_visit_count", "is_first_visit", "identity_conflict_flag",
    ):
        if column in row.index:
            out[column] = row[column]
    return out


def _session_source_dir(raw_root: Path, session: str) -> Path | None:
    if not raw_root.is_dir():
        return None
    try:
        return _subject_dir(raw_root, session)
    except FileNotFoundError:
        return None


def _science_config(config) -> RGBScienceConfig:
    windows = config.section("windows")
    science = config.data.get("science", {}) if isinstance(config.data.get("science", {}), dict) else {}
    return RGBScienceConfig(
        time_bin_seconds=float(windows.get("time_on_task_bin_seconds", 10)),
        bootstrap_replicates=int(science.get("participant_cluster_bootstrap_replicates", 1000)),
        bootstrap_seed=int(science.get("participant_cluster_bootstrap_seed", 20260830)),
        prediction_folds=int(science.get("prediction_folds", 5)),
        minimum_time_bins_for_slope=int(science.get("minimum_time_bins_for_slope", 3)),
    )


def run_rgb_formal_v2(
    config_path: str | Path = "configs/rgb_formal.yaml",
    *,
    paths_config: str | Path | None = None,
    subjects: Iterable[str] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path, paths_config=paths_config)
    raw_root = config.path_value("raw_root")
    ready_root = config.path_value("analysis_ready_root")
    output_root = config.path_value("output_root")
    ready_root.mkdir(parents=True, exist_ok=True)
    dirs = _mkdirs(output_root)

    included, identity, questionnaire, consistency = _governed_identity(config)
    sessions = _select_sessions(included, subjects)
    scfg = _science_config(config)
    inp = config.section("inputs")

    native_parts: list[pd.DataFrame] = []
    probe_parts: list[pd.DataFrame] = []
    qc_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    source_manifest_rows: list[dict[str, Any]] = []

    for session in sessions:
        idrow = _identity_row(identity, session)
        subject_dir = _session_source_dir(raw_root, session)
        if subject_dir is None:
            qc_rows.append({
                "session_id": session, "participant_group_id": idrow["participant_group_id"],
                "rgb_source_present": 0, "face_present": 0, "pose_present": 0, "motion_present": 0,
                "status": "modality_missing_rgb_source_directory",
            })
            source_manifest_rows.append({"session_id": session, "status": "rgb_source_absent"})
            continue
        try:
            face_path = _find_subject_file(subject_dir, session, str(inp["face_suffix"]))
            pose_path = _find_subject_file(subject_dir, session, str(inp["pose_suffix"]))
            motion_path = _find_subject_file(subject_dir, session, str(inp["motion_suffix"]))
            face_raw = _load_optional(face_path)
            pose_raw = _load_optional(pose_path)
            motion = derive_motion_features(_load_optional(motion_path))
            face, blink_events, face_status = derive_face_features(face_raw, config)
            pose = derive_pose_features(pose_raw)
            face = attach_behavior_context(face, motion)
            pose = attach_behavior_context(pose, motion)

            current_native: list[pd.DataFrame] = []
            for modality, frame in (("face", face), ("pose", pose), ("motion", motion)):
                if frame.empty:
                    continue
                current = _attach_identity(frame, idrow, session)
                current["modality"] = modality
                current_native.append(current)
                native_parts.append(current)
                target = ready_root / session
                target.mkdir(parents=True, exist_ok=True)
                current.to_parquet(target / f"{session}_{modality}_derived.parquet", index=False)

            if not motion.empty and {"block", "trial_num", "probe_onset_time"}.issubset(motion.columns):
                probe_mask = pd.to_numeric(motion.get("is_probe"), errors="coerce").eq(1) & pd.to_numeric(
                    motion["probe_onset_time"], errors="coerce"
                ).notna()
                probes = motion.loc[probe_mask].drop_duplicates(["block", "trial_num", "probe_onset_time"]).copy()
                if not probes.empty:
                    probe_parts.append(_attach_identity(probes, idrow, session))

            qc_rows.append({
                "session_id": session,
                "participant_group_id": idrow["participant_group_id"],
                "participant_identity_source": idrow.get("participant_identity_source", pd.NA),
                "questionnaire_present": int(session in set(questionnaire["session_id"].astype(str))),
                "rgb_source_present": 1,
                "face_present": int(face_path is not None), "pose_present": int(pose_path is not None),
                "motion_present": int(motion_path is not None),
                "face_rows": int(len(face)), "pose_rows": int(len(pose)), "motion_rows": int(len(motion)),
                "blink_event_candidate_n": int(len(blink_events)),
                "face_primary_status": face_status.get("primary_face_status"),
                "blink_threshold_status": face_status.get("blink_threshold_status"),
                "status": "available_with_explicit_component_coverage",
            })
            source_manifest_rows.append({
                "session_id": session, "status": "available",
                "face_file": str(face_path) if face_path else None,
                "face_sha256": _sha256_if_file(face_path),
                "pose_file": str(pose_path) if pose_path else None,
                "pose_sha256": _sha256_if_file(pose_path),
                "motion_file": str(motion_path) if motion_path else None,
                "motion_sha256": _sha256_if_file(motion_path),
            })
        except Exception as exc:
            failures.append({
                "session_id": session, "stage": "rgb_analysis_ready", "error_type": type(exc).__name__,
                "error": str(exc), "scientific_interpretation": "structural_failure_not_no_effect",
            })

    identity_audit = build_identity_audit(identity, questionnaire)
    _write_csv(identity_audit, dirs["qc"] / "rgb_identity_audit.csv")
    _write_csv(consistency, dirs["qc"] / "questionnaire_registry_consistency.csv")
    _write_csv(pd.DataFrame(qc_rows), dirs["qc"] / "session_qc.csv")
    _write_csv(pd.DataFrame(failures), dirs["qc"] / "rgb_failures.csv")
    _write_csv(pd.DataFrame(source_manifest_rows), dirs["provenance"] / "rgb_source_manifest.csv")

    if failures:
        blocked = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "blocked_structural_failures",
            "pipeline_version": RGB_FORMAL_RUNNER_VERSION, "governed_session_n": int(len(sessions)),
            "structural_failure_n": int(len(failures)), "expensive_models_rerun": False,
            "scientific_inference_authorized": False,
        }
        (dirs["provenance"] / "rgb_formal_manifest.json").write_text(json.dumps(blocked, ensure_ascii=False, indent=2), encoding="utf-8")
        return blocked

    features = pd.concat(native_parts, ignore_index=True, sort=False) if native_parts else pd.DataFrame()
    probes = pd.concat(probe_parts, ignore_index=True, sort=False) if probe_parts else pd.DataFrame()
    features.to_parquet(ready_root / "rgb_feature_native_long.parquet", index=False)

    if features.empty:
        summary = pd.DataFrame(); probe_summary = pd.DataFrame()
    else:
        summary, probe_summary = build_multiscale(features, probes, config.section("windows")["probe_pre_seconds"])
    # Add visit metadata to summary without changing metric values.
    visit_meta = identity[[c for c in ("session_id", "participant_group_id", "participant_key", "visit_order", "prior_visit_count") if c in identity]].drop_duplicates("session_id")
    if not summary.empty:
        summary = summary.merge(visit_meta, on=[c for c in ("session_id", "participant_group_id") if c in visit_meta], how="left", validate="many_to_one")
    if not probe_summary.empty:
        probe_summary = probe_summary.merge(visit_meta, on=[c for c in ("session_id", "participant_group_id") if c in visit_meta], how="left", validate="many_to_one")

    validation, redundancy, decisions = candidate_validation(summary)
    within_between = build_within_between(summary)
    visit_sensitivity = build_repeat_visit_sensitivity(summary, identity)
    time_bins, time_slopes = build_time_on_task(
        features, bin_seconds=scfg.time_bin_seconds, minimum_bins=scfg.minimum_time_bins_for_slope
    ) if not features.empty else (pd.DataFrame(), pd.DataFrame())
    cluster_ci = participant_cluster_bootstrap(
        summary, replicates=scfg.bootstrap_replicates, seed=scfg.bootstrap_seed
    ) if not summary.empty else pd.DataFrame()
    fold_table, prediction_audit = participant_exclusive_folds(identity, n_folds=scfg.prediction_folds)
    model_failures, deferred_models = model_contract_tables()

    _write_csv(summary, dirs["tables"] / "rgb_multiscale_metrics.csv")
    _write_csv(probe_summary, dirs["tables"] / "rgb_probe_metrics.csv")
    _write_csv(time_bins, dirs["tables"] / "rgb_time_on_task_bins.csv")
    _write_csv(time_slopes, dirs["tables"] / "rgb_time_on_task_slopes.csv")
    _write_csv(validation, dirs["validation"] / "rgb_candidate_metric_validation.csv")
    _write_csv(redundancy, dirs["validation"] / "rgb_metric_redundancy.csv")
    _write_csv(decisions, dirs["validation"] / "rgb_endpoint_decisions.csv")
    _write_csv(within_between, dirs["validation"] / "rgb_candidate_within_between.csv")
    _write_csv(visit_sensitivity, dirs["validation"] / "rgb_repeat_visit_sensitivity.csv")
    _write_csv(cluster_ci, dirs["statistics"] / "rgb_participant_cluster_uncertainty.csv")
    _write_csv(model_failures, dirs["statistics"] / "model_failures.csv")
    _write_csv(deferred_models, dirs["statistics"] / "deferred_models.csv")
    _write_csv(fold_table, dirs["prediction"] / "rgb_participant_folds.csv")
    _write_csv(prediction_audit, dirs["prediction"] / "rgb_prediction_audit.csv")

    figure_manifest, figure_audit = generate_rgb_figure_pack(
        summary, probe_summary, time_bins, validation, output_root
    )
    _write_csv(figure_manifest, dirs["figures"] / "figure_manifest.csv")
    _write_csv(figure_audit, dirs["figures"] / "figure_coverage_audit.csv")

    cohort_missing_rgb_n = int(sum(1 for row in qc_rows if int(row.get("rgb_source_present", 0)) == 0))
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_code_contract_real_data_freeze_pending",
        "pipeline_version": RGB_FORMAL_RUNNER_VERSION,
        "config_digest": config.digest,
        "governed_session_n": int(len(sessions)),
        "participant_group_n": int(identity[identity["session_id"].isin(sessions)]["participant_group_id"].nunique()),
        "rgb_source_missing_session_n": cohort_missing_rgb_n,
        "sessions_with_rgb_features_n": int(features["session_id"].nunique()) if not features.empty else 0,
        "expensive_models_rerun": False,
        "native_rates_preserved": True,
        "common_frame_rate_forced": False,
        "strict_preprobe_anchor_exclusion": True,
        "endpoint_freeze": "pending_real_data_scientific_review",
        "blink_threshold_freeze": "pending_representative_visual_distribution_qc",
        "perclos_role": "supportive_alertness_candidate_only",
        "rppg_in_scope": False,
        "inference_prediction_separated": True,
        "prediction_model_status": "deferred_pending_endpoint_freeze",
        "scientific_inference_authorized_by_code_alone": False,
        "multimodal_fusion_status": "deferred_not_release_ready",
        "figure_internal_titles_allowed": False,
        "figure_coverage_rows": int(len(figure_audit)),
        "model_attempt_failure_rows": int(len(model_failures)),
        "notes": [
            "Governed cohort defines sessions; questionnaire and RGB availability never redefine cohort membership.",
            "Missing RGB source is recorded as modality missingness, not silently excluded from the cohort.",
            "All participant uncertainty/folds use participant_group_id, with verified participant_key where available.",
            "No predictor-by-outcome Cartesian inference/prediction models run before real-data endpoint freeze.",
        ],
    }
    (dirs["provenance"] / "rgb_formal_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return manifest
