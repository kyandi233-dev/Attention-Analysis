"""Authoritative lightweight formal RGB downstream orchestrator.

Authority: ``scripts/rgb_formal_downstream.py -> run_rgb_formal_v2``.
This downstream runner consumes preserved Parquet outputs only; it never reruns Face/Pose
models and never writes to the mmWave result namespace.  The active default contract is
Motion Energy + exposure control, Pose confirmation/direction, and independent algorithm-
defined Blink candidate events.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.formal_analysis.cohort import canonical_session_id, included_cohort, load_cohort_manifest
from attention_pipeline.formal_analysis.identity_contract import (
    DEFAULT_ALLOWED_LEGACY_IDENTITY_STATUSES,
    reconcile_formal_identity,
)
from attention_pipeline.formal_analysis.identity_questionnaire import (
    build_identity_audit,
    load_questionnaire_data,
    load_repeat_registry,
    validate_questionnaire_registry_consistency,
)
from .blink_candidates import derive_blink_candidates, read_face_projection
from .motion_qc import derive_motion_qc
from .pose_direction import derive_pose_direction

RGB_FORMAL_RUNNER_VERSION = "rgb-formal-downstream-v2.2-lightweight"


def _mkdirs(root: Path) -> dict[str, Path]:
    result = {name: root / name for name in ("tables", "qc", "provenance")}
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


def _subject_dir(raw_root: Path, session_id: str) -> Path:
    exact = raw_root / session_id
    if exact.is_dir():
        return exact
    matches = [
        path for path in raw_root.iterdir()
        if path.is_dir() and canonical_session_id(path.name) == session_id
    ] if raw_root.is_dir() else []
    if len(matches) != 1:
        raise FileNotFoundError(f"RGB subject directory unresolved for {session_id}: {matches}")
    return matches[0]


def _find_subject_file(subject_dir: Path, session_id: str, suffix: str) -> Path | None:
    exact = subject_dir / f"{session_id}{suffix}"
    if exact.is_file():
        return exact
    matches = sorted(subject_dir.glob(f"*{suffix}"))
    return matches[0] if len(matches) == 1 else None


def _session_source_dir(raw_root: Path, session: str) -> Path | None:
    try:
        return _subject_dir(raw_root, session)
    except FileNotFoundError:
        return None


def _governed_identity(config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return governed session membership plus reconciled participant identity.

    ``reconcile_formal_identity`` applies the same participant contract used by Behavior.
    Cohort membership is selected before participant-level estimability is required, so an
    unresolved participant never disappears from lightweight RGB QC.
    """
    identity_cfg = config.section("identity")
    cohort = load_cohort_manifest(
        config,
        path_key=str(identity_cfg.get("cohort_manifest_path_key", "cohort_manifest")),
    )
    included = included_cohort(cohort, require_groups=False)
    registry = load_repeat_registry(
        config,
        path_key=str(identity_cfg.get("repeat_registry_path_key", "repeat_registry")),
    )
    questionnaire = load_questionnaire_data(
        config,
        path_key=str(identity_cfg.get("questionnaire_path_key", "questionnaire_derived_data")),
    )
    consistency = validate_questionnaire_registry_consistency(questionnaire, registry)
    raw_allowed = identity_cfg.get(
        "allowed_legacy_identity_statuses", sorted(DEFAULT_ALLOWED_LEGACY_IDENTITY_STATUSES)
    )
    if not isinstance(raw_allowed, list) or not raw_allowed:
        raise ValueError("identity.allowed_legacy_identity_statuses must be a non-empty list")
    identity = reconcile_formal_identity(
        included,
        registry,
        legacy_status_column=str(identity_cfg.get("legacy_identity_status_column", "identity_status")),
        allowed_legacy_statuses=[str(value) for value in raw_allowed],
    )
    return included, identity, questionnaire, consistency


def participant_inference_gate(identity: pd.DataFrame) -> dict[str, Any]:
    """Describe whether participant-cluster inference could be estimated later.

    This runner does not perform bootstrap/CV/models.  The gate is persisted so a later
    explicitly enabled science stage cannot silently replace unresolved identity with session_id.
    """
    if identity.empty or "participant_group_id" not in identity.columns:
        return {"status": "not_estimable", "reason": "participant_group_id_missing", "unresolved_session_n": int(len(identity))}
    unresolved = identity["participant_group_id"].isna()
    if unresolved.any():
        return {
            "status": "not_estimable",
            "reason": "participant_identity_unresolved_no_session_id_fallback",
            "unresolved_session_n": int(unresolved.sum()),
        }
    return {"status": "available_if_explicitly_enabled", "reason": "", "unresolved_session_n": 0}


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


def _component_row(
    session: str,
    component: str,
    status: dict[str, Any],
    idrow: pd.Series,
) -> dict[str, Any]:
    # Blink has its own component status because primary-face QC can be generated while
    # blink events remain not estimable. Never let generic metadata override blink_status.
    if component == "blink_candidates":
        component_status = status.get("blink_status") or status.get("status") or "not_estimable"
        component_reason = status.get("blink_reason") or status.get("reason") or ""
    else:
        component_status = status.get("status") or "not_estimable"
        component_reason = status.get("reason") or ""
    return {
        "session_id": session,
        "participant_group_id": idrow.get("participant_group_id", pd.NA),
        "participant_identity_source": idrow.get("participant_identity_source", pd.NA),
        "component": component,
        "status": component_status,
        "reason": component_reason,
        **{f"detail__{key}": value for key, value in status.items() if key not in {"status", "reason", "blink_status", "blink_reason"}},
    }


def _disabled_contract_rows() -> list[dict[str, Any]]:
    return [
        {"component": "perclos", "active": False, "status": "disabled_deferred", "reason": "no validated closure-event contract"},
        {"component": "au", "active": False, "status": "disabled_deferred", "reason": "outside lightweight RGB QC contract"},
        {"component": "emotion", "active": False, "status": "disabled_deferred", "reason": "outside lightweight RGB QC contract"},
        {"component": "rppg", "active": False, "status": "disabled_deferred", "reason": "outside lightweight RGB QC contract"},
        {"component": "full_head_pose", "active": False, "status": "disabled_deferred", "reason": "pose confirmation uses body direction candidates only"},
        {"component": "rgb_prediction", "active": False, "status": "disabled_deferred", "reason": "endpoint freeze and multimodal fusion deferred"},
        {"component": "multimodal_fusion", "active": False, "status": "disabled_deferred", "reason": "single-modality quality gates first"},
    ]


def mmwave_protection_contract() -> dict[str, Any]:
    return {
        "status": "enforced_by_interface",
        "rgb_writes_mmwave_results": False,
        "rgb_creates_mmwave_truth_table": False,
        "blink_combined_with_motion_pose_risk_score": False,
        "required_future_mmwave_tracks": [
            "original_mmwave_result",
            "rgb_strict_motion_exclusion_sensitivity",
            "rgb_continuous_motion_adjustment",
        ],
    }


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
    inp = config.section("inputs")
    execution = config.section("execution")
    pose_cfg = config.section("pose_confirmation")
    ocular = config.section("ocular")
    blink_cfg = ocular.get("blink_candidate", {})
    ref_cfg = ocular.get("open_reference", {})

    component_rows: list[dict[str, Any]] = []
    session_qc_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    blink_event_parts: list[pd.DataFrame] = []

    questionnaire_sessions = set(questionnaire["session_id"].astype(str)) if "session_id" in questionnaire.columns else set()

    for session in sessions:
        idrow = _identity_row(identity, session)
        subject_dir = _session_source_dir(raw_root, session)
        identity_resolved = pd.notna(idrow.get("participant_group_id", pd.NA))
        if subject_dir is None:
            for component in ("motion", "pose_confirmation", "blink_candidates"):
                component_rows.append(_component_row(session, component, {"status": "not_estimable", "reason": "rgb_source_directory_missing"}, idrow))
            session_qc_rows.append({
                "session_id": session,
                "participant_group_id": idrow.get("participant_group_id", pd.NA),
                "participant_identity_source": idrow.get("participant_identity_source", pd.NA),
                "participant_identity_resolved": bool(identity_resolved),
                "questionnaire_present": int(session in questionnaire_sessions),
                "rgb_source_present": 0,
                "motion_status": "not_estimable",
                "pose_status": "not_estimable",
                "blink_status": "not_estimable",
            })
            source_rows.append({"session_id": session, "status": "rgb_source_absent"})
            continue

        face_path = _find_subject_file(subject_dir, session, str(inp["face_suffix"]))
        pose_path = _find_subject_file(subject_dir, session, str(inp["pose_suffix"]))
        motion_path = _find_subject_file(subject_dir, session, str(inp["motion_suffix"]))
        source_rows.append({
            "session_id": session,
            "status": "source_directory_present",
            "face_file": str(face_path) if face_path else None,
            "face_sha256": _sha256_if_file(face_path),
            "pose_file": str(pose_path) if pose_path else None,
            "pose_sha256": _sha256_if_file(pose_path),
            "motion_file": str(motion_path) if motion_path else None,
            "motion_sha256": _sha256_if_file(motion_path),
        })

        component_status: dict[str, dict[str, Any]] = {}

        # Stage 1: Motion-only can complete independently and first.
        try:
            if motion_path is None:
                motion = pd.DataFrame()
                motion_status = {"status": "not_estimable", "reason": "motion_source_missing"}
            else:
                motion, motion_status = derive_motion_qc(pd.read_parquet(motion_path))
                if not motion.empty:
                    target = ready_root / session
                    target.mkdir(parents=True, exist_ok=True)
                    _attach_identity(motion, idrow, session).to_parquet(target / f"{session}_motion_qc.parquet", index=False)
        except Exception as exc:
            motion = pd.DataFrame()
            motion_status = {"status": "not_estimable", "reason": f"motion_exception:{type(exc).__name__}:{exc}"}
            failure_rows.append({"session_id": session, "component": "motion", "error_type": type(exc).__name__, "error": str(exc)})
        component_status["motion"] = motion_status
        component_rows.append(_component_row(session, "motion", motion_status, idrow))

        # Stage 2: Pose confirmation is active by default but cannot invalidate completed Motion.
        pose_active = bool(execution.get("pose_confirmation_active", True))
        try:
            if not pose_active:
                pose = pd.DataFrame()
                pose_status = {"status": "disabled_deferred", "reason": "pose_confirmation_disabled_by_config"}
            elif pose_path is None:
                pose = pd.DataFrame()
                pose_status = {"status": "not_estimable", "reason": "pose_source_missing"}
            else:
                raw_pose = pd.read_parquet(pose_path)
                pose, pose_status = derive_pose_direction(
                    raw_pose,
                    min_visibility=float(pose_cfg.get("minimum_visibility", 0.5)),
                    min_presence=float(pose_cfg.get("minimum_presence", 0.5)),
                    gap_reset_ms=float(pose_cfg.get("gap_reset_ms", 300.0)),
                )
                if not pose.empty:
                    target = ready_root / session
                    target.mkdir(parents=True, exist_ok=True)
                    _attach_identity(pose, idrow, session).to_parquet(target / f"{session}_pose_confirmation.parquet", index=False)
        except Exception as exc:
            pose = pd.DataFrame()
            pose_status = {"status": "not_estimable", "reason": f"pose_exception:{type(exc).__name__}:{exc}"}
            failure_rows.append({"session_id": session, "component": "pose_confirmation", "error_type": type(exc).__name__, "error": str(exc)})
        component_status["pose_confirmation"] = pose_status
        component_rows.append(_component_row(session, "pose_confirmation", pose_status, idrow))

        # Stage 3: Blink candidates are independent and never block Motion/Pose output.
        blink_active = bool(execution.get("blink_candidates_active", True))
        try:
            if not blink_active:
                blink_frames = pd.DataFrame(); blink_events = pd.DataFrame()
                blink_status = {"status": "disabled_deferred", "reason": "blink_candidates_disabled_by_config"}
            elif face_path is None:
                blink_frames = pd.DataFrame(); blink_events = pd.DataFrame()
                blink_status = {"blink_status": "not_estimable", "blink_reason": "face_source_missing"}
            else:
                face = read_face_projection(face_path)
                blink_frames, blink_events, blink_status = derive_blink_candidates(
                    face,
                    preferred_phase=str(ref_cfg.get("preferred_phase", "baseline")),
                    minimum_valid_frames=int(ref_cfg.get("minimum_valid_frames", 30)),
                    relative_openness_threshold=float(blink_cfg.get("relative_openness_threshold", 0.20)),
                    minimum_closed_duration_ms=float(blink_cfg.get("minimum_closed_duration_ms", 50)),
                    maximum_closed_duration_ms=float(blink_cfg.get("maximum_closed_duration_ms", 1000)),
                    gap_reset_ms=float(blink_cfg.get("gap_reset_ms", 250)),
                    maximum_bilateral_relative_difference=float(blink_cfg.get("maximum_bilateral_relative_difference", 0.35)),
                )
                target = ready_root / session
                target.mkdir(parents=True, exist_ok=True)
                if not blink_frames.empty:
                    _attach_identity(blink_frames, idrow, session).to_parquet(target / f"{session}_blink_candidate_frames.parquet", index=False)
                if not blink_events.empty:
                    current_events = _attach_identity(blink_events, idrow, session)
                    current_events.to_parquet(target / f"{session}_blink_candidate_events.parquet", index=False)
                    blink_event_parts.append(current_events)
        except Exception as exc:
            blink_frames = pd.DataFrame(); blink_events = pd.DataFrame()
            blink_status = {"blink_status": "not_estimable", "blink_reason": f"blink_exception:{type(exc).__name__}:{exc}"}
            failure_rows.append({"session_id": session, "component": "blink_candidates", "error_type": type(exc).__name__, "error": str(exc)})
        component_status["blink_candidates"] = blink_status
        component_rows.append(_component_row(session, "blink_candidates", blink_status, idrow))

        session_qc_rows.append({
            "session_id": session,
            "participant_group_id": idrow.get("participant_group_id", pd.NA),
            "participant_identity_source": idrow.get("participant_identity_source", pd.NA),
            "participant_identity_resolved": bool(identity_resolved),
            "questionnaire_present": int(session in questionnaire_sessions),
            "rgb_source_present": 1,
            "motion_status": motion_status.get("status", "not_estimable"),
            "pose_status": pose_status.get("status", "not_estimable"),
            "blink_status": blink_status.get("blink_status", blink_status.get("status", "not_estimable")),
            "blink_event_candidate_n": int(len(blink_events)),
            "motion_pose_blink_combined_risk_score": False,
        })

    identity_audit = build_identity_audit(identity, questionnaire)
    inference_gate = participant_inference_gate(identity[identity["session_id"].isin(sessions)].copy())
    _write_csv(identity_audit, dirs["qc"] / "rgb_identity_audit.csv")
    _write_csv(consistency, dirs["qc"] / "questionnaire_registry_consistency.csv")
    _write_csv(pd.DataFrame(session_qc_rows), dirs["qc"] / "session_qc.csv")
    _write_csv(pd.DataFrame(component_rows), dirs["qc"] / "rgb_component_status.csv")
    _write_csv(pd.DataFrame(failure_rows), dirs["qc"] / "rgb_component_failures.csv")
    _write_csv(pd.DataFrame(source_rows), dirs["provenance"] / "rgb_source_manifest.csv")
    _write_csv(pd.DataFrame(_disabled_contract_rows()), dirs["provenance"] / "rgb_deferred_contracts.csv")

    all_blinks = pd.concat(blink_event_parts, ignore_index=True, sort=False) if blink_event_parts else pd.DataFrame()
    _write_csv(all_blinks, dirs["tables"] / "rgb_blink_candidate_events.csv")

    mmwave_contract = mmwave_protection_contract()
    perclos = {
        "active": False,
        "status": "disabled_deferred",
        "reason": "no validated closure-event contract",
    }
    component_table = pd.DataFrame(component_rows)
    generated_n = int(component_table["status"].eq("generated").sum()) if not component_table.empty else 0
    not_estimable_n = int(component_table["status"].eq("not_estimable").sum()) if not component_table.empty else 0
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "lightweight_rgb_qc_complete_with_explicit_component_status",
        "pipeline_version": RGB_FORMAL_RUNNER_VERSION,
        "config_digest": config.digest,
        "governed_session_n": int(len(sessions)),
        "participant_group_n_resolved": int(identity.loc[identity["session_id"].isin(sessions), "participant_group_id"].nunique(dropna=True)),
        "participant_identity_unresolved_session_n": int(identity.loc[identity["session_id"].isin(sessions), "participant_group_id"].isna().sum()),
        "participant_inference_gate": inference_gate,
        "active_default_route": ["motion_energy_with_exposure_control", "pose_confirmation_and_direction", "algorithm_defined_blink_candidates"],
        "motion_only_can_complete_first": True,
        "pose_confirmation_nonblocking": True,
        "blink_nonblocking": True,
        "extended_science_default": False,
        "bootstrap_run": False,
        "prediction_cv_run": False,
        "formal_multimodal_fusion_run": False,
        "large_figure_suite_run": False,
        "manual_video_or_image_review_run": False,
        "perclos": perclos,
        "mmwave_protection_contract": mmwave_contract,
        "component_generated_rows": generated_n,
        "component_not_estimable_rows": not_estimable_n,
        "component_exception_rows": int(len(failure_rows)),
        "scientific_inference_authorized_by_code_alone": False,
        "real_data_smoke_status": "not_run_by_web_task",
        "full_44_session_status": "not_run_by_web_task",
        "notes": [
            "Governed cohort defines session membership; questionnaire presence never defines the cohort.",
            "participant_group_id comes from reconciled current or validated legacy identity; session_id is never used as a participant fallback.",
            "Motion Energy and exposure/gray-level change remain separate tracks.",
            "Pose direction is an auxiliary QC candidate and not physical displacement truth.",
            "Blink outputs are algorithm-defined candidate events without manual visual validation.",
            "PERCLOS is retained historically but disabled/deferred until a validated closure-event contract exists.",
        ],
    }
    (dirs["provenance"] / "rgb_formal_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return manifest
