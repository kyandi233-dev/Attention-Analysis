from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.nir_analysis_ready.pupil_only import load_source_manifest
from attention_pipeline.nir_pupil_only import adapt_session_rows
from attention_pipeline.nir_formal_analysis.pupil_blink_audit_config import (
    audit_signals,
    blink_buffers,
    cleaning_tracks,
    fixed_bin_widths,
    load_audit_config,
    read_table,
    resolve_source,
    selected_records,
)
from attention_pipeline.nir_formal_analysis.pupil_blink_audit_probe import append_probe_track
from attention_pipeline.nir_formal_analysis.pupil_blink_measurement import (
    AUDIT_SCHEMA_VERSION,
    GEOMETRY_SIGNAL,
    RSEG_HARD_SIGNAL,
    add_rgb_blink_mask,
    audit_buffer_loss,
    audit_rseg_quality_associations,
    audit_signal_availability,
    build_blink_recovery_bins,
    derive_eye_measurements,
)
from attention_pipeline.nir_formal_analysis.pupil_blink_binocular import (
    audit_binocular_source_modes,
    build_binocular_measurement_timepoints,
)
from attention_pipeline.nir_formal_analysis.pupil_blink_sync import (
    audit_rgb_nir_sync_with_frames,
    load_session_rgb_blink_frames,
    rgb_blink_source_availability,
)

RUNNER_VERSION = "nir-pupil-blink-measurement-audit-runner-v2"
FORMAL_PHASES = {"block1", "block2"}


def _session_events(blinks: pd.DataFrame, session_id: str) -> pd.DataFrame:
    if blinks.empty:
        return pd.DataFrame(columns=["session_id", "start_unix_ms", "end_unix_ms"])
    if "session_id" not in blinks.columns:
        return blinks.copy()
    return blinks[blinks["session_id"].astype(str).eq(session_id)].copy()


def _validate_probe_identity(session_probes: pd.DataFrame, adapted: pd.DataFrame, session_id: str) -> None:
    if session_probes.empty or "participant_group_id" not in session_probes.columns:
        return
    probe_ids = session_probes["participant_group_id"].dropna().astype(str).str.strip()
    probe_ids = set(probe_ids[probe_ids.ne("")])
    if len(probe_ids) > 1:
        raise ValueError(f"session {session_id} has multiple probe participant_group_id values")
    if "participant_group_id" not in adapted.columns:
        return
    nir_ids = adapted["participant_group_id"].dropna().astype(str).str.strip()
    nir_ids = set(nir_ids[nir_ids.ne("")])
    if len(nir_ids) > 1:
        raise ValueError(f"session {session_id} has multiple NIR participant_group_id values")
    if probe_ids and nir_ids and probe_ids != nir_ids:
        raise ValueError(
            f"participant_group_id mismatch for {session_id}: "
            f"probe={sorted(probe_ids)}, nir={sorted(nir_ids)}"
        )


def _tables(
    *,
    sync_parts: list[pd.DataFrame],
    availability_parts: list[pd.DataFrame],
    source_mode_parts: list[pd.DataFrame],
    rseg_parts: list[pd.DataFrame],
    buffer_loss_parts: list[pd.DataFrame],
    recovery_parts: list[pd.DataFrame],
    probe_rows: list[dict[str, object]],
    trajectory_rows: list[dict[str, object]],
    warning_rows: list[dict[str, object]],
    failure_rows: list[dict[str, object]],
) -> dict[str, pd.DataFrame]:
    concat = lambda parts: pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    return {
        "rgb_nir_sync_audit.csv": concat(sync_parts),
        "signal_availability_audit.csv": concat(availability_parts),
        "binocular_source_mode_audit.csv": concat(source_mode_parts),
        "rseg_measurement_sensitivity_audit.csv": concat(rseg_parts),
        "blink_buffer_loss_audit.csv": concat(buffer_loss_parts),
        "blink_recovery_bins.csv": concat(recovery_parts),
        "probe_measurement_candidates.csv": pd.DataFrame(probe_rows),
        "probe_fixed_bin_trajectories.csv": pd.DataFrame(trajectory_rows),
        "measurement_audit_warnings.csv": pd.DataFrame(
            warning_rows, columns=["session_id", "stage", "warning_type", "warning"]
        ),
        "measurement_audit_failures.csv": pd.DataFrame(
            failure_rows, columns=["session_id", "stage", "error_type", "error"]
        ),
    }


def run_pupil_blink_measurement_audit(
    *,
    nir_config_path: str | Path,
    audit_config_path: str | Path,
    rgb_blink_events_path: str | Path,
    probe_table_path: str | Path,
    output_root: str | Path,
    rgb_blink_frames_root: str | Path | None = None,
    paths_config: str | Path | None = None,
    subjects: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Generate measurement-only audit tables; never fit Q1/Q2 or supervised models."""
    nir_config = load_config(nir_config_path, paths_config=paths_config)
    _, records = load_source_manifest(nir_config)
    records = selected_records(records, subjects)
    audit_cfg = load_audit_config(audit_config_path)
    buffers = blink_buffers(audit_cfg)
    bin_widths = fixed_bin_widths(audit_cfg)
    tracks = cleaning_tracks(audit_cfg)
    window_sec = float(audit_cfg.get("probe_window_sec", 30.0))
    include_soft = bool(audit_cfg.get("include_soft_sensitivity", True))
    recovery_cfg = audit_cfg.get("blink_recovery", {}) or {}

    blinks = read_table(rgb_blink_events_path)
    probes = read_table(probe_table_path)
    if "session_id" not in probes.columns:
        raise ValueError("probe table missing session_id")
    probes["session_id"] = probes["session_id"].astype(str)
    if not blinks.empty and "session_id" in blinks.columns:
        blinks["session_id"] = blinks["session_id"].astype(str)
    elif not blinks.empty and len(records) > 1:
        raise ValueError("non-empty combined RGB blink table must contain session_id")

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    sync_parts: list[pd.DataFrame] = []
    availability_parts: list[pd.DataFrame] = []
    source_mode_parts: list[pd.DataFrame] = []
    rseg_parts: list[pd.DataFrame] = []
    buffer_loss_parts: list[pd.DataFrame] = []
    recovery_parts: list[pd.DataFrame] = []
    probe_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    processed_sessions: list[str] = []
    rgb_blink_available_sessions: list[str] = []
    rgb_blink_unavailable_sessions: list[str] = []

    for record in records:
        session_id = str(record["session_id"])
        try:
            source = resolve_source(nir_config, str(record["source_csv"]))
            raw = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)
            adapted = adapt_session_rows(raw, record)
            adapted = adapted[adapted["phase"].astype(str).isin(FORMAL_PHASES)].copy()
            if adapted.empty:
                raise ValueError("session has no block1/block2 pupil rows")
            session_probes = probes[probes["session_id"].eq(session_id)].copy()
            if session_probes.empty:
                raise ValueError("session has no Behavior probe rows")
            _validate_probe_identity(session_probes, adapted, session_id)

            eye = derive_eye_measurements(adapted)
            timepoints = build_binocular_measurement_timepoints(adapted)
            events = _session_events(blinks, session_id)

            # RGB is an optional device dependency for the RGB-assisted tracks. A
            # present-but-unreadable frame table is therefore a non-fatal RGB source
            # problem, not a reason to discard otherwise valid NIR-only measurements.
            try:
                rgb_frames = load_session_rgb_blink_frames(rgb_blink_frames_root, session_id)
            except Exception as rgb_exc:
                rgb_frames = None
                warning_rows.append(
                    {
                        "session_id": session_id,
                        "stage": "rgb_blink_frame_source",
                        "warning_type": type(rgb_exc).__name__,
                        "warning": str(rgb_exc),
                    }
                )

            rgb_blink_available, rgb_blink_basis = rgb_blink_source_availability(events, rgb_frames)
            if rgb_blink_available:
                rgb_blink_available_sessions.append(session_id)
            else:
                rgb_blink_unavailable_sessions.append(session_id)

            sync = audit_rgb_nir_sync_with_frames(timepoints, events, rgb_frames)
            sync["rgb_blink_source_available"] = bool(rgb_blink_available)
            sync["rgb_blink_availability_basis"] = rgb_blink_basis
            if not rgb_blink_available:
                sync["sync_status"] = "rgb_blink_source_unavailable"
                sync["sync_evidence_level"] = rgb_blink_basis
            elif events.empty:
                sync["sync_evidence_level"] = rgb_blink_basis
            sync_parts.append(sync)

            availability_parts.append(audit_signal_availability(eye))
            source_mode_parts.append(audit_binocular_source_modes(timepoints))
            rseg_parts.append(audit_rseg_quality_associations(eye))
            if rgb_blink_available:
                buffer_loss_parts.append(audit_buffer_loss(timepoints, events, buffers=buffers))
            if rgb_blink_available and not events.empty:
                recovery_parts.append(
                    build_blink_recovery_bins(
                        timepoints,
                        events,
                        pre_ms=float(recovery_cfg.get("pre_ms", 500.0)),
                        post_ms=float(recovery_cfg.get("post_ms", 1000.0)),
                        bin_ms=float(recovery_cfg.get("bin_ms", 50.0)),
                        anchors=tuple(recovery_cfg.get("anchors", ["start", "end"])),
                    )
                )

            signals = audit_signals(timepoints, include_soft)
            plain_tracks = tuple(x for x in tracks if x in {"original_nir", "nir_qc_only"})
            for _, probe in session_probes.iterrows():
                if plain_tracks:
                    append_probe_track(
                        out_rows=probe_rows,
                        trajectory_rows=trajectory_rows,
                        timepoints=timepoints,
                        probe=probe,
                        signals=signals,
                        tracks=plain_tracks,
                        bin_widths=bin_widths,
                        window_sec=window_sec,
                        buffer_id="none",
                        buffer_pre_ms=None,
                        buffer_post_ms=None,
                    )

            blink_tracks = tuple(x for x in tracks if x in {"rgb_blink_only", "rgb_plus_nir_qc"})
            if rgb_blink_available:
                for buffer in buffers:
                    masked = add_rgb_blink_mask(
                        timepoints,
                        events,
                        pre_buffer_ms=buffer.pre_ms,
                        post_buffer_ms=buffer.post_ms,
                    )
                    for _, probe in session_probes.iterrows():
                        if blink_tracks:
                            append_probe_track(
                                out_rows=probe_rows,
                                trajectory_rows=trajectory_rows,
                                timepoints=masked,
                                probe=probe,
                                signals=signals,
                                tracks=blink_tracks,
                                bin_widths=bin_widths,
                                window_sec=window_sec,
                                buffer_id=buffer.name,
                                buffer_pre_ms=buffer.pre_ms,
                                buffer_post_ms=buffer.post_ms,
                            )
            processed_sessions.append(session_id)
        except Exception as exc:
            failure_rows.append(
                {
                    "session_id": session_id,
                    "stage": "measurement_audit",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    tables = _tables(
        sync_parts=sync_parts,
        availability_parts=availability_parts,
        source_mode_parts=source_mode_parts,
        rseg_parts=rseg_parts,
        buffer_loss_parts=buffer_loss_parts,
        recovery_parts=recovery_parts,
        probe_rows=probe_rows,
        trajectory_rows=trajectory_rows,
        warning_rows=warning_rows,
        failure_rows=failure_rows,
    )
    for name, table in tables.items():
        table.to_csv(root / name, index=False, encoding="utf-8-sig")

    warning_sessions = sorted({str(row["session_id"]) for row in warning_rows})
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_version": RUNNER_VERSION,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "status": "measurement_audit_outputs_generated_not_formal_freeze",
        "formal_contract": "FocusWave-Formal-Analysis codex/code-fix-ledger 1.16.2 + 1.16.3",
        "probe_window_sec": window_sec,
        "blink_buffers": [
            {"id": x.name, "pre_ms": x.pre_ms, "post_ms": x.post_ms} for x in buffers
        ],
        "fixed_bin_width_sec_candidates": list(bin_widths),
        "fixed_bin_candidates_are_formal_parameters": False,
        "cleaning_tracks": list(tracks),
        "include_soft_sensitivity": include_soft,
        "main_signals": [GEOMETRY_SIGNAL, RSEG_HARD_SIGNAL],
        "soft_signal_role": "sensitivity_only" if include_soft else "not_run",
        "turning_point_feature_generated": False,
        "interpolation_used_for_formal_statistics": False,
        "q1_q2_used_for_parameter_selection": False,
        "supervised_model_run": False,
        "m0_m7_run": False,
        "source_session_n_requested": int(len(records)),
        "source_session_n_processed": int(len(processed_sessions)),
        "source_session_n_failed": int(len(failure_rows)),
        "source_session_n_warning": int(len(warning_sessions)),
        "source_warning_sessions": warning_sessions,
        "rgb_frame_axis_requested": rgb_blink_frames_root is not None,
        "rgb_blink_source_available_session_n": int(len(rgb_blink_available_sessions)),
        "rgb_blink_source_unavailable_session_n": int(len(rgb_blink_unavailable_sessions)),
        "rgb_blink_source_unavailable_sessions": rgb_blink_unavailable_sessions,
        "rgb_missing_is_zero_blinks": False,
        "processed_sessions": processed_sessions,
        "output_tables": {name: int(len(table)) for name, table in tables.items()},
    }
    (root / "measurement_audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest
