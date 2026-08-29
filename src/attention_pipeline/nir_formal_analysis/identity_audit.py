from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.formal_analysis.cohort import included_cohort, load_cohort_manifest
from attention_pipeline.formal_analysis.identity_questionnaire import (
    load_repeat_registry,
    reconcile_cohort_identity,
)
from .pupil_tables import selected_sessions

IDENTITY_AUDIT_VERSION = "nir-participant-identity-audit-v1"


def _resolve(config, key: str) -> Path:
    raw = config.section("paths").get(key)
    if raw in (None, ""):
        raise KeyError(f"formal pupil config missing paths.{key}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (config.path.parent.parent / path).resolve()
    return path


def _analysis_group_token(config, session_id: str) -> str:
    path = _resolve(config, "analysis_ready_root") / "frame_level" / session_id / f"{session_id}_nir_analysis_ready.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=["session_id", "analysis_group_token"], encoding="utf-8-sig")
    sessions = set(frame["session_id"].dropna().astype(str).unique())
    if sessions != {session_id}:
        raise ValueError(f"{session_id}: analysis-ready session_id mismatch: {sorted(sessions)}")
    tokens = frame["analysis_group_token"].dropna().astype(str).str.strip().unique()
    if len(tokens) != 1:
        raise ValueError(f"{session_id}: expected one analysis_group_token, got {tokens.tolist()}")
    return str(tokens[0])


def load_reconciled_identity(config_path: str | Path, *, paths_config: str | Path | None = None) -> pd.DataFrame:
    config = load_config(config_path, paths_config=paths_config)
    policy = config.section("identity")
    cohort = load_cohort_manifest(
        config,
        path_key=str(policy.get("cohort_manifest_path_key", "cohort_manifest")),
        session_column=str(policy.get("cohort_session_column", "session_id")),
        include_column=str(policy.get("cohort_include_column", "include")),
        group_column=str(policy.get("legacy_repeat_group_column", "repeat_participant_id")),
    )
    registry = load_repeat_registry(
        config,
        path_key=str(policy.get("repeat_registry_path_key", "repeat_registry")),
    )
    return reconcile_cohort_identity(cohort, registry)


def run_nir_identity_audit(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
    paths_config: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path, paths_config=paths_config)
    identity = load_reconciled_identity(config_path, paths_config=paths_config)
    governed = included_cohort(identity, require_groups=True)
    sessions = selected_sessions(config, subjects)
    if not sessions:
        raise ValueError("NIR identity audit has no selected sessions")

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    governed_map = governed.set_index("session_id", drop=False)
    for session_id in sessions:
        try:
            if session_id not in governed_map.index:
                raise ValueError("selected NIR session is outside governed cohort manifest")
            row = governed_map.loc[session_id]
            if isinstance(row, pd.DataFrame):
                raise ValueError("governed cohort contains duplicate session_id")
            token = _analysis_group_token(config, session_id)
            records.append({
                "session_id": session_id,
                "analysis_group_token": token,
                "participant_key": row.get("participant_key", pd.NA),
                "participant_group_id": row.get("participant_group_id", pd.NA),
                "participant_identity_source": row.get("participant_identity_source", pd.NA),
                "visit_order": row.get("visit_order", pd.NA),
                "prior_visit_count": row.get("prior_visit_count", pd.NA),
                "identity_conflict_flag": row.get("identity_conflict_flag", pd.NA),
                "legacy_repeat_participant_id": row.get("legacy_repeat_participant_id", pd.NA),
                "status": "loaded",
            })
        except Exception as exc:
            failures.append({
                "session_id": session_id,
                "stage": "nir_identity_load",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    root = _resolve(config, "output_root") / "identity_audit"
    root.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    failure_table = pd.DataFrame(failures, columns=["session_id", "stage", "error_type", "error"])

    partition_rows: list[dict[str, Any]] = []
    partition_ok = False
    if not table.empty and not failures:
        token_to_group = table.groupby("analysis_group_token")["participant_group_id"].nunique(dropna=True)
        group_to_token = table.groupby("participant_group_id")["analysis_group_token"].nunique(dropna=True)
        bad_tokens = token_to_group[token_to_group > 1]
        bad_groups = group_to_token[group_to_token > 1]
        missing_group = int(table["participant_group_id"].isna().sum())
        partition_ok = bool(bad_tokens.empty and bad_groups.empty and missing_group == 0)
        partition_rows.extend([
            {
                "audit": "analysis_group_token_to_participant_group_id",
                "status": "pass" if bad_tokens.empty else "fail",
                "conflict_n": int(len(bad_tokens)),
                "detail": ";".join(map(str, bad_tokens.index.tolist())),
            },
            {
                "audit": "participant_group_id_to_analysis_group_token",
                "status": "pass" if bad_groups.empty else "fail",
                "conflict_n": int(len(bad_groups)),
                "detail": ";".join(map(str, bad_groups.index.tolist())),
            },
            {
                "audit": "missing_participant_group_id",
                "status": "pass" if missing_group == 0 else "fail",
                "conflict_n": missing_group,
                "detail": "",
            },
        ])
        table["partition_parity_status"] = "pass" if partition_ok else "fail"
    else:
        partition_rows.append({
            "audit": "participant_partition_parity", "status": "not_estimable",
            "conflict_n": len(failures), "detail": "session load failure(s)",
        })

    table.to_csv(root / "nir_identity_group_audit.csv", index=False, encoding="utf-8-sig")
    failure_table.to_csv(root / "nir_identity_failures.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(partition_rows).to_csv(root / "nir_identity_partition_summary.csv", index=False, encoding="utf-8-sig")

    status = "complete" if not failures and partition_ok else "blocked"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": IDENTITY_AUDIT_VERSION,
        "status": status,
        "n_selected_nir_sessions": len(sessions),
        "n_loaded_sessions": int(len(table)),
        "n_failures": len(failures),
        "partition_parity": partition_ok,
        "grouping_semantics": "analysis_group_token may retain its storage label only when its partition is one-to-one equivalent to the governed participant_group_id partition",
        "participant_key_missing_policy": "preserve missing participant_key; do not invent it from session number",
        "outside_governed_cohort_allowed": False,
        "scientific_inference_authorized": bool(status == "complete"),
    }
    (root / "nir_identity_audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
