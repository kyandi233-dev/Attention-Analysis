"""Portable formal preflight plus explicitly legacy/deferred compatibility audits.

Current authoritative modality entrypoints are:
- Behavior: ``scripts/sart_formal_analysis.py``
- NIR: ``scripts/nir_formal_pipeline.py`` (staged pupil-only)
- RGB: ``scripts/rgb_formal_downstream.py``

The ``nir-adapt`` command below is retained only for historical CSV-adapter
reproducibility. ``merge-audit`` is a deferred scaffold and does not authorize
multimodal production analysis.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from attention_pipeline.config import Config, load_config
from attention_pipeline.formal_analysis.cohort import included_cohort, load_cohort_manifest, summarize_cohort
from attention_pipeline.formal_analysis.merge import UNIT_KEYS, merge_modalities
from attention_pipeline.formal_analysis.nir_adapter import adapt_nir_csv
from attention_pipeline.formal_analysis.provenance import collect_runtime_provenance


def _resolve_external(config: Config, value: str) -> Path:
    if config.path_registry is not None:
        return config.path_registry.resolve_spec(value, base_dir=Path.cwd())
    return Path(value).resolve()


def _cohort(config: Config) -> pd.DataFrame:
    cfg = config.section("cohort")
    return load_cohort_manifest(
        config,
        path_key=str(cfg.get("manifest_path_key", "cohort_manifest")),
        session_column=str(cfg.get("session_column", "session_id")),
        include_column=str(cfg.get("include_column", "include")),
        group_column=str(cfg.get("repeat_group_column", "repeat_participant_id")),
    )


def command_preflight(config: Config) -> dict[str, object]:
    if config.path_registry is None:
        raise ValueError("formal v2 preflight requires --paths-config or ATTENTION_ANALYSIS_PATHS_CONFIG")
    cohort = _cohort(config)
    summary = summarize_cohort(cohort)
    required_keys = list(config.section("path_contract").get("required_keys", []))
    paths: dict[str, object] = {}
    for key in required_keys:
        values = config.registry_paths(str(key))
        paths[str(key)] = [{"path": str(path), "exists": path.exists()} for path in values]
    return {
        "route_status": "current_single_modality_preflight",
        "science_config": str(config.path),
        "science_config_digest": config.digest,
        "paths_config": str(config.path_registry.path),
        "paths_config_digest": config.path_registry.digest,
        "cohort": {
            "sessions": summary.sessions,
            "groups": summary.groups,
            "repeated_groups": summary.repeated_groups,
            "repeated_sessions": summary.repeated_sessions,
        },
        "paths": paths,
    }


def _resolve_manifest_path(config: Config, manifest_path: Path, value: object) -> Path:
    raw = str(value).strip()
    if config.path_registry is not None:
        return config.path_registry.resolve_spec(raw, base_dir=manifest_path.parent)
    path = Path(raw)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _runtime_provenance(config: Config) -> dict[str, object]:
    provenance_cfg = config.section("provenance")
    evidence_env = str(
        provenance_cfg.get("evidence_repo_path_env", "ATTENTION_FORMAL_EVIDENCE_REPO")
    )
    evidence_path = os.environ.get(evidence_env)
    if not evidence_path:
        raise RuntimeError(
            f"缺少 {evidence_env}；无法在运行时解析真实 evidence Git commit，拒绝写入伪 provenance"
        )
    pipeline_cfg = config.section("pipeline")
    return collect_runtime_provenance(
        code_repo=Path(__file__).resolve().parents[1],
        evidence_repo=Path(evidence_path),
        evidence_repository=str(pipeline_cfg.get("evidence_repo", "")) or None,
        require_clean=bool(provenance_cfg.get("require_clean_git_checkout", True)),
    )


def _optional_record_text(record: dict[str, object], key: str) -> str | None:
    value = record.get(key)
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _legacy_nir_adapter_contract(config: Config) -> dict[str, object]:
    nir_cfg = config.section("nir")
    raw = nir_cfg.get("legacy_csv_adapter", {})
    legacy = dict(raw) if isinstance(raw, dict) else {}
    source_key = legacy.get("source_manifest_path_key", nir_cfg.get("source_manifest_path_key"))
    output_key = legacy.get("standardized_output_path_key", nir_cfg.get("standardized_output_path_key"))
    if source_key in (None, "") or output_key in (None, ""):
        raise ValueError("legacy NIR adapter path keys are not configured")
    return {
        "status": str(legacy.get("status", "legacy_compatibility_only")),
        "active_in_formal_pipeline": bool(legacy.get("active_in_formal_pipeline", False)),
        "source_manifest_path_key": str(source_key),
        "standardized_output_path_key": str(output_key),
        "may_authorize_formal_statistics": bool(legacy.get("may_authorize_formal_statistics", False)),
        "may_replace_staged_analysis_ready": bool(legacy.get("may_replace_staged_analysis_ready", False)),
    }


def command_nir_adapt(config: Config, *, sessions: list[str] | None, run_id: str | None) -> dict[str, object]:
    """Run the retained CSV NIR adapter for historical compatibility only."""
    if config.path_registry is None:
        raise ValueError("nir-adapt requires --paths-config or ATTENTION_ANALYSIS_PATHS_CONFIG")
    legacy = _legacy_nir_adapter_contract(config)
    if legacy["active_in_formal_pipeline"]:
        raise ValueError("legacy NIR adapter must not be marked active in the formal pipeline")
    source_manifest = config.registry_path(str(legacy["source_manifest_path_key"]))
    output_base = config.registry_path(str(legacy["standardized_output_path_key"]))
    manifest = pd.read_csv(source_manifest, encoding="utf-8-sig")
    required = {"session_id", "status", "eye_metrics_csv", "schema_version"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"NIR source manifest 缺少列: {sorted(missing)}")
    if manifest["session_id"].duplicated().any():
        dup = manifest.loc[manifest["session_id"].duplicated(keep=False), "session_id"].tolist()
        raise ValueError(f"NIR source manifest 每个session必须唯一: {sorted(set(map(str, dup)))}")

    cohort = included_cohort(_cohort(config), require_groups=False)
    cohort_summary = summarize_cohort(cohort)
    allowed = set(cohort["session_id"].astype(str))
    manifest = manifest.loc[manifest["status"].astype(str).str.lower().eq("complete")].copy()
    manifest = manifest.loc[manifest["session_id"].astype(str).isin(allowed)].copy()
    complete_eligible_sessions = int(len(manifest))
    if sessions:
        requested = set(sessions)
        manifest = manifest.loc[manifest["session_id"].astype(str).isin(requested)].copy()
        missing_requested = requested - set(manifest["session_id"].astype(str))
        if missing_requested:
            raise ValueError(
                "请求场次不在 complete legacy NIR source manifest/cohort 交集中: "
                + ", ".join(sorted(missing_requested))
            )
    if manifest.empty:
        raise ValueError("没有可适配的 complete legacy NIR session")

    provenance = _runtime_provenance(config)
    nir_cfg = config.section("nir")
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = output_base / run_id
    if run_root.exists():
        raise FileExistsError(f"run_id 已存在，拒绝覆盖: {run_root}")
    frame_root = run_root / "frame_level"
    frame_root.mkdir(parents=True, exist_ok=False)

    rows = []
    for record in manifest.sort_values("session_id").to_dict("records"):
        session = str(record["session_id"])
        source_csv = _resolve_manifest_path(config, source_manifest, record["eye_metrics_csv"])
        adapted = adapt_nir_csv(
            source_csv,
            frame_root / f"{session}.csv",
            session_id=session,
            schema_version=record.get("schema_version"),
            reject_pir=bool(nir_cfg.get("reject_historical_pir", True)),
        )
        adapted["source_provenance"] = {
            "status": _optional_record_text(record, "status"),
            "source_pipeline_version": _optional_record_text(record, "source_pipeline_version"),
            "source_commit": _optional_record_text(record, "source_commit"),
            "source_selection_reason": _optional_record_text(record, "source_selection_reason"),
            "eye_metrics_csv": str(source_csv),
        }
        adapted["qc_provenance"] = {
            "preserved_in_frame_level_output": True,
            "tracks": list(nir_cfg.get("qc_tracks", {}).keys()),
        }
        rows.append(adapted)

    result = {
        "route_status": "legacy_compatibility_only",
        "formal_statistics_authorized": False,
        "may_replace_staged_analysis_ready": False,
        "run_id": run_id,
        "science_config_digest": config.digest,
        "paths_config_digest": config.path_registry.digest,
        "provenance": provenance,
        "source_manifest": str(source_manifest),
        "source_manifest_rows": int(len(manifest)),
        "nir_input_snapshot": {
            "complete_eligible_sessions": complete_eligible_sessions,
            "selected_sessions": int(len(manifest)),
            "production_completion_is_input_availability_only": True,
            "measurement_validity_or_formal_statistics_implied": False,
        },
        "cohort_snapshot": {
            "sessions": cohort_summary.sessions,
            "groups": cohort_summary.groups,
            "repeated_groups": cohort_summary.repeated_groups,
            "repeated_sessions": cohort_summary.repeated_sessions,
        },
        "science_boundaries": {
            "authoritative_nir_route": "scripts/nir_formal_pipeline.py",
            "nir_primary_line": "pupil-only",
            "pir_or_iris_outer_formal_line_allowed": False,
            "oar_role": "eye-opening-or-eyelid-candidate-qc-only",
            "oar_is_not_blink_rate_or_perclos": True,
            "behavior_window_gate_modified_here": False,
            "mmwave_contract_modified_here": False,
        },
        "sessions": rows,
    }
    (run_root / "run_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def command_merge_audit(config: Config, *, unit: str, table_specs: list[str], how: str, output: str | None) -> dict[str, object]:
    """Audit the retained historical merge scaffold; never authorize production fusion."""
    fusion = config.section("fusion")
    if str(fusion.get("status", "disabled_deferred")) != "disabled_deferred":
        raise ValueError("multimodal fusion must remain disabled_deferred until explicitly re-frozen")
    tables: dict[str, pd.DataFrame] = {}
    for spec in table_specs:
        if "=" not in spec:
            raise ValueError("--table 必须写成 modality=PATH 或 modality=@path:key")
        modality, raw = spec.split("=", 1)
        modality = modality.strip()
        if not modality:
            raise ValueError("modality 名不能为空")
        tables[modality] = pd.read_csv(_resolve_external(config, raw.strip()), encoding="utf-8-sig")
    merged, audit = merge_modalities(tables, unit=unit, how=how)
    result: dict[str, object] = {
        "route_status": "legacy_deferred_audit_only",
        "production_fusion_authorized": False,
        "unit": unit,
        "how": how,
        "rows": int(len(merged)),
        "modalities": audit.to_dict("records"),
    }
    if output:
        target = _resolve_external(config, output)
        if target.exists():
            raise FileExistsError(f"拒绝覆盖已有 merge 输出: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(target, index=False, encoding="utf-8-sig")
        result["output"] = str(target)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FocusWave formal single-modality preflight with legacy/deferred audit surfaces"
    )
    parser.add_argument("--config", default="configs/formal_multimodal_v2.yaml")
    parser.add_argument("--paths-config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    nir = sub.add_parser("nir-adapt", help="LEGACY compatibility adapter; not the staged formal NIR route")
    nir.add_argument("--sessions", nargs="*", default=None)
    nir.add_argument("--run-id", default=None)
    merge = sub.add_parser("merge-audit", help="DEFERRED merge-contract audit; not production fusion")
    merge.add_argument("--unit", choices=sorted(UNIT_KEYS), required=True)
    merge.add_argument("--table", action="append", required=True, dest="tables")
    merge.add_argument("--how", choices=["inner", "outer"], default="inner")
    merge.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config, paths_config=args.paths_config)
    if args.command == "preflight":
        result = command_preflight(config)
    elif args.command == "nir-adapt":
        result = command_nir_adapt(config, sessions=args.sessions, run_id=args.run_id)
    else:
        result = command_merge_audit(config, unit=args.unit, table_specs=args.tables, how=args.how, output=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
