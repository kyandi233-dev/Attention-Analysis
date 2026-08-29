"""Portable formal multimodal preflight, NIR adaptation, and merge-contract audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from attention_pipeline.config import Config, load_config
from attention_pipeline.formal_analysis.cohort import included_cohort, load_cohort_manifest, summarize_cohort
from attention_pipeline.formal_analysis.merge import merge_modalities
from attention_pipeline.formal_analysis.nir_adapter import adapt_nir_csv


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


def command_nir_adapt(config: Config, *, sessions: list[str] | None, run_id: str | None) -> dict[str, object]:
    if config.path_registry is None:
        raise ValueError("nir-adapt requires --paths-config or ATTENTION_ANALYSIS_PATHS_CONFIG")
    nir_cfg = config.section("nir")
    source_manifest = config.registry_path(str(nir_cfg["source_manifest_path_key"]))
    output_base = config.registry_path(str(nir_cfg["standardized_output_path_key"]))
    manifest = pd.read_csv(source_manifest, encoding="utf-8-sig")
    required = {"session_id", "status", "eye_metrics_csv", "schema_version"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"NIR source manifest 缺少列: {sorted(missing)}")
    if manifest["session_id"].duplicated().any():
        dup = manifest.loc[manifest["session_id"].duplicated(keep=False), "session_id"].tolist()
        raise ValueError(f"NIR source manifest 每个session必须唯一: {sorted(set(map(str, dup)))}")

    cohort = included_cohort(_cohort(config), require_groups=False)
    allowed = set(cohort["session_id"].astype(str))
    manifest = manifest.loc[manifest["status"].astype(str).str.lower().eq("complete")].copy()
    manifest = manifest.loc[manifest["session_id"].astype(str).isin(allowed)].copy()
    if sessions:
        requested = set(sessions)
        manifest = manifest.loc[manifest["session_id"].astype(str).isin(requested)].copy()
        missing_requested = requested - set(manifest["session_id"].astype(str))
        if missing_requested:
            raise ValueError(
                "请求场次不在 complete NIR source manifest/cohort 交集中: "
                + ", ".join(sorted(missing_requested))
            )
    if manifest.empty:
        raise ValueError("没有可适配的 complete NIR session")

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
        rows.append(adapt_nir_csv(
            source_csv,
            frame_root / f"{session}.csv",
            session_id=session,
            schema_version=record.get("schema_version"),
            reject_pir=bool(nir_cfg.get("reject_historical_pir", True)),
        ))

    result = {
        "run_id": run_id,
        "science_config_digest": config.digest,
        "paths_config_digest": config.path_registry.digest,
        "source_manifest": str(source_manifest),
        "source_manifest_rows": int(len(manifest)),
        "sessions": rows,
    }
    (run_root / "run_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def command_merge_audit(config: Config, *, unit: str, table_specs: list[str], how: str, output: str | None) -> dict[str, object]:
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
    result = {"unit": unit, "how": how, "rows": int(len(merged)), "modalities": audit.to_dict("records")}
    if output:
        target = _resolve_external(config, output)
        if target.exists():
            raise FileExistsError(f"拒绝覆盖已有 merge 输出: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(target, index=False, encoding="utf-8-sig")
        result["output"] = str(target)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="FocusWave formal multimodal v2 portable analysis interface")
    parser.add_argument("--config", default="configs/formal_multimodal_v2.yaml")
    parser.add_argument("--paths-config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    nir = sub.add_parser("nir-adapt")
    nir.add_argument("--sessions", nargs="*", default=None)
    nir.add_argument("--run-id", default=None)
    merge = sub.add_parser("merge-audit")
    merge.add_argument("--unit", choices=["trial", "probe", "block", "session"], required=True)
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
