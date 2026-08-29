from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from attention_pipeline.config import Config, load_config
from attention_pipeline.nir_pupil_only import (
    SourceIdentity,
    adapt_session_rows,
    cohort_topology_summary,
    validate_cohort_topology,
)

ANALYSIS_READY_SCHEMA_VERSION = 2
ANALYSIS_READY_PIPELINE_VERSION = "nir-analysis-ready-pupil-only-v2"
ROBUST_Z_SCALE = 1.4826
FORMAL_PHASE_TO_BLOCK = {"block1": 1, "block2": 2}
PAIR_KEY = ["session_id", "phase", "phase_segment", "frame_idx"]
TIME_COLUMNS = ["unix_ms", "video_time_ms", "phase_time_ms"]


@dataclass(frozen=True)
class SessionMaterialization:
    session_id: str
    analysis_group_token: str
    output_csv: Path
    baseline_csv: Path
    n_eye_rows: int
    n_timepoints: int
    n_primary_valid: int
    n_strict_valid: int


def _safe_fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(config: Config, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config.path.parent.parent / path).resolve()


def _output_root(config: Config, override: str | Path | None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    raw = config.section("paths").get("output_root")
    if raw in (None, ""):
        raise KeyError("pupil-only analysis-ready config missing paths.output_root")
    return _resolve(config, str(raw))


def _manifest_path(config: Config) -> Path:
    raw = config.section("paths").get("source_manifest")
    if raw in (None, ""):
        raise KeyError(
            "pupil-only analysis-ready requires paths.source_manifest; "
            "keep the site-local manifest outside version control"
        )
    return _resolve(config, str(raw))


def _guard_output_root(output_root: Path, records: list[dict[str, Any]], config: Config) -> None:
    output = output_root.resolve()
    for record in records:
        source = _resolve(config, str(record["source_csv"]))
        if output == source or source in output.parents:
            raise ValueError(f"refusing derived output inside a source file path: {output}")
        source_parent = source.parent.resolve()
        if output == source_parent or source_parent in output.parents:
            raise ValueError(f"refusing derived output inside source directory: {output}")


def load_source_manifest(config: Config) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = _manifest_path(config)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("source manifest root must be an object")
    rows = payload.get("sessions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source manifest sessions must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            raise TypeError("each source manifest session must be an object")
        if raw.get("source_csv") in (None, ""):
            raise ValueError("source manifest session missing source_csv")
        identity = SourceIdentity.from_mapping(raw)
        row = dict(raw)
        row.update(
            {
                "session_id": identity.session_id,
                "analysis_group_token": identity.analysis_group_token,
                "source_schema_version": identity.source_schema_version,
                "repeat_group_size": identity.repeat_group_size,
            }
        )
        normalized.append(row)
    return payload, normalized


def _selected_records(
    records: list[dict[str, Any]], override: Iterable[str] | None
) -> list[dict[str, Any]]:
    if not override:
        return list(records)
    wanted = {str(value).strip() for value in override if str(value).strip()}
    selected = [row for row in records if row["session_id"] in wanted]
    missing = sorted(wanted - {row["session_id"] for row in selected})
    if missing:
        raise ValueError(f"requested sessions absent from source manifest: {missing}")
    return selected


def _median_mad(values: pd.Series) -> tuple[float, float, float, bool]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return float("nan"), float("nan"), float("nan"), False
    median = float(numeric.median())
    mad = float(np.median(np.abs(numeric.to_numpy(dtype=float) - median)))
    sigma = ROBUST_Z_SCALE * mad
    return median, mad, sigma, bool(np.isfinite(sigma) and sigma > 0)


def compute_session_eye_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (session_id, eye), group in frame.groupby(["session_id", "eye"], sort=True):
        primary = group.loc[group["pupil_valid_primary"], "pupil_geom_mean_diameter"]
        strict = group.loc[group["pupil_valid_strict"], "pupil_geom_mean_diameter"]
        p_med, p_mad, p_sigma, p_sigma_ok = _median_mad(primary)
        s_med, s_mad, s_sigma, s_sigma_ok = _median_mad(strict)
        row: dict[str, Any] = {
            "session_id": session_id,
            "subject": session_id,
            "analysis_group_token": str(group["analysis_group_token"].iloc[0]),
            "eye": eye,
            "n_formal_rows": int(len(group)),
            "n_primary_valid": int(group["pupil_valid_primary"].sum()),
            "primary_valid_fraction": _safe_fraction(
                int(group["pupil_valid_primary"].sum()), len(group)
            ),
            "primary_median_pupil": p_med,
            "primary_MAD_pupil": p_mad,
            "primary_robust_sigma_pupil": p_sigma,
            "primary_robust_sigma_valid": p_sigma_ok,
            "n_strict_valid": int(group["pupil_valid_strict"].sum()),
            "strict_valid_fraction": _safe_fraction(
                int(group["pupil_valid_strict"].sum()), len(group)
            ),
            "strict_median_pupil": s_med,
            "strict_MAD_pupil": s_mad,
            "strict_robust_sigma_pupil": s_sigma,
            "strict_robust_sigma_valid": s_sigma_ok,
        }
        for phase, block in FORMAL_PHASE_TO_BLOCK.items():
            current = group[group["phase"].astype(str).eq(phase)]
            row[f"block{block}_n_rows"] = int(len(current))
            row[f"block{block}_n_primary_valid"] = int(
                current["pupil_valid_primary"].sum()
            )
            row[f"block{block}_n_strict_valid"] = int(
                current["pupil_valid_strict"].sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def apply_session_eye_standardization(
    frame: pd.DataFrame, baselines: pd.DataFrame
) -> pd.DataFrame:
    merge_cols = [
        "session_id",
        "eye",
        "primary_median_pupil",
        "primary_robust_sigma_pupil",
        "primary_robust_sigma_valid",
        "strict_median_pupil",
        "strict_robust_sigma_pupil",
        "strict_robust_sigma_valid",
    ]
    result = frame.merge(
        baselines[merge_cols],
        on=["session_id", "eye"],
        how="left",
        validate="many_to_one",
    )
    pupil = pd.to_numeric(result["pupil_geom_mean_diameter"], errors="coerce")
    result["pupil_centered_primary"] = np.where(
        result["pupil_valid_primary"], pupil - result["primary_median_pupil"], np.nan
    )
    result["pupil_robust_z_primary"] = np.where(
        result["pupil_valid_primary"] & result["primary_robust_sigma_valid"],
        (pupil - result["primary_median_pupil"])
        / result["primary_robust_sigma_pupil"],
        np.nan,
    )
    result["pupil_centered_strict"] = np.where(
        result["pupil_valid_strict"], pupil - result["strict_median_pupil"], np.nan
    )
    result["pupil_robust_z_strict"] = np.where(
        result["pupil_valid_strict"] & result["strict_robust_sigma_valid"],
        (pupil - result["strict_median_pupil"])
        / result["strict_robust_sigma_pupil"],
        np.nan,
    )
    return result


def _finite(series: pd.Series) -> pd.Series:
    return pd.Series(
        np.isfinite(pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)),
        index=series.index,
    )


def _source_mode(left: pd.Series, right: pd.Series) -> pd.Series:
    l = left.fillna(False).astype(bool)
    r = right.fillna(False).astype(bool)
    return pd.Series(
        np.select(
            [l & r, l & ~r, ~l & r],
            ["binocular", "left_only", "right_only"],
            default="missing",
        ),
        index=left.index,
        dtype="object",
    )


def _fuse(
    left_value: pd.Series,
    right_value: pd.Series,
    left_valid: pd.Series,
    right_valid: pd.Series,
) -> pd.Series:
    lv = pd.to_numeric(left_value, errors="coerce")
    rv = pd.to_numeric(right_value, errors="coerce")
    l = left_valid.fillna(False).astype(bool) & _finite(lv)
    r = right_valid.fillna(False).astype(bool) & _finite(rv)
    return pd.Series(
        np.select([l & r, l & ~r, ~l & r], [(lv + rv) / 2.0, lv, rv], default=np.nan),
        index=left_value.index,
        dtype=float,
    )


def _eye_wide(frame: pd.DataFrame, eye: str) -> pd.DataFrame:
    current = frame[frame["eye"].astype(str).eq(eye)].copy()
    keep = [
        *PAIR_KEY,
        *TIME_COLUMNS,
        "subject",
        "analysis_group_token",
        "repeat_group_size",
        "is_repeat_session",
        "pupil_geom_mean_diameter",
        "pupil_valid_primary",
        "pupil_valid_strict",
        "pupil_centered_primary",
        "pupil_robust_z_primary",
        "pupil_centered_strict",
        "pupil_robust_z_strict",
        "quality_track",
        "roi_clipped",
        "temporal_flagged",
    ]
    current = current[[name for name in keep if name in current.columns]]
    rename = {
        "subject": f"{eye}_subject",
        "analysis_group_token": f"{eye}_analysis_group_token",
        "repeat_group_size": f"{eye}_repeat_group_size",
        "is_repeat_session": f"{eye}_is_repeat_session",
        "pupil_geom_mean_diameter": f"{eye}_raw_pupil_diameter",
        "pupil_valid_primary": f"{eye}_pupil_valid_primary",
        "pupil_valid_strict": f"{eye}_pupil_valid_strict",
        "pupil_centered_primary": f"{eye}_centered_pupil",
        "pupil_robust_z_primary": f"{eye}_robust_z_pupil",
        "pupil_centered_strict": f"{eye}_strict_centered_pupil",
        "pupil_robust_z_strict": f"{eye}_strict_robust_z_pupil",
        "quality_track": f"{eye}_quality_track",
        "roi_clipped": f"{eye}_roi_clipped",
        "temporal_flagged": f"{eye}_temporal_flagged",
    }
    return current.rename(columns=rename)


def build_wide_timepoints(frame: pd.DataFrame, *, time_tolerance_ms: float = 1.0) -> pd.DataFrame:
    duplicate = frame.duplicated(PAIR_KEY + ["eye"], keep=False)
    if duplicate.any():
        raise ValueError("duplicate eye rows for one pupil timepoint")
    left = _eye_wide(frame, "left")
    right = _eye_wide(frame, "right")
    wide = left.merge(
        right,
        on=PAIR_KEY,
        how="outer",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    for column in TIME_COLUMNS:
        l = pd.to_numeric(wide.get(f"{column}_left"), errors="coerce")
        r = pd.to_numeric(wide.get(f"{column}_right"), errors="coerce")
        both = l.notna() & r.notna()
        if both.any() and (l[both] - r[both]).abs().gt(time_tolerance_ms).any():
            raise ValueError(f"left/right {column} mismatch exceeds {time_tolerance_ms} ms")
        wide[column] = l.combine_first(r)
        wide = wide.drop(columns=[f"{column}_left", f"{column}_right"])

    for identity in ("subject", "analysis_group_token", "repeat_group_size", "is_repeat_session"):
        l_name = f"left_{identity}"
        r_name = f"right_{identity}"
        wide[identity] = wide.get(l_name).combine_first(wide.get(r_name))
        both = wide.get(l_name).notna() & wide.get(r_name).notna()
        if both.any() and (
            wide.loc[both, l_name].astype(str) != wide.loc[both, r_name].astype(str)
        ).any():
            raise ValueError(f"left/right {identity} mismatch")
        wide = wide.drop(columns=[l_name, r_name])

    wide["block"] = wide["phase"].map(FORMAL_PHASE_TO_BLOCK).astype("Int64")
    for name in (
        "left_pupil_valid_primary",
        "right_pupil_valid_primary",
        "left_pupil_valid_strict",
        "right_pupil_valid_strict",
    ):
        if name not in wide:
            wide[name] = False
        wide[name] = wide[name].fillna(False).astype(bool)

    wide["binocular_source_mode"] = _source_mode(
        wide["left_pupil_valid_primary"], wide["right_pupil_valid_primary"]
    )
    wide["binocular_pupil"] = _fuse(
        wide["left_centered_pupil"],
        wide["right_centered_pupil"],
        wide["left_pupil_valid_primary"],
        wide["right_pupil_valid_primary"],
    )
    wide["binocular_robust_z_pupil"] = _fuse(
        wide["left_robust_z_pupil"],
        wide["right_robust_z_pupil"],
        wide["left_pupil_valid_primary"],
        wide["right_pupil_valid_primary"],
    )
    wide["binocular_strict_source_mode"] = _source_mode(
        wide["left_pupil_valid_strict"], wide["right_pupil_valid_strict"]
    )
    wide["binocular_strict_pupil"] = _fuse(
        wide["left_strict_centered_pupil"],
        wide["right_strict_centered_pupil"],
        wide["left_pupil_valid_strict"],
        wide["right_pupil_valid_strict"],
    )
    wide["binocular_strict_robust_z_pupil"] = _fuse(
        wide["left_strict_robust_z_pupil"],
        wide["right_strict_robust_z_pupil"],
        wide["left_pupil_valid_strict"],
        wide["right_pupil_valid_strict"],
    )
    return wide.sort_values(
        ["block", "phase_segment", "frame_idx"], kind="stable"
    ).reset_index(drop=True)


def _load_session(config: Config, record: Mapping[str, Any]) -> pd.DataFrame:
    source = _resolve(config, str(record["source_csv"]))
    frame = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)
    adapted = adapt_session_rows(frame, record)
    adapted = adapted[adapted["phase"].astype(str).isin(FORMAL_PHASE_TO_BLOCK)].copy()
    if adapted.empty:
        raise ValueError("session has no block1/block2 pupil rows")
    return adapted


def _write_csv(frame: pd.DataFrame, path: Path, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"derived output already exists: {path}")
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def run_materialization(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
    output_root_override: str | Path | None = None,
    overwrite_derived: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    manifest_payload, all_records = load_source_manifest(config)
    topology = cohort_topology_summary(all_records)
    policy = config.section("analysis_policy")
    if bool(policy.get("enforce_expected_topology", True)):
        expected = config.section("cohort_topology")
        topology = validate_cohort_topology(
            all_records,
            expected_sessions=int(expected.get("sessions", 44)),
            expected_analysis_groups=int(expected.get("analysis_groups", 38)),
            expected_double_session_repeat_groups=int(
                expected.get("double_session_repeat_groups", 6)
            ),
        )
    records = _selected_records(all_records, subjects)
    output_root = _output_root(config, output_root_override)
    _guard_output_root(output_root, records, config)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[SessionMaterialization] = []
    baselines_all: list[pd.DataFrame] = []
    inclusion_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for record in records:
        session_id = str(record["session_id"])
        try:
            adapted = _load_session(config, record)
            baselines = compute_session_eye_baselines(adapted)
            standardized = apply_session_eye_standardization(adapted, baselines)
            wide = build_wide_timepoints(
                standardized,
                time_tolerance_ms=float(policy.get("max_eye_time_delta_ms", 1.0)),
            )
            frame_path = output_root / "frame_level" / session_id / f"{session_id}_nir_analysis_ready.csv"
            baseline_path = output_root / "baselines" / f"{session_id}_eye_baselines.csv"
            _write_csv(wide, frame_path, overwrite=overwrite_derived)
            _write_csv(baselines, baseline_path, overwrite=overwrite_derived)
            baselines_all.append(baselines)

            for (phase, eye), group in standardized.groupby(["phase", "eye"], sort=True):
                inclusion_rows.append(
                    {
                        "session_id": session_id,
                        "subject": session_id,
                        "block": FORMAL_PHASE_TO_BLOCK.get(str(phase)),
                        "phase": phase,
                        "eye": eye,
                        "n_rows": int(len(group)),
                        "n_primary_valid": int(group["pupil_valid_primary"].sum()),
                        "primary_valid_fraction": float(group["pupil_valid_primary"].mean()),
                        "n_strict_valid": int(group["pupil_valid_strict"].sum()),
                        "strict_valid_fraction": float(group["pupil_valid_strict"].mean()),
                    }
                )

            source_path = _resolve(config, str(record["source_csv"]))
            provenance_rows.append(
                {
                    "session_id": session_id,
                    "source_schema_version": int(record["source_schema_version"]),
                    "source_kind": str(record.get("source_kind", "ritnet-fullclass-pupil-only")),
                    "source_branch": record.get("source_branch"),
                    "source_commit": record.get("source_commit"),
                    "source_csv_sha256": _sha256(source_path),
                    "derived_frame_sha256": _sha256(frame_path),
                }
            )
            results.append(
                SessionMaterialization(
                    session_id=session_id,
                    analysis_group_token=str(record["analysis_group_token"]),
                    output_csv=frame_path,
                    baseline_csv=baseline_path,
                    n_eye_rows=int(len(standardized)),
                    n_timepoints=int(len(wide)),
                    n_primary_valid=int(standardized["pupil_valid_primary"].sum()),
                    n_strict_valid=int(standardized["pupil_valid_strict"].sum()),
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "session_id": session_id,
                    "stage": "analysis_ready",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    qc = output_root / "qc"
    provenance = output_root / "provenance"
    qc.mkdir(parents=True, exist_ok=True)
    provenance.mkdir(parents=True, exist_ok=True)
    failures_df = pd.DataFrame(
        failures, columns=["session_id", "stage", "error_type", "error"]
    )
    _write_csv(failures_df, qc / "session_load_failures.csv", overwrite=overwrite_derived)
    if not results:
        raise RuntimeError("all selected sessions failed pupil-only materialization")

    baselines_df = pd.concat(baselines_all, ignore_index=True)
    inclusion_df = pd.DataFrame(inclusion_rows)
    provenance_df = pd.DataFrame(provenance_rows)
    _write_csv(
        baselines_df,
        output_root / "baselines" / "session_eye_baselines.csv",
        overwrite=overwrite_derived,
    )
    _write_csv(
        inclusion_df,
        qc / "session_eye_block_inclusion.csv",
        overwrite=overwrite_derived,
    )
    _write_csv(
        provenance_df,
        provenance / "source_files.csv",
        overwrite=overwrite_derived,
    )

    summary = {
        "pipeline_version": ANALYSIS_READY_PIPELINE_VERSION,
        "schema_version": ANALYSIS_READY_SCHEMA_VERSION,
        "signal_semantics": "pupil_geometry_only",
        "iris_geometry_used": False,
        "pir_oar_allowed": False,
        "topology": topology,
        "n_sessions_requested_this_run": len(records),
        "n_sessions_materialized_this_run": len(results),
        "n_sessions_failed_this_run": len(failures),
        "n_eye_rows": int(sum(item.n_eye_rows for item in results)),
        "n_timepoints": int(sum(item.n_timepoints for item in results)),
        "n_primary_valid_eye_rows": int(sum(item.n_primary_valid for item in results)),
        "n_strict_valid_eye_rows": int(sum(item.n_strict_valid for item in results)),
        "interpretation_boundary": (
            "engineering/analysis readiness only; topology counts are not measurement validity "
            "or inferential conclusions"
        ),
    }
    summary_path = qc / "cohort_inclusion_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": ANALYSIS_READY_PIPELINE_VERSION,
        "schema_version": ANALYSIS_READY_SCHEMA_VERSION,
        "signal_semantics": "pupil_geometry_only",
        "source_manifest_schema": manifest_payload.get("schema_version"),
        "topology": topology,
        "production_read_only": True,
        "baseline": (
            "session×eye median-centered across block1+block2; repeat sessions remain "
            "separate baselines to prevent cross-session leakage"
        ),
        "primary_validity": (
            "source observed + RITnet success + positive finite fitted pupil geometry; "
            "interpolation-only rows excluded"
        ),
        "strict_sensitivity": "primary plus no ROI clipping and no temporal anomaly",
        "binocular_fusion": "equal-weight mean when both valid; single-eye fallback otherwise",
        "pir_oar_refused": True,
        "failure_table": str(qc / "session_load_failures.csv"),
        "summary": str(summary_path),
    }
    manifest_path = provenance / "analysis_ready_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "output_root": str(output_root),
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "failures_path": str(qc / "session_load_failures.csv"),
        "summary": summary,
    }
