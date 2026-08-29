"""Materialize pupil-only candidate metrics beside the canonical analysis-ready table.

The existing canonical frame table remains backward compatible.  This sidecar
prevents producer-provided pupil geometry/segmentation candidates from being
lost before scientific candidate validation.  Iris fractions are deliberately
excluded: they are QC proportions, not iris geometry.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from attention_pipeline.config import Config, load_config
from attention_pipeline.nir_pupil_only import adapt_session_rows
from .pupil_only import load_source_manifest

CANDIDATE_SCHEMA_VERSION = 1
CANDIDATE_PIPELINE_VERSION = "nir-analysis-ready-pupil-candidates-v1"
ROBUST_Z_SCALE = 1.4826
FORMAL_PHASES = {"block1", "block2"}

# These are pupil-only candidates explicitly supported by the formal evidence
# plan and already available from the accepted fullclass-final producer schema.
PUPIL_CANDIDATE_METRICS: dict[str, dict[str, str]] = {
    "pupil_geom_mean_diameter": {"kind": "positive", "unit": "px"},
    "pupil_equivalent_diameter": {"kind": "positive", "unit": "px"},
    "pupil_axis_a": {"kind": "positive", "unit": "px"},
    "pupil_axis_b": {"kind": "positive", "unit": "px"},
    "pupil_contour_area": {"kind": "positive", "unit": "px2"},
    "pupil_ellipse_area": {"kind": "positive", "unit": "px2"},
    "hard_pupil_fraction": {"kind": "fraction", "unit": "proportion"},
    "soft_pupil_fraction": {"kind": "fraction", "unit": "proportion"},
}


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
        raise KeyError("pupil candidate materializer requires paths.output_root")
    return _resolve(config, str(raw))


def _selected(records: list[dict[str, Any]], sessions: Iterable[str] | None) -> list[dict[str, Any]]:
    if not sessions:
        return list(records)
    wanted = {str(x).strip() for x in sessions if str(x).strip()}
    selected = [r for r in records if str(r["session_id"]) in wanted]
    missing = sorted(wanted - {str(r["session_id"]) for r in selected})
    if missing:
        raise ValueError(f"requested sessions absent from source manifest: {missing}")
    return selected


def _metric_valid(values: pd.Series, *, kind: str, base_valid: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = base_valid.fillna(False).astype(bool) & np.isfinite(numeric)
    if kind == "positive":
        valid &= numeric.gt(0)
    elif kind == "fraction":
        valid &= numeric.between(0.0, 1.0, inclusive="both")
    else:
        raise ValueError(f"unknown pupil candidate kind: {kind}")
    return valid


def _median_mad(values: pd.Series) -> tuple[float, float, float, bool]:
    x = pd.to_numeric(values, errors="coerce")
    x = x[np.isfinite(x)]
    if x.empty:
        return np.nan, np.nan, np.nan, False
    median = float(x.median())
    mad = float(np.median(np.abs(x.to_numpy(dtype=float) - median)))
    sigma = ROBUST_Z_SCALE * mad
    return median, mad, sigma, bool(np.isfinite(sigma) and sigma > 0)


def compute_candidate_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (session_id, eye), group in frame.groupby(["session_id", "eye"], sort=True):
        row: dict[str, Any] = {
            "session_id": str(session_id),
            "eye": str(eye),
            "analysis_group_token": str(group["analysis_group_token"].iloc[0]),
        }
        for metric, spec in PUPIL_CANDIDATE_METRICS.items():
            values = pd.to_numeric(group.get(metric), errors="coerce")
            for quality, base_col in (("primary", "pupil_valid_primary"), ("strict", "pupil_valid_strict")):
                valid = _metric_valid(values, kind=spec["kind"], base_valid=group[base_col])
                med, mad, sigma, scale_ok = _median_mad(values[valid])
                prefix = f"{metric}__{quality}"
                row[f"{prefix}__n_valid"] = int(valid.sum())
                row[f"{prefix}__valid_fraction"] = float(valid.mean()) if len(valid) else np.nan
                row[f"{prefix}__median"] = med
                row[f"{prefix}__mad"] = mad
                row[f"{prefix}__robust_sigma"] = sigma
                row[f"{prefix}__scale_valid"] = scale_ok
        rows.append(row)
    return pd.DataFrame(rows)


def apply_candidate_standardization(frame: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    baseline = baselines.set_index(["session_id", "eye"])
    keys = pd.MultiIndex.from_frame(out[["session_id", "eye"]])
    for metric, spec in PUPIL_CANDIDATE_METRICS.items():
        values = pd.to_numeric(out.get(metric), errors="coerce")
        out[f"{metric}__raw"] = values
        for quality, base_col in (("primary", "pupil_valid_primary"), ("strict", "pupil_valid_strict")):
            valid = _metric_valid(values, kind=spec["kind"], base_valid=out[base_col])
            med_map = baseline[f"{metric}__{quality}__median"]
            sig_map = baseline[f"{metric}__{quality}__robust_sigma"]
            scale_map = baseline[f"{metric}__{quality}__scale_valid"]
            med = pd.Series(keys.map(med_map), index=out.index, dtype=float)
            sigma = pd.Series(keys.map(sig_map), index=out.index, dtype=float)
            scale_ok = pd.Series(keys.map(scale_map), index=out.index).fillna(False).astype(bool)
            out[f"{metric}__valid_{quality}"] = valid
            out[f"{metric}__centered_{quality}"] = np.where(valid, values - med, np.nan)
            out[f"{metric}__robust_z_{quality}"] = np.where(
                valid & scale_ok & np.isfinite(sigma) & sigma.gt(0),
                (values - med) / sigma,
                np.nan,
            )
    return out


def _candidate_columns() -> list[str]:
    cols = [
        "session_id", "subject", "analysis_group_token", "repeat_group_size",
        "is_repeat_session", "phase", "phase_segment", "frame_idx", "eye",
        "eye_raw", "unix_ms", "video_time_ms", "phase_time_ms",
        "pupil_valid_primary", "pupil_valid_strict", "quality_track",
        "roi_clipped", "temporal_flagged",
    ]
    for metric in PUPIL_CANDIDATE_METRICS:
        cols.append(f"{metric}__raw")
        for quality in ("primary", "strict"):
            cols.extend([
                f"{metric}__valid_{quality}",
                f"{metric}__centered_{quality}",
                f"{metric}__robust_z_{quality}",
            ])
    return cols


def _availability_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (session_id, eye), group in frame.groupby(["session_id", "eye"], sort=True):
        for metric, spec in PUPIL_CANDIDATE_METRICS.items():
            raw = pd.to_numeric(group[f"{metric}__raw"], errors="coerce")
            for quality in ("primary", "strict"):
                valid = group[f"{metric}__valid_{quality}"].fillna(False).astype(bool)
                rows.append({
                    "session_id": str(session_id),
                    "analysis_group_token": str(group["analysis_group_token"].iloc[0]),
                    "eye": str(eye),
                    "metric": metric,
                    "metric_kind": spec["kind"],
                    "unit": spec["unit"],
                    "quality_track": quality,
                    "n_rows": int(len(group)),
                    "n_finite_raw": int(np.isfinite(raw).sum()),
                    "n_valid": int(valid.sum()),
                    "valid_fraction": float(valid.mean()) if len(valid) else np.nan,
                })
    return rows


def run_candidate_materialization(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
    output_root_override: str | Path | None = None,
    overwrite_derived: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    _, records = load_source_manifest(config)
    records = _selected(records, subjects)
    root = _output_root(config, output_root_override)
    root.mkdir(parents=True, exist_ok=True)

    availability: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    materialized: list[str] = []

    for record in records:
        session_id = str(record["session_id"])
        try:
            source = _resolve(config, str(record["source_csv"]))
            source_rows = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)
            adapted = adapt_session_rows(source_rows, record)
            adapted = adapted[adapted["phase"].astype(str).isin(FORMAL_PHASES)].copy()
            if adapted.empty:
                raise ValueError("session has no block1/block2 pupil rows")
            baselines = compute_candidate_baselines(adapted)
            candidates = apply_candidate_standardization(adapted, baselines)
            sidecar = candidates[[c for c in _candidate_columns() if c in candidates.columns]].copy()

            frame_path = root / "candidate_frame_level" / session_id / f"{session_id}_nir_pupil_candidates.csv"
            baseline_path = root / "candidate_baselines" / f"{session_id}_candidate_eye_baselines.csv"
            for path, table in ((frame_path, sidecar), (baseline_path, baselines)):
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and not overwrite_derived:
                    raise FileExistsError(f"derived candidate output already exists: {path}")
                table.to_csv(path, index=False, encoding="utf-8-sig")
            availability.extend(_availability_rows(candidates))
            materialized.append(session_id)
        except Exception as exc:
            failures.append({
                "session_id": session_id,
                "stage": "candidate_metrics",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    qc = root / "qc"
    provenance = root / "provenance"
    qc.mkdir(parents=True, exist_ok=True)
    provenance.mkdir(parents=True, exist_ok=True)
    availability_df = pd.DataFrame(availability)
    failures_df = pd.DataFrame(failures, columns=["session_id", "stage", "error_type", "error"])
    for path, table in (
        (qc / "candidate_metric_availability.csv", availability_df),
        (qc / "candidate_metric_failures.csv", failures_df),
    ):
        if path.exists() and not overwrite_derived:
            raise FileExistsError(f"derived candidate QC output already exists: {path}")
        table.to_csv(path, index=False, encoding="utf-8-sig")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": CANDIDATE_PIPELINE_VERSION,
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "source": "accepted fullclass-final pupil-only adapter fields",
        "candidate_metrics": PUPIL_CANDIDATE_METRICS,
        "iris_geometry_used": False,
        "pir_oar_allowed": False,
        "iris_fraction_policy": "retained elsewhere as segmentation QC only; excluded from candidate geometry",
        "baseline_contract": "session×eye per candidate; repeat sessions never share baselines",
        "n_sessions_requested": len(records),
        "n_sessions_materialized": len(materialized),
        "n_sessions_failed": len(failures),
        "endpoint_freeze": "not_performed_here",
    }
    manifest_path = provenance / "candidate_metrics_manifest.json"
    if manifest_path.exists() and not overwrite_derived:
        raise FileExistsError(f"derived candidate manifest already exists: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "output_root": str(root),
        "manifest_path": str(manifest_path),
        "availability_path": str(qc / "candidate_metric_availability.csv"),
        "failures_path": str(qc / "candidate_metric_failures.csv"),
        **manifest,
    }
