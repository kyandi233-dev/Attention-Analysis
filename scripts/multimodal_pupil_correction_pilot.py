"""Validation-only pilot for same-camera and RGB pupil scale correction.

This module compares M0--M3 using unlabeled measurement properties only.  It
never reads Behavior/Probe outcomes, never runs NIR/RGB models, and never
turns a pilot correction into a formal pupil measure.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from scripts.multimodal_pupil_audit import (
        build_paired,
        read_nir,
        read_repeat_registry_csv,
        read_rgb,
        safe_corr,
        summarize_numeric,
        write_json,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from multimodal_pupil_audit import (
        build_paired,
        read_nir,
        read_repeat_registry_csv,
        read_rgb,
        safe_corr,
        summarize_numeric,
        write_json,
    )


PILOT_VERSION = "issue22-multimodal-pupil-correction-pilot-v1"
RGB_SCALE_COLUMNS = (
    "rgb_face_bbox_scale_px",
    "rgb_eye_outer_corner_distance_px",
    "rgb_eye_inner_canthus_distance_px",
    "rgb_iris_diameter_px",
    "rgb_iris_center_distance_px",
)
POSE_COLUMNS = ("Pitch", "Roll", "Yaw", "gaze_pitch", "gaze_yaw")
METHOD_SPECS: dict[str, dict[str, Any]] = {
    "M0": {"label": "uncorrected_reference", "predictors": (), "requires_rgb_50ms": False, "fit_scope": "none"},
    "M1": {"label": "nir_bbox_baseline_residualization", "predictors": ("log_nir_bbox_geom",), "requires_rgb_50ms": False, "fit_scope": "baseline_only"},
    "M2a": {"label": "rgb_outer_eye_baseline_residualization", "predictors": ("log_rgb_outer_eye",), "requires_rgb_50ms": True, "fit_scope": "baseline_only"},
    "M2b": {"label": "rgb_inner_canthus_baseline_residualization", "predictors": ("log_rgb_inner_canthus",), "requires_rgb_50ms": True, "fit_scope": "baseline_only"},
    "M3": {"label": "nir_bbox_plus_pitch_sensitivity", "predictors": ("log_nir_bbox_geom", "Pitch"), "requires_rgb_50ms": True, "fit_scope": "baseline_only"},
}


def _finite(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.notna() & np.isfinite(numeric)


def _phase_mask(frame: pd.DataFrame, phase: str) -> pd.Series:
    return frame["nir_phase"].astype("string").eq(phase)


def add_nir_bbox_features(paired: pd.DataFrame) -> pd.DataFrame:
    """Add per-eye NIR YOLO bbox width, height and geometric scale."""

    result = paired.copy()
    x1 = pd.to_numeric(result["bbox_x1"], errors="coerce")
    y1 = pd.to_numeric(result["bbox_y1"], errors="coerce")
    x2 = pd.to_numeric(result["bbox_x2"], errors="coerce")
    y2 = pd.to_numeric(result["bbox_y2"], errors="coerce")
    result["nir_bbox_width_px"] = (x2 - x1).abs()
    result["nir_bbox_height_px"] = (y2 - y1).abs()
    result["nir_bbox_geom_scale_px"] = np.sqrt(result["nir_bbox_width_px"] * result["nir_bbox_height_px"])
    result["log_nir_bbox_geom"] = np.log(result["nir_bbox_geom_scale_px"].where(result["nir_bbox_geom_scale_px"].gt(0)))
    result["log_rgb_outer_eye"] = np.log(pd.to_numeric(result["rgb_eye_outer_corner_distance_px"], errors="coerce").where(pd.to_numeric(result["rgb_eye_outer_corner_distance_px"], errors="coerce").gt(0)))
    result["log_rgb_inner_canthus"] = np.log(pd.to_numeric(result["rgb_eye_inner_canthus_distance_px"], errors="coerce").where(pd.to_numeric(result["rgb_eye_inner_canthus_distance_px"], errors="coerce").gt(0)))
    return result


def _successive_abs_difference(frame: pd.DataFrame, value_column: str) -> dict[str, Any]:
    if value_column not in frame:
        return {"n": 0, "median": None, "p95": None}
    diffs: list[pd.Series] = []
    for _, group in frame.sort_values("unix_ms_nir").groupby("eye", dropna=False):
        values = pd.to_numeric(group[value_column], errors="coerce")
        diffs.append(values.diff().abs())
    if not diffs:
        return {"n": 0, "median": None, "p95": None}
    values = pd.concat(diffs, ignore_index=True).dropna()
    if values.empty:
        return {"n": 0, "median": None, "p95": None}
    return {"n": int(len(values)), "median": float(values.median()), "p95": float(values.quantile(0.95))}


def bbox_stability(paired: pd.DataFrame) -> dict[str, Any]:
    """Summarize bbox stability separately for each NIR eye."""

    output: dict[str, Any] = {}
    for eye, group in paired.groupby("eye", dropna=False):
        eye_key = str(eye)
        output[eye_key] = {
            "rows": int(len(group)),
            "metrics": {},
        }
        for column in ("nir_bbox_width_px", "nir_bbox_height_px", "nir_bbox_geom_scale_px"):
            metric = summarize_numeric(group[column])
            baseline = group.loc[_phase_mask(group, "baseline"), column]
            task = group.loc[group["nir_phase"].astype("string").isin({"block1", "block2"}), column]
            metric["baseline"] = summarize_numeric(baseline)
            metric["task_blocks"] = summarize_numeric(task)
            metric["successive_abs_difference"] = _successive_abs_difference(group, column)
            output[eye_key]["metrics"][column] = metric
    return output


def _baseline_residualize(frame: pd.DataFrame, predictors: Sequence[str]) -> tuple[pd.Series, dict[str, Any]]:
    """Fit per-eye baseline-only linear residualization and preserve baseline center."""

    corrected = pd.Series(np.nan, index=frame.index, dtype=float)
    fit_details: dict[str, Any] = {"scope": "baseline_only", "by_eye": {}}
    for eye, group in frame.groupby("eye", dropna=False):
        y = pd.to_numeric(group["log_pupil_diameter"], errors="coerce")
        baseline = _phase_mask(group, "baseline")
        x_columns = [column for column in predictors if column in group]
        design_values = [pd.to_numeric(group[column], errors="coerce") for column in x_columns]
        if not x_columns:
            continue
        complete = _finite(y)
        for values in design_values:
            complete &= _finite(values)
        fit_mask = baseline & complete
        if int(fit_mask.sum()) < max(10, len(x_columns) + 2):
            fit_details["by_eye"][str(eye)] = {"fit_rows": int(fit_mask.sum()), "status": "insufficient_baseline_rows"}
            continue
        x = np.column_stack([np.ones(int(fit_mask.sum()))] + [values.loc[fit_mask].to_numpy(float) for values in design_values])
        y_fit = y.loc[fit_mask].to_numpy(float)
        coefficients, _, _, _ = np.linalg.lstsq(x, y_fit, rcond=None)
        centers = np.array([values.loc[fit_mask].median() for values in design_values], dtype=float)
        prediction_rows = complete
        x_all = np.column_stack([np.ones(int(prediction_rows.sum()))] + [values.loc[prediction_rows].to_numpy(float) for values in design_values])
        y_all = y.loc[prediction_rows].to_numpy(float)
        correction = y_all - (x_all[:, 1:] - centers) @ coefficients[1:]
        corrected.loc[group.index[prediction_rows]] = correction
        fit_details["by_eye"][str(eye)] = {
            "fit_rows": int(fit_mask.sum()),
            "status": "fit",
            "predictors": list(x_columns),
            "coefficients": [float(value) for value in coefficients],
            "baseline_centers": [float(value) for value in centers],
        }
    return corrected, fit_details


def _metric_summary(frame: pd.DataFrame, value_column: str) -> dict[str, Any]:
    value = pd.to_numeric(frame[value_column], errors="coerce")
    summary = summarize_numeric(value)
    baseline = frame.loc[_phase_mask(frame, "baseline"), value_column]
    summary["baseline"] = summarize_numeric(baseline)
    summary["baseline_successive_abs_difference"] = _successive_abs_difference(frame.loc[_phase_mask(frame, "baseline")], value_column)
    return summary


def _method_frame(paired: pd.DataFrame, method: str) -> pd.DataFrame:
    frame = paired.loc[paired["pupil_valid"]].copy()
    if METHOD_SPECS[method]["requires_rgb_50ms"]:
        frame = frame.loc[pd.to_numeric(frame["abs_delta_ms"], errors="coerce").le(50)].copy()
    return frame


def evaluate_method(paired: pd.DataFrame, method: str) -> dict[str, Any]:
    spec = METHOD_SPECS[method]
    frame = _method_frame(paired, method)
    if method == "M0":
        corrected = pd.to_numeric(frame["log_pupil_diameter"], errors="coerce")
        fit_details = {"scope": "none"}
    else:
        corrected, fit_details = _baseline_residualize(frame, spec["predictors"])
    frame = frame.copy()
    frame["corrected_log_pupil"] = corrected
    frame["corrected_pupil_diameter_px"] = np.exp(corrected)
    frame = frame.loc[_finite(frame["corrected_log_pupil"])].copy()
    total_nir = len(paired)
    coverage = {
        "rows": int(len(frame)),
        "fraction_of_all_nir_rows": float(len(frame) / total_nir) if total_nir else 0.0,
        "fraction_of_valid_pupil_rows": float(len(frame) / int(paired["pupil_valid"].sum())) if int(paired["pupil_valid"].sum()) else 0.0,
        "requires_rgb_abs_delta_ms_le_50": bool(spec["requires_rgb_50ms"]),
    }
    geometry_columns = ["nir_bbox_geom_scale_px", *RGB_SCALE_COLUMNS, *POSE_COLUMNS]
    geometry_relations = {column: safe_corr(frame["corrected_log_pupil"], frame[column], method="spearman") for column in geometry_columns if column in frame}
    raw_corrected = safe_corr(frame["log_pupil_diameter"], frame["corrected_log_pupil"], method="pearson")
    raw_diameter = _metric_summary(frame, "pupil_diameter_px")
    corrected_diameter = _metric_summary(frame, "corrected_pupil_diameter_px")
    return {
        "method": method,
        "label": spec["label"],
        "fit_scope": spec["fit_scope"],
        "predictors": list(spec["predictors"]),
        "coverage": coverage,
        "fit": fit_details,
        "geometry_residual_relations": geometry_relations,
        "raw_vs_corrected": raw_corrected,
        "raw_pupil_measurement": raw_diameter,
        "corrected_pupil_measurement": corrected_diameter,
        "raw_corrected_successive_abs_difference": {
            "raw": _successive_abs_difference(frame, "pupil_diameter_px"),
            "corrected": _successive_abs_difference(frame, "corrected_pupil_diameter_px"),
        },
        "frame_rows": int(len(frame)),
    }


def _find_nir_csv(nir_root: Path, experiment_id: str) -> Path | None:
    candidates = sorted(nir_root.glob(f"sub-{experiment_id}_*/sub-{experiment_id}_ritnet_fullclass.csv"))
    return candidates[0] if candidates else None


def _find_rgb_parquet(rgb_root: Path, experiment_id: str) -> Path | None:
    candidate = rgb_root / f"sub-{experiment_id}" / f"sub-{experiment_id}_face_raw.parquet"
    return candidate if candidate.exists() else None


def _registry_group(registry: Mapping[str, Any], group_id: str) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    for entry in registry["by_experiment_id"].values():
        if entry["local_repeat_participant_id"] == group_id:
            entries[group_id] = entry
    if group_id not in entries:
        raise ValueError(f"repeat group not found in registry: {group_id}")
    return entries[group_id]


def run_pilot(
    repeat_registry_path: Path,
    repeat_group: str,
    nir_root: Path,
    rgb_root: Path,
    output_dir: Path,
    tolerance_ms: int = 1000,
) -> dict[str, Any]:
    registry = read_repeat_registry_csv(repeat_registry_path)
    group_entry = _registry_group(registry, repeat_group)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []
    session_results: list[dict[str, Any]] = []
    for experiment_id in group_entry["experiment_ids"]:
        nir_path = _find_nir_csv(nir_root, experiment_id)
        rgb_path = _find_rgb_parquet(rgb_root, experiment_id)
        inventory_item = {
            "experiment_id": experiment_id,
            "subject": f"sub-{experiment_id}",
            "nir_path": str(nir_path) if nir_path else None,
            "rgb_path": str(rgb_path) if rgb_path else None,
            "nir_present": nir_path is not None,
            "rgb_present": rgb_path is not None,
        }
        inventory.append(inventory_item)
        if nir_path is None or rgb_path is None:
            inventory_item["status"] = "missing_existing_output"
            continue
        nir = read_nir(nir_path)
        rgb = read_rgb(rgb_path)
        paired = add_nir_bbox_features(build_paired(nir, rgb, tolerance_ms=tolerance_ms))
        identity = {
            "site": "Beijing",
            "session_id": f"sub-{experiment_id}",
            "local_repeat_participant_id": repeat_group,
            "global_repeat_participant_id": None,
            "repeat_session_count": group_entry["session_count"],
        }
        session_results.append(
            {
                "experiment_id": experiment_id,
                "subject": f"sub-{experiment_id}",
                "identity": identity,
                "input_rows": {"nir": int(len(nir)), "rgb_primary": int(len(rgb)), "paired": int(len(paired))},
                "alignment": {
                    "matched_rows": int(paired["unix_ms_rgb"].notna().sum()),
                    "valid_pupil_rows": int(paired["pupil_valid"].sum()),
                    "valid_pupil_rows_abs_delta_le_50ms": int((paired["pupil_valid"] & pd.to_numeric(paired["abs_delta_ms"], errors="coerce").le(50)).sum()),
                },
                "nir_bbox_stability_by_eye": bbox_stability(paired),
                "methods": {method: evaluate_method(paired, method) for method in METHOD_SPECS},
            }
        )
        inventory_item["status"] = "available_and_evaluated"

    cross_session: dict[str, Any] = {
        "status": "not_estimable" if len(session_results) < 2 else "estimated",
        "available_sessions": [item["experiment_id"] for item in session_results],
        "missing_sessions": [item["experiment_id"] for item in inventory if item["status"] != "available_and_evaluated"],
        "note": "No session time series were concatenated; cross-session consistency is computed only when at least two existing complete sessions are available.",
        "by_method": {},
    }
    if len(session_results) >= 2:
        for method in METHOD_SPECS:
            centers = []
            scales = []
            for session in session_results:
                metric = session["methods"][method]["corrected_pupil_measurement"]["baseline"]
                if metric["n"]:
                    centers.append(float(metric["quantiles"]["q50"]))
                    scales.append(float(metric["quantiles"]["q90"] - metric["quantiles"]["q10"]))
            cross_session["by_method"][method] = {
                "sessions_with_baseline": len(centers),
                "baseline_center_cv": float(np.std(centers, ddof=1) / np.mean(centers)) if len(centers) > 1 and np.mean(centers) else None,
                "baseline_scale_cv": float(np.std(scales, ddof=1) / np.mean(scales)) if len(scales) > 1 and np.mean(scales) else None,
                "baseline_centers_px": centers,
                "baseline_scales_q90_minus_q10_px": scales,
            }
    summary = {
        "pilot_version": PILOT_VERSION,
        "validation_only": True,
        "outcome_data_used": False,
        "fit_scope": "baseline_only_for_M1_M2a_M2b_M3; M0_unadjusted_reference",
        "tolerance_ms": tolerance_ms,
        "evaluation_window_ms": 50,
        "repeat_registry": {
            "path": str(repeat_registry_path),
            "local_repeat_participant_id": repeat_group,
            "experiment_ids": group_entry["experiment_ids"],
            "session_count": group_entry["session_count"],
            "global_repeat_participant_id": None,
        },
        "session_inventory": inventory,
        "sessions": session_results,
        "cross_session_consistency": cross_session,
        "decision_boundary": "Pilot metrics are measurement/QC evidence only. Do not select a final correction formula or enter Behavior/Probe/ML statistics from this output.",
    }
    write_json(output_dir / f"repeat_group_{repeat_group}_cross_session_summary.json", summary)
    for session in session_results:
        write_json(output_dir / f"sub-{session['experiment_id']}_correction_pilot_summary.json", session)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat-registry", type=Path, required=True)
    parser.add_argument("--repeat-group", required=True)
    parser.add_argument("--nir-root", type=Path, default=Path(r"D:\_AttentionData\Beijing-NIR\amd-directml"))
    parser.add_argument("--rgb-root", type=Path, default=Path(r"D:\_AttentionData\Beijing-RGB"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/multimodal_pupil_correction_pilot"))
    parser.add_argument("--tolerance-ms", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_pilot(args.repeat_registry, args.repeat_group, args.nir_root, args.rgb_root, args.output_dir, args.tolerance_ms)
    print(pd.Series({
        "output_dir": str(args.output_dir),
        "available_sessions": len(summary["sessions"]),
        "missing_sessions": summary["cross_session_consistency"]["missing_sessions"],
        "cross_session_status": summary["cross_session_consistency"]["status"],
    }).to_json(force_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
