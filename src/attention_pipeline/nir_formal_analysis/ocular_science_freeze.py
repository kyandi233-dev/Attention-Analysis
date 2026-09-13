from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from attention_pipeline.nir_formal_analysis.ocular_science_output import (
    FORMAL_BIN_WIDTH_SEC,
    FORMAL_RGB_BUFFER,
    GEOMETRY,
    HANDOFF_COLUMNS,
    RSEG_HARD,
    TRACKS,
    _config_mask,
    _normalize_block,
    _probe_keys,
    _read,
    build_ocular_science_output as _build_candidate_output,
)

FORMAL_MIN_TEMPORAL_SPAN_SEC = 20.0
FORMAL_BIN_N = 15
DYNAMIC = {"linear_slope_per_sec", "quadratic_curvature_per_sec2"}


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.lower().isin({"true", "1", "yes"})


def _metric(predictor: str) -> str | None:
    for name in ("level_mean", "level_median", "variability_sd", "variability_mad", "linear_slope_per_sec", "quadratic_curvature_per_sec2"):
        if f"__{name}__" in predictor:
            return name
    return None


def _role(predictor: str) -> str:
    if predictor == "ocular__blink_event_rate_per_min__pre30s":
        return "first_round_selected"
    metric = _metric(predictor)
    if "__geometry__" in predictor:
        return "cross_representation_sensitivity"
    if metric in {"level_median", "variability_sd"}:
        return "sensitivity_alternative"
    if "__nir_qc__" in predictor:
        return "nir_only_device_alternative"
    return "first_round_selected"


def build_frozen_ocular_science_output(
    g1_probe_candidates_path: str | Path,
    science_root: str | Path,
    *,
    rgb_probe_features_path: str | Path | None = None,
    movement_artifact_audit_path: str | Path | None = None,
    temporal_support_summary_path: str | Path | None = None,
    authoritative: bool = True,
    replace: bool = True,
) -> dict[str, object]:
    if authoritative and (rgb_probe_features_path is None or temporal_support_summary_path is None):
        raise ValueError("authoritative frozen Ocular output requires RGB blink input and temporal freeze evidence")

    source = _normalize_block(_read(g1_probe_candidates_path))
    keys = _probe_keys(source)
    if temporal_support_summary_path is not None:
        evidence = _read(temporal_support_summary_path)
        required = {"signal", "cleaning_track", "buffer_id", "bin_width_sec", "linear_span_ge_20s_fraction", "quadratic_span_ge_20s_fraction"}
        missing = sorted(required - set(evidence.columns))
        if missing:
            raise ValueError(f"temporal freeze evidence missing fields: {missing}")
        formal = evidence[
            evidence["signal"].astype(str).eq(RSEG_HARD)
            & evidence["cleaning_track"].astype(str).eq("rgb_plus_nir_qc")
            & evidence["buffer_id"].astype(str).eq(FORMAL_RGB_BUFFER)
            & pd.to_numeric(evidence["bin_width_sec"], errors="coerce").eq(FORMAL_BIN_WIDTH_SEC)
        ]
        if len(formal) != 1:
            raise ValueError("temporal freeze evidence lacks the unique formal hard-Rseg 2s row")

    _build_candidate_output(
        g1_probe_candidates_path,
        science_root,
        rgb_probe_features_path=rgb_probe_features_path,
        movement_artifact_audit_path=movement_artifact_audit_path,
        temporal_support_summary_path=temporal_support_summary_path,
        authoritative=authoritative,
        replace=replace,
    )

    root = Path(science_root).expanduser().resolve() / "Ocular"
    wide_path = root / "tables/ocular_probe_features_wide.csv"
    handoff_path = root / "tables/ocular_feature_handoff.csv"
    manifest_path = root / "manifests/ocular_science_output_manifest.json"
    wide = _normalize_block(_read(wide_path))

    for signal in (GEOMETRY, RSEG_HARD):
        sig = "geometry" if signal == GEOMETRY else "rseg_hard"
        for track, buffer_id, _devices, _label in TRACKS:
            trk = "nir_qc" if track == "nir_qc_only" else "rgb_nir_qc"
            subset = source[source["signal"].astype(str).eq(signal) & _config_mask(source, track, buffer_id)].copy()
            for metric in DYNAMIC:
                predictor = f"ocular__{sig}__{trk}__{metric}__2s"
                if subset.empty or predictor not in wide.columns:
                    continue
                status = "linear_status" if metric == "linear_slope_per_sec" else "quadratic_status"
                required = {status, "temporal_span_sec", "has_early_support", "has_late_support"}
                missing = sorted(required - set(subset.columns))
                if missing:
                    raise ValueError(f"G1 dynamic fields missing: {missing}")
                flag = subset[keys].copy()
                flag["__ok"] = (
                    subset[status].astype(str).eq("computable")
                    & _truthy(subset["has_early_support"])
                    & _truthy(subset["has_late_support"])
                    & pd.to_numeric(subset["temporal_span_sec"], errors="coerce").ge(FORMAL_MIN_TEMPORAL_SPAN_SEC)
                ).to_numpy()
                wide = wide.merge(flag, on=keys, how="left", validate="one_to_one")
                wide[predictor] = pd.to_numeric(wide[predictor], errors="coerce").where(wide["__ok"].fillna(False))
                wide = wide.drop(columns="__ok")
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")

    handoff = _read(handoff_path)
    for i, row in handoff.iterrows():
        predictor = str(row["predictor_column"])
        handoff.at[i, "measurement_qc_status"] = "g1_complete_parameters_frozen" if predictor != "ocular__blink_event_rate_per_min__pre30s" else "existing_rgb55_event_source"
        handoff.at[i, "time_legality_status"] = "verified_pre_probe_only"
        handoff.at[i, "report_role"] = _role(predictor)
        handoff.at[i, "registry_ready"] = True
        handoff.at[i, "researcher_freeze_required"] = False
        metric = _metric(predictor)
        if metric in DYNAMIC:
            handoff.at[i, "estimability_rule"] = "computable + early/late support + temporal_span_sec >= 20"
            handoff.at[i, "estimability_status"] = "estimable_frozen_rule"
            handoff.at[i, "time_legality_evidence"] = str(row["time_legality_evidence"]) + "; frozen span >=20s"
    handoff[HANDOFF_COLUMNS].to_csv(handoff_path, index=False, encoding="utf-8-sig")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "technical_freeze": {"blink_buffer": FORMAL_RGB_BUFFER, "bin_width_sec": FORMAL_BIN_WIDTH_SEC, "bin_n": FORMAL_BIN_N, "minimum_temporal_span_sec": FORMAL_MIN_TEMPORAL_SPAN_SEC, "requires_early_and_late_support": True, "interpolation": False},
        "first_round_pupil_representation": RSEG_HARD,
        "cross_representation_sensitivity": GEOMETRY,
        "segmentation_sensitivity": "seg_pupil_fraction_within_pupil_iris_soft",
        "first_round_level_metric": "level_mean",
        "first_round_variability_metric": "variability_mad",
        "sensitivity_metrics": ["level_median", "variability_sd"],
        "dynamic_metrics": ["linear_slope_per_sec", "quadratic_curvature_per_sec2"],
        "researcher_freeze_required": False,
        "p3_measurement_parameters_frozen": True,
        "final_feature_registry_mutated": False,
        "supervised_model_run": False,
        "multimodal_model_run": False,
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
