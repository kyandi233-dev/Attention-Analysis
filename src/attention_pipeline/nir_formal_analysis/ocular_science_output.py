from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HANDOFF_COLUMNS = [
    "scientific_feature_id",
    "candidate_representation_id",
    "predictor_columns",
    "predictor_column",
    "display_name",
    "unit",
    "scientific_modality",
    "source_namespace",
    "required_devices",
    "preprocessing_dependencies",
    "measurement_qc_status",
    "estimability_rule",
    "estimability_status",
    "coverage_summary",
    "redundancy_relation",
    "temporal_anchor",
    "temporal_scope",
    "time_legality_status",
    "time_legality_evidence",
    "report_role",
    "registry_ready",
    "researcher_freeze_required",
    "selection_policy",
]

SELECTION_POLICY = "Q1/Q2 significance and supervised outer-test performance are not freeze criteria"
GEOMETRY = "pupil_geom_mean_diameter"
RSEG_HARD = "seg_pupil_fraction_within_pupil_iris_hard"
FORMAL_BIN_WIDTH_SEC = 2.0
FORMAL_RGB_BUFFER = "pre200_post200"

METRICS = (
    ("level_mean", "pupil_level", "level_mean", "primary_candidate", "level_mean_vs_median_alternative"),
    ("level_median", "pupil_level", "level_median", "sensitivity_alternative", "level_mean_vs_median_alternative"),
    ("variability_sd", "pupil_variability", "variability_sd", "primary_candidate", "sd_vs_mad_alternative"),
    ("variability_mad", "pupil_variability", "variability_mad", "alternative_candidate", "sd_vs_mad_alternative"),
    ("linear_slope_per_sec", "pupil_linear_trend", "linear_slope", "dynamic_candidate", "trend_temporal_span_pending"),
    ("quadratic_curvature_per_sec2", "pupil_quadratic_curvature", "quadratic_curvature", "dynamic_candidate", "trend_temporal_span_pending"),
)

TRACKS = (
    ("nir_qc_only", None, ["nir"], "NIR QC only"),
    ("rgb_plus_nir_qc", FORMAL_RGB_BUFFER, ["nir", "rgb"], "RGB blink mask + NIR QC"),
)


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p, encoding="utf-8-sig", low_memory=False)


def _normalize_block(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "block_id" not in out.columns and "block_num" in out.columns:
        nums = pd.to_numeric(out["block_num"], errors="coerce")
        out["block_id"] = nums.map(lambda x: f"b{int(x)}" if pd.notna(x) else pd.NA)
    if "block_id" in out.columns:
        text = out["block_id"].astype("string").str.strip().str.lower()
        out["block_id"] = text.replace({"1": "b1", "2": "b2", "block1": "b1", "block2": "b2", "B1": "b1", "B2": "b2"})
    return out


def _probe_keys(frame: pd.DataFrame) -> list[str]:
    candidates = ["session_id", "participant_group_id", "block_id", "probe_index_in_block"]
    if all(name in frame.columns for name in candidates):
        return candidates
    fallback = ["session_id", "participant_group_id", "probe_event_id"]
    if all(name in frame.columns for name in fallback):
        return fallback
    fallback = ["session_id", "participant_group_id", "probe_index_global"]
    if all(name in frame.columns for name in fallback):
        return fallback
    raise ValueError("G1 probe candidates lack a canonical probe identity")


def _config_mask(frame: pd.DataFrame, track: str, buffer_id: str | None) -> pd.Series:
    mask = frame["cleaning_track"].astype(str).eq(track)
    mask &= pd.to_numeric(frame["bin_width_sec"], errors="coerce").eq(FORMAL_BIN_WIDTH_SEC)
    if buffer_id is not None:
        mask &= frame["buffer_id"].astype(str).eq(buffer_id)
    return mask


def _predictor_name(signal: str, track: str, metric: str) -> str:
    signal_token = "geometry" if signal == GEOMETRY else "rseg_hard"
    track_token = "nir_qc" if track == "nir_qc_only" else "rgb_nir_qc"
    return f"ocular__{signal_token}__{track_token}__{metric}__2s"


def _coverage(frame: pd.DataFrame, metric: str) -> dict[str, object]:
    values = pd.to_numeric(frame[metric], errors="coerce") if metric in frame.columns else pd.Series(np.nan, index=frame.index)
    finite = np.isfinite(values)
    return {
        "probe_total_n": int(len(frame)),
        "finite_probe_n": int(finite.sum()),
        "finite_fraction": float(finite.mean()) if len(frame) else None,
        "participant_group_n": int(frame.loc[finite, "participant_group_id"].dropna().astype(str).nunique()) if "participant_group_id" in frame else 0,
        "session_n": int(frame.loc[finite, "session_id"].dropna().astype(str).nunique()) if "session_id" in frame else 0,
    }


def _dynamic_status(frame: pd.DataFrame, metric: str) -> str:
    status_col = "linear_status" if metric == "linear_slope_per_sec" else "quadratic_status"
    if status_col not in frame.columns:
        return "pending_temporal_span_freeze"
    computable = frame[status_col].astype(str).eq("computable")
    return "pending_temporal_span_freeze" if computable.any() else "not_estimable"


def build_ocular_science_output(
    g1_probe_candidates_path: str | Path,
    science_root: str | Path,
    *,
    rgb_probe_features_path: str | Path | None = None,
    movement_artifact_audit_path: str | Path | None = None,
    temporal_support_summary_path: str | Path | None = None,
    authoritative: bool = True,
    replace: bool = True,
) -> dict[str, object]:
    """Materialize the narrow post-G1 Ocular candidate interface without freezing it.

    The function reads existing G1/RGB products only.  It does not reopen frame-level
    NIR/RGB data, rerun G1, select a pupil representation, mutate the final registry,
    or run supervised/multimodal prediction.
    """
    source = _normalize_block(_read(g1_probe_candidates_path))
    required = {"session_id", "participant_group_id", "signal", "cleaning_track", "buffer_id", "bin_width_sec"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"G1 probe candidates missing fields: {missing}")
    keys = _probe_keys(source)

    out_root = Path(science_root).expanduser().resolve() / "Ocular"
    if out_root.exists() and replace:
        shutil.rmtree(out_root)
    if out_root.exists() and not replace:
        raise FileExistsError(out_root)
    for rel in ("tables", "figures/main", "figures/qualification", "figures/qc", "figures/sensitivity", "manifests"):
        (out_root / rel).mkdir(parents=True, exist_ok=True)

    handoff_rows: list[dict[str, object]] = []
    wide: pd.DataFrame | None = None
    coverage_rows: list[dict[str, object]] = []

    for signal in (GEOMETRY, RSEG_HARD):
        for track, buffer_id, devices, track_label in TRACKS:
            subset = source[source["signal"].astype(str).eq(signal) & _config_mask(source, track, buffer_id)].copy()
            if subset.empty:
                continue
            if subset.duplicated(keys).any():
                raise ValueError(f"duplicate G1 probe identity for signal={signal}, track={track}, buffer={buffer_id}")
            base = subset[keys].copy()
            if wide is None:
                wide = base.drop_duplicates(keys).copy()
            else:
                wide = wide.merge(base.drop_duplicates(keys), on=keys, how="outer", validate="one_to_one")

            for metric, feature_id, metric_label, role, redundancy in METRICS:
                if metric not in subset.columns:
                    continue
                predictor = _predictor_name(signal, track, metric)
                values = subset[keys + [metric]].rename(columns={metric: predictor})
                wide = wide.merge(values, on=keys, how="outer", validate="one_to_one")
                coverage = _coverage(subset, metric)
                coverage_rows.append({"predictor_column": predictor, **coverage})
                dynamic = metric in {"linear_slope_per_sec", "quadratic_curvature_per_sec2"}
                estimability_status = _dynamic_status(subset, metric) if dynamic else ("estimable_candidate" if coverage["finite_probe_n"] else "not_estimable")
                unit = (
                    "signal_units_per_second" if metric == "linear_slope_per_sec" else
                    "signal_units_per_second_squared" if metric == "quadratic_curvature_per_sec2" else
                    "signal_units"
                )
                dependencies = [track, "fixed_2s_bins", "no_interpolation"]
                if buffer_id is not None:
                    dependencies.append(buffer_id)
                handoff_rows.append(
                    {
                        "scientific_feature_id": f"ocular.{feature_id}",
                        "candidate_representation_id": f"ocular.{feature_id}.{signal}.{track}.{metric_label}.v1",
                        "predictor_columns": json.dumps([predictor], ensure_ascii=False),
                        "predictor_column": predictor,
                        "display_name": f"{signal} | {track_label} | {metric_label}",
                        "unit": unit,
                        "scientific_modality": "ocular",
                        "source_namespace": "nir",
                        "required_devices": json.dumps(devices, ensure_ascii=False),
                        "preprocessing_dependencies": json.dumps(dependencies, ensure_ascii=False),
                        "measurement_qc_status": "g1_complete_parameter_freeze_pending",
                        "estimability_rule": "finite existing G1 probe-level candidate; dynamic terms additionally require the final temporal-span rule",
                        "estimability_status": estimability_status,
                        "coverage_summary": json.dumps(coverage, ensure_ascii=False, sort_keys=True),
                        "redundancy_relation": redundancy,
                        "temporal_anchor": "probe_time_ms",
                        "temporal_scope": "pre_probe_only",
                        "time_legality_status": "verified_pre_probe_only" if not dynamic else "pending_temporal_span_freeze",
                        "time_legality_evidence": "G1 fixed probe-relative pre-30s bins; 2s x 15; no interpolation; pre200/post200 for RGB-assisted track",
                        "report_role": role,
                        "registry_ready": False,
                        "researcher_freeze_required": True,
                        "selection_policy": SELECTION_POLICY,
                    }
                )

    if wide is None:
        raise ValueError("no post-G1 geometry/hard-Rseg candidates found for the frozen tracks/bin width")

    blink_status: object = "not_supplied"
    if rgb_probe_features_path is not None:
        rgb = _normalize_block(_read(rgb_probe_features_path))
        rgb_keys = [k for k in keys if k in rgb.columns]
        if set(keys) != set(rgb_keys):
            raise ValueError(f"RGB blink table missing Ocular probe keys: {sorted(set(keys) - set(rgb.columns))}")
        if "blink_event_rate_per_min" not in rgb.columns:
            raise ValueError("RGB probe table missing blink_event_rate_per_min")
        blink = rgb[keys + ["blink_event_rate_per_min"]].copy()
        if blink.duplicated(keys).any():
            raise ValueError("RGB blink probe table is not unique on canonical probe key")
        predictor = "ocular__blink_event_rate_per_min__pre30s"
        blink = blink.rename(columns={"blink_event_rate_per_min": predictor})
        wide = wide.merge(blink, on=keys, how="outer", validate="one_to_one")
        coverage = _coverage(rgb.rename(columns={"blink_event_rate_per_min": predictor}), predictor)
        coverage_rows.append({"predictor_column": predictor, **coverage})
        handoff_rows.append(
            {
                "scientific_feature_id": "ocular.blink_rate",
                "candidate_representation_id": "ocular.blink_rate.rgb_event_rate_per_min.pre30s.v1",
                "predictor_columns": json.dumps([predictor], ensure_ascii=False),
                "predictor_column": predictor,
                "display_name": "眨眼事件率（每分钟）",
                "unit": "events_per_min",
                "scientific_modality": "ocular",
                "source_namespace": "rgb",
                "required_devices": json.dumps(["rgb"], ensure_ascii=False),
                "preprocessing_dependencies": json.dumps(["rgb_blink_events", "strict_probe_pre30s_window"], ensure_ascii=False),
                "measurement_qc_status": "existing_rgb55_event_source",
                "estimability_rule": "finite blink event rate; missing RGB source remains missing and is never zero-filled",
                "estimability_status": "estimable_candidate" if coverage["finite_probe_n"] else "not_estimable",
                "coverage_summary": json.dumps(coverage, ensure_ascii=False, sort_keys=True),
                "redundancy_relation": "first_round_blink_scientific_dimension",
                "temporal_anchor": "probe_time_ms",
                "temporal_scope": "pre_probe_only",
                "time_legality_status": "verified_pre_probe_only",
                "time_legality_evidence": "existing RGB 5.5 strict same-block [-30s,0) probe window; anchor trial excluded",
                "report_role": "primary_candidate",
                "registry_ready": False,
                "researcher_freeze_required": True,
                "selection_policy": SELECTION_POLICY,
            }
        )
        blink_status = {"status": "included", "finite_probe_n": int(coverage["finite_probe_n"])}

    handoff = pd.DataFrame(handoff_rows, columns=HANDOFF_COLUMNS)
    coverage_table = pd.DataFrame(coverage_rows)
    handoff.to_csv(out_root / "tables/ocular_feature_handoff.csv", index=False, encoding="utf-8-sig")
    coverage_table.to_csv(out_root / "tables/ocular_feature_coverage.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(out_root / "tables/ocular_probe_features_wide.csv", index=False, encoding="utf-8-sig")

    artifact_status: object = "not_supplied"
    if movement_artifact_audit_path is not None:
        artifact = _read(movement_artifact_audit_path)
        artifact.to_csv(out_root / "tables/ocular_movement_artifact_sensitivity.csv", index=False, encoding="utf-8-sig")
        artifact_status = {"status": "included_measurement_sensitivity_only", "row_n": int(len(artifact))}

    temporal_status: object = "not_supplied"
    if temporal_support_summary_path is not None:
        temporal = _read(temporal_support_summary_path)
        temporal.to_csv(out_root / "tables/g1_temporal_support_freeze_grid.csv", index=False, encoding="utf-8-sig")
        temporal_status = {
            "status": "evidence_included_no_threshold_auto_selected",
            "row_n": int(len(temporal)),
            "formal_minimum_span_frozen": False,
        }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "authoritative_scientific_output" if authoritative else "non_authoritative_smoke_output",
        "science_modality": "ocular",
        "source_g1_probe_candidates": str(Path(g1_probe_candidates_path).expanduser().resolve()),
        "technical_freeze": {"blink_buffer": FORMAL_RGB_BUFFER, "bin_width_sec": 2.0, "bin_n": 15},
        "signals_kept_for_researcher_freeze": [GEOMETRY, RSEG_HARD],
        "level_candidates": ["level_mean", "level_median"],
        "variability_candidates": ["variability_sd", "variability_mad"],
        "dynamic_candidates": ["linear_slope_per_sec", "quadratic_curvature_per_sec2"],
        "blink": blink_status,
        "movement_cross_artifact_audit": artifact_status,
        "temporal_support": temporal_status,
        "handoff_interface_columns": HANDOFF_COLUMNS,
        "selection_policy": SELECTION_POLICY,
        "final_feature_registry_mutated": False,
        "supervised_model_run": False,
        "multimodal_model_run": False,
        "researcher_freeze_required": True,
    }
    (out_root / "manifests/ocular_science_output_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest
