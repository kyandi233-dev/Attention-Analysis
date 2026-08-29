"""Audit unadjusted vs adjusted pupil-behavior effects without overstating confound control.

The reference-model runner can fit nuisance-adjusted models even when no visual
covariate is estimable.  This module prevents those fits from being mislabeled
as formal visual-confound adjustment: visual adjustment is admitted only when an
adjusted fit actually contains at least one current/previous visual covariate and
the configured visual table was available.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from attention_pipeline.config import Config, load_config

ADJUSTMENT_AUDIT_VERSION = "nir-adjustment-audit-v1"
VISUAL_TOKENS = (
    "current_central_rel_lum_mean",
    "previous_central_rel_lum_mean",
    "current_central_rms_contrast",
    "previous_central_rms_contrast",
    "current_fruit_visible_area_fraction_central_roi",
    "previous_fruit_visible_area_fraction_central_roi",
    "stimulus_size",
    "prev_stimulus_size",
)


def _resolve(config: Config, key: str) -> Path:
    raw = config.section("paths").get(key)
    if raw in (None, ""):
        raise KeyError(f"formal pupil config missing paths.{key}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (config.path.parent.parent / path).resolve()
    return path


def _is_visual_covariate_set(text: str) -> bool:
    covariates = {item.strip() for item in str(text).split(";") if item.strip()}
    return any(token in covariates for token in VISUAL_TOKENS)


def build_adjustment_comparison(
    effects: pd.DataFrame,
    *,
    visual_status: dict[str, Any],
) -> pd.DataFrame:
    required = {
        "model_name", "outcome", "adjusted", "pupil_term", "estimate",
        "ci_low", "ci_high", "participant_group_n", "session_n", "n_rows",
        "covariates",
    }
    if effects.empty:
        return pd.DataFrame()
    missing = sorted(required - set(effects.columns))
    if missing:
        raise ValueError(f"reference effect table missing fields: {missing}")

    current = effects.copy()
    current["adjusted"] = current["adjusted"].astype(str).str.lower().isin({"true", "1", "yes"})
    rows: list[dict[str, Any]] = []
    for (outcome, pupil_term), group in current.groupby(["outcome", "pupil_term"], sort=True):
        unadjusted = group[~group["adjusted"]]
        adjusted = group[group["adjusted"]]
        if len(unadjusted) != 1 or len(adjusted) != 1:
            rows.append({
                "outcome": outcome,
                "pupil_term": pupil_term,
                "comparison_status": "not_estimable",
                "reason": f"expected one unadjusted and one adjusted effect; got {len(unadjusted)}/{len(adjusted)}",
                "formal_visual_adjustment_status": "not_estimable",
            })
            continue
        u = unadjusted.iloc[0]
        a = adjusted.iloc[0]
        visual_covariates_used = _is_visual_covariate_set(str(a["covariates"]))
        visual_table_available = visual_status.get("status") == "available"
        if visual_table_available and visual_covariates_used:
            adjustment_set = "visual_time_quality_adjusted"
            formal_visual_status = "estimable"
            reason = "configured visual table available and adjusted fit includes visual covariates"
        else:
            adjustment_set = "time_quality_adjusted_only"
            formal_visual_status = "not_estimable"
            if not visual_table_available:
                reason = "visual table unavailable; adjusted fit cannot be called visual-adjusted"
            else:
                reason = "visual table available but no visual covariate entered this fitted adjustment set"
        u_est = float(u["estimate"])
        a_est = float(a["estimate"])
        rows.append({
            "outcome": outcome,
            "pupil_term": pupil_term,
            "comparison_status": "estimable_pair",
            "unadjusted_estimate": u_est,
            "unadjusted_ci_low": float(u["ci_low"]),
            "unadjusted_ci_high": float(u["ci_high"]),
            "adjusted_estimate": a_est,
            "adjusted_ci_low": float(a["ci_low"]),
            "adjusted_ci_high": float(a["ci_high"]),
            "adjusted_minus_unadjusted": a_est - u_est,
            "absolute_estimate_change": abs(a_est - u_est),
            "relative_absolute_change": (
                abs(a_est - u_est) / abs(u_est)
                if np.isfinite(u_est) and u_est != 0 else np.nan
            ),
            "adjustment_set": adjustment_set,
            "formal_visual_adjustment_status": formal_visual_status,
            "visual_covariates_used": bool(visual_covariates_used),
            "visual_table_available": bool(visual_table_available),
            "adjusted_covariates": str(a["covariates"]),
            "participant_group_n_unadjusted": int(u["participant_group_n"]),
            "participant_group_n_adjusted": int(a["participant_group_n"]),
            "session_n_unadjusted": int(u["session_n"]),
            "session_n_adjusted": int(a["session_n"]),
            "n_rows_unadjusted": int(u["n_rows"]),
            "n_rows_adjusted": int(a["n_rows"]),
            "reason": reason,
            "reference_signal_only": True,
            "endpoint_selection_allowed": False,
        })
    return pd.DataFrame(rows)


def run_adjustment_audit(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = _resolve(config, "output_root") / "reference_adjusted_models"
    effect_path = root / "trial_unadjusted_adjusted_effects.csv"
    manifest_path = root / "reference_adjusted_models_manifest.json"
    failure_path = root / "model_failures.csv"
    root.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, Any]] = []
    if not effect_path.is_file() or not manifest_path.is_file():
        failures.append({
            "stage": "adjustment_audit",
            "status": "not_estimable",
            "reason": "reference adjusted-model outputs missing",
        })
        comparison = pd.DataFrame()
        visual_status = {"status": "unknown"}
    else:
        try:
            effects = pd.read_csv(effect_path, encoding="utf-8-sig", low_memory=False)
            source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            visual_status = source_manifest.get("visual_status", {"status": "unknown"})
            comparison = build_adjustment_comparison(effects, visual_status=visual_status)
        except Exception as exc:
            comparison = pd.DataFrame()
            visual_status = {"status": "unknown"}
            failures.append({
                "stage": "adjustment_audit",
                "status": "not_estimable",
                "reason": f"{type(exc).__name__}: {exc}",
            })

    comparison_path = root / "formal_unadjusted_vs_adjusted_comparison.csv"
    audit_failure_path = root / "adjustment_audit_failures.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(failures, columns=["stage", "status", "reason"]).to_csv(
        audit_failure_path, index=False, encoding="utf-8-sig"
    )
    formal_visual_n = (
        int(comparison["formal_visual_adjustment_status"].eq("estimable").sum())
        if "formal_visual_adjustment_status" in comparison else 0
    )
    nonvisual_adjusted_n = (
        int(comparison["adjustment_set"].eq("time_quality_adjusted_only").sum())
        if "adjustment_set" in comparison else 0
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": ADJUSTMENT_AUDIT_VERSION,
        "status": "complete" if not failures else "not_estimable",
        "visual_status": visual_status,
        "n_comparison_rows": int(len(comparison)),
        "n_formal_visual_adjusted_rows": formal_visual_n,
        "n_time_quality_only_rows": nonvisual_adjusted_n,
        "formal_visual_claim_allowed": bool(formal_visual_n > 0),
        "rule": "an adjusted fit is visual-adjusted only when a configured visual table is available and visual covariates actually enter that fit",
        "scientific_inference_authorized_by_code_alone": False,
        "effect_table": str(effect_path),
        "model_failure_table": str(failure_path),
        "comparison_table": str(comparison_path),
        "audit_failure_table": str(audit_failure_path),
    }
    (root / "adjustment_audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
