"""Repair and audit strict pre-probe behavior/visual exposure semantics.

A pre-probe window must contain only trials that occurred before the anchoring
probe trial.  ``trial_onset < probe_onset`` alone is insufficient when the probe
occurs after its anchoring trial has started.  This module therefore requires
both temporal precedence and ``trial_num < probe_trial_num`` within the block.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from attention_pipeline.config import Config, load_config
from attention_pipeline.nir_behavior.alignment import _behavior_window_features
from .pupil_tables import selected_sessions

PROBE_CONTRACT_VERSION = "nir-probe-prewindow-contract-v1"
VISUAL_METRICS = (
    "central_rel_lum_mean",
    "central_rms_contrast",
    "fruit_support_rel_lum_mean",
    "fruit_support_rms_contrast",
    "fruit_visible_area_fraction_central_roi",
    "delta_central_rel_lum_vs_background",
    "delta_central_rel_lum_vs_mask",
)
BEHAVIOR_FEATURES = (
    "n_trials",
    "n_go",
    "n_nogo",
    "n_commission",
    "n_omission",
    "n_prestimulus_press",
    "n_ambiguous_omission",
    "n_anticipatory_candidate",
    "go_rt_median_ms",
    "go_rt_mad_ms",
    "go_rt_iqr_ms",
    "go_rt_cv",
)


def _resolve(config: Config, key: str) -> Path:
    raw = config.section("paths").get(key)
    if raw in (None, ""):
        raise KeyError(f"formal pupil config missing paths.{key}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (config.path.parent.parent / path).resolve()
    return path


def _optional_path(config: Config, key: str) -> Path | None:
    raw = config.section("paths").get(key)
    if raw in (None, ""):
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (config.path.parent.parent / path).resolve()
    return path


def _session_paths(config: Config, session_id: str) -> tuple[Path, Path]:
    root = _resolve(config, "output_root") / "sessions" / session_id
    return (
        root / f"{session_id}_trial_level.csv",
        root / f"{session_id}_probe_pupil_windows.csv",
    )


def _strict_preprobe_trials(
    trials: pd.DataFrame,
    *,
    block_num: int,
    probe_trial_num: int,
    probe_onset_ms: float,
    start_ms: float,
    end_ms: float,
) -> tuple[pd.DataFrame, bool, int]:
    onset = pd.to_numeric(trials["absolute_onset_time"], errors="coerce")
    trial_num = pd.to_numeric(trials["trial_num"], errors="coerce")
    block = pd.to_numeric(trials["block_num"], errors="coerce")
    temporal = (
        block.eq(block_num)
        & onset.ge(start_ms)
        & onset.lt(end_ms)
        & onset.lt(probe_onset_ms)
    )
    old = trials[temporal]
    anchor_under_old_rule = bool(trial_num[temporal].eq(probe_trial_num).any())
    strict = trials[
        temporal
        & trial_num.lt(probe_trial_num)
    ].copy()
    if pd.to_numeric(strict["trial_num"], errors="coerce").ge(probe_trial_num).any():
        raise AssertionError("anchoring/future trial leaked into strict pre-probe window")
    return strict, anchor_under_old_rule, int(len(old))


def _visual_lookup(visual: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if visual.empty:
        return visual, []
    required = {"stimulus_name", "stimulus_size_pct"}
    missing = sorted(required - set(visual.columns))
    if missing:
        raise ValueError(f"stimulus visual table missing fields: {missing}")
    metrics = [m for m in VISUAL_METRICS if m in visual.columns]
    keep = ["stimulus_name", "stimulus_size_pct", *metrics]
    lookup = visual[keep].copy()
    lookup["stimulus_size_pct"] = pd.to_numeric(lookup["stimulus_size_pct"], errors="coerce")
    if lookup.duplicated(["stimulus_name", "stimulus_size_pct"]).any():
        raise ValueError("stimulus visual table has duplicate stimulus_name×size rows")
    return lookup, metrics


def _visual_exposure(
    strict_trials: pd.DataFrame,
    visual: pd.DataFrame,
    metrics: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "visual_trial_n": int(len(strict_trials)),
        "visual_joined_trial_n": 0,
        "visual_trial_coverage": np.nan,
    }
    if strict_trials.empty or visual.empty or not metrics:
        for metric in metrics:
            result[f"{metric}__preprobe_mean"] = np.nan
            result[f"{metric}__preprobe_median"] = np.nan
        return result
    current = strict_trials.copy()
    current["stimulus_size"] = pd.to_numeric(current["stimulus_size"], errors="coerce")
    joined = current.merge(
        visual,
        left_on=["stimulus_name", "stimulus_size"],
        right_on=["stimulus_name", "stimulus_size_pct"],
        how="left",
        validate="many_to_one",
    )
    joined_any = joined[metrics].notna().any(axis=1)
    result["visual_joined_trial_n"] = int(joined_any.sum())
    result["visual_trial_coverage"] = float(joined_any.mean()) if len(joined) else np.nan
    for metric in metrics:
        x = pd.to_numeric(joined[metric], errors="coerce")
        result[f"{metric}__preprobe_mean"] = float(x.mean()) if x.notna().any() else np.nan
        result[f"{metric}__preprobe_median"] = float(x.median()) if x.notna().any() else np.nan
    return result


def repair_session_probe_contract(
    config: Config,
    session_id: str,
    *,
    visual_lookup: pd.DataFrame,
    visual_metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_path, probe_path = _session_paths(config, session_id)
    if not trial_path.is_file() or not probe_path.is_file():
        raise FileNotFoundError(f"{session_id}: trial/probe table missing")
    trials = pd.read_csv(trial_path, encoding="utf-8-sig", low_memory=False)
    probes = pd.read_csv(probe_path, encoding="utf-8-sig", low_memory=False)
    required_probe = {
        "block_num", "probe_index_global", "probe_trial_num", "probe_onset_ms",
        "window_name", "window_start_ms", "window_end_ms",
    }
    missing = sorted(required_probe - set(probes.columns))
    if missing:
        raise ValueError(f"{session_id}: probe table missing fields {missing}")

    key_cols = [
        "block_num", "probe_index_global", "probe_trial_num", "probe_onset_ms",
        "window_name", "window_start_ms", "window_end_ms",
    ]
    unique = probes[key_cols].drop_duplicates().copy()
    features_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []

    for row in unique.itertuples(index=False):
        strict, anchor_old, old_n = _strict_preprobe_trials(
            trials,
            block_num=int(row.block_num),
            probe_trial_num=int(row.probe_trial_num),
            probe_onset_ms=float(row.probe_onset_ms),
            start_ms=float(row.window_start_ms),
            end_ms=float(row.window_end_ms),
        )
        behavior = _behavior_window_features(strict)
        key = tuple(getattr(row, c) for c in key_cols)
        features_by_key[key] = behavior
        base = {
            "session_id": session_id,
            **{c: getattr(row, c) for c in key_cols},
            "old_temporal_rule_trial_n": old_n,
            "corrected_strict_trial_n": int(len(strict)),
            "anchor_would_enter_old_temporal_rule": anchor_old,
            "anchor_present_after_correction": False,
            "contract_version": PROBE_CONTRACT_VERSION,
        }
        audit_rows.append(base)
        visual_rows.append({
            **base,
            **_visual_exposure(strict, visual_lookup, visual_metrics),
        })

    for idx, row in probes.iterrows():
        key = tuple(row[c] for c in key_cols)
        corrected = features_by_key[key]
        for field in BEHAVIOR_FEATURES:
            probes.at[idx, field] = corrected.get(field)
    probes["anchoring_probe_trial_excluded"] = True
    probes["probe_behavior_contract_version"] = PROBE_CONTRACT_VERSION
    probes.to_csv(probe_path, index=False, encoding="utf-8-sig")
    return pd.DataFrame(audit_rows), pd.DataFrame(visual_rows)


def run_probe_contract_repair(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    sessions = selected_sessions(config, subjects)
    visual_path = _optional_path(config, "stimulus_visual_table")
    visual_status: dict[str, Any]
    if visual_path is None:
        visual = pd.DataFrame()
        visual_status = {"status": "not_available", "reason": "paths.stimulus_visual_table_not_configured"}
    elif not visual_path.is_file():
        visual = pd.DataFrame()
        visual_status = {"status": "not_available", "reason": "visual_table_missing", "path": str(visual_path)}
    else:
        visual = pd.read_csv(visual_path, encoding="utf-8-sig", low_memory=False)
        visual_status = {"status": "available", "path": str(visual_path), "rows": int(len(visual))}
    visual_lookup, visual_metrics = _visual_lookup(visual)

    audits: list[pd.DataFrame] = []
    exposures: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    for session_id in sessions:
        try:
            audit, exposure = repair_session_probe_contract(
                config,
                session_id,
                visual_lookup=visual_lookup,
                visual_metrics=visual_metrics,
            )
            audits.append(audit)
            exposures.append(exposure)
        except Exception as exc:
            failures.append({
                "session_id": session_id,
                "stage": "probe_contract",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    root = _resolve(config, "output_root") / "probe_contract"
    root.mkdir(parents=True, exist_ok=True)
    audit_df = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    exposure_df = pd.concat(exposures, ignore_index=True) if exposures else pd.DataFrame()
    failure_df = pd.DataFrame(failures, columns=["session_id", "stage", "error_type", "error"])
    audit_df.to_csv(root / "probe_behavior_window_audit.csv", index=False, encoding="utf-8-sig")
    exposure_df.to_csv(root / "probe_visual_exposure.csv", index=False, encoding="utf-8-sig")
    failure_df.to_csv(root / "probe_contract_failures.csv", index=False, encoding="utf-8-sig")

    leaked = (
        bool(audit_df["anchor_present_after_correction"].fillna(False).any())
        if "anchor_present_after_correction" in audit_df else False
    )
    status = "complete" if not failures and not leaked else "blocked"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "contract_version": PROBE_CONTRACT_VERSION,
        "n_sessions_requested": len(sessions),
        "n_sessions_failed": len(failures),
        "anchor_leak_after_correction": leaked,
        "visual_status": visual_status,
        "visual_metrics": visual_metrics,
        "visual_semantics": "actual stimuli experienced inside strict pre-probe interval; anchoring probe trial excluded",
        "future_information_allowed": False,
        "outputs": {
            "behavior_audit": str(root / "probe_behavior_window_audit.csv"),
            "visual_exposure": str(root / "probe_visual_exposure.csv"),
            "failures": str(root / "probe_contract_failures.csv"),
        },
    }
    (root / "probe_contract_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
