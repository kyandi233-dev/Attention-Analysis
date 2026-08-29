"""Final validation layer for the pupil-only NIR formal chain.

Validation is deliberately downstream-only: it consumes completed
``11_analysis_tables`` session products, never raw RITnet production output.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .pupil_figures import FIGURE_SUITE_VERSION, write_pupil_figure_suite
from .scientific_contract import (
    MODEL_FAILURE_COLUMNS,
    attach_causal_visual_covariates,
    audit_feature_family_columns,
    binary_classification_metrics,
    build_phasic_pupil_features,
    participant_exclusive_outer_folds,
    qc_count_axes,
    repeat_session_descriptive_summary,
    report_admission,
    split_sart_error_targets,
)


class ValidationContractError(ValueError):
    pass


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_stage_manifests(root: Path) -> list[dict[str, str]]:
    sessions_root = root / "sessions"
    if not sessions_root.is_dir():
        raise FileNotFoundError(sessions_root)
    rows: list[dict[str, str]] = []
    for session_dir in sorted(p for p in sessions_root.iterdir() if p.is_dir()):
        sid = session_dir.name
        path = session_dir / f"{sid}_analysis_tables_manifest.json"
        if not path.is_file():
            raise ValidationContractError(f"{sid}: missing 11_analysis_tables manifest")
        rows.append({"session_id": sid, "path": str(path), "sha256": _sha256(path)})
    return rows


def load_analysis_table_cohort(root: Path) -> dict[str, pd.DataFrame]:
    """Load only completed pupil-only 11_analysis_tables sessions."""
    sessions_root = root / "sessions"
    if not sessions_root.is_dir():
        raise FileNotFoundError(sessions_root)
    buckets: dict[str, list[pd.DataFrame]] = {
        "trials": [],
        "trial_windows": [],
        "probe_windows": [],
        "time_on_task": [],
        "dependency": [],
    }
    for session_dir in sorted(p for p in sessions_root.iterdir() if p.is_dir()):
        sid = session_dir.name
        manifest_path = session_dir / f"{sid}_analysis_tables_manifest.json"
        completion_path = session_dir / f"{sid}_analysis_tables_completion.json"
        if not manifest_path.is_file() or not completion_path.is_file():
            raise ValidationContractError(
                f"{sid}: validation refuses a session without stage manifest/completion"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") != "complete":
            raise ValidationContractError(f"{sid}: analysis tables are not complete")
        if manifest.get("signal_semantics") != "pupil_geometry_only":
            raise ValidationContractError(f"{sid}: non-pupil-only manifest is forbidden")
        expected = {
            "trials": session_dir / f"{sid}_trial_level.csv",
            "trial_windows": session_dir / f"{sid}_trial_pupil_windows.csv",
            "probe_windows": session_dir / f"{sid}_probe_pupil_windows.csv",
            "time_on_task": session_dir / f"{sid}_time_on_task_1s.csv",
            "dependency": session_dir / f"{sid}_window_dependency_audit.csv",
        }
        for key, path in expected.items():
            frame = _read(path)
            if "session_id" in frame:
                actual = set(frame["session_id"].dropna().astype(str))
                if actual not in ({sid}, set()):
                    raise ValidationContractError(f"{path}: session identity mismatch")
            buckets[key].append(frame)
    if not buckets["trials"]:
        raise ValidationContractError("no completed pupil-only analysis-table sessions")
    return {
        key: pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        for key, frames in buckets.items()
    }


def attach_visual_with_temporal_gate(
    trial_windows: pd.DataFrame,
    trials: pd.DataFrame,
    visual: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach stimulus covariates while rejecting current/future visual leakage.

    Current-trial visual properties are unavailable to strictly pre-stimulus
    windows. Previous-trial properties remain causally available.
    """
    linked_trials = attach_causal_visual_covariates(trials, visual)
    keys = [
        c
        for c in [
            "session_id",
            "analysis_group_token",
            "block_num",
            "trial_num",
            "global_trial_index",
        ]
        if c in trial_windows.columns and c in linked_trials.columns
    ]
    if not keys:
        raise ValidationContractError("no common formal trial key for visual join")
    visual_cols = [
        c
        for c in linked_trials.columns
        if c.startswith("current_visual__") or c.startswith("previous_visual__")
    ]
    meta_cols = [
        c
        for c in [
            "current_visual_matched",
            "previous_visual_matched",
            "visual_time_direction",
            "visual_multiple_component_policy",
        ]
        if c in linked_trials.columns
    ]
    right = linked_trials[[*keys, *visual_cols, *meta_cols]].drop_duplicates(keys)
    out = trial_windows.merge(right, on=keys, how="left", validate="many_to_one")
    end = pd.to_numeric(out.get("window_end_offset_ms"), errors="coerce")
    pre = end.le(0)
    current_cols = [c for c in out.columns if c.startswith("current_visual__")]
    if current_cols:
        out.loc[pre, current_cols] = np.nan
    out["current_visual_allowed"] = ~pre
    out["previous_visual_allowed"] = True
    out["visual_temporal_gate_status"] = np.where(
        pre,
        "current_rejected_pre_stimulus",
        "current_allowed_post_or_concurrent",
    )
    out["visual_temporal_tolerance_ms"] = 0.0
    audit = (
        out.groupby(["window_name", "visual_temporal_gate_status"], dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
    )
    audit["future_or_current_brightness_in_pre_window"] = False
    return out, audit


def _average_precision(y: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(score)
    y, score = y[valid].astype(int), score[valid]
    if len(y) == 0 or not np.any(y == 1):
        return math.nan
    order = np.argsort(-score, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    precision = tp / np.maximum(1, tp + fp)
    return float(np.sum(precision[ys == 1]) / np.sum(ys == 1))


def participant_exclusive_prediction(
    trials: pd.DataFrame,
    trial_windows: pd.DataFrame,
    *,
    n_splits: int = 5,
    feature_columns: Sequence[str] = (
        "pupil_median",
        "pupil_mad",
        "pupil_iqr",
        "pupil_slope_per_sec",
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leakage-safe pupil increment screen for the two SART error targets."""
    targets = split_sart_error_targets(trials)
    track = trial_windows.get("track", pd.Series(index=trial_windows.index, dtype=str))
    window = trial_windows.get("window_name", pd.Series(index=trial_windows.index, dtype=str))
    pre = trial_windows[
        track.astype(str).eq("binocular_primary") & window.astype(str).eq("pre_200ms")
    ].copy()
    join_candidates = ["session_id", "block_num", "trial_num", "global_trial_index"]
    keys = [c for c in join_candidates if c in targets.columns and c in pre.columns]
    if not keys:
        failure = {
            "analysis_question": "prediction",
            "target": "all",
            "model_stage": "nir",
            "fold": pd.NA,
            "model_name": "participant_exclusive_prediction",
            "status": "not_estimable",
            "failure_type": "missing_join_key",
            "failure_detail": "trial/NIR windows have no common formal key",
        }
        return pd.DataFrame(), pd.DataFrame([failure], columns=MODEL_FAILURE_COLUMNS)
    features = [c for c in feature_columns if c in pre.columns]
    if not features:
        failure = {
            "analysis_question": "prediction",
            "target": "all",
            "model_stage": "nir",
            "fold": pd.NA,
            "model_name": "participant_exclusive_prediction",
            "status": "not_estimable",
            "failure_type": "missing_features",
            "failure_detail": "pre_200ms pupil features unavailable",
        }
        return pd.DataFrame(), pd.DataFrame([failure], columns=MODEL_FAILURE_COLUMNS)
    right_cols = [*keys, *features]
    merged = targets.merge(
        pre[right_cols].drop_duplicates(keys),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    metrics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for target_name in ["go_omission_target", "nogo_commission_target"]:
        d = merged[merged[target_name].isin([0, 1])].copy()
        d = d.dropna(subset=["analysis_group_token"])
        if d["analysis_group_token"].nunique() < 2:
            failures.append(
                {
                    "analysis_question": "prediction",
                    "target": target_name,
                    "model_stage": "nir",
                    "fold": pd.NA,
                    "model_name": "Logit",
                    "status": "not_estimable",
                    "failure_type": "insufficient_groups",
                    "failure_detail": "<2 analysis groups",
                }
            )
            continue
        assignments = participant_exclusive_outer_folds(
            d, target_col=target_name, n_splits=n_splits
        )
        d = d.merge(
            assignments[["analysis_group_token", "outer_fold"]],
            on="analysis_group_token",
            how="left",
            validate="many_to_one",
        )
        for fold in sorted(d["outer_fold"].dropna().astype(int).unique()):
            train = d[d["outer_fold"].ne(fold)].copy()
            test = d[d["outer_fold"].eq(fold)].copy()
            y_train = pd.to_numeric(train[target_name], errors="coerce").astype(int)
            y_test = pd.to_numeric(test[target_name], errors="coerce").astype(int).to_numpy()
            if y_train.nunique() < 2 or len(test) == 0:
                failures.append(
                    {
                        "analysis_question": "prediction",
                        "target": target_name,
                        "model_stage": "nir",
                        "fold": int(fold),
                        "model_name": "Logit",
                        "status": "not_estimable",
                        "failure_type": "single_class_or_empty_fold",
                        "failure_detail": "training class or test fold unavailable",
                    }
                )
                continue
            majority_class = int(y_train.mean() >= 0.5)
            base_score = np.full(len(test), float(y_train.mean()))
            base_pred = np.full(len(test), majority_class, dtype=int)
            base = binary_classification_metrics(y_test, y_pred=base_pred, score=base_score)
            metrics.append(
                {
                    "target": target_name,
                    "outer_fold": int(fold),
                    "model_stage": "majority_baseline",
                    **base,
                    "pr_auc": _average_precision(y_test, base_score),
                }
            )
            try:
                import statsmodels.api as sm

                train_x = train[features].apply(pd.to_numeric, errors="coerce")
                test_x = test[features].apply(pd.to_numeric, errors="coerce")
                keep = train_x.notna().mean().ge(0.5)
                used = list(keep[keep].index)
                if not used:
                    raise ValueError("all pupil features fail training coverage gate")
                med = train_x[used].median()
                train_x = train_x[used].fillna(med)
                test_x = test_x[used].fillna(med)
                mean = train_x.mean()
                sd = train_x.std(ddof=0).replace(0, 1.0)
                train_x = sm.add_constant((train_x - mean) / sd, has_constant="add")
                test_x = sm.add_constant((test_x - mean) / sd, has_constant="add")
                fit = sm.Logit(y_train.to_numpy(), train_x.to_numpy()).fit(
                    disp=False, maxiter=200
                )
                if getattr(fit, "mle_retvals", {}).get("converged", True) is False:
                    raise ValueError("logit did not converge")
                if not np.isfinite(np.asarray(fit.params, dtype=float)).all():
                    raise ValueError("non-finite model parameters")
                score = np.asarray(fit.predict(test_x.to_numpy()), dtype=float)
                pred = (score >= 0.5).astype(int)
                row = binary_classification_metrics(y_test, y_pred=pred, score=score)
                metrics.append(
                    {
                        "target": target_name,
                        "outer_fold": int(fold),
                        "model_stage": "nir_pre200ms",
                        **row,
                        "pr_auc": _average_precision(y_test, score),
                        "feature_count": len(used),
                        "features": ";".join(used),
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "analysis_question": "prediction",
                        "target": target_name,
                        "model_stage": "nir",
                        "fold": int(fold),
                        "model_name": "Logit",
                        "status": "not_estimable",
                        "failure_type": type(exc).__name__,
                        "failure_detail": str(exc),
                    }
                )
    return pd.DataFrame(metrics), pd.DataFrame(failures, columns=MODEL_FAILURE_COLUMNS)


def _admission_windows(trials: pd.DataFrame, trial_windows: pd.DataFrame) -> pd.DataFrame:
    """Attach trial-level target semantics to each pupil window for report gates."""
    targets = split_sart_error_targets(trials)
    keys = [
        c
        for c in ["session_id", "block_num", "trial_num", "global_trial_index"]
        if c in targets.columns and c in trial_windows.columns
    ]
    if not keys:
        raise ValidationContractError("cannot build admission windows without trial key")
    target_cols = [
        c
        for c in ["go_omission_target", "nogo_commission_target", "target_denominator"]
        if c in targets.columns
    ]
    right = targets[[*keys, *target_cols]].drop_duplicates(keys)
    return trial_windows.merge(right, on=keys, how="left", validate="many_to_one")


def run_validation(
    *,
    tables_root: Path,
    output_root: Path,
    visual_properties: pd.DataFrame | None = None,
    formats: Iterable[str] = ("png",),
    dpi: int = 220,
    expected_topology: dict[str, int] | None = None,
    runtime_provenance: Mapping[str, Any] | None = None,
    config_digest: str | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite validation output: {output_root}")
    output_root.mkdir(parents=True)

    manifests = source_stage_manifests(tables_root)
    data = load_analysis_table_cohort(tables_root)
    trials = split_sart_error_targets(data["trials"])
    admission_windows = _admission_windows(trials, data["trial_windows"])

    visual_trials = pd.DataFrame()
    visual_audit = pd.DataFrame(
        columns=[
            "window_name",
            "visual_temporal_gate_status",
            "n_rows",
            "future_or_current_brightness_in_pre_window",
        ]
    )
    if visual_properties is not None and not visual_properties.empty:
        visual_trials, visual_audit = attach_visual_with_temporal_gate(
            data["trial_windows"], trials, visual_properties
        )

    phasic = build_phasic_pupil_features(data["trial_windows"])
    feature_audit = audit_feature_family_columns(data["trial_windows"])
    cv_metrics, model_failures = participant_exclusive_prediction(
        trials, data["trial_windows"]
    )
    repeat_summary = repeat_session_descriptive_summary(data["time_on_task"])

    topology = {
        "n_sessions": int(trials["session_id"].astype(str).nunique()),
        "n_analysis_groups": int(trials["analysis_group_token"].astype(str).nunique()),
    }
    group_sessions = (
        trials[["analysis_group_token", "session_id"]]
        .drop_duplicates()
        .groupby("analysis_group_token")["session_id"]
        .nunique()
    )
    topology["n_double_session_repeat_groups"] = int(group_sessions.eq(2).sum())
    if expected_topology is not None and topology != expected_topology:
        raise ValidationContractError(
            f"topology mismatch: observed={topology}, expected={expected_topology}"
        )

    qc_axes = qc_count_axes(
        sessions=trials["session_id"],
        analysis_groups=trials["analysis_group_token"],
        eye_rows=0,
        timepoints=len(data["time_on_task"]),
        trial_windows=len(data["trial_windows"]),
        probe_windows=len(data["probe_windows"]),
        failures=len(model_failures),
    )
    expected_names = [f"Figure{i:02d}" for i in range(1, 11)]
    admission_pre = report_admission(
        figure_names=expected_names,
        trial_windows=admission_windows,
        probe_windows=data["probe_windows"],
        model_failures=model_failures,
        failure_tables_written=True,
        topology=topology,
    )
    figures = write_pupil_figure_suite(
        output_dir=output_root / "figures",
        formats=formats,
        dpi=dpi,
        time_on_task=data["time_on_task"],
        trials=trials,
        trial_windows=data["trial_windows"],
        probe_windows=data["probe_windows"],
        dependency=data["dependency"],
        visual_trials=visual_trials,
        feature_audit=feature_audit,
        phasic=phasic,
        repeat_summary=repeat_summary,
        qc_axes=qc_axes,
        cv_metrics=cv_metrics,
        model_failures=model_failures,
        admission=admission_pre,
    )
    admission = report_admission(
        figure_names=figures.keys(),
        trial_windows=admission_windows,
        probe_windows=data["probe_windows"],
        model_failures=model_failures,
        failure_tables_written=True,
        topology=topology,
    )

    model_failures.to_csv(output_root / "model_failures.csv", index=False, encoding="utf-8-sig")
    cv_metrics.to_csv(
        output_root / "participant_exclusive_prediction.csv", index=False, encoding="utf-8-sig"
    )
    visual_audit.to_csv(output_root / "visual_temporal_audit.csv", index=False, encoding="utf-8-sig")
    feature_audit.to_csv(output_root / "feature_family_audit.csv", index=False, encoding="utf-8-sig")
    repeat_summary.to_csv(
        output_root / "repeat_session_descriptive.csv", index=False, encoding="utf-8-sig"
    )
    qc_axes.to_csv(output_root / "qc_count_axes.csv", index=False, encoding="utf-8-sig")
    (output_root / "report_admission.json").write_text(
        json.dumps(admission, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "pipeline": "nir-pupil-validation-v2",
        "schema_version": 2,
        "signal_semantics": "pupil_geometry_only",
        "figure_suite": FIGURE_SUITE_VERSION,
        "source_stage": "11_analysis_tables",
        "source_tables_root": str(tables_root),
        "source_stage_manifests": manifests,
        "config_digest": config_digest,
        "runtime_provenance": dict(runtime_provenance or {}),
        "topology": topology,
        "figures": figures,
        "model_failure_rows": int(len(model_failures)),
        "visual_temporal_gate_rows": int(len(visual_audit)),
        "report_package_admitted": bool(admission["report_package_admitted"]),
        "scientific_inference_authorized": False,
        "pir_oar_formal_use_allowed": False,
        "direct_raw_production_read_allowed": False,
    }
    (output_root / "validation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
