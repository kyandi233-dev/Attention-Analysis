from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

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
STRICT_WINDOW_EVIDENCE = (
    "existing RGB 5.5 producer; strict same-block [-30 s, 0) probe window; "
    "anchor trial excluded; no future information"
)

# Scientific reclassification of the existing RGB 5.5 producer fields.  Blink is
# intentionally absent: it belongs to Ocular.  Exposure/coverage are QC/device
# support and therefore do not become Movement handoff candidates.
MOVEMENT_CANDIDATES = (
    {
        "scientific_feature_id": "movement.body_motion_energy",
        "candidate_representation_id": "movement.body_motion_energy.median.pre30s.v1",
        "predictor": "body_motion_energy_median",
        "display_name": "身体动作能量（30秒中位数）",
        "unit": "dimensionless_motion_energy",
        "role": "primary_candidate",
        "registry_ready": False,
        "redundancy_relation": "first_round_primary_representation",
    },
    {
        "scientific_feature_id": "movement.pose_lateral_direction",
        "candidate_representation_id": "movement.pose_lateral_direction.median.pre30s.v1",
        "predictor": "pose_lateral_right_per_sec_median",
        "display_name": "姿态横向变化（30秒中位数）",
        "unit": "normalized_direction_score_per_sec",
        "role": "sensitivity_auxiliary",
        "registry_ready": False,
        "redundancy_relation": "pose_auxiliary_not_parallel_primary",
    },
    {
        "scientific_feature_id": "movement.pose_vertical_direction",
        "candidate_representation_id": "movement.pose_vertical_direction.median.pre30s.v1",
        "predictor": "pose_vertical_up_per_sec_median",
        "display_name": "姿态纵向变化（30秒中位数）",
        "unit": "normalized_direction_score_per_sec",
        "role": "sensitivity_auxiliary",
        "registry_ready": False,
        "redundancy_relation": "pose_auxiliary_not_parallel_primary",
    },
    {
        "scientific_feature_id": "movement.radial_proximity_proxy",
        "candidate_representation_id": "movement.radial_proximity_proxy.median.pre30s.v1",
        "predictor": "pose_radial_proximity_direction_score_median",
        "display_name": "径向接近方向代理（30秒中位数）",
        "unit": "dimensionless_proxy_score",
        "role": "qc_sensitivity_only",
        "registry_ready": False,
        "redundancy_relation": "dimensionless_nonphysical_proxy_not_displacement",
    },
)

MODEL_FILES = {
    "movement_task_progression.csv": "models/rgb_block_cycle_gee.csv",
    "movement_q1_models.csv": "models/rgb_q1_mnlogit.csv",
    "movement_q2_models.csv": "models/rgb_q2_ordinal_gee.csv",
    "movement_behavior_links.csv": "models/rgb_behavior_window_gee.csv",
    "movement_within_between.csv": "models/rgb_probe_within_between.csv",
}

PROBE_TABLE_RELATIVE = "tables/rgb_probe_pre30s_strict_features.csv"
SESSION_COVERAGE_RELATIVE = "tables/rgb_session_coverage.csv"
BLOCK_CYCLE_RELATIVE = "tables/rgb_block_cycle_table.csv"


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _finite(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.isfinite(values.to_numpy(float)), index=series.index)


def _participant_col(frame: pd.DataFrame) -> str:
    for name in ("participant_group_id", "analysis_group_token"):
        if name in frame.columns:
            return name
    raise ValueError("Movement probe table requires participant_group_id or analysis_group_token")


def _session_col(frame: pd.DataFrame) -> str:
    if "session_id" not in frame.columns:
        raise ValueError("Movement probe table requires session_id")
    return "session_id"


def _coverage(frame: pd.DataFrame, predictor: str) -> dict[str, object]:
    total = int(len(frame))
    if predictor not in frame.columns:
        return {
            "probe_total_n": total,
            "finite_probe_n": 0,
            "finite_fraction": 0.0 if total else None,
            "participant_group_n": 0,
            "session_n": 0,
        }
    mask = _finite(frame[predictor])
    pcol = _participant_col(frame)
    scol = _session_col(frame)
    return {
        "probe_total_n": total,
        "finite_probe_n": int(mask.sum()),
        "finite_fraction": float(mask.mean()) if total else None,
        "participant_group_n": int(frame.loc[mask, pcol].dropna().astype(str).nunique()),
        "session_n": int(frame.loc[mask, scol].dropna().astype(str).nunique()),
    }


def _filter_rows_for_predictors(frame: pd.DataFrame, predictors: Iterable[str]) -> pd.DataFrame:
    """Keep only rows whose model identity explicitly names a Movement predictor.

    Older RGB 5.5 tables used slightly different identity-column names, so this
    deliberately checks all scalar cells for an *exact* predictor value rather
    than guessing one historical schema.  It never uses substring matching.
    """
    wanted = {str(x) for x in predictors}
    if frame.empty:
        return frame.copy()
    object_cols = list(frame.columns)
    keep = pd.Series(False, index=frame.index)
    for col in object_cols:
        values = frame[col].astype("string").str.strip()
        keep = keep | values.isin(wanted)
    return frame.loc[keep].copy()


def _build_handoff(probes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in MOVEMENT_CANDIDATES:
        predictor = str(spec["predictor"])
        coverage = _coverage(probes, predictor)
        finite_n = int(coverage["finite_probe_n"])
        estimable = predictor in probes.columns and finite_n > 0
        rows.append(
            {
                "scientific_feature_id": spec["scientific_feature_id"],
                "candidate_representation_id": spec["candidate_representation_id"],
                "predictor_columns": json.dumps([predictor], ensure_ascii=False),
                "predictor_column": predictor,
                "display_name": spec["display_name"],
                "unit": spec["unit"],
                "scientific_modality": "movement",
                "source_namespace": "rgb",
                "required_devices": json.dumps(["rgb"], ensure_ascii=False),
                "preprocessing_dependencies": json.dumps(
                    ["rgb_formal_analysis_ready", "strict_probe_pre30s_window"],
                    ensure_ascii=False,
                ),
                "measurement_qc_status": (
                    "existing_rgb55_real_run_available_scientific_freeze_pending"
                    if estimable
                    else "not_estimable_in_source_table"
                ),
                "estimability_rule": "finite numeric value in the existing strict pre-probe RGB 5.5 table",
                "estimability_status": "estimable_candidate" if estimable else "not_estimable",
                "coverage_summary": json.dumps(coverage, ensure_ascii=False, sort_keys=True),
                "redundancy_relation": spec["redundancy_relation"],
                "temporal_anchor": "probe_time_ms",
                "temporal_scope": "pre_probe_only",
                "time_legality_status": "verified_pre_probe_only" if estimable else "pending_upstream_availability",
                "time_legality_evidence": STRICT_WINDOW_EVIDENCE if estimable else "",
                "report_role": spec["role"],
                # P4 deliberately stops before researcher freeze / final registry.
                "registry_ready": False,
                "researcher_freeze_required": True,
                "selection_policy": SELECTION_POLICY,
            }
        )
    return pd.DataFrame(rows, columns=HANDOFF_COLUMNS)


def _coverage_table(probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in MOVEMENT_CANDIDATES:
        predictor = str(spec["predictor"])
        row = {"predictor_column": predictor, "report_role": spec["role"]}
        row.update(_coverage(probes, predictor))
        rows.append(row)
    for predictor, role in (
        ("exposure_change_abs_median", "device_qc_only"),
        ("global_motion_energy_median", "device_qc_only"),
        ("gray_mean_median", "device_qc_only"),
        ("pose_coverage_fraction", "device_qc_only"),
        ("body_motion_coverage_fraction", "device_qc_only"),
    ):
        if predictor in probes.columns:
            row = {"predictor_column": predictor, "report_role": role}
            row.update(_coverage(probes, predictor))
            rows.append(row)
    return pd.DataFrame(rows)


def _pairwise_qc(probes: pd.DataFrame) -> pd.DataFrame:
    """Descriptive QC only; never substitutes participant-clustered scientific models."""
    primary = "body_motion_energy_median"
    auxiliaries = [
        "exposure_change_abs_median",
        "gray_mean_median",
        "pose_lateral_right_per_sec_median",
        "pose_vertical_up_per_sec_median",
        "pose_radial_proximity_direction_score_median",
        "pose_coverage_fraction",
        "body_motion_coverage_fraction",
    ]
    rows: list[dict[str, object]] = []
    if primary not in probes.columns:
        return pd.DataFrame(rows)
    x = pd.to_numeric(probes[primary], errors="coerce")
    for name in auxiliaries:
        if name not in probes.columns:
            continue
        y = pd.to_numeric(probes[name], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)
        n = int(mask.sum())
        corr = float(x[mask].corr(y[mask], method="spearman")) if n >= 3 else np.nan
        rows.append(
            {
                "movement_feature": primary,
                "qc_or_auxiliary_feature": name,
                "finite_pair_n": n,
                "spearman_rho": corr,
                "interpretation_scope": "descriptive_qc_only_not_repeated_measure_inference",
            }
        )
    return pd.DataFrame(rows)


def _plot_histogram(probes: pd.DataFrame, path: Path) -> bool:
    import matplotlib.pyplot as plt

    name = "body_motion_energy_median"
    if name not in probes.columns:
        return False
    values = pd.to_numeric(probes[name], errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(values.to_numpy(float), bins=30)
    ax.set_xlabel("Body motion energy median (dimensionless)")
    ax.set_ylabel("Probe windows")
    ax.set_title("Movement candidate distribution")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _plot_coverage(coverage: pd.DataFrame, path: Path) -> bool:
    import matplotlib.pyplot as plt

    if coverage.empty:
        return False
    data = coverage.copy()
    data["finite_fraction"] = pd.to_numeric(data["finite_fraction"], errors="coerce")
    data = data[np.isfinite(data["finite_fraction"])]
    if data.empty:
        return False
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.45 * len(data))))
    y = np.arange(len(data))
    ax.barh(y, data["finite_fraction"].to_numpy(float))
    ax.set_yticks(y, labels=data["predictor_column"].astype(str).tolist())
    ax.set_xlim(0, 1)
    ax.set_xlabel("Finite probe fraction")
    ax.set_title("Movement / RGB support coverage")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _plot_qc_scatter(probes: pd.DataFrame, path: Path) -> bool:
    import matplotlib.pyplot as plt

    xname = "body_motion_energy_median"
    yname = "exposure_change_abs_median"
    if xname not in probes.columns or yname not in probes.columns:
        return False
    x = pd.to_numeric(probes[xname], errors="coerce")
    y = pd.to_numeric(probes[yname], errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return False
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.scatter(x[mask], y[mask], s=10, alpha=0.35)
    ax.set_xlabel("Body motion energy median")
    ax.set_ylabel("Absolute exposure change median")
    ax.set_title("Movement vs exposure: QC view")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _find_level_column(frame: pd.DataFrame) -> str:
    # G1 is intentionally not frozen.  The cross-audit uses a level summary only
    # and accepts the current schema explicitly rather than silently deriving a new
    # representation.  Keep aliases for hardening-branch schema evolution.
    aliases = (
        "median",
        "level_median",
        "signal_median",
        "median_value",
        "window_median",
    )
    for name in aliases:
        if name in frame.columns:
            return name
    raise ValueError(
        "G1 probe_measurement_candidates has no recognized level-median column; "
        f"available columns={sorted(frame.columns.tolist())}"
    )


def _normalize_join_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "block_id" not in out.columns and "block_num" in out.columns:
        nums = pd.to_numeric(out["block_num"], errors="coerce")
        out["block_id"] = nums.map(lambda x: f"b{int(x)}" if pd.notna(x) else pd.NA)
    if "block_id" in out.columns:
        text = out["block_id"].astype("string").str.strip().str.lower()
        text = text.replace({"1": "b1", "2": "b2", "block1": "b1", "block2": "b2"})
        out["block_id"] = text
    if "probe_index_in_block" not in out.columns:
        for alias in ("probe_order_in_block", "probe_index_in_block"):
            if alias in out.columns:
                out["probe_index_in_block"] = pd.to_numeric(out[alias], errors="coerce")
                break
    return out


def _artifact_gee(
    data: pd.DataFrame,
    *,
    outcome: str,
    predictor: str,
    group: str,
) -> dict[str, object]:
    import statsmodels.api as sm
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.generalized_estimating_equations import GEE

    work = data[[outcome, predictor, group]].copy()
    work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
    work[predictor] = pd.to_numeric(work[predictor], errors="coerce")
    work = work[np.isfinite(work[outcome]) & np.isfinite(work[predictor]) & work[group].notna()].copy()
    n = int(len(work))
    participant_n = int(work[group].astype(str).nunique()) if n else 0
    if n < 10 or participant_n < 3 or float(work[predictor].std(ddof=0)) == 0.0:
        return {
            "status": "not_estimable",
            "n_probe": n,
            "participant_n": participant_n,
            "estimate": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
        }
    # z scaling makes geometry and ratio outcomes comparable as artifact-sensitivity
    # diagnostics without asserting a physical unit relation.
    x = (work[predictor] - work[predictor].mean()) / work[predictor].std(ddof=0)
    y = (work[outcome] - work[outcome].mean()) / work[outcome].std(ddof=0)
    exog = sm.add_constant(pd.DataFrame({predictor: x}), has_constant="add")
    try:
        fit = GEE(
            y,
            exog,
            groups=work[group].astype(str),
            family=sm.families.Gaussian(),
            cov_struct=Exchangeable(),
        ).fit()
        estimate = float(fit.params[predictor])
        se = float(fit.bse[predictor])
        return {
            "status": "estimable",
            "n_probe": n,
            "participant_n": participant_n,
            "estimate": estimate,
            "se": se,
            "ci_low": estimate - 1.96 * se,
            "ci_high": estimate + 1.96 * se,
            "p_value": float(fit.pvalues[predictor]),
        }
    except Exception as exc:  # explicit failure row; never fabricate an effect
        return {
            "status": "failed",
            "n_probe": n,
            "participant_n": participant_n,
            "estimate": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def build_ocular_movement_artifact_audit(
    g1_probe_candidates: pd.DataFrame,
    movement_probes: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-audit Ocular candidates against Movement/device artifacts.

    This is measurement-sensitivity evidence only.  It never selects geometry vs
    hard R_seg and never consults Q1/Q2 significance.
    """
    g1 = _normalize_join_keys(g1_probe_candidates)
    rgb = _normalize_join_keys(movement_probes)
    required = {"session_id", "block_id", "participant_group_id"}
    missing_g1 = sorted(required - set(g1.columns))
    missing_rgb = sorted(required - set(rgb.columns))
    if missing_g1 or missing_rgb:
        raise ValueError(f"cross-audit missing identity keys: g1={missing_g1}, rgb={missing_rgb}")

    # Prefer canonical probe key.  G1 always carries probe onset; RGB formal tables
    # generally carry probe_index_in_block.  Use only keys present on both sides.
    keys = ["session_id", "block_id", "participant_group_id"]
    if "probe_index_in_block" in g1.columns and "probe_index_in_block" in rgb.columns:
        keys.append("probe_index_in_block")
    elif "probe_onset_ms" in g1.columns and "probe_time_ms" in rgb.columns:
        g1 = g1.rename(columns={"probe_onset_ms": "probe_time_ms"})
        keys.append("probe_time_ms")
    else:
        raise ValueError("cross-audit requires shared probe_index_in_block or probe time identity")

    level_col = _find_level_column(g1)
    if "signal" not in g1.columns:
        raise ValueError("G1 probe_measurement_candidates missing signal column")

    movement_cols = [
        name
        for name in (
            "body_motion_energy_median",
            "pose_lateral_right_per_sec_median",
            "pose_vertical_up_per_sec_median",
            "pose_radial_proximity_direction_score_median",
            "exposure_change_abs_median",
        )
        if name in rgb.columns
    ]
    if not movement_cols:
        raise ValueError("Movement table has no P4 artifact predictors")

    rgb_unique = rgb[keys + movement_cols].copy()
    if rgb_unique.duplicated(keys).any():
        raise ValueError("Movement probe table is not unique on the cross-audit probe key")
    merged = g1.merge(rgb_unique, on=keys, how="left", validate="many_to_one", suffixes=("", "_rgb"))

    config_cols = [
        name
        for name in ("signal", "cleaning_track", "buffer_id", "bin_width_sec", "window_sec")
        if name in merged.columns
    ]
    rows: list[dict[str, object]] = []
    for config, part in merged.groupby(config_cols, dropna=False, sort=False):
        if not isinstance(config, tuple):
            config = (config,)
        metadata = dict(zip(config_cols, config))
        signal = str(metadata.get("signal", ""))
        if signal not in {"pupil_geom_mean_diameter", "seg_pupil_fraction_within_pupil_iris_hard"}:
            continue
        for predictor in movement_cols:
            result = _artifact_gee(part, outcome=level_col, predictor=predictor, group="participant_group_id")
            rows.append(
                {
                    **metadata,
                    "ocular_level_column": level_col,
                    "artifact_predictor": predictor,
                    **result,
                    "model": "Gaussian GEE; exchangeable participant cluster; standardized outcome/predictor",
                    "interpretation_scope": "measurement_artifact_sensitivity_only",
                    "representation_selection_allowed": False,
                    "q1_q2_used": False,
                }
            )
    return pd.DataFrame(rows)


def build_movement_science_output(
    rgb55_root: str | Path,
    science_root: str | Path,
    *,
    g1_probe_candidates_path: str | Path | None = None,
    authoritative: bool = True,
    replace: bool = True,
) -> dict[str, object]:
    """Build Movement scientific outputs from an already completed RGB 5.5 run.

    No video producer, final feature registry, supervised LOSO model, or multimodal
    model is run by this function.
    """
    source_root = Path(rgb55_root).expanduser().resolve()
    out_root = Path(science_root).expanduser().resolve() / "Movement"
    if out_root.exists() and replace:
        shutil.rmtree(out_root)
    if out_root.exists() and not replace:
        raise FileExistsError(out_root)
    for relative in (
        "tables",
        "figures/main",
        "figures/qualification",
        "figures/qc",
        "figures/sensitivity",
        "manifests",
    ):
        (out_root / relative).mkdir(parents=True, exist_ok=True)

    probes = _read_table(source_root / PROBE_TABLE_RELATIVE)
    handoff = _build_handoff(probes)
    coverage = _coverage_table(probes)
    qc = _pairwise_qc(probes)
    _write_csv(handoff, out_root / "tables/movement_feature_handoff.csv")
    _write_csv(coverage, out_root / "tables/movement_feature_coverage.csv")
    _write_csv(qc, out_root / "tables/movement_qc_associations.csv")

    source_support = {}
    for dest_name, source_relative in MODEL_FILES.items():
        source = source_root / source_relative
        if not source.exists():
            source_support[source_relative] = "missing"
            continue
        raw = _read_table(source)
        filtered = _filter_rows_for_predictors(raw, [str(x["predictor"]) for x in MOVEMENT_CANDIDATES])
        _write_csv(filtered, out_root / "tables" / dest_name)
        source_support[source_relative] = {"raw_n": int(len(raw)), "movement_n": int(len(filtered))}

    for dest_name, source_relative in (
        ("rgb_session_coverage_source.csv", SESSION_COVERAGE_RELATIVE),
        ("rgb_block_cycle_source.csv", BLOCK_CYCLE_RELATIVE),
    ):
        source = source_root / source_relative
        if source.exists():
            shutil.copy2(source, out_root / "tables" / dest_name)

    generated_figures = []
    if _plot_histogram(probes, out_root / "figures/qualification/movement_body_motion_distribution.png"):
        generated_figures.append("figures/qualification/movement_body_motion_distribution.png")
    if _plot_coverage(coverage, out_root / "figures/qualification/movement_feature_coverage.png"):
        generated_figures.append("figures/qualification/movement_feature_coverage.png")
    if _plot_qc_scatter(probes, out_root / "figures/qc/movement_vs_exposure_qc.png"):
        generated_figures.append("figures/qc/movement_vs_exposure_qc.png")

    ocular_audit_status: object = "not_requested"
    if g1_probe_candidates_path is not None:
        g1_path = Path(g1_probe_candidates_path).expanduser().resolve()
        g1 = _read_table(g1_path)
        ocular_audit = build_ocular_movement_artifact_audit(g1, probes)
        _write_csv(ocular_audit, out_root / "tables/ocular_cross_artifact_audit.csv")
        ocular_audit_status = {
            "status": "generated_measurement_sensitivity_only",
            "rows": int(len(ocular_audit)),
            "source": str(g1_path),
            "representation_frozen": False,
        }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "authoritative_scientific_output" if authoritative else "non_authoritative_smoke_output",
        "science_modality": "movement",
        "source_namespace": "rgb",
        "source_rgb55_root": str(source_root),
        "probe_table": str(source_root / PROBE_TABLE_RELATIVE),
        "probe_n": int(len(probes)),
        "participant_group_n": int(probes[_participant_col(probes)].dropna().astype(str).nunique()),
        "session_n": int(probes[_session_col(probes)].dropna().astype(str).nunique()),
        "primary_scientific_candidate": "body_motion_energy_median",
        "pose_role": "auxiliary_or_sensitivity_only",
        "radial_proxy_role": "dimensionless_nonphysical_qc_or_sensitivity_only",
        "blink_role": "excluded_from_movement_belongs_to_ocular",
        "exposure_coverage_role": "device_qc_only",
        "source_model_support": source_support,
        "generated_figures": generated_figures,
        "ocular_cross_artifact_audit": ocular_audit_status,
        "handoff_interface_columns": HANDOFF_COLUMNS,
        "selection_policy": SELECTION_POLICY,
        "final_feature_registry_mutated": False,
        "supervised_model_run": False,
        "multimodal_model_run": False,
        "researcher_freeze_required": True,
    }
    (out_root / "manifests/movement_science_output_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest
