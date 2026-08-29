from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from attention_pipeline.config import load_config
from .identity_audit import load_reconciled_identity

VISIT_SENSITIVITY_VERSION = "nir-candidate-visit-sensitivity-v1"


def _resolve(config, key: str) -> Path:
    raw = config.section("paths").get(key)
    if raw in (None, ""):
        raise KeyError(f"formal pupil config missing paths.{key}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (config.path.parent.parent / path).resolve()
    return path


def run_candidate_visit_sensitivity(
    config_path: str | Path,
    *,
    subjects: Iterable[str] | None = None,
    paths_config: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path, paths_config=paths_config)
    root = _resolve(config, "output_root") / "candidate_validation"
    summary_path = root / "nir_candidate_session_block_metrics.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path, encoding="utf-8-sig", low_memory=False)
    if subjects:
        wanted = {str(x).strip() for x in subjects if str(x).strip()}
        summary = summary[summary["session_id"].astype(str).isin(wanted)].copy()

    identity = load_reconciled_identity(config_path, paths_config=paths_config)
    metadata_cols = [
        "session_id", "participant_key", "participant_group_id", "participant_identity_source",
        "visit_order", "prior_visit_count", "prior_same_stage_count", "is_first_visit",
        "identity_conflict_flag", "identity_conflict_fields",
    ]
    metadata = identity[[c for c in metadata_cols if c in identity.columns]].drop_duplicates("session_id")
    joined = summary.merge(metadata, on="session_id", how="left", validate="many_to_one")

    verified = joined["participant_key"].notna() & pd.to_numeric(joined["visit_order"], errors="coerce").notna()
    first_any = joined[verified & pd.to_numeric(joined["visit_order"], errors="coerce").eq(1)].copy()
    if "prior_same_stage_count" in joined:
        first_same_stage = joined[
            joined["participant_key"].notna()
            & pd.to_numeric(joined["prior_same_stage_count"], errors="coerce").eq(0)
        ].copy()
    else:
        first_same_stage = joined.iloc[0:0].copy()

    session_metric = (
        joined.groupby(
            ["participant_key", "session_id", "metric", "unit", "visit_order"],
            as_index=False, dropna=False,
        )["binocular_raw_median"].median()
    )
    changes: list[dict[str, Any]] = []
    for (participant, metric), current in session_metric.dropna(subset=["participant_key", "visit_order"]).groupby(
        ["participant_key", "metric"], sort=True
    ):
        current = current.sort_values(["visit_order", "session_id"], kind="stable")
        if len(current) < 2:
            continue
        rows = list(current.itertuples(index=False))
        for left, right in zip(rows[:-1], rows[1:], strict=False):
            a = float(left.binocular_raw_median) if pd.notna(left.binocular_raw_median) else np.nan
            b = float(right.binocular_raw_median) if pd.notna(right.binocular_raw_median) else np.nan
            changes.append({
                "participant_key": str(participant),
                "metric": str(metric),
                "unit": str(left.unit),
                "from_session_id": str(left.session_id),
                "to_session_id": str(right.session_id),
                "from_visit_order": int(left.visit_order),
                "to_visit_order": int(right.visit_order),
                "visit_order_gap": int(right.visit_order - left.visit_order),
                "from_value": a,
                "to_value": b,
                "directional_change": b - a if np.isfinite(a) and np.isfinite(b) else np.nan,
                "absolute_change": abs(b - a) if np.isfinite(a) and np.isfinite(b) else np.nan,
                "status": "estimable" if np.isfinite(a) and np.isfinite(b) else "not_estimable_missing_metric_value",
                "order_source": "questionnaire_repeat_registry",
                "endpoint_selection_allowed": False,
            })
    changes_df = pd.DataFrame(changes)

    status_rows = [
        {
            "analysis": "all_nir_candidate_session_blocks",
            "status": "ready",
            "n_rows": int(len(joined)),
            "participant_key_present_rows": int(joined["participant_key"].notna().sum()),
            "verified_visit_order_rows": int(verified.sum()),
            "reason": "all modality-available rows retained for primary candidate validation",
        },
        {
            "analysis": "first_any_visit_only",
            "status": "ready_complete_case_only" if int(verified.sum()) < len(joined) else "ready",
            "n_rows": int(len(first_any)),
            "participant_key_present_rows": int(first_any["participant_key"].notna().sum()),
            "verified_visit_order_rows": int(len(first_any)),
            "reason": "visit_order==1 from verified questionnaire registry; missing questionnaire rows are not imputed",
        },
        {
            "analysis": "first_same_stage_visit_only",
            "status": "ready_complete_case_only" if len(first_same_stage) else "not_estimable",
            "n_rows": int(len(first_same_stage)),
            "participant_key_present_rows": int(first_same_stage["participant_key"].notna().sum()) if len(first_same_stage) else 0,
            "verified_visit_order_rows": int(len(first_same_stage)),
            "reason": "prior_same_stage_count==0; distinct from first-ever project visit",
        },
        {
            "analysis": "directional_successive_visit_change",
            "status": "ready" if len(changes_df) else "not_estimable_no_repeat_nir_sessions_with_verified_order",
            "n_rows": int(len(changes_df)),
            "participant_key_present_rows": int(changes_df["participant_key"].nunique()) if len(changes_df) else 0,
            "verified_visit_order_rows": int(len(changes_df)),
            "reason": "ordered by verified visit_order; no ordering is inferred from sub/session number",
        },
    ]
    status = pd.DataFrame(status_rows)

    joined.to_csv(root / "nir_candidate_identity_visit_overlay.csv", index=False, encoding="utf-8-sig")
    first_any.to_csv(root / "nir_candidate_first_any_visit_metrics.csv", index=False, encoding="utf-8-sig")
    first_same_stage.to_csv(root / "nir_candidate_first_same_stage_visit_metrics.csv", index=False, encoding="utf-8-sig")
    changes_df.to_csv(root / "nir_candidate_repeat_visit_changes.csv", index=False, encoding="utf-8-sig")
    status.to_csv(root / "nir_candidate_visit_sensitivity_status.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": VISIT_SENSITIVITY_VERSION,
        "status": "complete",
        "n_candidate_rows": int(len(joined)),
        "n_rows_with_participant_key": int(joined["participant_key"].notna().sum()),
        "n_directional_change_rows": int(len(changes_df)),
        "visit_order_source": "questionnaire-derived subject_repeat_registry",
        "session_number_ordering_allowed": False,
        "missing_visit_order_imputation_allowed": False,
        "first_any_visit_and_first_same_stage_visit_are_distinct": True,
        "scientific_endpoint_freeze": "pending_real_data_scientific_review",
    }
    (root / "nir_candidate_visit_sensitivity_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
