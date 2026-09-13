from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PRIMARY_PREDICTOR = "body_motion_energy_median"


def finalize_movement_science_handoff(movement_root: str | Path) -> dict[str, object]:
    """Synchronize the accepted P4 research decision into the compact handoff.

    This is metadata-only. It does not alter producer values, scientific model
    estimates, figures, the unified feature registry, or any supervised model.
    """
    root = Path(movement_root).expanduser().resolve()
    handoff_path = root / "tables/movement_feature_handoff.csv"
    manifest_path = root / "manifests/movement_science_output_manifest.json"
    if not handoff_path.is_file():
        raise FileNotFoundError(handoff_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    handoff = pd.read_csv(handoff_path, encoding="utf-8-sig", low_memory=False)
    required = {
        "predictor_column",
        "report_role",
        "measurement_qc_status",
        "estimability_status",
        "registry_ready",
        "researcher_freeze_required",
    }
    missing = sorted(required - set(handoff.columns))
    if missing:
        raise ValueError(f"Movement handoff missing fields: {missing}")

    primary = handoff["predictor_column"].astype(str).eq(PRIMARY_PREDICTOR)
    if int(primary.sum()) != 1:
        raise ValueError(f"Movement handoff must contain exactly one {PRIMARY_PREDICTOR} row")
    primary_row = handoff.loc[primary].iloc[0]
    if str(primary_row["estimability_status"]).startswith("not_estimable"):
        raise ValueError("P4 primary Movement predictor is not estimable")

    handoff["researcher_freeze_required"] = False
    handoff.loc[primary, "registry_ready"] = True
    handoff.loc[~primary, "registry_ready"] = False
    handoff.loc[primary, "report_role"] = "first_round_selected"
    handoff.loc[primary, "measurement_qc_status"] = "p4_real_run_accepted_first_round_frozen"
    handoff.loc[primary, "estimability_status"] = "estimable_frozen"
    handoff.loc[~primary, "measurement_qc_status"] = "p4_real_run_accepted_role_frozen"
    handoff.to_csv(handoff_path, index=False, encoding="utf-8-sig")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "p4_real_run_accepted": True,
            "first_round_feature_frozen": True,
            "first_round_feature": PRIMARY_PREDICTOR,
            "researcher_freeze_required": False,
            "final_feature_registry_mutated": False,
            "supervised_model_run": False,
            "multimodal_model_run": False,
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
