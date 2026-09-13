import json
from pathlib import Path

import pandas as pd

from attention_pipeline.rgb_formal.movement_science_freeze import (
    PRIMARY_PREDICTOR,
    finalize_movement_science_handoff,
)


def test_finalize_movement_handoff_selects_only_primary_for_registry(tmp_path: Path) -> None:
    root = tmp_path / "Movement"
    (root / "tables").mkdir(parents=True)
    (root / "manifests").mkdir(parents=True)
    handoff = pd.DataFrame(
        [
            {
                "predictor_column": PRIMARY_PREDICTOR,
                "report_role": "primary_candidate",
                "measurement_qc_status": "p4_candidate",
                "estimability_status": "estimable_candidate",
                "registry_ready": False,
                "researcher_freeze_required": True,
            },
            {
                "predictor_column": "pose_lateral_right_per_sec_median",
                "report_role": "sensitivity_auxiliary",
                "measurement_qc_status": "p4_candidate",
                "estimability_status": "estimable_candidate",
                "registry_ready": False,
                "researcher_freeze_required": True,
            },
        ]
    )
    handoff.to_csv(root / "tables/movement_feature_handoff.csv", index=False)
    (root / "manifests/movement_science_output_manifest.json").write_text(
        json.dumps(
            {
                "final_feature_registry_mutated": False,
                "supervised_model_run": False,
                "multimodal_model_run": False,
                "researcher_freeze_required": True,
            }
        ),
        encoding="utf-8",
    )

    manifest = finalize_movement_science_handoff(root)
    actual = pd.read_csv(root / "tables/movement_feature_handoff.csv")
    primary = actual[actual["predictor_column"].eq(PRIMARY_PREDICTOR)].iloc[0]
    pose = actual[actual["predictor_column"].str.startswith("pose_")].iloc[0]

    assert bool(primary["registry_ready"]) is True
    assert bool(primary["researcher_freeze_required"]) is False
    assert primary["report_role"] == "first_round_selected"
    assert bool(pose["registry_ready"]) is False
    assert bool(pose["researcher_freeze_required"]) is False
    assert manifest["first_round_feature_frozen"] is True
    assert manifest["first_round_feature"] == PRIMARY_PREDICTOR
    assert manifest["final_feature_registry_mutated"] is False
    assert manifest["supervised_model_run"] is False
    assert manifest["multimodal_model_run"] is False
