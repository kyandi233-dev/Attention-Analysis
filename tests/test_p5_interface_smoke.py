import json
from pathlib import Path

import pandas as pd

from attention_pipeline.multimodal_formal.p5_interface_smoke import (
    REQUIRED_HANDOFF_COLUMNS,
    run_p5_interface_smoke,
)


def _handoff_row(
    modality: str,
    predictor: str,
    *,
    namespace: str,
    devices: list[str],
    ready: bool = True,
    pending: bool = False,
) -> dict[str, object]:
    row = {column: "x" for column in REQUIRED_HANDOFF_COLUMNS}
    row.update(
        {
            "scientific_feature_id": f"{modality}.{predictor}",
            "candidate_representation_id": f"{modality}.{predictor}.v1",
            "predictor_column": predictor,
            "scientific_modality": modality,
            "source_namespace": namespace,
            "required_devices": json.dumps(devices),
            "preprocessing_dependencies": "[]",
            "measurement_qc_status": "accepted",
            "estimability_status": "estimable",
            "temporal_anchor": "probe_time_ms",
            "temporal_scope": "pre_probe_only",
            "time_legality_status": "verified_pre_probe_only",
            "time_legality_evidence": "strict pre-probe window",
            "report_role": "first_round_selected",
            "registry_ready": ready,
            "researcher_freeze_required": pending,
        }
    )
    return row


def _probe_rows(extra: dict[str, list[float]]) -> pd.DataFrame:
    base = {
        "participant_group_id": ["p01", "p02"],
        "session_id": ["sub-001", "sub-002"],
        "block_id": ["B1", "B1"],
        "probe_index_in_block": [1, 1],
    }
    base.update(extra)
    return pd.DataFrame(base)


def _build_science_root(tmp_path: Path, *, missing_movement_predictor: bool = False) -> Path:
    science = tmp_path / "FormalScience"
    behavior = science / "Behavior"
    ocular = science / "Ocular"
    movement = science / "Movement"
    for root in (behavior, ocular, movement):
        (root / "tables").mkdir(parents=True)
        (root / "manifests").mkdir(parents=True)

    producer = tmp_path / "BehaviorProducer"
    producer.mkdir()
    behavior_probe = _probe_rows({"behavior_rt_cv": [0.1, 0.2], "behavior_rt_mean": [300.0, 320.0]})
    behavior_probe.to_csv(producer / "probe_primary_30s.csv", index=False)
    pd.DataFrame(
        [
            _handoff_row("behavior", "behavior_rt_cv", namespace="behavior", devices=[]),
            _handoff_row(
                "behavior",
                "behavior_rt_mean",
                namespace="behavior",
                devices=[],
                ready=True,
                pending=True,
            ),
        ]
    ).to_csv(behavior / "tables/behavior_feature_handoff.csv", index=False)
    (behavior / "manifests/science_output_manifest.json").write_text(
        json.dumps({"producer_formal_root": str(producer), "formal_registry_mutated": False}),
        encoding="utf-8",
    )

    _probe_rows({"ocular_main": [0.4, 0.5]}).to_csv(
        ocular / "tables/ocular_probe_features_wide.csv", index=False
    )
    pd.DataFrame(
        [_handoff_row("ocular", "ocular_main", namespace="nir", devices=["nir", "rgb"])]
    ).to_csv(ocular / "tables/ocular_feature_handoff.csv", index=False)
    (ocular / "manifests/ocular_science_output_manifest.json").write_text(
        json.dumps({"final_feature_registry_mutated": False, "supervised_model_run": False}),
        encoding="utf-8",
    )

    movement_column = "movement_missing" if missing_movement_predictor else "body_motion_energy_median"
    _probe_rows({"body_motion_energy_median": [0.01, 0.03]}).to_csv(
        movement / "tables/movement_probe_descriptive_source.csv", index=False
    )
    pd.DataFrame(
        [_handoff_row("movement", movement_column, namespace="rgb", devices=["rgb"])]
    ).to_csv(movement / "tables/movement_feature_handoff.csv", index=False)
    (movement / "manifests/movement_science_output_manifest.json").write_text(
        json.dumps(
            {
                "final_feature_registry_mutated": False,
                "supervised_model_run": False,
                "multimodal_model_run": False,
            }
        ),
        encoding="utf-8",
    )
    return science


def test_p5_smoke_passes_with_pending_behavior_freeze_as_warning(tmp_path: Path) -> None:
    science = _build_science_root(tmp_path)
    manifest = run_p5_interface_smoke(science)
    assert manifest["status"] == "pass"
    assert manifest["common_probe_n"] == 2
    assert manifest["common_session_n"] == 2
    assert manifest["common_participant_group_n"] == 2
    assert any(item.startswith("behavior:researcher_freeze_pending:") for item in manifest["warnings"])
    assert manifest["final_feature_registry_mutated"] is False
    assert manifest["supervised_model_run"] is False
    assert manifest["multimodal_model_run"] is False
    overlap = pd.read_csv(science / "P5_InterfaceSmoke/tables/p5_key_overlap.csv")
    assert int(overlap.loc[overlap["key_set"].eq("behavior_x_ocular_x_movement"), "probe_n"].iloc[0]) == 2


def test_p5_smoke_fails_closed_when_ready_predictor_is_missing(tmp_path: Path) -> None:
    science = _build_science_root(tmp_path, missing_movement_predictor=True)
    manifest = run_p5_interface_smoke(science)
    assert manifest["status"] == "fail"
    assert "movement:registry_ready_predictor_missing:movement_missing" in manifest["errors"]
