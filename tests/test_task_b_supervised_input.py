from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.multimodal_formal.analysis_sets import build_analysis_sets
from attention_pipeline.multimodal_formal.quality_admission import audit_quality
from attention_pipeline.multimodal_formal.supervised_input import (
    SupervisedInputMaterializationError,
    materialize_supervised_input,
)


def _tables() -> dict[str, pd.DataFrame]:
    behavior = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s2", "s2"],
            "participant_group_id": ["p1", "p1", "p2", "p2"],
            "block_id": ["b1", "b1", "b1", "b1"],
            "probe_index_in_block": [1, 2, 1, 2],
            "probe_event_id": ["s1|b1|p1", "s1|b1|p2", "s2|b1|p1", "s2|b1|p2"],
            "probe_order_in_block": [1, 2, 1, 2],
            "probe_time_ms": [1000, 2000, 1000, 2000],
            "q1_nominal_4class": [1, 2, 1, 3],
            "q2_ordinal_4level": [4, 3, 4, 2],
            "window_name": ["pre_30s"] * 4,
            "b": [1.0, 2.0, 3.0, 4.0],
        }
    )
    nir = behavior[
        ["session_id", "participant_group_id", "block_id", "probe_index_in_block"]
    ].copy()
    nir["source_present"] = True
    nir["source_readable"] = True
    nir["window_name"] = "pre_30s"
    nir["n"] = [10.0, np.nan, 12.0, 13.0]
    return {"behavior": behavior, "nir": nir}


def _task_b_outputs(tables: dict[str, pd.DataFrame] | None = None):
    tables = tables or _tables()
    audit = audit_quality(tables, {"behavior": ["b"], "nir": ["n"]})
    sets, summary = build_analysis_sets(
        audit["formal_probe_identity"],
        audit["probe_feature_status"],
        {
            "behavior_plus_nir": {
                "models": ["M0", "M1"],
                "required_features": {"behavior": ["b"], "nir": ["n"]},
                "required_outcomes": ["q1_nominal_4class"],
            }
        },
    )
    return audit, sets, summary


def test_analysis_sets_persist_exact_required_feature_mapping() -> None:
    _, sets, summary = _task_b_outputs()
    encoded = sets["required_features"].drop_duplicates().tolist()
    assert len(encoded) == 1
    assert json.loads(encoded[0]) == {"behavior": ["b"], "nir": ["n"]}
    assert json.loads(summary.loc[0, "required_features"]) == {
        "behavior": ["b"],
        "nir": ["n"],
    }
    assert set(sets["feature_identity_mode"]) == {"legacy_source_feature"}
    legacy_records = json.loads(sets["required_feature_records"].iloc[0])
    assert {(record["source_namespace"], record["predictor_column"]) for record in legacy_records} == {
        ("behavior", "b"),
        ("nir", "n"),
    }


def test_complete_and_missing_aware_materialize_different_valid_memberships() -> None:
    tables = _tables()
    audit, sets, _ = _task_b_outputs(tables)

    complete = materialize_supervised_input(
        sets,
        audit["probe_feature_status"],
        analysis_set_id="behavior_plus_nir",
        membership_type="included_complete",
        probe_metadata=tables["behavior"],
    )
    missing_aware = materialize_supervised_input(
        sets,
        audit["probe_feature_status"],
        analysis_set_id="behavior_plus_nir",
        membership_type="included_missing_aware",
        probe_metadata=tables["behavior"],
    )

    assert len(complete) == 3
    assert complete[["b", "n"]].notna().all().all()
    assert set(complete["membership_type"]) == {"included_complete"}
    assert set(complete["comparison_models"]) == {json.dumps(["M0", "M1"])}

    assert len(missing_aware) == 4
    assert missing_aware["n"].isna().sum() == 1
    assert set(missing_aware["membership_type"]) == {"included_missing_aware"}
    assert missing_aware["probe_time_ms"].tolist() == [1000, 2000, 1000, 2000]
    assert set(missing_aware["feature_identity_mode"]) == {"legacy_source_feature"}
    missing_row = missing_aware[missing_aware["n"].isna()].iloc[0]
    assert missing_row["session_id"] == "s1"
    assert missing_row["probe_index_in_block"] == 2


def test_structural_missing_probe_never_enters_missing_aware_materialization() -> None:
    tables = _tables()
    tables["nir"] = tables["nir"].iloc[:3].copy()
    audit, sets, _ = _task_b_outputs(tables)
    frame = materialize_supervised_input(
        sets,
        audit["probe_feature_status"],
        analysis_set_id="behavior_plus_nir",
        membership_type="included_missing_aware",
        probe_metadata=tables["behavior"],
    )
    assert len(frame) == 3
    assert not (
        frame["session_id"].eq("s2") & frame["probe_index_in_block"].eq(2)
    ).any()


def test_complete_materialization_refuses_nonfinite_value_even_if_membership_is_corrupted() -> None:
    tables = _tables()
    audit, sets, _ = _task_b_outputs(tables)
    corrupted = sets.copy()
    target = corrupted["session_id"].eq("s1") & corrupted["probe_index_in_block"].eq(2)
    corrupted.loc[target, "included_complete"] = True

    with pytest.raises(
        SupervisedInputMaterializationError,
        match="included_complete contains non-finite required feature",
    ):
        materialize_supervised_input(
            corrupted,
            audit["probe_feature_status"],
            analysis_set_id="behavior_plus_nir",
            membership_type="included_complete",
            probe_metadata=tables["behavior"],
        )


def test_missing_aware_refuses_missing_value_outside_residual_feature_missingness() -> None:
    tables = _tables()
    audit, sets, _ = _task_b_outputs(tables)
    status = audit["probe_feature_status"].copy()
    target = (
        status["session_id"].eq("s1")
        & status["probe_index_in_block"].eq(2)
        & status["modality"].eq("nir")
        & status["feature"].eq("n")
    )
    status.loc[target, "missing_kind"] = "native_qc_invalid"

    with pytest.raises(
        SupervisedInputMaterializationError,
        match="outside residual single-feature missingness",
    ):
        materialize_supervised_input(
            sets,
            status,
            analysis_set_id="behavior_plus_nir",
            membership_type="included_missing_aware",
            probe_metadata=tables["behavior"],
        )


def test_explicit_feature_identity_maps_one_rgb_source_to_ocular_and_movement() -> None:
    tables = _tables()
    rgb = tables["behavior"][[
        "session_id", "participant_group_id", "block_id", "probe_index_in_block"
    ]].copy()
    rgb["source_present"] = True
    rgb["source_readable"] = True
    rgb["window_name"] = "pre_30s"
    rgb["blink_event_rate_per_min"] = [12.0, 13.0, 10.0, 11.0]
    rgb["body_motion_energy_median"] = [0.2, 0.3, 0.1, 0.4]
    tables = {"behavior": tables["behavior"], "rgb": rgb}

    audit = audit_quality(
        tables,
        {
            "behavior": ["b"],
            "rgb": ["blink_event_rate_per_min", "body_motion_energy_median"],
        },
    )
    feature_records = [
        {
            "feature_id": "behavior_b",
            "scientific_modality": "behavior",
            "source_namespace": "behavior",
            "predictor_column": "b",
        },
        {
            "feature_id": "blink_rate",
            "scientific_modality": "ocular",
            "source_namespace": "rgb",
            "predictor_column": "blink_event_rate_per_min",
        },
        {
            "feature_id": "body_motion",
            "scientific_modality": "movement",
            "source_namespace": "rgb",
            "predictor_column": "body_motion_energy_median",
        },
    ]
    sets, summary = build_analysis_sets(
        audit["formal_probe_identity"],
        audit["probe_feature_status"],
        {
            "behavior_plus_rgb_science": {
                "models": ["behavior_reference", "behavior_plus_rgb_science"],
                "required_features": {
                    "behavior": ["b"],
                    "ocular": ["blink_event_rate_per_min"],
                    "movement": ["body_motion_energy_median"],
                },
                "required_feature_records": feature_records,
                "required_outcomes": ["q1_nominal_4class"],
            }
        },
    )

    assert set(sets["feature_identity_mode"]) == {"explicit_per_feature"}
    assert summary["feature_identity_mode"].eq("explicit_per_feature").all()
    serialized_records = json.loads(sets["required_feature_records"].iloc[0])
    rgb_records = [record for record in serialized_records if record["source_namespace"] == "rgb"]
    assert {record["scientific_modality"] for record in rgb_records} == {"ocular", "movement"}

    frame = materialize_supervised_input(
        sets,
        audit["probe_feature_status"],
        analysis_set_id="behavior_plus_rgb_science",
        membership_type="included_complete",
        probe_metadata=tables["behavior"],
    )
    assert len(frame) == 4
    assert frame[["b", "blink_event_rate_per_min", "body_motion_energy_median"]].notna().all().all()
    assert set(frame["feature_identity_mode"]) == {"explicit_per_feature"}
    assert json.loads(frame["required_features"].iloc[0]) == {
        "behavior": ["b"],
        "movement": ["body_motion_energy_median"],
        "ocular": ["blink_event_rate_per_min"],
    }


def test_explicit_feature_records_must_match_scientific_required_feature_union() -> None:
    tables = _tables()
    audit = audit_quality(tables, {"behavior": ["b"], "nir": ["n"]})
    with pytest.raises(ValueError, match="must cover exactly"):
        build_analysis_sets(
            audit["formal_probe_identity"],
            audit["probe_feature_status"],
            {
                "bad": {
                    "models": ["B", "B+n"],
                    "required_features": {"behavior": ["b"], "ocular": ["n"]},
                    "required_feature_records": [
                        {
                            "feature_id": "behavior_b",
                            "scientific_modality": "behavior",
                            "source_namespace": "behavior",
                            "predictor_column": "b",
                        }
                    ],
                    "required_outcomes": ["q1_nominal_4class"],
                }
            },
        )
