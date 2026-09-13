from __future__ import annotations

import pandas as pd

from attention_pipeline.multimodal_formal.analysis_sets import build_analysis_sets
from attention_pipeline.multimodal_formal.quality_admission import audit_quality
from attention_pipeline.multimodal_formal.supervised_input import materialize_supervised_input
from attention_pipeline.supervised_learning.feature_registry import (
    RegisteredFeature,
    build_feature_comparison_plan,
)


def _registry() -> list[RegisteredFeature]:
    return [
        RegisteredFeature(
            feature_id="behavior_b",
            scientific_feature_id="behavior_b",
            columns=("b",),
            role="behavior",
            modality="behavior",
            raw_source="synthetic behavior",
            source_namespace="behavior",
            required_devices=(),
            behavior_reference_eligible=True,
            modality_model_eligible=True,
        ),
        RegisteredFeature(
            feature_id="pupil_level",
            scientific_feature_id="pupil_level",
            columns=("pupil_level",),
            role="sensor",
            modality="ocular",
            feature_type="pupil",
            raw_source="NIR pupil with RGB-assisted cleaning",
            source_namespace="nir",
            required_devices=("nir", "rgb"),
            preprocessing_dependencies=("RGB blink mask",),
            behavior_increment_eligible=True,
            modality_model_eligible=True,
        ),
        RegisteredFeature(
            feature_id="blink_rate",
            scientific_feature_id="blink_rate",
            columns=("blink_rate",),
            role="sensor",
            modality="ocular",
            feature_type="blink",
            raw_source="RGB blink producer",
            source_namespace="rgb",
            required_devices=("rgb",),
            behavior_increment_eligible=True,
            modality_model_eligible=True,
        ),
        RegisteredFeature(
            feature_id="body_motion",
            scientific_feature_id="body_motion",
            columns=("body_motion",),
            role="sensor",
            modality="movement",
            feature_type="body_motion",
            raw_source="RGB motion producer",
            source_namespace="rgb",
            required_devices=("rgb",),
            modality_model_eligible=True,
        ),
    ]


def _tables() -> dict[str, pd.DataFrame]:
    behavior = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s2", "s2"],
            "participant_group_id": ["p1", "p1", "p2", "p2"],
            "block_id": ["b1"] * 4,
            "probe_index_in_block": [1, 2, 1, 2],
            "probe_event_id": ["s1|b1|1", "s1|b1|2", "s2|b1|1", "s2|b1|2"],
            "q1_nominal_4class": [1, 2, 1, 3],
            "window_name": ["pre_30s"] * 4,
            "b": [0.1, 0.2, 0.3, 0.4],
        }
    )
    keys = ["session_id", "participant_group_id", "block_id", "probe_index_in_block"]

    nir = behavior[keys].copy()
    nir["source_present"] = True
    nir["source_readable"] = True
    nir["window_name"] = "pre_30s"
    nir["pupil_level"] = [3.0, 3.1, 3.2, 3.3]

    rgb = behavior[keys].copy()
    rgb["source_present"] = True
    rgb["source_readable"] = True
    rgb["window_name"] = "pre_30s"
    rgb["blink_rate"] = [10.0, 11.0, 12.0, 13.0]
    # Movement is intentionally missing for one probe. It must not shrink the
    # Behavior-vs-Ocular comparison even though it shares the RGB producer.
    rgb["body_motion"] = [1.0, None, 1.2, 1.3]
    return {"behavior": behavior, "nir": nir, "rgb": rgb}


def test_registry_generated_task_b_spec_preserves_science_and_source_identity() -> None:
    plan = build_feature_comparison_plan(_registry())
    spec = plan.task_b_comparison_spec(
        ["behavior_reference", "behavior_plus_modality::ocular"],
        required_outcomes=["q1_nominal_4class"],
    )

    assert spec["required_features"] == {
        "behavior": ["b"],
        "ocular": ["pupil_level", "blink_rate"],
    }
    records = {row["predictor_column"]: row for row in spec["required_feature_records"]}
    assert records["pupil_level"]["source_namespace"] == "nir"
    assert records["pupil_level"]["scientific_modality"] == "ocular"
    assert records["blink_rate"]["source_namespace"] == "rgb"
    assert "body_motion" not in records

    tables = _tables()
    audited = audit_quality(
        tables,
        {
            "behavior": ["b"],
            "nir": ["pupil_level"],
            "rgb": ["blink_rate", "body_motion"],
        },
    )
    analysis_sets, _ = build_analysis_sets(
        audited["formal_probe_identity"],
        audited["probe_feature_status"],
        {"behavior_vs_ocular": spec},
    )

    scoped = analysis_sets[analysis_sets["analysis_set_id"].eq("behavior_vs_ocular")]
    assert scoped["included_complete"].sum() == 4
    assert set(scoped["feature_identity_mode"]) == {"explicit_per_feature"}

    materialized = materialize_supervised_input(
        analysis_sets,
        audited["probe_feature_status"],
        analysis_set_id="behavior_vs_ocular",
        membership_type="included_complete",
        probe_metadata=tables["behavior"],
    )
    assert list(materialized[["b", "pupil_level", "blink_rate"]].columns) == [
        "b",
        "pupil_level",
        "blink_rate",
    ]
    assert len(materialized) == 4
