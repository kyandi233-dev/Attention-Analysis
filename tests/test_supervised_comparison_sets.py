"""Contract tests for the comparison-specific Task-B analysis-set builder.

Covers the four properties the supervised programme depends on:

1. every planned model is covered by exactly one analysis set, and each set's
   ``required_features`` equals the predictor union of the models inside it (so a direct
   comparison really is trained on one common sample);
2. probe-key alias handling fails closed instead of silently changing probe identity;
3. a missing registered predictor column fails closed instead of silently shrinking the
   namespace;
4. a non-finite frozen value is recorded as an upstream failure and is NEVER eligible for
   the training-fold missing strategy, so no imputation is exercised and the two
   memberships coincide.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from attention_pipeline.multimodal_formal.supervised_comparison_sets import (
    SupervisedComparisonSetError,
    analysis_set_groups,
    build_supervised_comparison_sets,
    comparison_specs,
    write_supervised_comparison_sets,
)
from attention_pipeline.supervised_learning.feature_registry import (
    build_feature_comparison_plan,
    load_registered_features,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "supervised_learning_v1.yaml"

PUPIL_LEVEL = "ocular__rseg_hard__rgb_nir_qc__level_mean__2s"
PUPIL_MAD = "ocular__rseg_hard__rgb_nir_qc__variability_mad__2s"
PUPIL_SLOPE = "ocular__rseg_hard__rgb_nir_qc__linear_slope_per_sec__2s"
PUPIL_CURV = "ocular__rseg_hard__rgb_nir_qc__quadratic_curvature_per_sec2__2s"
BLINK = "ocular__blink_event_rate_per_min__pre30s"
MOTION = "body_motion_energy_median"

BEHAVIOR_COLUMNS = (
    "go_correct_rt_median_ms",
    "go_correct_rt_cv",
    "go_correct_rt_theilsen_slope_ms_per_s",
    "raw_go_omission_rate",
    "commission_rate",
)


def _registry_features() -> list[dict]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return config["feature_registry"]["features"]


def _plan():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_feature_comparison_plan(load_registered_features(config["feature_registry"]))


def _frames(n_per_group: int = 4, n_groups: int = 2):
    """Synthetic but canonically keyed producer tables."""
    rows = []
    for g in range(n_groups):
        for i in range(n_per_group):
            rows.append(
                {
                    "participant_group_id": f"P-{g}",
                    "session_id": f"S-{g}",
                    "block_id": "b1" if i < n_per_group // 2 else "b2",
                    "probe_order_in_block": (i % (n_per_group // 2)) + 1,
                    "q1_nominal_4class": [1, 2, 3, 4][i % 4],
                    "correct_go_rt_opportunities": 5,
                    "go_correct_rt_median_ms": 400.0 + i,
                    "go_correct_rt_cv": 0.20 + 0.01 * i,
                    "go_correct_rt_theilsen_slope_ms_per_s": 0.5 + i,
                    "raw_go_omission_rate": 0.01 * i,
                    "commission_rate": 0.02 * i,
                    PUPIL_LEVEL: 0.30 + 0.001 * i,
                    PUPIL_MAD: 0.010 + 0.001 * i,
                    PUPIL_SLOPE: 0.0001 * i,
                    PUPIL_CURV: 0.00001 * i,
                    BLINK: 12.0 + i,
                    MOTION: 0.5 + 0.01 * i,
                }
            )
    behavior = pd.DataFrame(rows)
    key = ["participant_group_id", "session_id", "block_id", "probe_order_in_block"]
    ocular = behavior[key + [PUPIL_LEVEL, PUPIL_MAD, PUPIL_SLOPE, PUPIL_CURV, BLINK]].copy()
    movement = behavior[key + [MOTION]].copy()
    return behavior, ocular, movement


def _build(behavior=None, ocular=None, movement=None):
    b, o, m = _frames()
    return build_supervised_comparison_sets(
        plan=_plan(),
        registry_features=_registry_features(),
        behavior_probes=behavior if behavior is not None else b,
        ocular_probes=ocular if ocular is not None else o,
        movement_probes=movement if movement is not None else m,
    )


def test_groups_cover_every_planned_model() -> None:
    plan = _plan()
    groups = analysis_set_groups(plan)
    seen: list[str] = []
    for model_ids in groups.values():
        assert model_ids, "an analysis set may not be empty"
        seen.extend(model_ids)

    # Coverage: the grouping must not drop any planned model.
    assert set(seen) == {model.model_id for model in plan.models}

    # A shared baseline deliberately appears in more than one set: it is retrained inside
    # each comparison set so that the pair is evaluated on one common sample. That
    # multiplicity is the mechanism, not a defect.
    assert seen.count("behavior_reference") == 3
    # A per-feature standalone model, by contrast, belongs to exactly one set.
    assert seen.count("standalone::ocular.pupil_level.rseg_hard.rgb_nir_qc.v1") == 1


def test_required_features_equal_the_predictor_union_of_the_models() -> None:
    plan = _plan()
    groups = analysis_set_groups(plan)
    specs = comparison_specs(plan, groups)
    models = plan.model_map()

    for set_id, spec in specs.items():
        union = {
            column
            for model_id in groups[set_id]
            for column in models[model_id].columns
        }
        declared = {
            column
            for columns in spec["required_features"].values()
            for column in columns
        }
        assert declared == union, f"{set_id}: required_features != predictor union"
        # required_feature_records must cover the same union with explicit source identity
        assert {r["predictor_column"] for r in spec["required_feature_records"]} == union


def test_every_analysis_set_has_both_memberships_reported() -> None:
    result = _build()
    sets = set(result.analysis_sets["analysis_set_id"].astype(str))
    assert sets == set(analysis_set_groups(_plan()))
    assert set(result.analysis_set_summary["membership"]) == {
        "included_complete",
        "included_missing_aware",
    }


def test_namespaces_are_derived_from_registry_not_inferred() -> None:
    result = _build()
    status = result.quality["probe_feature_status"]
    by_namespace = {
        str(namespace): sorted(set(group["feature"].astype(str)))
        for namespace, group in status.groupby("modality")
    }
    assert set(by_namespace) == {"behavior", "nir", "rgb"}
    assert by_namespace["behavior"] == sorted(BEHAVIOR_COLUMNS)
    assert by_namespace["nir"] == sorted([PUPIL_LEVEL, PUPIL_MAD, PUPIL_SLOPE, PUPIL_CURV])
    # blink is produced by the RGB device namespace and joins the rgb block alongside motion
    assert by_namespace["rgb"] == sorted([BLINK, MOTION])


def test_probe_index_alias_mismatch_fails_closed() -> None:
    behavior, _, _ = _frames()
    behavior["probe_index_in_block"] = behavior["probe_order_in_block"] + 1
    with pytest.raises(SupervisedComparisonSetError, match="aliases disagree"):
        _build(behavior=behavior)


def test_missing_registered_predictor_column_fails_closed() -> None:
    _, ocular, _ = _frames()
    ocular = ocular.drop(columns=[PUPIL_MAD])
    with pytest.raises(SupervisedComparisonSetError, match="registered predictor column missing"):
        _build(ocular=ocular)


def test_non_finite_frozen_value_is_never_imputable() -> None:
    behavior, ocular, _ = _frames()
    # Break one probe's MAD value: under the frozen rule that representation was not
    # produced, so it must not become imputable missingness.
    dead_identity = {"session_id": "S-0", "block_id": "b1", "probe_index": 1}

    def _on_dead_probe(frame: pd.DataFrame) -> pd.Series:
        """Locate the affected probe on any frame, whichever probe-index alias it uses."""
        probe_column = next(
            (
                column
                for column in ("probe_index_in_block", "probe_order_in_block")
                if column in frame.columns
            ),
            None,
        )
        if probe_column is None:
            raise AssertionError(f"frame has no probe-index column: {list(frame.columns)[:8]}")
        mask = pd.Series(True, index=frame.index)
        for column, value in dead_identity.items():
            resolved = probe_column if column == "probe_index" else column
            mask &= frame[resolved].astype(str).eq(str(value))
        return mask

    dead = _on_dead_probe(behavior)
    ocular = ocular.copy()
    ocular.loc[dead.to_numpy(), PUPIL_MAD] = np.nan

    result = _build(behavior=behavior, ocular=ocular)
    assert result.manifest["imputation_exercised"] is False
    assert result.manifest["non_finite_frozen_values_made_non_imputable"] == 1

    status = result.quality["probe_feature_status"]
    row = status[status["feature"].eq(PUPIL_MAD) & _on_dead_probe(status)]
    assert len(row) == 1
    assert bool(row.iloc[0]["feature_computable"]) is False
    assert bool(row.iloc[0]["eligible_for_missing_strategy"]) is False
    assert row.iloc[0]["missing_kind"] == "native_qc_invalid"

    # And the affected probe is excluded from a set that requires that feature.
    membership = "included_missing_aware"
    excluded = result.analysis_sets[
        result.analysis_sets["analysis_set_id"].astype(str).eq("AS.full")
        & _on_dead_probe(result.analysis_sets)
    ]
    assert len(excluded) == 1
    assert bool(excluded.iloc[0][membership]) is False


def test_memberships_coincide_and_no_imputation_is_claimed() -> None:
    result = _build()
    summary = result.analysis_set_summary
    complete = summary[summary["membership"].eq("included_complete")].set_index("analysis_set_id")
    aware = summary[summary["membership"].eq("included_missing_aware")].set_index("analysis_set_id")
    assert complete.index.equals(aware.index)
    assert (complete["probe_n"] == aware["probe_n"]).all()
    assert result.manifest["imputation_exercised"] is False


def test_directly_compared_models_share_one_analysis_set() -> None:
    """The behaviour reference and its conditioned increments must share a set."""
    plan = _plan()
    groups = analysis_set_groups(plan)
    for baseline, added, _feature_id in plan.behavior_increment_pairs:
        shared = [set_id for set_id, ids in groups.items() if baseline in ids and added in ids]
        assert shared, f"no analysis set contains both {baseline} and {added}"
    for baseline, added, _modality in plan.modality_increment_pairs:
        shared = [set_id for set_id, ids in groups.items() if baseline in ids and added in ids]
        assert shared, f"no analysis set contains both {baseline} and {added}"
    for reduced, full, _modality in plan.full_leave_one_modality_out_pairs:
        shared = [set_id for set_id, ids in groups.items() if reduced in ids and full in ids]
        assert shared


def test_writer_refuses_to_overwrite(tmp_path) -> None:
    result = _build()
    out = tmp_path / "sets"
    write_supervised_comparison_sets(out, result)
    assert (out / "analysis_sets.csv").is_file()
    assert (out / "probe_feature_status.csv").is_file()
    assert (out / "comparison_sets_manifest.json").is_file()
    with pytest.raises(FileExistsError):
        write_supervised_comparison_sets(out, result)
