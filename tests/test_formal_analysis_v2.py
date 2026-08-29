from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from attention_pipeline.config import load_config
from attention_pipeline.formal_analysis.cohort import (
    load_cohort_manifest,
    summarize_cohort,
    validate_participant_disjoint_folds,
)
from attention_pipeline.formal_analysis.merge import merge_modalities, validate_merge_ready
from attention_pipeline.formal_analysis.nir_adapter import adapt_nir_frame_table
from attention_pipeline.path_registry import load_path_registry


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def test_path_registry_is_separate_from_science_config(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setenv("FORMAL_RAW_TEST", str(raw))
    registry_path = tmp_path / "paths.local.yaml"
    _write_yaml(registry_path, {"version": 1, "paths": {"formal_raw_roots": ["${FORMAL_RAW_TEST}"], "out": "derived"}})
    registry = load_path_registry(registry_path)
    assert registry.path_values("formal_raw_roots") == [raw.resolve()]
    assert registry.path_value("out") == (tmp_path / "derived").resolve()

    science_path = tmp_path / "science.yaml"
    _write_yaml(science_path, {"paths": {"output_root": {"path_key": "out"}}})
    config = load_config(science_path, paths_config=registry_path)
    assert config.path_value("output_root") == (tmp_path / "derived").resolve()


def test_cohort_summary_and_fold_leakage_gate(tmp_path):
    cohort_path = tmp_path / "cohort.csv"
    pd.DataFrame([
        {"session_id": "sub-031", "include": True, "repeat_participant_id": "g1"},
        {"session_id": "sub-032", "include": True, "repeat_participant_id": "g1"},
        {"session_id": "sub-033", "include": True, "repeat_participant_id": "g2"},
    ]).to_csv(cohort_path, index=False)
    cohort = load_cohort_manifest(cohort_path)
    summary = summarize_cohort(cohort)
    assert summary.sessions == 3
    assert summary.groups == 2
    assert summary.repeated_groups == 1
    assert summary.repeated_sessions == 2

    ok = pd.DataFrame({"repeat_participant_id": ["g1", "g1", "g2"], "fold": [0, 0, 1]})
    validate_participant_disjoint_folds(ok)
    leaking = ok.copy()
    leaking.loc[1, "fold"] = 1
    with pytest.raises(ValueError, match="多个折"):
        validate_participant_disjoint_folds(leaking)


def _nir_base() -> pd.DataFrame:
    return pd.DataFrame({
        "subject": ["sub-031", "sub-031"],
        "phase": ["block1", "block1"],
        "phase_segment": ["block1", "block1"],
        "frame_idx": [1, 1],
        "eye": ["frame_left", "frame_right"],
        "unix_ms": [1000, 1000],
        "pupil_found": [1, 1],
        "pupil_fit_valid": [1, 1],
        "pupil_equivalent_diameter": [30.0, 31.0],
        "pupil_geom_mean_diameter": [29.0, 30.0],
        "hard_pupil_fraction": [0.1, 0.11],
        "soft_pupil_fraction": [0.12, 0.13],
        "hard_iris_fraction": [0.2, 0.21],
        "soft_iris_fraction": [0.22, 0.23],
        "temporal_anomaly": [False, True],
    })


def test_nir_adapter_is_pupil_only_and_preserves_eye_raw():
    frame = _nir_base()
    frame["fullclass_ocular_aperture_ratio_median"] = [0.30, 0.31]
    out = adapt_nir_frame_table(frame, schema_version=7)
    assert out["session_id"].unique().tolist() == ["sub-031"]
    assert out["eye_raw"].tolist() == ["frame_left", "frame_right"]
    assert out["eye"].tolist() == ["left", "right"]
    assert "pupil_equivalent_diameter" in out
    assert "hard_iris_fraction" in out
    assert out["ocular_aperture_available"].all()
    assert set(out["ocular_aperture_role"]) == {"nir_eye_opening_candidate_qc"}
    assert "binocular_PIR" not in out.columns


def test_nir_adapter_rejects_historical_pir_and_fake_time():
    frame = _nir_base()
    frame["fullclass_pupil_to_iris_diameter_ratio"] = [0.2, 0.2]
    with pytest.raises(ValueError, match="拒绝 PIR"):
        adapt_nir_frame_table(frame)
    no_time = _nir_base().drop(columns=["unix_ms"])
    with pytest.raises(ValueError, match="真实时间键"):
        adapt_nir_frame_table(no_time)


def test_merge_contract_is_key_strict_and_modality_prefixed():
    keys = {
        "repeat_participant_id": ["g1", "g2"],
        "session_id": ["sub-031", "sub-033"],
        "block_id": [1, 1],
    }
    behavior = pd.DataFrame({**keys, "rt_cv": [0.1, 0.2]})
    nir = pd.DataFrame({**keys, "pupil": [30.0, 31.0]})
    validate_merge_ready(behavior, unit="block", modality="behavior")
    merged, audit = merge_modalities({"behavior": behavior, "nir": nir}, unit="block", how="inner")
    assert len(merged) == 2
    assert "behavior__rt_cv" in merged
    assert "nir__pupil" in merged
    assert set(audit["modality"]) == {"behavior", "nir"}

    duplicated = pd.concat([nir, nir.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="主键重复"):
        validate_merge_ready(duplicated, unit="block", modality="nir")
