"""Small synthetic acceptance cases for Issue #30; no upstream/model execution."""
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from attention_pipeline.multimodal_formal.quality_admission import audit_quality, run_quality_admission, validate_omission_inputs


def tables(n=100):
    b = pd.DataFrame({"session_id": ["s"] * n, "participant_group_id": ["p"] * n,
                      "block_id": ["b1"] * n, "probe_index_in_block": np.arange(1, n + 1),
                      "x": np.arange(n, dtype=float), "window_name": "pre_30s"})
    r = b.copy()
    r["source_present"] = True
    return {"behavior": b, "rgb": r}


def feature(result, name="x", modality="rgb"):
    f = result["feature_admission"]
    return f[(f.feature == name) & (f.modality == modality)].iloc[0]


def test_missing_modality_and_nan_have_distinct_denominators():
    t = tables()
    t["rgb"] = t["rgb"].iloc[:90].copy()
    t["rgb"].loc[:9, "x"] = np.nan
    r = audit_quality(t, {"behavior": ["x"], "rgb": ["x"]})
    f = feature(r)
    assert f.formal_probe_n == 100 and f.opportunity_n == 90 and f.computable_n == 80
    assert f.modality_availability_rate == .90
    assert f.feature_computable_coverage == 80/90 and f.overall_effective_rate == .80
    assert feature(r, modality="behavior").computable_n == 100
    t["rgb"] = pd.DataFrame()
    f = feature(audit_quality(t, {"behavior": ["x"], "rgb": ["x"]}))
    assert f.opportunity_n == 0 and not f.main_candidate and pd.isna(f.feature_computable_coverage)


@pytest.mark.parametrize("n,admitted", [(79, False), (80, True)])
def test_exact_coverage_boundary(n, admitted):
    t = tables()
    t["rgb"].loc[n:, "x"] = np.nan
    f = feature(audit_quality(t, {"behavior": ["x"], "rgb": ["x"]}))
    assert bool(f.main_candidate) == admitted


def test_distribution_and_redundancy_are_only_review_flags():
    t = tables()
    r = t["rgb"]
    r["constant"], r["binary"], r["empty"] = 1., np.arange(100) % 2, np.nan
    r["floor"] = np.arange(100) / 10000
    r["ceiling"] = 1 - r["floor"]
    r["other_science"] = r["floor"]
    rules = {"proportion_features": ["floor", "ceiling"], "measurement_families": {"same": ["floor", "ceiling"]}}
    out = audit_quality(t, {"behavior": ["x"], "rgb": ["constant", "binary", "empty", "absent", "floor", "ceiling", "other_science"]}, rules)
    assert feature(out, "constant").zero_variance
    assert not feature(out, "binary").main_candidate
    assert "column_missing" in feature(out, "absent").admission_reason
    assert "all_missing" in feature(out, "empty").admission_reason
    assert feature(out, "floor").severe_floor and feature(out, "floor").main_candidate
    assert feature(out, "ceiling").severe_ceiling and feature(out, "ceiling").main_candidate
    assert len(out["redundancy_review"]) == 1


def test_native_qc_nonfinite_alignment_and_placeholder_source():
    t = tables(10)
    r = t["rgb"]
    r["native_qc_valid"] = True
    r.loc[0, "source_present"] = False
    r.loc[1, "native_qc_valid"] = False
    r.loc[2, "x"] = np.inf
    r.loc[3, "window_name"] = "post_30s"
    out = audit_quality(t, {"behavior": ["x"], "rgb": ["x"]})
    f = feature(out)
    assert f.opportunity_n == 8 and f.computable_n == 6
    reasons = set(out["probe_quality"].inclusion_reason)
    assert {"native_qc_invalid", "nonfinite_value", "alignment_invalid", "source_missing_or_unverified"} <= reasons
    t["rgb"] = r.drop(columns=["source_present"])
    assert feature(audit_quality(t, {"rgb": ["x"]})).opportunity_n == 0


@pytest.mark.parametrize("fault", ["duplicate", "null", "identity", "illegal_key"])
def test_invalid_keys_fail_closed(fault):
    t = tables()
    if fault == "duplicate": t["rgb"] = pd.concat([t["rgb"], t["rgb"].iloc[:1]])
    if fault == "null": t["rgb"].loc[0, "probe_index_in_block"] = np.nan
    if fault == "identity": t["rgb"]["participant_group_id"] = "other"
    if fault == "illegal_key": t["rgb"].loc[0, "block_id"] = "b3"
    with pytest.raises(ValueError): audit_quality(t, {"behavior": ["x"], "rgb": ["x"]})


def test_mmwave_inherited_window_and_explicit_bad_boundary():
    t = tables(10)
    m = t.pop("rgb")
    t["mmwave"] = m
    m["mmwave_observed"], m["mmwave_loadable"] = True, True
    m["window_start_unix_ms"], m["window_end_unix_ms"], m["probe_onset_unix_ms"] = 10000, 40000, 40000
    m["window_effective_start_unix_ms"] = 10000
    m["block_start_unix_ms"], m["block_end_unix_ms"] = np.nan, np.nan
    m["window_boundary_source"] = "probe_primary_30s"
    m.loc[0, "block_start_unix_ms"], m.loc[0, "block_end_unix_ms"] = 20000, 80000
    m.loc[1, "mmwave_loadable"] = False
    out = audit_quality(t, {"behavior": ["x"], "mmwave": ["x"]})
    assert feature(out, modality="mmwave").opportunity_n == 8
    assert out["analysis_sets"].included.sum() == 8


def test_omission_configuration_and_model_input_guard():
    config = yaml.safe_load(Path("configs/multimodal_fusion.yaml").read_text(encoding="utf-8"))
    b = config["feature_blocks"]["behavior"]
    assert {"clean_go_omission_rate", "timing_ambiguous_go_omission_rate"} <= set(b)
    assert not {"omission_rate", "raw_go_omission_rate"} & set(b)
    validate_omission_inputs(b)
    for invalid in (["omission_rate", "raw_go_omission_rate"], b + ["raw_go_omission_rate"]):
        with pytest.raises(ValueError): validate_omission_inputs(invalid)
    from attention_pipeline.multimodal_formal.runner import _feature_columns
    with pytest.raises(ValueError): _feature_columns({"behavior": b + ["raw_go_omission_rate"]}, ("behavior",))


def test_independent_runner_no_models_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "attention_pipeline.multimodal_formal.runner", None)
    monkeypatch.setitem(sys.modules, "attention_pipeline.multimodal_formal.models", None)
    root = tmp_path / "input"
    target = root / "Behavior/formal_v3/probe_primary_30s.csv"
    target.parent.mkdir(parents=True)
    b = tables(10)["behavior"].rename(columns={"probe_index_in_block": "probe_order_in_block"})
    b.to_csv(target, index=False)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"paths": {"data_root": str(root), "output_root": str(tmp_path / "output")},
                                    "feature_blocks": {"behavior": ["x"], "rgb": ["x"]}, "quality_admission": {},
                                    "combinations": {"B": ["behavior"], "BR": ["behavior", "rgb"]}}))
    before = target.read_bytes()
    summary = run_quality_admission(config, run_id="synthetic")
    assert summary["models_trained"] is False and target.read_bytes() == before
    assert len(summary["input_problems"]) > 0
    assert (tmp_path / "output/synthetic/quality_admission/summary.json").is_file()
    with pytest.raises(FileExistsError): run_quality_admission(config, run_id="synthetic")
    monkeypatch.setattr(sys, "argv", ["multimodal_fusion_analysis.py", "--config", str(config), "--quality-only", "--run-id", "cli"])
    with pytest.raises(SystemExit) as exited:
        runpy.run_path("scripts/multimodal_fusion_analysis.py", run_name="__main__")
    assert exited.value.code == 0
