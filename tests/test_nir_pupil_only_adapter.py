from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml

from attention_pipeline.nir_pupil_only import (
    IrisGeometryUnavailableError,
    adapt_session,
    attach_behavior_and_visual,
    refuse_pir_without_iris_geometry,
)


FIXTURES = Path(__file__).parent / "fixtures" / "nir_pupil_only"
SOURCE_DIRS = (
    "sub-031-schema-v7",
    "sub-032-schema-v6",
    "sub-033-schema-v6",
    "sub-035-historical-yolo-b8-schema-v6",
)


def _load_source(name: str) -> tuple[pd.DataFrame, dict, Path]:
    root = FIXTURES / name
    eyes = pd.read_csv(root / "eye_metrics.csv")
    manifest_path = root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return eyes, manifest, manifest_path


@pytest.mark.parametrize("source_dir", SOURCE_DIRS)
def test_v6_v7_sources_map_by_field_name_and_preserve_provenance(source_dir: str) -> None:
    eyes, manifest, manifest_path = _load_source(source_dir)
    out = adapt_session(eyes, manifest, source_manifest_path=manifest_path)

    assert out["source_schema_version"].eq(manifest["source_schema_version"]).all()
    assert out["source_path"].eq(manifest["source_path"]).all()
    assert set(out["eye_raw"]) == {"frame_left", "frame_right"}
    assert set(out["eye"]) == {"left", "right"}
    assert out.loc[out["pupil_fit_valid"].astype(str).str.lower().eq("true"), "pupil_axis_a"].notna().all()
    assert out.loc[out["pupil_fit_valid"].astype(str).str.lower().eq("true"), "pupil_equivalent_diameter"].notna().all()
    assert not any("pupil_to_iris" in name or name in {"pir", "oar"} for name in out)


def test_primary_key_uniqueness_is_fail_closed() -> None:
    eyes, manifest, manifest_path = _load_source(SOURCE_DIRS[0])
    duplicated = pd.concat([eyes, eyes.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate pupil-only primary key"):
        adapt_session(duplicated, manifest, source_manifest_path=manifest_path)


def test_time_must_be_monotonic_within_phase_segment_and_eye() -> None:
    eyes, manifest, manifest_path = _load_source(SOURCE_DIRS[1])
    # Preserve a unique frame key but put a later frame before an earlier time.
    bad = eyes.copy()
    left = bad.index[bad["eye"].eq("frame_left")].tolist()
    bad.loc[left[1], "unix_ms"] = 900
    with pytest.raises(ValueError, match="non-monotonic unix_ms"):
        adapt_session(bad, manifest, source_manifest_path=manifest_path)


def test_quality_tracks_are_independent_and_exclusive_label_is_fail_closed() -> None:
    eyes, manifest, manifest_path = _load_source(SOURCE_DIRS[0])
    out = adapt_session(eyes, manifest, source_manifest_path=manifest_path)
    missing = out[(out["frame_idx"] == 1) & out["eye"].eq("right")].iloc[0]
    clipped = out[(out["frame_idx"] == 2) & out["eye"].eq("left")].iloc[0]
    invalid = out[(out["frame_idx"] == 2) & out["eye"].eq("right")].iloc[0]

    assert bool(missing["ritnet_missing"])
    assert missing["quality_track"] == "ritnet_missing"
    assert bool(clipped["roi_clipped"])
    assert clipped["quality_track"] == "roi_clipped"
    assert bool(invalid["geometry_invalid"])
    assert bool(invalid["temporal_flagged"])
    assert invalid["quality_track"] == "geometry_invalid"


def test_source_missing_and_interpolation_are_separate_tracks() -> None:
    eyes, manifest, manifest_path = _load_source(SOURCE_DIRS[1])
    eyes.loc[0, "source_eye_status"] = "source_missing"
    source_missing = adapt_session(eyes, manifest, source_manifest_path=manifest_path)
    assert bool(source_missing.loc[0, "source_missing"])
    assert source_missing.loc[0, "quality_track"] == "source_missing"

    # Interpolation is never inferred from producer rows.  If a future caller
    # explicitly marks a derived row, it remains a separate, highest-priority track.
    from attention_pipeline.nir_pupil_only.adapter import classify_quality_tracks

    future_derived = source_missing.iloc[[1]].copy()
    future_derived["interpolation_only"] = True
    classified = classify_quality_tracks(future_derived)
    assert bool(classified.iloc[0]["interpolation_only"])
    assert classified.iloc[0]["quality_track"] == "interpolation_only"


def test_unix_ms_join_keeps_delta_current_previous_and_first_trial_null() -> None:
    eyes, manifest, manifest_path = _load_source(SOURCE_DIRS[0])
    pupil = adapt_session(eyes, manifest, source_manifest_path=manifest_path)
    behavior = pd.read_csv(FIXTURES / "behavior_trials.csv")
    visual = pd.read_csv(FIXTURES / "stimulus_visual_properties.csv")
    linked = attach_behavior_and_visual(pupil, behavior, visual)

    first = linked[linked["frame_idx"].eq(1)]
    second = linked[linked["frame_idx"].eq(2)]
    assert first["behavior_match_status"].eq("matched").all()
    assert first["behavior_match_delta_ms"].eq(100).all()
    assert first["previous_stimulus_name"].isna().all()
    assert first["previous_visual_failure_reason"].eq("block_first_trial").all()
    assert second["current_stimulus_code"].eq("banana").all()
    assert second["previous_stimulus_code"].eq("apple").all()
    assert second["current_screen_rel_lum_mean"].eq(0.38).all()
    assert second["previous_screen_rms_contrast"].eq(0.12).all()
    assert linked["visual_luminance_semantics"].str.contains("not physical cd/m²", regex=False).all()


def test_missing_visual_key_is_explicit() -> None:
    eyes, manifest, manifest_path = _load_source("sub-033-schema-v6")
    pupil = adapt_session(eyes, manifest, source_manifest_path=manifest_path)
    behavior = pd.read_csv(FIXTURES / "behavior_trials.csv")
    visual = pd.read_csv(FIXTURES / "stimulus_visual_properties.csv")
    linked = attach_behavior_and_visual(pupil, behavior, visual)
    missing = linked[linked["frame_idx"].eq(2)]
    assert missing["current_visual_match_status"].eq("missing").all()
    assert missing["current_visual_failure_reason"].eq("visual_key_not_found").all()


def test_pir_is_explicitly_refused_without_independent_iris_geometry() -> None:
    with pytest.raises(IrisGeometryUnavailableError, match="hard_iris_fraction"):
        refuse_pir_without_iris_geometry()


def test_versioned_cli_writes_rows_and_auditable_manifest(tmp_path: Path) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "nir_pupil_only_adapter.py"
    spec = importlib.util.spec_from_file_location("nir_pupil_only_cli_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    sources = []
    for source_dir in SOURCE_DIRS:
        root = (FIXTURES / source_dir).resolve()
        sources.append(
            {
                "manifest": str(root / "source_manifest.json"),
                "eye_metrics": str(root / "eye_metrics.csv"),
            }
        )
    config = {
        "adapter": {
            "version": "nir-pupil-only-adapter-v1.0.0",
            "output_schema_version": 1,
        },
        "inputs": {
            "behavior_trials_csv": str((FIXTURES / "behavior_trials.csv").resolve()),
            "visual_properties_csv": str(
                (FIXTURES / "stimulus_visual_properties.csv").resolve()
            ),
            "sources": sources,
        },
        "output": {"root": str(tmp_path / "out")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    manifest = module.run(config_path)

    assert manifest["status"] == "validation_complete"
    assert manifest["output"]["row_count"] == 16
    assert len(manifest["sources"]) == 4
    assert Path(manifest["output"]["path"]).is_file()
    assert (tmp_path / "out" / "manifest.json").is_file()
