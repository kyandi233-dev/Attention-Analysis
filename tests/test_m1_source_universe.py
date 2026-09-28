"""Regression coverage for FW-1313-INPUT-01; synthetic data are not study evidence."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/build_m1_cardiopulmonary_taskb_source.py"
SPEC = spec_from_file_location("m1_source_universe_script", PATH)
MOD = module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


@pytest.fixture
def frames():
    m1 = pd.DataFrame({
        "session_id": ["s1", "s1", "s2", "s2"],
        "block_id": ["block-1", "block-1", "block-2", "block-2"],
        "legacy_block_num": [1, 1, 2, 2], "probe_index_in_block": [1, 2, 1, 2],
        "repeat_participant_id": ["r1", "r1", "r2", "r2"],
        MOD.MMWAVE_COLUMNS[0]: [70.5, 71.5, 72.5, np.nan],
        MOD.MMWAVE_COLUMNS[1]: [14., 15., 16., np.nan],
        "mmwave_state": ["OBSERVED"] * 3 + ["STRUCTURAL_MISSING"],
        "mmwave_missing_reason": [None, None, None, "source unavailable"],
        "window_effective_start_unix_ms": [100, 200, 300, 400],
        "window_end_unix_ms": [150, 250, 350, 450],
        "probe_onset_unix_ms": [150, 250, 350, 450],
        "mmwave_source_commit": ["producer-frozen"] * 4,
    })
    universe = m1[MOD.JOIN_KEY].copy()
    universe["block_id"] = ["b1", "b1", "b2", "b2"]
    universe["participant_group_id"] = ["p1", "p1", "p2", "p2"]
    universe["q1_nominal_4class"] = [1, 2, 3, np.nan]
    bridge = m1[["session_id", "repeat_participant_id"]].drop_duplicates()
    bridge["participant_group_id"] = ["p1", "p2"]
    return m1, bridge, universe


def test_full_source_does_not_inherit_behavior_filter(frames):
    m1, bridge, universe = frames
    legacy = universe.iloc[[0, 3]].copy()
    original = m1.copy(deep=True)
    out, report = MOD.build_source(m1, bridge, universe, legacy)
    assert len(out) == len(universe) == 4
    assert report["m1_hr_finite"] == report["m1_br_finite"] == 3
    assert report["legacy_base_audit"] == {
        "base_rows": 2, "would_exclude_rows": 2,
        "would_exclude_finite_hr": 2, "would_exclude_finite_br": 2,
    }
    pd.testing.assert_frame_equal(m1, original)
    for column in MOD.MMWAVE_COLUMNS + ["mmwave_state", "mmwave_missing_reason", "probe_onset_unix_ms", "mmwave_source_commit"]:
        pd.testing.assert_series_equal(out[column], m1[column])
    assert report["behavior_availability_used_for_filtering"] is False


def test_missing_q1_and_missing_measurements_not_filtered_at_source(frames):
    out, _ = MOD.build_source(*frames)
    assert len(out) == 4
    assert out.iloc[-1].mmwave_state == "STRUCTURAL_MISSING"
    assert pd.isna(out.iloc[-1][MOD.MMWAVE_COLUMNS[0]])


def test_independent_sensor_membership_keeps_two_extra_rows(frames):
    m1, bridge, universe = frames
    out, _ = MOD.build_source(m1, bridge, universe, universe.iloc[[0, 3]])
    sensor = out[np.isfinite(out[MOD.MMWAVE_COLUMNS[0]])]
    paired_behavior = sensor.merge(universe.iloc[[0, 3]][MOD.JOIN_KEY], on=MOD.JOIN_KEY)
    assert len(sensor) == 3 and len(paired_behavior) == 1


@pytest.mark.parametrize("which", [0, 2])
def test_duplicate_probe_rejected(frames, which):
    values = list(frames)
    values[which] = pd.concat([values[which], values[which].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        MOD.build_source(*values)


@pytest.mark.parametrize("column", MOD.JOIN_KEY)
def test_missing_key_rejected(frames, column):
    m1, bridge, universe = frames
    universe.loc[0, column] = None
    with pytest.raises(ValueError, match="null/blank"):
        MOD.build_source(m1, bridge, universe)


def test_filtered_universe_rejected(frames):
    m1, bridge, universe = frames
    with pytest.raises(ValueError, match="missing_from_universe=1"):
        MOD.build_source(m1, bridge, universe.iloc[:-1])


def test_extra_universe_probe_rejected(frames):
    m1, bridge, universe = frames
    row = universe.iloc[[0]].copy()
    row["probe_index_in_block"] = 3
    with pytest.raises(ValueError, match="extra_in_universe=1"):
        MOD.build_source(m1, bridge, pd.concat([universe, row]))


def test_session_identity_mismatch_stops_write(frames):
    m1, bridge, universe = frames
    bridge["session_id"] = ["s2", "s1"]
    with pytest.raises(ValueError, match="session identity"):
        MOD.build_source(m1, bridge, universe)


def test_one_producer_to_two_governed_groups_rejected(frames):
    m1, bridge, universe = frames
    extra = bridge.iloc[[0]].copy()
    extra["participant_group_id"] = "p3"
    with pytest.raises(ValueError, match="multiple governed groups"):
        MOD.build_source(m1, pd.concat([bridge, extra]), universe)


def test_multiple_producer_ids_in_session_rejected(frames):
    m1, bridge, universe = frames
    m1.loc[0, "repeat_participant_id"] = "r2"
    with pytest.raises(ValueError, match="multiple producer"):
        MOD.build_source(m1, bridge, universe)


def test_many_producer_ids_to_one_governed_participant_allowed(frames):
    m1, bridge, universe = frames
    bridge["participant_group_id"] = "p1"
    universe["participant_group_id"] = "p1"
    _, report = MOD.build_source(m1, bridge, universe)
    assert report["m1_producer_participants"] == 2 and report["participant_group_n"] == 1


def test_governed_identity_mismatch_rejected(frames):
    m1, bridge, universe = frames
    universe.loc[universe.session_id.eq("s1"), "participant_group_id"] = "p3"
    with pytest.raises(ValueError, match="governed participant identity"):
        MOD.build_source(m1, bridge, universe)


def test_legacy_block_mismatch_rejected(frames):
    m1, bridge, universe = frames
    m1.loc[0, "legacy_block_num"] = 2
    with pytest.raises(ValueError, match="legacy_block_num"):
        MOD.build_source(m1, bridge, universe)


def test_probe_alias_supported_and_conflict_rejected(frames):
    m1, bridge, universe = frames
    aliased = universe.rename(columns={"probe_index_in_block": "probe_order_in_block"})
    out, _ = MOD.build_source(m1, bridge, aliased)
    assert len(out) == 4
    universe["probe_order_in_block"] = 9
    with pytest.raises(ValueError, match="aliases disagree"):
        MOD.build_source(m1, bridge, universe)


def test_unmatched_legacy_key_rejected(frames):
    m1, bridge, universe = frames
    legacy = universe.copy()
    legacy.loc[0, "probe_index_in_block"] = 5
    with pytest.raises(ValueError, match="legacy base contains"):
        MOD.build_source(m1, bridge, universe, legacy)


def test_inf_not_counted_as_finite(frames):
    m1, bridge, universe = frames
    m1.loc[0, MOD.MMWAVE_COLUMNS[0]] = np.inf
    _, report = MOD.build_source(m1, bridge, universe)
    assert report["m1_hr_finite"] == 2


def _cli_inputs(tmp_path, frames):
    m1, bridge, universe = frames
    root = tmp_path / "producer"
    root.mkdir()
    m1.iloc[:2].to_csv(root / "mmwave_probe_merge_ready.csv", index=False)
    m1.iloc[2:].to_csv(root / "mmwave_probe_merge_ready_E.csv", index=False)
    bridge.to_csv(tmp_path / "bridge.csv", index=False)
    universe.to_csv(tmp_path / "universe.csv", index=False)
    return ["--m1-dir", str(root), "--identity-bridge", str(tmp_path / "bridge.csv"),
            "--probe-universe", str(tmp_path / "universe.csv"), "--output", str(tmp_path / "out")]


def test_cli_records_hash_and_refuses_overwrite(tmp_path, frames):
    args = _cli_inputs(tmp_path, frames)
    assert MOD.main(args) == 0
    output = tmp_path / "out/m1_cardiopulmonary_taskb_source.csv"
    report = json.loads((tmp_path / "out/m1_source_report.json").read_text())
    assert report["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert len(report["inputs"]) == 4
    assert report["output_rows"] == 4
    with pytest.raises(FileExistsError):
        MOD.main(args)


def test_bad_universe_fails_before_output_created(tmp_path, frames):
    args = _cli_inputs(tmp_path, frames)
    frames[2].iloc[:-1].to_csv(tmp_path / "universe.csv", index=False)
    with pytest.raises(ValueError, match="probe universe mismatch"):
        MOD.main(args)
    assert not (tmp_path / "out").exists()


def test_old_cli_cannot_silently_reproduce_filter(tmp_path, frames):
    args = _cli_inputs(tmp_path, frames)
    idx = args.index("--probe-universe")
    args[idx] = "--base-table"
    with pytest.raises(SystemExit) as error:
        MOD.main(args)
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()
