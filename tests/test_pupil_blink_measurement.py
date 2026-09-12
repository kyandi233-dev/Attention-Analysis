import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from attention_pipeline.nir_formal_analysis import pupil_blink_measurement as pbm
from attention_pipeline.nir_formal_analysis import pupil_blink_measurement_runner as pbr


def eye_frame():
    return pd.DataFrame({
        "session_id": ["s1"] * 6,
        "phase": ["block1"] * 6,
        "frame_idx": [1, 1, 2, 2, 3, 3],
        "eye": ["left", "right"] * 3,
        "unix_ms": [1000, 1000, 2000, 2000, 3000, 3000],
        "source_observed": [True] * 6,
        "ritnet_missing": [False] * 6,
        "interpolation_only": [False] * 6,
        "temporal_flagged": [False] * 6,
        "pupil_geom_mean_diameter": [10, 14, np.nan, 20, 30, 34],
        "hard_pupil_fraction": [0.2, 0.3, 0.4, 0.5, 0.0, 0.6],
        "hard_iris_fraction": [0.3, 0.2, 0.1, 0.5, 0.0, 0.4],
        "soft_pupil_fraction": [0.2, 0.3, 0.4, 0.5, 0.0, 0.6],
        "soft_iris_fraction": [0.3, 0.2, 0.1, 0.5, 0.0, 0.4],
    })


def test_rseg_independent_of_geometry_validity():
    out = pbm.derive_eye_measurements(eye_frame())
    row = out.iloc[2]
    assert not row["pupil_geom_mean_diameter__audit_valid"]
    assert row["seg_pupil_fraction_within_pupil_iris_hard__audit_valid"]
    assert row["seg_pupil_fraction_within_pupil_iris_hard"] == pytest.approx(0.8)


def test_rseg_denominator_zero_is_invalid():
    out = pbm.derive_eye_measurements(eye_frame())
    row = out.iloc[4]
    assert not row["seg_pupil_fraction_within_pupil_iris_hard__audit_valid"]
    assert "denominator_nonpositive" in row["seg_pupil_fraction_within_pupil_iris_hard__audit_invalid_reason"]


def test_binocular_fusion_is_signal_specific():
    out = pbm.build_binocular_measurement_timepoints(eye_frame())
    assert out.loc[0, "pupil_geom_mean_diameter"] == pytest.approx(12)
    assert out.loc[1, "pupil_geom_mean_diameter"] == pytest.approx(20)
    assert out.loc[1, "seg_pupil_fraction_within_pupil_iris_hard"] == pytest.approx(0.65)
    assert out.loc[1, "seg_pupil_fraction_within_pupil_iris_hard__source_mode"] == "binocular"


def test_blink_mask_buffers_and_unions_overlap():
    tp = pd.DataFrame({"session_id": ["s1"] * 7, "unix_ms": [900, 1000, 1050, 1100, 1150, 1200, 1300]})
    ev = pd.DataFrame({"session_id": ["s1", "s1"], "start_unix_ms": [1000, 1100], "end_unix_ms": [1050, 1150]})
    out = pbm.add_rgb_blink_mask(tp, ev, pre_buffer_ms=50, post_buffer_ms=50)
    assert out["rgb_blink_mask"].tolist() == [False, True, True, True, True, True, False]
    assert out.loc[out["unix_ms"].eq(1100), "rgb_blink_event_coverage_count"].iloc[0] == 2


def test_blink_mask_never_crosses_session():
    tp = pd.DataFrame({"session_id": ["s1", "s2"], "unix_ms": [1000, 1000]})
    ev = pd.DataFrame({"session_id": ["s1"], "start_unix_ms": [900], "end_unix_ms": [1100]})
    out = pbm.add_rgb_blink_mask(tp, ev, pre_buffer_ms=0, post_buffer_ms=0)
    assert out["rgb_blink_mask"].tolist() == [True, False]


def test_fixed_bins_no_interpolation_and_linear_slope():
    tp = pd.DataFrame({
        "session_id": ["s1"] * 4,
        "block": [1] * 4,
        "unix_ms": [1000, 3000, 7000, 9000],
        "pupil_geom_mean_diameter": [1, 3, 7, 9],
        "seg_pupil_fraction_within_pupil_iris_hard": [0.1, 0.3, 0.7, 0.9],
        "rgb_blink_mask": [False] * 4,
    })
    w = pbm.select_probe_window(tp, session_id="s1", block_num=1, probe_onset_ms=11000, window_sec=10)
    bins = pbm.fixed_probe_bins(w, signal="pupil_geom_mean_diameter", window_sec=10, bin_width_sec=2, cleaning_track="original_nir")
    assert bins["bin_value_median"].isna().sum() == 1
    fit = pbm.fit_fixed_bin_dynamics(bins, window_sec=10)
    assert fit["linear_status"] == "computable"
    assert fit["linear_slope_per_sec"] == pytest.approx(1.0, abs=0.05)


def test_quadratic_curvature_known_sequence():
    centers = np.array([-9, -7, -5, -3, -1], float)
    midpoint = -5
    y = 2 * (centers - midpoint) ** 2 + 3 * (centers - midpoint) + 10
    bins = pd.DataFrame({"bin_center_sec": centers, "bin_value_median": y})
    fit = pbm.fit_fixed_bin_dynamics(bins, window_sec=10)
    assert fit["quadratic_status"] == "computable"
    assert fit["quadratic_curvature_per_sec2"] == pytest.approx(2.0)


def test_sync_residuals_are_session_local():
    tp = pd.DataFrame({"session_id": ["s1"] * 4, "unix_ms": [0, 100, 200, 300]})
    ev = pd.DataFrame({"session_id": ["s1"], "start_unix_ms": [90], "end_unix_ms": [210]})
    out = pbm.audit_rgb_nir_sync(tp, ev).iloc[0]
    assert out["nearest_residual_abs_median_ms"] == pytest.approx(10)


def test_original_and_nir_qc_tracks_are_distinct():
    frame = eye_frame()
    frame.loc[0, "temporal_flagged"] = True
    tp = pbm.build_binocular_measurement_timepoints(frame)
    assert tp.loc[0, "pupil_geom_mean_diameter__raw"] == pytest.approx(12.0)
    assert tp.loc[0, "pupil_geom_mean_diameter"] == pytest.approx(14.0)


def test_soft_ratio_is_explicit_sensitivity_role():
    out = pbm.derive_eye_measurements(eye_frame())
    assert pbm.SIGNAL_ROLES[pbm.RSEG_SOFT_SIGNAL].startswith("sensitivity_candidate")
    assert out.loc[0, pbm.RSEG_SOFT_SIGNAL] == pytest.approx(0.4)


def test_probe_window_never_crosses_block():
    tp = pd.DataFrame({
        "session_id": ["s1", "s1", "s1"],
        "block": [1, 2, 1],
        "unix_ms": [9000, 9500, 9900],
        "pupil_geom_mean_diameter": [1, 999, 2],
        "pupil_geom_mean_diameter__raw": [1, 999, 2],
        "seg_pupil_fraction_within_pupil_iris_hard": [0.1, 0.9, 0.2],
        "seg_pupil_fraction_within_pupil_iris_hard__raw": [0.1, 0.9, 0.2],
    })
    w = pbm.select_probe_window(tp, session_id="s1", block_num=1, probe_onset_ms=10000, window_sec=2)
    assert w["pupil_geom_mean_diameter"].tolist() == [1, 2]


def test_recovery_bins_preserve_raw_and_qc_availability_separately():
    tp = pd.DataFrame({
        "session_id": ["s1"] * 3,
        "unix_ms": [900, 1000, 1100],
        "pupil_geom_mean_diameter__raw": [10, 11, 12],
        "pupil_geom_mean_diameter": [10, np.nan, 12],
        "seg_pupil_fraction_within_pupil_iris_hard__raw": [0.4, 0.5, 0.6],
        "seg_pupil_fraction_within_pupil_iris_hard": [0.4, np.nan, 0.6],
    })
    ev = pd.DataFrame({"session_id": ["s1"], "blink_event_id": [1], "start_unix_ms": [1000], "end_unix_ms": [1000]})
    out = pbm.build_blink_recovery_bins(tp, ev, pre_ms=100, post_ms=200, bin_ms=100, anchors=("start",))
    row = out[(out["signal"].eq(pbm.GEOMETRY_SIGNAL)) & (out["relative_bin_start_ms"].eq(0))].iloc[0]
    assert row["raw_computable_n"] == 1
    assert row["nir_qc_valid_n"] == 0


def test_recovery_bins_skip_event_anchors_without_nir_time_support():
    tp = pd.DataFrame({
        "session_id": ["s1"] * 3,
        "unix_ms": [900, 1000, 1100],
        "pupil_geom_mean_diameter__raw": [10, 11, 12],
        "pupil_geom_mean_diameter": [10, 11, 12],
        "seg_pupil_fraction_within_pupil_iris_hard__raw": [0.4, 0.5, 0.6],
        "seg_pupil_fraction_within_pupil_iris_hard": [0.4, 0.5, 0.6],
    })
    ev = pd.DataFrame({
        "session_id": ["s1", "s1"],
        "blink_event_id": [1, 2],
        "start_unix_ms": [1000, 10000],
        "end_unix_ms": [1000, 10000],
    })
    out = pbm.build_blink_recovery_bins(tp, ev, pre_ms=100, post_ms=200, bin_ms=100, anchors=("start",))
    assert set(out["blink_event_id"]) == {1}


def test_empty_blink_table_without_columns_is_valid_no_event_case():
    tp = pd.DataFrame({"session_id": ["s1", "s1"], "unix_ms": [1000, 1100]})
    out = pbm.add_rgb_blink_mask(tp, pd.DataFrame(), pre_buffer_ms=100, post_buffer_ms=300)
    assert not out["rgb_blink_mask"].any()
    sync = pbm.audit_rgb_nir_sync(tp, pd.DataFrame()).iloc[0]
    assert sync["sync_status"] == "no_blink_events"


def test_fixed_bins_support_nondivisible_candidate_width_without_crossing_zero():
    tp = pd.DataFrame({
        "session_id": ["s1"] * 3,
        "block": [1] * 3,
        "unix_ms": [1000, 5000, 9000],
        "pupil_geom_mean_diameter": [1, 5, 9],
    })
    w = pbm.select_probe_window(tp, session_id="s1", block_num=1, probe_onset_ms=11000, window_sec=10)
    bins = pbm.fixed_probe_bins(
        w, signal="pupil_geom_mean_diameter", window_sec=10,
        bin_width_sec=3, cleaning_track="original_nir",
    )
    assert bins.iloc[0]["bin_start_sec"] == pytest.approx(-10)
    assert bins.iloc[-1]["bin_end_sec"] == pytest.approx(0)
    assert (bins["bin_start_sec"] < bins["bin_end_sec"]).all()


def _run_runner_with_rgb_loader(monkeypatch, tmp_path, rgb_loader):
    source = tmp_path / "nir.csv"
    pd.DataFrame({"placeholder": [1]}).to_csv(source, index=False)
    events_path = tmp_path / "events.csv"
    pd.DataFrame(columns=["session_id", "start_unix_ms", "end_unix_ms"]).to_csv(events_path, index=False)
    probes_path = tmp_path / "probes.csv"
    pd.DataFrame({"session_id": ["s1"]}).to_csv(probes_path, index=False)
    out = tmp_path / "out"
    records = [{"session_id": "s1", "source_csv": str(source)}]

    monkeypatch.setattr(pbr, "load_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(pbr, "load_source_manifest", lambda cfg: (None, records))
    monkeypatch.setattr(pbr, "selected_records", lambda rows, subjects: rows)
    monkeypatch.setattr(
        pbr,
        "load_audit_config",
        lambda *args, **kwargs: {
            "probe_window_sec": 30.0,
            "include_soft_sensitivity": False,
            "blink_recovery": {},
        },
    )
    monkeypatch.setattr(
        pbr,
        "blink_buffers",
        lambda cfg: [SimpleNamespace(name="b100_300", pre_ms=100.0, post_ms=300.0)],
    )
    monkeypatch.setattr(pbr, "fixed_bin_widths", lambda cfg: (2.0,))
    monkeypatch.setattr(
        pbr,
        "cleaning_tracks",
        lambda cfg: ("original_nir", "nir_qc_only", "rgb_blink_only", "rgb_plus_nir_qc"),
    )
    monkeypatch.setattr(pbr, "read_table", lambda path: pd.read_csv(path))
    monkeypatch.setattr(pbr, "resolve_source", lambda cfg, path: source)
    monkeypatch.setattr(
        pbr,
        "adapt_session_rows",
        lambda raw, record: pd.DataFrame({"phase": ["block1"]}),
    )
    monkeypatch.setattr(
        pbr,
        "derive_eye_measurements",
        lambda adapted: pd.DataFrame({"session_id": ["s1"]}),
    )
    monkeypatch.setattr(
        pbr,
        "build_binocular_measurement_timepoints",
        lambda adapted: pd.DataFrame({"session_id": ["s1"], "unix_ms": [1000.0]}),
    )
    monkeypatch.setattr(pbr, "load_session_rgb_blink_frames", rgb_loader)
    monkeypatch.setattr(
        pbr,
        "audit_rgb_nir_sync_with_frames",
        lambda tp, events, frames: pd.DataFrame({"session_id": ["s1"], "sync_status": ["no_blink_events"]}),
    )
    monkeypatch.setattr(
        pbr,
        "audit_signal_availability",
        lambda eye: pd.DataFrame({"session_id": ["s1"]}),
    )
    monkeypatch.setattr(
        pbr,
        "audit_binocular_source_modes",
        lambda tp: pd.DataFrame({"session_id": ["s1"]}),
    )
    monkeypatch.setattr(
        pbr,
        "audit_rseg_quality_associations",
        lambda eye: pd.DataFrame({"session_id": ["s1"]}),
    )
    monkeypatch.setattr(
        pbr,
        "audit_signals",
        lambda tp, include_soft: ("pupil_geom_mean_diameter",),
    )

    def fake_append_probe_track(**kwargs):
        for track in kwargs["tracks"]:
            kwargs["out_rows"].append({"session_id": "s1", "cleaning_track": track})

    monkeypatch.setattr(pbr, "append_probe_track", fake_append_probe_track)

    def rgb_assisted_must_not_run(*args, **kwargs):
        raise AssertionError("RGB-assisted audit must not run when RGB source is unavailable")

    monkeypatch.setattr(pbr, "audit_buffer_loss", rgb_assisted_must_not_run)
    monkeypatch.setattr(pbr, "build_blink_recovery_bins", rgb_assisted_must_not_run)
    monkeypatch.setattr(pbr, "add_rgb_blink_mask", rgb_assisted_must_not_run)

    manifest = pbr.run_pupil_blink_measurement_audit(
        nir_config_path=tmp_path / "nir.yaml",
        audit_config_path=tmp_path / "audit.yaml",
        rgb_blink_events_path=events_path,
        probe_table_path=probes_path,
        output_root=out,
        rgb_blink_frames_root=tmp_path / "rgb",
    )
    candidates = pd.read_csv(out / "probe_measurement_candidates.csv")
    warnings = pd.read_csv(out / "measurement_audit_warnings.csv")
    failures = pd.read_csv(out / "measurement_audit_failures.csv")
    return manifest, candidates, warnings, failures


def test_runner_missing_rgb_keeps_nir_only_tracks(monkeypatch, tmp_path):
    manifest, candidates, warnings, failures = _run_runner_with_rgb_loader(
        monkeypatch,
        tmp_path,
        lambda root, session_id: None,
    )
    assert manifest["source_session_n_requested"] == 1
    assert manifest["source_session_n_processed"] == 1
    assert manifest["source_session_n_failed"] == 0
    assert manifest["source_session_n_warning"] == 0
    assert manifest["rgb_blink_source_unavailable_sessions"] == ["s1"]
    assert manifest["rgb_missing_is_zero_blinks"] is False
    assert set(candidates["cleaning_track"]) == {"original_nir", "nir_qc_only"}
    assert warnings.empty
    assert failures.empty


def test_runner_corrupt_rgb_is_nonfatal_and_keeps_nir_only_tracks(monkeypatch, tmp_path):
    def corrupt_loader(root, session_id):
        raise OSError("truncated parquet")

    manifest, candidates, warnings, failures = _run_runner_with_rgb_loader(
        monkeypatch,
        tmp_path,
        corrupt_loader,
    )
    assert manifest["source_session_n_requested"] == 1
    assert manifest["source_session_n_processed"] == 1
    assert manifest["source_session_n_failed"] == 0
    assert manifest["source_session_n_warning"] == 1
    assert manifest["source_warning_sessions"] == ["s1"]
    assert manifest["rgb_blink_source_unavailable_sessions"] == ["s1"]
    assert set(candidates["cleaning_track"]) == {"original_nir", "nir_qc_only"}
    assert len(warnings) == 1
    assert warnings.iloc[0]["session_id"] == "s1"
    assert warnings.iloc[0]["stage"] == "rgb_blink_frame_source"
    assert warnings.iloc[0]["warning_type"] == "OSError"
    assert "truncated parquet" in warnings.iloc[0]["warning"]
    assert failures.empty
