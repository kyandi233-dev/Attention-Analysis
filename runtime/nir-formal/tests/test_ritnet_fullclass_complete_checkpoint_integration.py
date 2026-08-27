from __future__ import annotations

from types import SimpleNamespace

import ritnet_fullclass_final_engine as engine
from ritnet_fullclass_io import csv_fieldnames, iter_csv
from ritnet_fullclass_schema import EYE_METRIC_FIELDS, FRAME_COVERAGE_FIELDS


def test_complete_checkpoint_serializes_plain_csv_and_carries_rows_without_runtime(monkeypatch, tmp_path):
    run_dir = tmp_path / "sub-034_formal_source"
    run_dir.mkdir()
    model = tmp_path / "model.onnx"
    external = tmp_path / "model.onnx.data"
    model.write_bytes(b"model")
    external.write_bytes(b"external")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")

    source_eye = {
        "phase": "block1",
        "phase_segment": 1,
        "frame_idx": 10,
        "video_time_ms": 1000.0,
        "unix_ms": 2000.0,
        "phase_time_ms": 500.0,
        "eye": "frame_left",
        "source": "yolo",
        "frame_status": "single_eye",
        "status": "observed",
        "redetect_reason": "tracker_disabled",
        "anchor_yolo_confidence": 0.9,
        "bbox_x1": 10,
        "bbox_y1": 20,
        "bbox_x2": 50,
        "bbox_y2": 60,
        "yolo_batch_size": 8,
    }
    source_frame = {
        "phase": "block1",
        "phase_segment": 1,
        "frame_idx": 10,
        "video_time_ms": 1000.0,
        "unix_ms": 2000.0,
        "phase_time_ms": 500.0,
        "status": "single_eye",
        "raw_detection_count": 1,
        "selected_eye_count": 1,
    }
    stored_atomic = {
        "eye_metrics_schema_version": 6,
        "subject": "sub-034",
        "phase": "block1",
        "phase_segment": 1,
        "frame_idx": 10,
        "video_time_ms": 1000.0,
        "unix_ms": 2000.0,
        "phase_time_ms": 500.0,
        "eye": "frame_left",
        "ritnet_status": "success",
        "ritnet_failure_reason": None,
        "hard_pupil_fraction": 0.1,
        "hard_ocular_fraction": 0.5,
        "pupil_center_x": 320.0,
        "pupil_center_y": 200.0,
        "ocular_max_probability_mean": 0.9,
        "ocular_top1_top2_margin_mean": 0.7,
        "ocular_entropy_mean": 0.3,
    }
    config = {
        "models": {
            "ritnet_fullclass_final": str(model),
            "ritnet_fullclass_final_external_data": str(external),
        },
        "fullclass": {
            "output_dirname": "ritnet-fullclass-final",
            "roi": {
                "target_width": 640,
                "target_height": 400,
                "aspect_ratio": 1.6,
                "expand_horizontal_each_side": 0.30,
                "expand_vertical_each_side": 0.45,
                "padding_mode": "replicate",
            },
            "qc_interval_sec": 30,
            "checkpoint_rows": 128,
            "progress_every_batches": 100,
            "summary_workers": 2,
            "max_pending_summaries": 2,
        },
    }
    context = SimpleNamespace(
        subject="sub-034",
        run_dir=run_dir,
        config=config,
        eye_rows=(source_eye,),
        frame_rows=(source_frame,),
    )

    class FakeStore:
        def __init__(self, path, *, identity):
            self.path = path
            self.identity = identity

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def validate_prefix(self, rows):
            assert rows is context.eye_rows
            return len(rows)

        def iter_rows(self):
            yield dict(stored_atomic)

    class RuntimeMustNotLoad:
        FIXED_BATCH_SIZE = 16

        def __init__(self, *args, **kwargs):
            raise AssertionError("complete checkpoint recovery must not initialize RITnet runtime")

    monkeypatch.setattr(engine, "load_source_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(engine, "_work_identity", lambda **_kwargs: {"test_identity": True})
    monkeypatch.setattr(engine, "FullClassWorkStore", FakeStore)
    monkeypatch.setattr(engine, "RitnetFullClassFinalRuntime", RuntimeMustNotLoad)

    core = engine.run_numeric_core(
        run_dir=run_dir,
        config_path=config_path,
        device="0",
    )

    assert core.eye_row_count == 1
    assert core.frame_row_count == 1
    assert core.eye_metrics.name == "eye_metrics.csv"
    assert core.frame_coverage.name == "frame_coverage.csv"
    assert csv_fieldnames(core.eye_metrics) == EYE_METRIC_FIELDS
    assert csv_fieldnames(core.frame_coverage) == FRAME_COVERAGE_FIELDS

    assert len(core.eye_metric_rows) == 1
    assert len(core.frame_coverage_rows) == 1
    assert core.eye_metric_rows[0]["subject"] == "sub-034"
    assert core.eye_metric_rows[0]["temporal_reset_reason"] == "first_observation"
    assert core.frame_coverage_rows[0]["coverage_status"] == "single_eye_success"
    assert core.frame_coverage_rows[0]["fixed_qc_anchor"] is True

    # Disk serialization is an independent final artifact contract; the runner
    # no longer reparses these files merely to feed QC.
    disk_eye_rows = list(iter_csv(core.eye_metrics))
    disk_coverage_rows = list(iter_csv(core.frame_coverage))
    assert len(disk_eye_rows) == len(core.eye_metric_rows)
    assert len(disk_coverage_rows) == len(core.frame_coverage_rows)
    assert disk_eye_rows[0]["subject"] == core.eye_metric_rows[0]["subject"]
    assert disk_coverage_rows[0]["coverage_status"] == core.frame_coverage_rows[0]["coverage_status"]
