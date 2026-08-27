from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import ritnet_fullclass_final_engine as engine
from ritnet_fullclass_schema import EYE_METRICS_SCHEMA_VERSION


def _source_eye():
    return {
        "phase": "block1",
        "phase_segment": 1,
        "frame_idx": 10,
        "video_time_ms": 1000.0,
        "unix_ms": 2000.0,
        "phase_time_ms": 500.0,
        "eye": "frame_left",
        "source": "yolo",
        "redetect_reason": "tracker_disabled",
        "frame_status": "single_eye",
        "status": "observed",
        "anchor_yolo_confidence": 0.9,
        "bbox_x1": 10.0,
        "bbox_y1": 20.0,
        "bbox_x2": 50.0,
        "bbox_y2": 60.0,
        "yolo_batch_size": 8,
    }


def _source_frame():
    return {
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


def _numeric_eye():
    return {
        "eye_metrics_schema_version": EYE_METRICS_SCHEMA_VERSION,
        "subject": "sub-031",
        "phase": "block1",
        "phase_segment": 1,
        "frame_idx": 10,
        "video_time_ms": 1000.0,
        "unix_ms": 2000.0,
        "phase_time_ms": 500.0,
        "eye": "frame_left",
        "ritnet_status": "success",
        "ritnet_failure_reason": None,
    }


def test_complete_checkpoint_finalizes_numeric_tables_without_runtime_or_gzip(tmp_path, monkeypatch):
    run_dir = tmp_path / "sub-031_formal_v3.1.3_yolo_b16_fp32"
    run_dir.mkdir()
    model = tmp_path / "model.onnx"
    external = tmp_path / "model.onnx.data"
    model.write_bytes(b"model")
    external.write_bytes(b"external")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")

    context = SimpleNamespace(
        run_dir=run_dir,
        subject="sub-031",
        video=tmp_path / "source.avi",
        eye_rows=(_source_eye(),),
        frame_rows=(_source_frame(),),
        source_identity={"source_video_sha256": "a" * 64},
        config={
            "models": {
                "ritnet_fullclass_final": "model.onnx",
                "ritnet_fullclass_final_external_data": "model.onnx.data",
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
        },
    )

    numeric_row = _numeric_eye()

    class FakeStore:
        def __init__(self, path, *, identity):
            self.path = Path(path)
            self.identity = identity

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def validate_prefix(self, rows):
            assert rows is context.eye_rows
            return len(rows)

        def iter_rows(self):
            yield dict(numeric_row)

    class RuntimeMustNotLoad:
        FIXED_BATCH_SIZE = 16

        def __init__(self, *args, **kwargs):
            raise AssertionError("complete checkpoint must not initialize DirectML runtime")

    monkeypatch.setattr(engine, "load_source_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(engine, "FullClassWorkStore", FakeStore)
    monkeypatch.setattr(engine, "RitnetFullClassFinalRuntime", RuntimeMustNotLoad)
    monkeypatch.setattr(engine, "iter_temporal_facts", lambda rows: (dict(row) for row in rows))
    monkeypatch.setattr(engine, "_work_identity", lambda **_kwargs: {"identity": "test"})
    monkeypatch.setattr(
        engine,
        "resolve_package_path",
        lambda value: model if str(value) == "model.onnx" else external,
    )

    core = engine.run_numeric_core(
        run_dir=run_dir,
        config_path=config_path,
        device="0",
    )

    assert core.source_context is context
    assert core.eye_row_count == 1
    assert core.frame_row_count == 1
    assert core.eye_metrics.name == "eye_metrics.csv"
    assert core.frame_coverage.name == "frame_coverage.csv"
    assert not (core.eye_metrics.parent / "eye_metrics.csv.gz").exists()
    assert not (core.frame_coverage.parent / "frame_coverage.csv.gz").exists()

    with core.eye_metrics.open(encoding="utf-8", newline="") as handle:
        eye_rows = list(csv.DictReader(handle))
    with core.frame_coverage.open(encoding="utf-8", newline="") as handle:
        coverage_rows = list(csv.DictReader(handle))

    assert len(eye_rows) == 1
    assert eye_rows[0]["subject"] == "sub-031"
    assert eye_rows[0]["eye"] == "frame_left"
    assert len(coverage_rows) == 1
    assert coverage_rows[0]["coverage_status"] == "single_eye_success"
