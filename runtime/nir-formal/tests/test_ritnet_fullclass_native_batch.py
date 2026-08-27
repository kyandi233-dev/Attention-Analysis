from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import run_ritnet_fullclass_native_batch as native_batch
import ritnet_fullclass_source as source_loader


def _candidate(tmp_path: Path, subject: str, *, yolo_batch_size=8, run_id: str | None = None) -> dict:
    run_dir = tmp_path / f"{subject}_formal_v3.1.3_yolo8_b16_fp32"
    marker = {
        "run_id": run_id or f"{subject}-run",
        "subject": subject,
        "video": f"J:/Data/{subject}_/nir/{subject}_nir.avi",
        "focuswave_release": "v3.1.3",
        "phases": ["baseline2", "block1"],
        "expected_frames": 10,
        "yolo_model_sha256": "yolo",
        "ritnet_enabled": True,
        "ritnet_batch_size": 16,
        "ritnet_precision": "fp32",
        "ritnet_model_sha256": "ritnet",
    }
    if yolo_batch_size is not None:
        marker["yolo_batch_size"] = yolo_batch_size
    return {
        "run_dir": run_dir,
        "marker": marker,
        "validation_reason": "valid completion marker",
        "eyes_sha256": "eyes",
        "formal_identity": native_batch._formal_identity(marker),
    }


def test_legacy_completion_without_yolo_batch_is_accepted_and_recorded(tmp_path):
    selected, alternatives, reason = native_batch.select_run(
        [_candidate(tmp_path, "sub-031", yolo_batch_size=None)],
        expected_yolo_batch_size=8,
    )

    assert selected["marker"].get("yolo_batch_size") is None
    assert alternatives == []
    assert "legacy_source_yolo_batch_size_not_recorded_accepted" in reason


def test_explicit_production_batch_is_preferred_over_legacy_candidate(tmp_path):
    legacy = _candidate(tmp_path, "sub-031", yolo_batch_size=None, run_id="legacy")
    explicit = _candidate(tmp_path, "sub-031", yolo_batch_size=8, run_id="explicit")

    selected, alternatives, reason = native_batch.select_run(
        [legacy, explicit],
        expected_yolo_batch_size=8,
    )

    assert selected is explicit
    assert alternatives == [legacy["run_dir"]]
    assert reason == "unique_validated_source_matching_configured_yolo_batch_size"


def test_wrong_explicit_yolo_batch_is_not_treated_as_legacy(tmp_path):
    with pytest.raises(RuntimeError, match=r"available=\[4\]"):
        native_batch.select_run(
            [_candidate(tmp_path, "sub-031", yolo_batch_size=4)],
            expected_yolo_batch_size=8,
        )


def test_subject_filter_limits_directory_scan_before_validation(tmp_path, monkeypatch):
    selected_dir = tmp_path / "sub-031_formal_v3.1.3_yolo8_b16_fp32"
    unrelated_dir = tmp_path / "sub-032_formal_v3.1.3_yolo8_b16_fp32"
    selected_dir.mkdir()
    unrelated_dir.mkdir()
    (selected_dir / "eyes.csv").touch()
    (unrelated_dir / "eyes.csv").touch()
    seen: list[str] = []

    def fake_validate(run_dir: Path):
        seen.append(run_dir.name)
        if run_dir == unrelated_dir:
            raise AssertionError("unrelated subject was validated")
        return SimpleNamespace(
            valid=True,
            reason="valid completion marker",
            marker={"subject": "sub-031", "yolo_batch_size": 8},
        )

    monkeypatch.setattr(native_batch, "validate_completion", fake_validate)
    monkeypatch.setattr(native_batch, "sha256_file", lambda path: "eyes")

    grouped = native_batch.discover_source_runs(tmp_path, subjects={"sub-031"})

    assert seen == [selected_dir.name]
    assert list(grouped) == ["sub-031"]


def test_source_context_preserves_missing_legacy_yolo_batch(tmp_path, monkeypatch):
    run_dir = tmp_path / "sub-031_formal_legacy"
    run_dir.mkdir()
    video = tmp_path / "sub-031_nir.avi"
    video.touch()
    marker = {
        "run_id": "legacy",
        "subject": "sub-031",
        "video": str(video),
        "focuswave_release": "v3.1.3",
        "phases": ["baseline2"],
        "expected_frames": 2,
        "yolo_model_sha256": "yolo",
    }
    monkeypatch.setattr(
        source_loader,
        "validate_completion",
        lambda path: SimpleNamespace(valid=True, marker=marker, reason="valid completion marker"),
    )
    monkeypatch.setattr(source_loader, "load_config", lambda path: {"yolo": {"batch_size": 8}})
    monkeypatch.setattr(
        source_loader,
        "resolve_source_video",
        lambda **kwargs: (video, {"content_sha256": "video"}),
    )
    monkeypatch.setattr(source_loader, "load_source_eye_rows", lambda path, subject: (("subject",), ()))
    monkeypatch.setattr(source_loader, "load_source_frames", lambda path, subject: ())
    monkeypatch.setattr(source_loader, "sha256_file", lambda path: "digest")
    source_loader._load_source_context_cached.cache_clear()

    context = source_loader.load_source_context(run_dir, tmp_path / "config.yaml")

    assert context.source_identity["source_yolo_batch_size"] is None
    assert context.source_identity["source_yolo_batch_size_recorded"] is False
    source_loader._load_source_context_cached.cache_clear()
