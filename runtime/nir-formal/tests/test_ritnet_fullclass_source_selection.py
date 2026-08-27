from __future__ import annotations

from pathlib import Path

import pytest

import run_ritnet_fullclass_native_batch as batch


def candidate(
    path: Path,
    *,
    yolo_batch_size: int | None = 8,
    yolo_model_sha256: str | None = "yolo",
    run_id: str = "run-a",
    eyes_sha256: str = "eyes-a",
):
    marker = {
        "run_id": run_id,
        "subject": "sub-031",
        "video": "E:/Data/sub-031_/nir/sub-031_nir.avi",
        "focuswave_release": "v3.1.3",
        "phases": ["baseline", "instructions", "practice", "block1", "block2"],
        "expected_frames": 45000,
        "ritnet_enabled": True,
        "ritnet_batch_size": 16,
        "ritnet_precision": "fp32",
        "ritnet_model_sha256": "ritnet",
    }
    if yolo_batch_size is not None:
        marker["yolo_batch_size"] = yolo_batch_size
    if yolo_model_sha256 is not None:
        marker["yolo_model_sha256"] = yolo_model_sha256
    return {
        "run_dir": path,
        "marker": marker,
        "validation_reason": "valid completion marker",
        "eyes_sha256": eyes_sha256,
        "formal_identity": batch._formal_identity(marker),
    }


def test_select_run_uses_unique_validated_historical_source_regardless_of_old_yolo_metadata(tmp_path):
    selected, alternatives, reason = batch.select_run([candidate(tmp_path / "historical", yolo_batch_size=16)])
    assert selected["run_dir"] == tmp_path / "historical"
    assert alternatives == []
    assert reason == "unique_validated_historical_formal_source"


def test_select_run_allows_unique_legacy_source_without_yolo_metadata(tmp_path):
    selected, alternatives, reason = batch.select_run([
        candidate(tmp_path / "legacy", yolo_batch_size=None, yolo_model_sha256=None)
    ])
    assert selected["run_dir"] == tmp_path / "legacy"
    assert alternatives == []
    assert reason == "unique_validated_historical_formal_source"
    assert selected["marker"].get("yolo_batch_size") is None
    assert selected["marker"].get("yolo_model_sha256") is None


def test_select_run_refuses_distinct_historical_sources_instead_of_guessing(tmp_path):
    with pytest.raises(RuntimeError, match="Ambiguous validated historical formal sources"):
        batch.select_run([
            candidate(tmp_path / "first", eyes_sha256="eyes-a"),
            candidate(tmp_path / "second", eyes_sha256="eyes-b"),
        ])


def test_select_run_does_not_prefer_current_batch_when_sources_are_distinct(tmp_path):
    with pytest.raises(RuntimeError, match="Ambiguous validated historical formal sources"):
        batch.select_run([
            candidate(tmp_path / "b8", yolo_batch_size=8, run_id="b8", eyes_sha256="eyes-a"),
            candidate(tmp_path / "b16", yolo_batch_size=16, run_id="b16", eyes_sha256="eyes-b"),
        ])


def test_select_run_allows_true_duplicate_deterministically(tmp_path):
    selected, alternatives, reason = batch.select_run([
        candidate(tmp_path / "z-copy", run_id="same", eyes_sha256="same-eyes"),
        candidate(tmp_path / "a-copy", run_id="same", eyes_sha256="same-eyes"),
    ])
    assert selected["run_dir"] == tmp_path / "a-copy"
    assert alternatives == [tmp_path / "z-copy"]
    assert reason == "equivalent_duplicate_historical_sources_same_identity_and_eyes_sha256"


def test_discovery_filters_to_requested_subject_before_validation(monkeypatch, tmp_path):
    sub31 = tmp_path / "sub-031_formal_good"
    sub32 = tmp_path / "sub-032_formal_good"
    sub31.mkdir()
    sub32.mkdir()
    (sub31 / "eyes.csv").write_text("frame_idx,eye\n", encoding="utf-8")
    (sub32 / "eyes.csv").write_text("frame_idx,eye\n", encoding="utf-8")

    class Result:
        valid = True
        reason = "valid completion marker"

        def __init__(self, subject):
            self.marker = {
                "run_id": subject,
                "subject": subject,
                "video": f"E:/Data/{subject}_/nir/{subject}_nir.avi",
            }

    validated = []

    def fake_validate(path):
        validated.append(path)
        return Result(path.name.split("_formal_", 1)[0])

    monkeypatch.setattr(batch, "validate_completion", fake_validate)
    monkeypatch.setattr(batch, "sha256_file", lambda _: "eyes-hash")

    grouped = batch.discover_source_runs(tmp_path, requested_subjects=["sub-031"])
    assert set(grouped) == {"sub-031"}
    assert validated == [sub31]
