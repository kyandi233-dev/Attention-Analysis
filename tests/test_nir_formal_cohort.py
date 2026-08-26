from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from attention_pipeline.nir_formal_analysis import cohort


def _fake_config() -> SimpleNamespace:
    return SimpleNamespace(path=Path("configs/nir_formal_analysis.yaml"), digest="test-digest")


def _patch_common(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _fake_config()
    monkeypatch.setattr(cohort, "load_config", lambda _path: config)
    monkeypatch.setattr(cohort, "selected_subjects", lambda _config, _subjects: ["sub-031", "sub-032"])
    monkeypatch.setattr(cohort, "_output_root", lambda _config: tmp_path / "11_analysis_tables")
    monkeypatch.setattr(cohort, "_analysis_ready_root", lambda _config: tmp_path / "10_analysis_ready")


def test_run_cohort_preserves_validated_skip_and_completes(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)

    def fake_run_subject(_config, subject, *, force=False):
        assert force is False
        if subject == "sub-031":
            return {"subject": subject, "status": "skipped", "reason": "validated_completion"}
        return {"subject": subject, "status": "complete"}

    monkeypatch.setattr(cohort, "run_subject", fake_run_subject)
    result = cohort.run_cohort("unused.yaml")

    assert result["status"] == "complete"
    assert result["n_subjects_requested"] == 2
    assert result["n_subjects_skipped_validated"] == 1
    assert result["n_subjects_completed"] == 1
    assert result["n_subjects_validated"] == 2

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["resume_safe"] is True
    assert manifest["n_subjects_validated"] == 2
    assert [item["status"] for item in manifest["results"]] == ["skipped", "complete"]


def test_run_cohort_writes_interrupted_manifest(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)

    def fake_run_subject(_config, subject, *, force=False):
        if subject == "sub-031":
            return {"subject": subject, "status": "complete"}
        raise KeyboardInterrupt()

    monkeypatch.setattr(cohort, "run_subject", fake_run_subject)

    with pytest.raises(KeyboardInterrupt):
        cohort.run_cohort("unused.yaml")

    manifest_path = tmp_path / "11_analysis_tables" / "cohort_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert manifest["current_subject"] == "sub-032"
    assert manifest["n_subjects_processed"] == 1
    assert manifest["n_subjects_validated"] == 1
    assert manifest["n_subjects_remaining_including_current"] == 1
