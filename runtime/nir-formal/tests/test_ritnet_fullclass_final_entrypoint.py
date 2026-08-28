from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

RUNTIME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME))

import run_ritnet_fullclass_extension as runner


def _context(tmp_path: Path) -> tuple[SimpleNamespace, Path, dict]:
    output_root = tmp_path / "output"
    run_dir = output_root / "source-run"
    run_dir.mkdir(parents=True)
    context = SimpleNamespace(
        subject="sub-031",
        run_dir=run_dir,
        config={"fullclass": {"output_dirname": "ritnet-fullclass-final"}},
    )
    identity = {
        "core_version": "fullclass-final-core-v7",
        "subject": "sub-031",
        "git_commit": "a" * 40,
        "git_branch": "nvidia-cuda",
        "config_sha256": "c" * 64,
        "ritnet_model_sha256": "m" * 64,
        "ritnet_external_data_sha256": "d" * 64,
        "source_identity": {"source_video_sha256": "v" * 64},
    }
    return context, output_root / "ritnet-fullclass-final" / "sub-031", identity


def _write_candidate(subject_dir: Path, identity: dict, *, explicit_identity: bool = True) -> None:
    subject_dir.mkdir(parents=True)
    (subject_dir / "completion.json").write_text("{}\n", encoding="utf-8")
    manifest = {"work_identity": dict(identity)}
    if explicit_identity:
        manifest["scientific_identity"] = {"result_code_sha256": "s" * 64}
        manifest["provenance_identity"] = {
            "git_commit": identity.get("git_commit"),
            "git_branch": identity.get("git_branch"),
        }
    (subject_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (subject_dir / "old-output.txt").write_text("preserve me\n", encoding="utf-8")


def _valid_completion(monkeypatch, expected: dict) -> None:
    monkeypatch.setattr(
        runner,
        "validate_final_completion",
        lambda *args, **kwargs: SimpleNamespace(valid=True, reason="valid final completion"),
    )
    monkeypatch.setattr(runner, "_expected_work_identity", lambda *args, **kwargs: expected)


def test_missing_git_provenance_is_archived_instead_of_skipped(tmp_path, monkeypatch):
    context, subject_dir, expected = _context(tmp_path)
    stored = {key: value for key, value in expected.items() if key not in {"git_commit", "git_branch"}}
    _write_candidate(subject_dir, stored)
    _valid_completion(monkeypatch, expected)

    runner._strict_skip_or_preflight(context, tmp_path / "config.yaml")

    assert not subject_dir.exists()
    archived = list((subject_dir.parent.parent / "_archive" / subject_dir.parent.name / subject_dir.name).iterdir())
    assert len(archived) == 1
    assert (archived[0] / "old-output.txt").read_text(encoding="utf-8") == "preserve me\n"
    record = json.loads((archived[0] / "_archive_reason.json").read_text(encoding="utf-8"))
    assert record["policy"] == "preserve-by-move-no-delete"
    assert record["reason"] == "valid-completion-identity-incomplete"


def test_incomplete_scientific_identity_is_archived_instead_of_skipped(tmp_path, monkeypatch):
    context, subject_dir, expected = _context(tmp_path)
    _write_candidate(subject_dir, expected, explicit_identity=False)
    _valid_completion(monkeypatch, expected)

    runner._strict_skip_or_preflight(context, tmp_path / "config.yaml")

    assert not subject_dir.exists()
    assert len(list((subject_dir.parent.parent / "_archive" / subject_dir.parent.name / subject_dir.name).iterdir())) == 1


def test_mismatched_completion_is_archived_and_valid_identity_is_skipped(tmp_path, monkeypatch, capsys):
    context, subject_dir, expected = _context(tmp_path)
    _write_candidate(subject_dir, expected)
    _valid_completion(monkeypatch, expected)

    runner._strict_skip_or_preflight(context, tmp_path / "config.yaml")
    assert subject_dir.exists()
    assert json.loads(capsys.readouterr().out)["status"] == "skipped_valid_completion"

    context2, subject_dir2, expected2 = _context(tmp_path / "second")
    _write_candidate(subject_dir2, expected2)
    monkeypatch.setattr(
        runner,
        "validate_final_completion",
        lambda *args, **kwargs: SimpleNamespace(valid=False, reason="scientific identity mismatch"),
    )

    runner._strict_skip_or_preflight(context2, tmp_path / "second-config.yaml")

    assert not subject_dir2.exists()
    archive_root = subject_dir2.parent.parent / "_archive" / subject_dir2.parent.name / subject_dir2.name
    assert len(list(archive_root.iterdir())) == 1


def test_incomplete_directory_without_completion_is_archived(tmp_path, monkeypatch):
    context, subject_dir, expected = _context(tmp_path)
    subject_dir.mkdir(parents=True)
    (subject_dir / "partial.csv").write_text("partial\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_expected_work_identity", lambda *args, **kwargs: expected)

    runner._strict_skip_or_preflight(context, tmp_path / "config.yaml")

    assert not subject_dir.exists()
    archive_root = subject_dir.parent.parent / "_archive" / subject_dir.parent.name / subject_dir.name
    archived = list(archive_root.iterdir())
    assert len(archived) == 1
    assert (archived[0] / "partial.csv").read_text(encoding="utf-8") == "partial\n"
