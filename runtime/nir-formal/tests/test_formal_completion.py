from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


RUNTIME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME))

from formal_completion import REQUIRED_ARTIFACTS, validate_completion, write_completion


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _make_run(tmp_path: Path, *, status: str = "complete") -> tuple[Path, dict]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    video = tmp_path / "sub-031_nir.avi"
    video.touch()
    identity = {
        "subject": "sub-031",
        "video": str(video.resolve()),
        "package_version": "1.0.1",
        "focuswave_release": "v3.1.3",
        "phases": ["baseline2", "sart1"],
        "ritnet_enabled": True,
        "ritnet_precision": "fp32",
        "ritnet_batch_size": 16,
        "max_frames": None,
        "partial_phase_selection": False,
    }
    windows = [
        {"phase": "baseline2", "segment": 1, "start_frame_idx": 10, "end_frame_idx": 11},
        {"phase": "sart1", "segment": 1, "start_frame_idx": 20, "end_frame_idx": 20},
    ]
    rows = [
        ("baseline2", 1, 10, "ok"),
        ("baseline2", 1, 11, "no_eye"),
        ("sart1", 1, 20, "ok"),
    ]
    _write_json(run_dir / "phase_windows.json", windows)
    with (run_dir / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "phase_segment", "frame_idx", "status"])
        writer.writerows(rows)
    (run_dir / "eyes.csv").write_text("", encoding="utf-8")
    _write_json(
        run_dir / "summary.json",
        {
            "subject": identity["subject"],
            "video": identity["video"],
            "phases": identity["phases"],
            "processed_frames": len(rows),
            "truncated_for_smoke_test": False,
        },
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            "package": {"version": identity["package_version"]},
            "effective_parameters": {"phases": identity["phases"], "max_frames": None},
        },
    )
    write_completion(
        run_dir,
        {
            "schema_version": 1,
            "status": status,
            **identity,
            "expected_frames": len(rows),
            "processed_frames": len(rows),
            "decoded_frames": len(rows),
            "video_read_failure_count": 0,
            "missing_expected_frame_count": 0,
            "unexpected_frame_count": 0,
            "truncated_for_smoke_test": False,
            "required_artifacts": list(REQUIRED_ARTIFACTS),
        },
    )
    return run_dir, identity


def test_valid_complete_run_passes_strict_validation(tmp_path):
    run_dir, identity = _make_run(tmp_path)
    assert validate_completion(run_dir, identity).valid


def test_summary_without_completion_marker_is_not_complete(tmp_path):
    run_dir, identity = _make_run(tmp_path)
    (run_dir / "completion.json").unlink()
    result = validate_completion(run_dir, identity)
    assert not result.valid
    assert "missing completion.json" in result.reason


def test_smoke_marker_is_never_accepted_as_formal_complete(tmp_path):
    run_dir, identity = _make_run(tmp_path, status="smoke_complete")
    result = validate_completion(run_dir, identity)
    assert not result.valid
    assert "smoke_complete" in result.reason


def test_identity_mismatch_forces_rerun(tmp_path):
    run_dir, identity = _make_run(tmp_path)
    result = validate_completion(run_dir, {**identity, "package_version": "9.9.9"})
    assert not result.valid
    assert "identity mismatch" in result.reason


def test_missing_frame_fails_even_when_marker_claims_complete(tmp_path):
    run_dir, identity = _make_run(tmp_path)
    lines = (run_dir / "frames.csv").read_text(encoding="utf-8").splitlines()
    (run_dir / "frames.csv").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    marker = json.loads((run_dir / "completion.json").read_text(encoding="utf-8"))
    write_completion(run_dir, marker)
    result = validate_completion(run_dir, identity)
    assert not result.valid
    assert "processed_frames" in result.reason or "missing" in result.reason


def test_video_read_failure_cannot_validate_as_complete(tmp_path):
    run_dir, identity = _make_run(tmp_path)
    with (run_dir / "frames.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["status"] = "video_read_failed"
    with (run_dir / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    marker = json.loads((run_dir / "completion.json").read_text(encoding="utf-8"))
    marker["video_read_failure_count"] = 1
    write_completion(run_dir, marker)
    result = validate_completion(run_dir, identity)
    assert not result.valid
    assert "failed frames" in result.reason
