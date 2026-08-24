from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest


RUNTIME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME))

import formal_completion as completion


def test_run_lock_rejects_second_live_owner(tmp_path):
    run_dir = tmp_path / "run"
    lock = completion.acquire_run_lock(run_dir)
    try:
        with pytest.raises(completion.RunLockError, match="already active"):
            completion.acquire_run_lock(run_dir)
    finally:
        completion.release_run_lock(lock)


def test_run_lock_recovers_provably_stale_owner(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lock_path = run_dir / completion.LOCK_NAME
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "token": "stale",
                "pid": 424242,
                "host": socket.gethostname(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(completion, "_pid_is_running", lambda pid: False)

    lock = completion.acquire_run_lock(run_dir)
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["token"] == lock.token
        assert payload["pid"] == lock.pid
    finally:
        completion.release_run_lock(lock)


def test_release_does_not_remove_replaced_lock(tmp_path):
    run_dir = tmp_path / "run"
    lock = completion.acquire_run_lock(run_dir)
    replacement = {
        "schema_version": 1,
        "token": "replacement",
        "pid": lock.pid,
        "host": lock.host,
    }
    lock.path.write_text(json.dumps(replacement), encoding="utf-8")

    completion.release_run_lock(lock)

    assert lock.path.exists()
    assert json.loads(lock.path.read_text(encoding="utf-8"))["token"] == "replacement"


def test_formal_guard_spec_matches_runtime_output_name(tmp_path, monkeypatch):
    video = tmp_path / "sub-078_nir.avi"
    video.touch()
    output = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        """
package:
  version: 1.0.1
formal:
  focuswave_release: v3.1.3
  phases: [baseline2, practice, block1, block2]
ritnet:
  enabled: true
  precision: fp32
  batch_size: 16
inference:
  backend: pytorch-cuda
  yolo_batch_size: 8
output:
  root: ignored
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--config",
            str(config),
            "formal",
            "--video",
            str(video),
            "--output",
            str(output),
            "--device",
            "0",
        ],
    )

    spec = completion._formal_guard_spec()

    assert spec is not None
    run_dir, marker = spec
    assert run_dir == output / "sub-078_formal_v3.1.3_yolo8_b16_fp32"
    assert marker["status"] == "initializing"
    assert marker["subject"] == "sub-078"
    assert marker["processed_frames"] == 0
    assert marker["failure_stage"] == "initialization"
