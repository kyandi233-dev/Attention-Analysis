from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import ritnet_fullclass_git as git_gate
import run_ritnet_fullclass_extension as single
import run_ritnet_fullclass_native_batch as batch


def test_single_entrypoint_no_longer_routes_to_legacy_native_module():
    assert not hasattr(single, "implementation")
    assert callable(single.run_numeric_core)
    assert callable(single.produce_qc_artifacts)
    assert callable(single.finalize_subject)


def test_single_strict_skip_does_not_rerun_numeric_core(monkeypatch, tmp_path):
    run_dir = tmp_path / "formal"
    run_dir.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("fullclass: {}\n", encoding="utf-8")
    subject_dir = tmp_path / "ritnet-fullclass-final" / "sub-031"
    subject_dir.mkdir(parents=True)
    (subject_dir / "completion.json").write_text("{}", encoding="utf-8")

    context = SimpleNamespace(subject="sub-031", config={})
    monkeypatch.setattr(single, "parse_args", lambda: Namespace(run_dir=run_dir, config=config, device="0"))
    monkeypatch.setattr(single, "load_source_context", lambda *_: context)
    monkeypatch.setattr(single, "require_clean_code_worktree", lambda *_: None)
    monkeypatch.setattr(single, "_strict_skip_or_preflight", lambda *_: (subject_dir, {"identity": 1}))
    monkeypatch.setattr(
        single,
        "run_numeric_core",
        lambda **_: (_ for _ in ()).throw(AssertionError("numeric core must not run on strict skip")),
    )
    assert single.main() == 0


def test_batch_canonical_cli_rejects_removed_chunk_storage_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ritnet_fullclass_batch.py",
            "--output",
            str(tmp_path),
            "--chunk-rows",
            "128",
        ],
    )
    with pytest.raises(SystemExit):
        batch.parse_args()


def test_batch_cli_keeps_only_final_operational_controls(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ritnet_fullclass_batch.py",
            "--output",
            str(tmp_path),
            "--subjects",
            "sub-031,sub-032",
            "--device",
            "0",
            "--dry-run",
        ],
    )
    args = batch.parse_args()
    assert args.output == Path(tmp_path)
    assert args.subjects == "sub-031,sub-032"
    assert args.device == "0"
    assert args.dry_run is True
    assert not hasattr(args, "chunk_rows")
    assert not hasattr(args, "compression")


def test_git_gate_allows_only_generated_final_model_artifacts(monkeypatch):
    config = {
        "models": {
            "ritnet_fullclass_final": "models/ritnet-b16-fp32-uncertainty.onnx",
            "ritnet_fullclass_final_external_data": "models/ritnet-b16-fp32-uncertainty.onnx.data",
        }
    }
    allowed = sorted(git_gate.allowed_generated_model_paths(config))
    status = "\n".join(f"?? {path}" for path in allowed) + "\n"
    monkeypatch.setattr(git_gate.subprocess, "check_output", lambda *_, **__: status)
    git_gate.require_clean_code_worktree(config)

    dirty = status + " M runtime/nir-formal/run_ritnet_fullclass_extension.py\n"
    monkeypatch.setattr(git_gate.subprocess, "check_output", lambda *_, **__: dirty)
    with pytest.raises(RuntimeError, match="Unexpected status"):
        git_gate.require_clean_code_worktree(config)
