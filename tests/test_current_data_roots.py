from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOTS = ["J:/Data"]


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_nir_batch_module():
    path = REPO_ROOT / "runtime" / "nir-formal" / "run_formal_batch.py"
    spec = importlib.util.spec_from_file_location("nir_formal_batch_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_behavior_and_nir_configs_share_dynamic_roots():
    behavior = _load_yaml(REPO_ROOT / "configs" / "behavior_formal.yaml")
    nir = _load_yaml(REPO_ROOT / "runtime" / "nir-formal" / "config.yaml")

    assert behavior["data"]["roots"] == EXPECTED_ROOTS
    assert nir["data"]["roots"] == EXPECTED_ROOTS


def test_nir_discovery_ignores_missing_candidate_roots(tmp_path):
    batch = _load_nir_batch_module()
    missing_a = tmp_path / "missing-a"
    active = tmp_path / "active"
    missing_b = tmp_path / "missing-b"
    video = active / "sub-031_" / "nir" / "sub-031_nir.avi"
    video.parent.mkdir(parents=True)
    video.touch()

    config = {
        "data": {
            "roots": [str(missing_a), str(active), str(missing_b)],
            "subject_pattern": "sub-*_/nir/*_nir.avi",
        },
        "formal": {"min_subject_number": 31},
    }

    discovered = batch.discover(config)

    assert discovered == {"sub-031": video}


def test_nir_discovery_rejects_duplicate_subject_video(tmp_path):
    batch = _load_nir_batch_module()
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    video_a = root_a / "sub-031_" / "nir" / "sub-031_nir.avi"
    video_b = root_b / "sub-031_" / "nir" / "sub-031_nir.avi"
    for video in (video_a, video_b):
        video.parent.mkdir(parents=True)
        video.touch()

    config = {
        "data": {
            "roots": [str(root_a), str(root_b)],
            "subject_pattern": "sub-*_/nir/*_nir.avi",
        },
        "formal": {"min_subject_number": 31},
    }

    with pytest.raises(RuntimeError, match="Duplicate subject videos found across data roots"):
        batch.discover(config)


def test_nir_subject_exclusion_applies_even_to_cli_selection():
    batch = _load_nir_batch_module()
    discovered = {
        "sub-031": Path("sub-031_nir.avi"),
        "sub-9504": Path("sub-9504_nir.avi"),
    }
    config = {"batch": {"subjects": {"include": [], "exclude": ["sub-9504"]}}}

    selected = batch.selected_subjects(config, discovered, ["sub-031", "sub-9504"])

    assert selected == ["sub-031"]


def test_batch_rejects_exit_zero_without_valid_completion_marker(monkeypatch, tmp_path):
    batch = _load_nir_batch_module()
    data_root = tmp_path / "data"
    video = data_root / "sub-031_" / "nir" / "sub-031_nir.avi"
    video.parent.mkdir(parents=True)
    video.touch()
    output_root = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
package:
  version: 1.0.1
data:
  roots: [DATA_ROOT]
  subject_pattern: sub-*_/nir/*_nir.avi
formal:
  min_subject_number: 31
  focuswave_release: v3.1.3
  phases: [baseline2, sart1]
ritnet:
  enabled: true
  precision: fp32
  batch_size: 16
batch:
  subjects:
    include: []
    exclude: []
  output_root: OUTPUT_ROOT
  device: "0"
  continue_on_error: true
  skip_completed: true
output:
  root: OUTPUT_ROOT
""".replace("DATA_ROOT", json.dumps(str(data_root))).replace(
            "OUTPUT_ROOT", json.dumps(str(output_root))
        ),
        encoding="utf-8",
    )

    class Completed:
        returncode = 0

    monkeypatch.setattr(batch.subprocess, "run", lambda *args, **kwargs: Completed())
    monkeypatch.setattr(
        batch,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "config": config_path,
                "subjects": None,
                "device": None,
                "ritnet_precision": None,
                "ritnet_batch_size": None,
                "phases": None,
                "backend": None,
                "output": None,
                "force": False,
                "dry_run": False,
            },
        )(),
    )

    return_code = batch.main()
    results = json.loads((output_root / "batch_run_summary.json").read_text(encoding="utf-8"))

    assert return_code == 1
    assert results[0]["status"] == "failed"
    assert "missing completion.json" in results[0]["validation_error"]
