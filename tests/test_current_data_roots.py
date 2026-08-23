from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOTS = [
    "E:/正式实验",
    "F:/正式实验",
    "E:/Data",
    "F:/Data",
]


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
