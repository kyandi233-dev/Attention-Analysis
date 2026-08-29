from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "setup_formal_environment.py"
    module_name = "setup_formal_environment"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_formal_analysis_environment_mapping_is_isolated():
    module = _load_module()
    assert module.ENVIRONMENTS["behavior"].env_name == "attention-behavior-formal"
    assert module.ENVIRONMENTS["nir"].env_name == "attention-nir-formal"
    assert module.ENVIRONMENTS["rgb"].env_name == "attention-rgb-formal"
    assert len({spec.env_name for spec in module.ENVIRONMENTS.values()}) == 3


def test_environment_yaml_files_exist():
    module = _load_module()
    for spec in module.ENVIRONMENTS.values():
        assert spec.yaml_path.is_file(), spec.yaml_path


def test_placeholder_detection_blocks_unedited_local_paths(tmp_path):
    module = _load_module()
    unresolved = tmp_path / "paths.local.yaml"
    unresolved.write_text('paths:\n  cohort_manifest: "${FOCUSWAVE_COHORT_MANIFEST}"\n', encoding="utf-8")
    resolved = tmp_path / "paths.ready.yaml"
    resolved.write_text('paths:\n  cohort_manifest: "Q:/project/cohort_manifest.csv"\n', encoding="utf-8")
    assert module._contains_unresolved_placeholders(unresolved) is True
    assert module._contains_unresolved_placeholders(resolved) is False


def test_paths_only_parser_does_not_require_conda():
    module = _load_module()
    args = module.parse_args(["rgb", "--paths-only"])
    assert args.analysis == "rgb"
    assert args.paths_only is True


def test_rgb_environment_is_downstream_only():
    module = _load_module()
    text = module.ENVIRONMENTS["rgb"].yaml_path.read_text(encoding="utf-8").lower()
    assert "pyarrow" in text
    assert "mediapipe" not in text
    assert "onnxruntime-directml" not in text
    assert "pytorch" not in text
