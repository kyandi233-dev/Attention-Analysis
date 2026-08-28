from pathlib import Path

import yaml


RUNTIME = Path(__file__).resolve().parents[1] / "runtime" / "nir-formal"


def test_formal_runtime_has_required_entrypoints_and_assets():
    required = [
        "README.md",
        "INSTALL.md",
        "config.yaml",
        "run_pipeline.py",
        "run_formal_batch.py",
        "formal_completion.py",
        "phase_windows.py",
        "cuda_runtime.py",
        "ritnet_fullclass_final_runtime.py",
        "ritnet_runtime.py",
        "requirements.txt",
        "models/nir-eye-yolo26n-best.onnx",
        "models/ritnet-b16-fp32.onnx",
        "models/ritnet-b16-fp32.onnx.data",
    ]
    missing = [item for item in required if not (RUNTIME / item).exists()]
    assert not missing, f"formal NIR runtime is missing required assets: {missing}"


def test_formal_runtime_config_is_current_nvidia_cuda_v8_baseline():
    config = yaml.safe_load((RUNTIME / "config.yaml").read_text(encoding="utf-8"))
    assert config["formal"]["focuswave_release"] == "v3.1.3"
    assert config["formal"]["expected_formal_blocks"] == 2
    assert config["formal"]["min_subject_number"] == 31
    assert config["tracking"]["method"] == "none"
    assert config["ritnet"]["batch_size"] == 16
    assert config["ritnet"]["precision"] == "fp32"
    assert config["package"]["version"] == "0.2.0"
    assert "NVIDIA CUDA" in config["package"]["purpose"]
    assert config["batch"]["subjects"]["exclude"] == ["sub-9504"]
    assert "nvidia-cuda" in config["output"]["root"]
    assert "nvidia-cuda" in config["batch"]["output_root"]
    assert config["models"]["yolo"].endswith(".onnx")
    assert config["models"]["ritnet"].endswith(".onnx")


def test_legacy_pytorch_weights_are_not_in_nvidia_v8_runtime():
    assert not (RUNTIME / "models/nir-eye-yolo26n-best.pt").exists()
    assert not (RUNTIME / "models/ritnet-best_model.pkl").exists()


def test_deleted_tracking_runtime_is_not_current_package():
    old_runtime = RUNTIME.parent / "nir-yolo-tracking-ritnet-v1"
    assert not old_runtime.exists()
