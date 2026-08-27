"""NVIDIA/CUDA qualification gate for the final batch-16 RITnet ONNX export."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ritnet_fullclass_final_runtime import RitnetFullClassFinalRuntime

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PACKAGE_ROOT / "models" / "ritnet-b16-fp32-uncertainty.onnx"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate final RITnet b16 CUDA output contract")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def validate_model(model: Path, *, device: str = "0") -> dict:
    model = Path(model).expanduser().resolve()
    external = model.with_name(model.name + ".data")
    if not model.is_file() or not external.is_file():
        raise FileNotFoundError(f"final ONNX/export data missing: {model} / {external}")
    runtime = RitnetFullClassFinalRuntime(model, device=str(device))
    roi_a = np.zeros((400, 640), dtype=np.uint8)
    roi_b = np.tile(np.arange(640, dtype=np.uint16) % 256, (400, 1)).astype(np.uint8)
    outputs, timing = runtime.infer_batch([roi_a, roi_b])
    probability = outputs["class_probability"]
    class_mass = probability.sum(axis=1)
    result = {
        "status": "pass",
        "model": str(model),
        "model_sha256": _sha256(model),
        "external_data": str(external),
        "external_data_sha256": _sha256(external),
        "device": runtime.device,
        "providers": runtime.providers,
        "input_size": list(runtime.input_size),
        "fixed_batch_size": runtime.FIXED_BATCH_SIZE,
        "requested_roi_count": 2,
        "outputs": {name: list(value.shape) for name, value in outputs.items()},
        "output_dtypes": {name: str(value.dtype) for name, value in outputs.items()},
        "label_values": sorted(int(value) for value in np.unique(outputs["labels"])),
        "class_probability_mass_max_abs_deviation": float(np.max(np.abs(class_mass - 1.0))),
        "timing": timing,
    }
    if not runtime.providers or runtime.providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA is not the primary provider: {runtime.providers}")
    expected_shapes = {
        "labels": [2, 400, 640],
        "class_probability": [2, 4, 400, 640],
        "max_probability": [2, 400, 640],
        "top1_top2_margin": [2, 400, 640],
        "entropy": [2, 400, 640],
    }
    if result["outputs"] != expected_shapes:
        raise RuntimeError(f"unexpected final output shapes: {result['outputs']}")
    if result["class_probability_mass_max_abs_deviation"] > 1e-5:
        raise RuntimeError("four-class per-pixel probability mass is not normalized")
    return result


def main() -> int:
    args = parse_args()
    print(json.dumps(validate_model(args.model, device=str(args.device)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
