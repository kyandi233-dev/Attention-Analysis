from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import inspect
import platform
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_frame_manifest(root: Path) -> Path:
    candidates = sorted(root.glob("*_face-benchmark_frames.csv")) + sorted(root.glob("*_face-continuous_frames.csv"))
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise RuntimeError(f"Expected exactly one supported frame manifest in {root}, found {len(unique)}")
    return unique[0]


def _image_shape_from_manifest(manifest_path: Path) -> tuple[int, int]:
    sample = pd.read_csv(manifest_path)
    if sample.empty or "image_path" not in sample:
        raise ValueError(f"Invalid frame manifest: {manifest_path}")
    shapes = set()
    for p in sample["image_path"].tolist():
        path = Path(str(p))
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as img:
            shapes.add((img.height, img.width))
        if len(shapes) > 1:
            raise RuntimeError(
                "DirectML RetinaFace export currently requires one fixed HxW for the shared 300-frame benchmark; "
                f"multiple shapes observed: {sorted(shapes)}"
            )
    return next(iter(shapes))


def _export(
    torch: Any,
    model: Any,
    example_input: Any,
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    opset: int,
    dynamic_axes: dict[str, dict[int, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        # Keep the exporter dependency-light in the candidate reference envs.
        # The classic exporter is sufficient for these feed-forward inference graphs.
        export_kwargs["dynamo"] = False
    torch.onnx.export(
        model,
        example_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        **export_kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Py-Feat 2.1.1 Detectorv2 RetinaFace + multitask scientific core to ONNX for DirectML."
    )
    parser.add_argument("--benchmark-dir", required=True, help="Existing shared 300-frame continuous benchmark directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    import onnx
    import torch
    import torch.nn as nn

    from feat import Detectorv2
    from feat.multitask.inference import (
        EXPAND_BBOX,
        HF_REPO,
        HF_WEIGHTS_FILE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        MODEL_INPUT,
    )
    from feat.utils import hf_hub_download_with_fallback
    from feat.utils.io import get_resource_path

    benchmark_dir = Path(args.benchmark_dir).resolve()
    manifest_path = _find_frame_manifest(benchmark_dir)
    height, width = _image_shape_from_manifest(manifest_path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = Detectorv2(device="cpu", identity_model=None, amp=False, compile=False)

    class MultitaskScientificCore(nn.Module):
        def __init__(self, base: nn.Module):
            super().__init__()
            self.base = base

        def forward(self, image):
            out = self.base(image)
            blendshapes = out["blendshapes"]
            if blendshapes is None:
                blendshapes = torch.full((image.shape[0], 52), float("nan"), device=image.device)
            return (
                out["p_au"],
                torch.softmax(out["emotion_logits"], dim=-1),
                out["va"],
                out["gaze"],
                out["pose"],
                out["mesh"],
                blendshapes,
            )

    multitask_model = MultitaskScientificCore(detector.multitask.model).eval()
    retinaface_model = detector.face_detector.net.eval()

    retina_path = output_dir / "pyfeat211_retinaface_r34.onnx"
    multitask_path = output_dir / "pyfeat211_multitask_scientific_core.onnx"

    # RetinaFace ONNX receives the same RGB float32 frame after [123,117,104] mean subtraction
    # that the Py-Feat wrapper applies immediately before its PyTorch network.
    _export(
        torch,
        retinaface_model,
        torch.zeros(1, 3, height, width, dtype=torch.float32),
        retina_path,
        ["image_mean_subtracted_rgb255"],
        ["bbox_regression", "face_probability", "landmark5_regression"],
        args.opset,
        {
            "image_mean_subtracted_rgb255": {0: "batch"},
            "bbox_regression": {0: "batch"},
            "face_probability": {0: "batch"},
            "landmark5_regression": {0: "batch"},
        },
    )
    _export(
        torch,
        multitask_model,
        torch.zeros(1, 3, MODEL_INPUT, MODEL_INPUT, dtype=torch.float32),
        multitask_path,
        ["face_chip_imagenet_224"],
        ["au_prob", "emotion_prob", "valence_arousal", "gaze_raw", "pose_raw", "mesh478_normalized", "blendshapes"],
        args.opset,
        {
            "face_chip_imagenet_224": {0: "batch"},
            "au_prob": {0: "batch"},
            "emotion_prob": {0: "batch"},
            "valence_arousal": {0: "batch"},
            "gaze_raw": {0: "batch"},
            "pose_raw": {0: "batch"},
            "mesh478_normalized": {0: "batch"},
            "blendshapes": {0: "batch"},
        },
    )

    for p in (retina_path, multitask_path):
        onnx.checker.check_model(onnx.load(str(p)))

    multitask_weight = detector.multitask._resolve_weights(None)
    retina_weight = hf_hub_download_with_fallback(
        repo_id="py-feat/retinaface_r34",
        filename="model.safetensors",
        fallback_filename="retinaface_r34.safetensors",
        cache_dir=get_resource_path(),
    )

    try:
        pyfeat_version = importlib.metadata.version("py-feat")
    except Exception:
        pyfeat_version = None

    manifest = {
        "schema_version": "rgb-face-pyfeat211-onnx-export-v0.1",
        "candidate": "pyfeat_detectorv2_scientific_core",
        "purpose": "Export current Py-Feat Detectorv2 RetinaFace and multitask scientific core without rerunning the saved CPU benchmark.",
        "benchmark_frame_manifest": str(manifest_path),
        "fixed_retinaface_input_hw": [height, width],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "py_feat": pyfeat_version,
        },
        "opset": args.opset,
        "models": [
            {"role": "retinaface_r34", "path": str(retina_path), "sha256": _sha256(retina_path), "size_bytes": retina_path.stat().st_size},
            {"role": "multitask_scientific_core", "path": str(multitask_path), "sha256": _sha256(multitask_path), "size_bytes": multitask_path.stat().st_size},
        ],
        "source_weights": [
            {"role": "retinaface_r34", "path": str(Path(retina_weight).resolve()), "sha256": _sha256(Path(retina_weight))},
            {"role": "multitask_v2", "path": str(Path(multitask_weight).resolve()), "sha256": _sha256(Path(multitask_weight))},
        ],
        "contracts": {
            "retinaface_input": "RGB float32 [0,255] minus per-channel mean [123,117,104]; fixed HxW from the shared 300-frame manifest",
            "retinaface_postprocess": {
                "min_sizes": [[16, 32], [64, 128], [256, 512]],
                "steps": [8, 16, 32],
                "variance": [0.1, 0.2],
                "confidence_threshold_pre_nms": 0.02,
                "nms_iou": 0.4,
                "face_detection_threshold": 0.5,
                "keep_top_k": 750,
            },
            "crop": f"isotropic square-pad, expand_bbox={EXPAND_BBOX}, 256x256, reflection padding",
            "multitask_input": f"256 RGB chip [0,1] resized over full field to {MODEL_INPUT}x{MODEL_INPUT}, ImageNet normalize mean={list(IMAGENET_MEAN)} std={list(IMAGENET_STD)}",
            "multitask_outputs": "20 AU probabilities, 7 emotion probabilities, V/A, raw gaze [yaw,pitch] rad, raw pose, normalized 478x3 mesh, 52 blendshapes",
            "canonical_postprocess": {
                "head_pose": "Pitch=-pose_raw[:,0], Roll=-pose_raw[:,2], Yaw=pose_raw[:,1], XYZ=pose_raw[:,3:6]",
                "gaze": "gaze_pitch=-gaze_raw[:,1], gaze_yaw=gaze_raw[:,0], gaze_angle=acos(cos(pitch)*cos(yaw))",
            },
        },
        "identity_branch": "excluded from first scientific-core DirectML benchmark by project decision",
    }
    manifest_path_out = output_dir / "pyfeat211_onnx_export_manifest.json"
    manifest_path_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
