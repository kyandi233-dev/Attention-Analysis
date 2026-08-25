from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import inspect
import platform
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _libreface_path(path: Path) -> str:
    """Return a Windows-safe path for LibreFace's slash-splitting downloader.

    LibreFace 0.2.0 derives the checkpoint parent with
    ``model_path.split('/')`` instead of ``os.path.dirname``.  A normal
    ``str(Path(...))`` on Windows therefore becomes a backslash-only path and
    makes LibreFace call ``os.makedirs('')``.  Forward slashes are valid on
    Windows, so normalize only the paths passed into LibreFace.
    """
    return path.resolve().as_posix()


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


def _au_opts(weights_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        seed=0,
        ckpt_path=_libreface_path(weights_dir / "AU_Recognition" / "weights" / "combined_repvgg.pt"),
        weights_download_id="1CbnBr8OBt8Wb73sL1ENcrtrWAFWSSRv0",
        image_inference=False,
        au_recognition_data_root="",
        au_recognition_data="DISFA",
        au_detection_data_root="",
        au_detection_data="BP4D",
        fer_train_csv="training_filtered.csv",
        fer_test_csv="validation_filtered.csv",
        fer_data_root="",
        fer_data="AffectNet",
        fold="all",
        image_size=256,
        crop_size=224,
        au_recognition_num_labels=12,
        au_detection_num_labels=12,
        fer_num_labels=8,
        sigma=10.0,
        jitter=False,
        copy_classifier=False,
        model_name="resnet",
        dropout=0.1,
        ffhq_pretrain="",
        hidden_dim=128,
        fm_distillation=False,
        num_epochs=30,
        interval=500,
        threshold=0,
        batch_size=32,
        learning_rate=3e-5,
        weight_decay=1e-4,
        clip=1.0,
        when=10,
        patience=5,
        device="cpu",
    )


def _expression_opts(weights_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        seed=0,
        train_csv="training_filtered.csv",
        test_csv="validation_filtered.csv",
        data_root="",
        ckpt_path=_libreface_path(weights_dir / "Facial_Expression_Recognition" / "weights" / "repvgg.pt"),
        weights_download_id="1yPBUjPhkwcIkRLt47-JJRLRD7rCwPIVU",
        data="AffectNet",
        image_size=224,
        num_labels=8,
        dropout=0.1,
        hidden_dim=128,
        sigma=10.0,
        student_model_name="repvgg",
        student_model_choices=["resnet_heatmap", "resnet", "swin", "mae", "emotionnet_mae", "gh_feat"],
        alpha=1.0,
        T=1.0,
        fm_distillation=True,
        grad=True,
        interval=500,
        threshold=0.0,
        loss="unweighted",
        num_epochs=50,
        batch_size=32,
        learning_rate="3e-5",
        weight_decay="1e-4",
        clip=1.0,
        when=10,
        patience=10,
        device="cpu",
    )


def _gaze_opts(weights_dir: Path, feat_dim: int) -> SimpleNamespace:
    return SimpleNamespace(
        seed=0,
        data_root="",
        ckpt_path=_libreface_path(weights_dir / "gaze_estimation" / "weights" / "mlp.pt"),
        weights_download_id="1AC20oXAV37I-OPfTtkalSJA5SfdCqXXJ",
        data="Gaze360",
        fold="all",
        num_labels=2,
        model_name="mlp",
        mlp_input_size=feat_dim,
        dropout=0.1,
        hidden_dim=128,
        half_precision=False,
        batch_size=32,
        num_workers=0,
        device="cpu",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the exact LibreFace 2.0 Python-reference AU/expression/gaze models to ONNX for DirectML."
    )
    parser.add_argument("--weights-dir", required=True, help="Existing LibreFace weights root; missing weights may be downloaded by LibreFace")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    import onnx
    import torch
    import torch.nn as nn

    from libreface.AU_Recognition.solver_inference_combine import solver_inference_image_task_combine
    from libreface.Facial_Expression_Recognition.solver_inference_image import solver_inference_image
    from libreface.gaze_estimation.solver_inference_image import GAZE_FEAT_DIM, solver_gaze_image

    weights_dir = Path(args.weights_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class JointAU(nn.Module):
        def __init__(self, base: nn.Module):
            super().__init__()
            self.encoder = base.encoder
            self.intensity = base.classifier
            self.detection = base.classifier_2

        def forward(self, image):
            features = self.encoder(image).reshape(image.shape[0], -1)
            return self.intensity(features), self.detection(features)

    class Expression(nn.Module):
        def __init__(self, base: nn.Module):
            super().__init__()
            self.base = base

        def forward(self, image):
            out = self.base(image)
            if isinstance(out, tuple):
                return out[0]
            return out

    au_solver = solver_inference_image_task_combine(_au_opts(weights_dir)).to("cpu")
    au_model = JointAU(au_solver.model).eval()

    expr_solver = solver_inference_image(_expression_opts(weights_dir)).to("cpu")
    expr_model = Expression(expr_solver.student_model).eval()

    gaze_solver = solver_gaze_image(_gaze_opts(weights_dir, GAZE_FEAT_DIM)).to("cpu")
    gaze_solver.load_best_ckpt()
    gaze_model = gaze_solver.model.eval()

    au_path = output_dir / "libreface2_au_joint.onnx"
    expr_path = output_dir / "libreface2_expression.onnx"
    gaze_path = output_dir / "libreface2_gaze_mlp.onnx"

    _export(
        torch,
        au_model,
        torch.zeros(1, 3, 224, 224, dtype=torch.float32),
        au_path,
        ["image"],
        ["au_intensity_prob", "au_detection_prob"],
        args.opset,
        {
            "image": {0: "batch"},
            "au_intensity_prob": {0: "batch"},
            "au_detection_prob": {0: "batch"},
        },
    )
    _export(
        torch,
        expr_model,
        torch.zeros(1, 3, 224, 224, dtype=torch.float32),
        expr_path,
        ["image"],
        ["expression_score"],
        args.opset,
        {"image": {0: "batch"}, "expression_score": {0: "batch"}},
    )
    _export(
        torch,
        gaze_model,
        torch.zeros(1, GAZE_FEAT_DIM, dtype=torch.float32),
        gaze_path,
        ["landmarks_468_xyz_px"],
        ["gaze_yaw_pitch_deg"],
        args.opset,
        {"landmarks_468_xyz_px": {0: "batch"}, "gaze_yaw_pitch_deg": {0: "batch"}},
    )

    models = []
    for role, path in [("au_joint", au_path), ("expression", expr_path), ("gaze_mlp", gaze_path)]:
        onnx.checker.check_model(onnx.load(str(path)))
        models.append({"role": role, "path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size})

    source_weights = []
    for role, path_str in [
        ("au_joint", _au_opts(weights_dir).ckpt_path),
        ("expression", _expression_opts(weights_dir).ckpt_path),
        ("gaze_mlp", _gaze_opts(weights_dir, GAZE_FEAT_DIM).ckpt_path),
    ]:
        path = Path(path_str)
        source_weights.append(
            {
                "role": role,
                "path": str(path.resolve()),
                "exists": path.exists(),
                "sha256": _sha256(path) if path.exists() else None,
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
        )

    try:
        libreface_version = importlib.metadata.version("libreface")
    except Exception:
        libreface_version = None

    manifest = {
        "schema_version": "rgb-face-libreface2-onnx-export-v0.1",
        "candidate": "libreface2_current_python_reference",
        "purpose": "Export the same model families/weights used by the saved LibreFace CPU reference; no CPU benchmark is rerun.",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "libreface": libreface_version,
        },
        "opset": args.opset,
        "models": models,
        "source_weights": source_weights,
        "contracts": {
            "au_input": "aligned RGB -> Resize(shorter_side=256) -> CenterCrop(224) -> ImageNet normalize; ONNX receives NCHW float32",
            "au_postprocess": "intensity = au_intensity_prob * 5.0; detection = au_detection_prob >= 0.5",
            "au_intensity_order": [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26],
            "au_detection_order": [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24],
            "expression_input": "aligned RGB resized to 224x224 -> ImageNet normalize; argmax over 8 scores",
            "expression_order": ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"],
            "gaze_input": "MediaPipe refine_landmarks=True; first 468 landmarks flattened as x*w,y*h,z*w -> 1404 float32",
            "gaze_output": "[yaw, pitch] in degrees, matching current LibreFace Python reference",
        },
        "notes": [
            "This export intentionally targets the current LibreFace Python reference used by this project, rather than assuming an older derivative ONNX/NuGet package has identical weights.",
            "MediaPipe alignment/head pose/landmarks and gaze feature extraction remain CPU-side preprocessing; only the learned AU/expression/gaze models are moved to ONNX Runtime DirectML.",
            "LibreFace 0.2.0 uses slash-based checkpoint parent parsing; checkpoint paths are normalized to forward slashes for Windows compatibility.",
        ],
    }
    manifest_path = output_dir / "libreface2_onnx_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
