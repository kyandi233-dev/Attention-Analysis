from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import onnx
import torch


PACKAGE_ROOT = Path(__file__).resolve().parent
RITNET_DIR = PACKAGE_ROOT / "ritnet"
if str(RITNET_DIR) not in sys.path:
    sys.path.insert(0, str(RITNET_DIR))

from densenet import DenseNet2D


class ExportWrapper(torch.nn.Module):
    """Match the AMD runtime outputs: uint8 labels plus FP32 pupil probability."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(image)
        labels = torch.argmax(logits, dim=1).to(torch.uint8)
        pupil_probability = torch.softmax(logits, dim=1)[:, 3]
        return labels, pupil_probability


def parse_batches(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("--batches must contain positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed-batch RITnet FP32 ONNX variants")
    parser.add_argument("--weights", type=Path, default=PACKAGE_ROOT / "models" / "ritnet-best_model.pkl")
    parser.add_argument("--batches", default="8,10,12,14")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing variant")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)

    model = DenseNet2D(dropout=True, prob=0.2)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    wrapped = ExportWrapper(model).eval()

    outputs: list[Path] = []
    for batch in parse_batches(args.batches):
        output = PACKAGE_ROOT / "models" / f"ritnet-b{batch}-fp32.onnx"
        output_data = output.with_name(output.name + ".data")
        if not args.force and (output.exists() or output_data.exists()):
            raise FileExistsError(
                f"Refusing to overwrite existing RITnet variant: {output} / {output_data}. "
                "Use --force only if replacement is intentional."
            )

        dummy = torch.zeros((batch, 1, int(args.height), int(args.width)), dtype=torch.float32)
        with tempfile.TemporaryDirectory(prefix="attention-ritnet-batch-") as temp_dir:
            temp_onnx = Path(temp_dir) / f"ritnet-b{batch}-fp32.onnx"
            with torch.inference_mode():
                torch.onnx.export(
                    wrapped,
                    dummy,
                    str(temp_onnx),
                    input_names=["image"],
                    output_names=["labels", "pupil_probability"],
                    opset_version=int(args.opset),
                    do_constant_folding=True,
                    dynamo=False,
                )

            graph = onnx.load(str(temp_onnx), load_external_data=True)
            onnx.checker.check_model(graph)
            output.parent.mkdir(parents=True, exist_ok=True)
            onnx.save_model(
                graph,
                str(output),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=output_data.name,
                size_threshold=0,
                convert_attribute=False,
            )

        check = onnx.load(str(output), load_external_data=True)
        onnx.checker.check_model(check)
        print(f"batch={batch}: {output}")
        print(f"           {output_data}")
        outputs.append(output)

    print(f"Exported {len(outputs)} RITnet variant(s) from unchanged weights: {weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
