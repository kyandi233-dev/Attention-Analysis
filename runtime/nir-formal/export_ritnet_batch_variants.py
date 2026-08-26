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
    """Match the frozen AMD runtime: project ArgMax label + class-3 softmax map."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(image)
        labels = torch.argmax(logits, dim=1).to(torch.uint8)
        pupil_probability = torch.softmax(logits, dim=1)[:, 3]
        return labels, pupil_probability


class EvidenceSummaryExportWrapper(torch.nn.Module):
    """Experimental small-output confidence summaries for DirectML qualification.

    This wrapper does not change network weights or argmax labels. It adds
    deterministic reductions after the upstream logits. It must remain a
    separate *-evidence.onnx artifact until DirectML parity/performance tests pass.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, image: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        logits = self.model(image)
        probs = torch.softmax(logits, dim=1)
        top1_probability, labels_long = torch.max(probs, dim=1)
        labels = labels_long.to(torch.uint8)
        pupil_probability = probs[:, 3]

        # Mean probability assigned to the winning pixels of each class. A class
        # absent from the hard argmax mask yields NaN via 0/0; runtime must retain
        # availability separately rather than inventing a confidence value.
        one_hot = torch.nn.functional.one_hot(labels_long, num_classes=4).permute(0, 3, 1, 2).to(probs.dtype)
        class_prob_sum = torch.sum(probs * one_hot, dim=(2, 3))
        class_pixel_count = torch.sum(one_hot, dim=(2, 3))
        class_mean_on_argmax = class_prob_sum / class_pixel_count
        top1_probability_mean = torch.mean(top1_probability, dim=(1, 2))
        entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-12)), dim=1)
        entropy_mean = torch.mean(entropy, dim=(1, 2))
        return labels, pupil_probability, class_mean_on_argmax, top1_probability_mean, entropy_mean


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
    parser.add_argument(
        "--evidence-summary",
        action="store_true",
        help=(
            "Export a separate *-evidence.onnx with small all-class confidence summaries. "
            "It is experimental and must not replace the production ONNX before DirectML parity/performance qualification."
        ),
    )
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
    wrapped = (
        EvidenceSummaryExportWrapper(model).eval()
        if args.evidence_summary
        else ExportWrapper(model).eval()
    )
    output_names = (
        [
            "labels",
            "pupil_probability",
            "class_mean_probability_on_argmax_mask",
            "top1_probability_mean",
            "entropy_mean",
        ]
        if args.evidence_summary
        else ["labels", "pupil_probability"]
    )

    outputs: list[Path] = []
    for batch in parse_batches(args.batches):
        suffix = "-evidence" if args.evidence_summary else ""
        output = PACKAGE_ROOT / "models" / f"ritnet-b{batch}-fp32{suffix}.onnx"
        output_data = output.with_name(output.name + ".data")
        if not args.force and (output.exists() or output_data.exists()):
            raise FileExistsError(
                f"Refusing to overwrite existing RITnet variant: {output} / {output_data}. "
                "Use --force only if replacement is intentional."
            )

        dummy = torch.zeros((batch, 1, int(args.height), int(args.width)), dtype=torch.float32)
        with tempfile.TemporaryDirectory(prefix="attention-ritnet-batch-") as temp_dir:
            temp_onnx = Path(temp_dir) / f"ritnet-b{batch}-fp32{suffix}.onnx"
            with torch.inference_mode():
                torch.onnx.export(
                    wrapped,
                    dummy,
                    str(temp_onnx),
                    input_names=["image"],
                    output_names=output_names,
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

    mode = "experimental evidence-summary" if args.evidence_summary else "production-interface"
    print(f"Exported {len(outputs)} {mode} RITnet variant(s) from unchanged weights: {weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
