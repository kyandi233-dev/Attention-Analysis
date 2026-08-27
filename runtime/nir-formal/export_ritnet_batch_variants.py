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
    """Historical production interface: project ArgMax label + class-3 probability."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(image)
        labels = torch.argmax(logits, dim=1).to(torch.uint8)
        pupil_probability = torch.softmax(logits, dim=1)[:, 3]
        return labels, pupil_probability


class EvidenceSummaryExportWrapper(torch.nn.Module):
    """Historical experimental small-output confidence summaries.

    Retained for provenance only. The final full-class path uses
    ``FinalUncertaintyExportWrapper`` because mean-only summaries cannot support
    percentile, top1-vs-top2 or boundary uncertainty QC.
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

        one_hot = torch.nn.functional.one_hot(labels_long, num_classes=4).permute(0, 3, 1, 2).to(probs.dtype)
        class_prob_sum = torch.sum(probs * one_hot, dim=(2, 3))
        class_pixel_count = torch.sum(one_hot, dim=(2, 3))
        class_mean_on_argmax = class_prob_sum / class_pixel_count
        top1_probability_mean = torch.mean(top1_probability, dim=(1, 2))
        entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-12)), dim=1)
        entropy_mean = torch.mean(entropy, dim=(1, 2))
        return labels, pupil_probability, class_mean_on_argmax, top1_probability_mean, entropy_mean


class FinalUncertaintyExportWrapper(torch.nn.Module):
    """Final all-class probability/uncertainty interface for the <=1 GiB workflow.

    RITnet weights and logits are unchanged. The graph adds deterministic
    post-processing only. Four-class pixel probabilities plus three uncertainty
    maps are returned temporarily to CPU and MUST be reduced immediately; they
    are not final on-disk artifacts.

    Outputs:
      labels: uint8 [B,H,W]
      class_probability: float32 [B,4,H,W]
      max_probability: float32 [B,H,W]
      top1_top2_margin: float32 [B,H,W]
      entropy: float32 [B,H,W]
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
        top2_probability, top2_class = torch.topk(probs, k=2, dim=1, largest=True, sorted=True)
        labels = top2_class[:, 0].to(torch.uint8)
        max_probability = top2_probability[:, 0]
        top1_top2_margin = top2_probability[:, 0] - top2_probability[:, 1]
        entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-12)), dim=1)
        return labels, probs, max_probability, top1_top2_margin, entropy


def parse_batches(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("--batches must contain positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed-batch RITnet FP32 ONNX variants")
    parser.add_argument("--weights", type=Path, default=PACKAGE_ROOT / "models" / "ritnet-best_model.pkl")
    parser.add_argument(
        "--batches",
        default=None,
        help=(
            "Comma-separated fixed batch sizes. Default is 16 for --final-uncertainty; "
            "historical modes retain 8,10,12,14."
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing variant")
    parser.add_argument(
        "--evidence-summary",
        action="store_true",
        help="Export historical experimental *-evidence.onnx mean-only summaries.",
    )
    parser.add_argument(
        "--final-uncertainty",
        action="store_true",
        help=(
            "Export the final *-uncertainty.onnx interface: labels + transient four-class "
            "probabilities + max-probability/margin/entropy maps."
        ),
    )
    args = parser.parse_args()
    if args.evidence_summary and args.final_uncertainty:
        parser.error("--evidence-summary and --final-uncertainty are mutually exclusive")
    if args.batches is None:
        args.batches = "16" if args.final_uncertainty else "8,10,12,14"
    return args


def _export_mode(args: argparse.Namespace) -> tuple[torch.nn.Module, list[str], str, str]:
    model = DenseNet2D(dropout=True, prob=0.2)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    if args.final_uncertainty:
        return (
            FinalUncertaintyExportWrapper(model).eval(),
            [
                "labels",
                "class_probability",
                "max_probability",
                "top1_top2_margin",
                "entropy",
            ],
            "-uncertainty",
            "final uncertainty",
        )
    if args.evidence_summary:
        return (
            EvidenceSummaryExportWrapper(model).eval(),
            [
                "labels",
                "pupil_probability",
                "class_mean_probability_on_argmax_mask",
                "top1_probability_mean",
                "entropy_mean",
            ],
            "-evidence",
            "historical experimental evidence-summary",
        )
    return (
        ExportWrapper(model).eval(),
        ["labels", "pupil_probability"],
        "",
        "historical production-interface",
    )


def main() -> int:
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    args.weights = weights
    if int(args.width) != 640 or int(args.height) != 400:
        if args.final_uncertainty:
            raise ValueError("final uncertainty RITnet export is frozen to 640x400")

    wrapped, output_names, suffix, mode = _export_mode(args)

    outputs: list[Path] = []
    for batch in parse_batches(args.batches):
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
        actual_output_names = [value.name for value in check.graph.output]
        if actual_output_names != output_names:
            raise RuntimeError(
                f"exported ONNX output contract mismatch: expected={output_names}, got={actual_output_names}"
            )
        print(f"batch={batch}: {output}")
        print(f"           {output_data}")
        outputs.append(output)

    print(f"Exported {len(outputs)} {mode} RITnet variant(s) from unchanged weights: {weights}")
    if args.final_uncertainty:
        print(
            "Final four-class probability and uncertainty maps are transient runtime outputs only; "
            "production code must reduce them to compact source-valid per-eye summaries and must "
            "not persist them per frame."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
