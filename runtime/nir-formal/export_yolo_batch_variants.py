from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fixed-batch YOLO26n ONNX variants for DirectML benchmarking."
    )
    parser.add_argument("--pt", required=True, help="Path to nir-eye-yolo26n-best.pt")
    parser.add_argument(
        "--batches",
        default="4,8",
        help="Comma-separated fixed batch sizes to export (default: 4,8)",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Output directory, relative to this script unless absolute",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Ultralytics export device. CPU is sufficient for ONNX export.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pt = Path(args.pt).expanduser().resolve()
    if not pt.is_file():
        raise FileNotFoundError(pt)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = [int(value.strip()) for value in args.batches.split(",") if value.strip()]
    if not batches or any(value <= 0 for value in batches):
        raise ValueError("--batches must contain positive integers")

    model = YOLO(str(pt))
    for batch in batches:
        exported = Path(
            model.export(
                format="onnx",
                imgsz=int(args.imgsz),
                batch=batch,
                dynamic=False,
                device=args.device,
            )
        ).resolve()
        target = output_dir / f"nir-eye-yolo26n-best-b{batch}.onnx"
        shutil.copy2(exported, target)
        print(f"batch={batch}: {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
