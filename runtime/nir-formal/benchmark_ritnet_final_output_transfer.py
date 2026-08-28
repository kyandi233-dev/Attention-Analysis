from __future__ import annotations

"""Isolated CUDA benchmark for the current fixed-b16 final RITnet ONNX.

This script does not write scientific outputs and does not touch the checkpoint.
It prepares one real b16 tensor using the canonical source/ROI/preprocess path,
then repeatedly requests different output subsets from the *same* ONNX session:

1. labels only
2. labels + class_probability (current cohort transfer contract)
3. all five qualified outputs

The calls are interleaved to reduce thermal/order bias. Labels and probability
outputs are parity-checked before timings are reported.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ritnet_fullclass_final_engine import _iter_prepared_items, resolve_package_path
from ritnet_fullclass_final_runtime import OUTPUT_NAMES, RitnetFullClassFinalRuntime
from ritnet_fullclass_source import load_source_context
from ritnet_label_store import sha256_file


PACKAGE_ROOT = Path(__file__).resolve().parent
MODES = {
    "labels_only": ("labels",),
    "cohort_current": ("labels", "class_probability"),
    "full_five_output": OUTPUT_NAMES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark output-transfer cost of the current final RITnet b16 ONNX on CUDA"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "nvidia-cuda" / "ritnet-final-output-transfer.json",
    )
    return parser.parse_args()


def _first_full_batch(context) -> list[np.ndarray]:
    rois: list[np.ndarray] = []
    for item in _iter_prepared_items(context, 0):
        roi = item.get("roi")
        if roi is None:
            continue
        rois.append(np.asarray(roi))
        if len(rois) == RitnetFullClassFinalRuntime.FIXED_BATCH_SIZE:
            return rois
    raise RuntimeError("source run does not contain 16 successful ROI crops for benchmark")


def _output_bytes(values) -> int:
    return int(sum(np.asarray(value).nbytes for value in values))


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "calls": int(array.size),
        "mean_ms": float(np.mean(array)),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
    }


def main() -> int:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    run_dir = args.run_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    context = load_source_context(run_dir, config_path)
    model = resolve_package_path(context.config["models"]["ritnet_fullclass_final"]).resolve()
    if not model.is_file():
        raise FileNotFoundError(model)

    runtime = RitnetFullClassFinalRuntime(model, device=str(args.device))
    rois = _first_full_batch(context)
    tensor, valid, prep = runtime.prepare_batch(rois)
    if valid != runtime.FIXED_BATCH_SIZE:
        raise AssertionError(f"benchmark requires a full b16 tensor; got {valid}")

    feed = {runtime.input_name: tensor}
    reference = {}
    output_bytes = {}
    for mode, names in MODES.items():
        values = runtime.session.run(list(names), feed)
        reference[mode] = values
        output_bytes[mode] = _output_bytes(values)

    labels_reference = np.asarray(reference["labels_only"][0])
    for mode in ("cohort_current", "full_five_output"):
        if not np.array_equal(labels_reference, np.asarray(reference[mode][0])):
            raise RuntimeError(f"labels parity failed before timing for mode={mode}")

    compact_probability = np.asarray(reference["cohort_current"][1])
    full_probability = np.asarray(reference["full_five_output"][1])
    if compact_probability.shape != full_probability.shape:
        raise RuntimeError("class_probability shape differs between requested-output modes")
    probability_max_abs = float(
        np.max(np.abs(compact_probability.astype(np.float64) - full_probability.astype(np.float64)))
    )
    if probability_max_abs != 0.0:
        raise RuntimeError(
            "class_probability parity failed before timing: "
            f"max_abs={probability_max_abs}"
        )

    # Warm every requested-output path before measured interleaving.
    mode_names = list(MODES)
    for _ in range(args.warmup):
        for mode in mode_names:
            runtime.session.run(list(MODES[mode]), feed)

    timings: dict[str, list[float]] = {mode: [] for mode in mode_names}
    for iteration in range(args.iterations):
        offset = iteration % len(mode_names)
        order = mode_names[offset:] + mode_names[:offset]
        for mode in order:
            started = time.perf_counter()
            runtime.session.run(list(MODES[mode]), feed)
            timings[mode].append((time.perf_counter() - started) * 1000.0)

    result = {
        "benchmark": "ritnet-final-output-transfer-v1",
        "subject": context.subject,
        "source_run": str(context.run_dir),
        "model": str(model),
        "model_sha256": sha256_file(model),
        "device": str(args.device),
        "providers": list(runtime.providers),
        "batch_size": runtime.FIXED_BATCH_SIZE,
        "input_shape": list(tensor.shape),
        "preprocess_ms_for_reference_batch": float(prep.get("preprocess_ms", 0.0)),
        "warmup_rounds": int(args.warmup),
        "measured_rounds": int(args.iterations),
        "class_probability_requested_mode_max_abs": probability_max_abs,
        "modes": {
            mode: {
                "requested_outputs": list(MODES[mode]),
                "returned_bytes_per_call": output_bytes[mode],
                "returned_mib_per_call": float(output_bytes[mode] / (1024 ** 2)),
                **_stats(timings[mode]),
            }
            for mode in mode_names
        },
    }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Benchmark JSON -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
