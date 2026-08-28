"""CUDA transport benchmark for the frozen NVIDIA RITnet v8 model.

Validation-only tool. It does not alter production inference, scientific formulas,
Topology pupil geometry, schema, QC, or completion behavior.

It compares three execution paths for the exact same fixed-B16 FP32 RITnet graph:

1. baseline: ordinary ``InferenceSession.run``;
2. iobinding: fixed CUDA input/output OrtValues with I/O Binding;
3. cudagraph: the same fixed CUDA buffers plus ORT CUDA Graph replay.

All modes request the production cohort outputs ``labels`` and
``class_probability``. The benchmark includes host->device input update and
GPU->host output retrieval in the measured path so the comparison reflects the
current production transport contract. Hard labels must match bit-for-bit and
class probabilities must remain within the configured absolute tolerance.

The default 128 timed batches equal 2048 eyes at fixed batch size 16.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from cuda_runtime import CUDA_PROVIDER, _import_onnxruntime, parse_device_id


FIXED_BATCH_SIZE = 16
INPUT_SHAPE = (16, 1, 400, 640)
LABELS_SHAPE = (16, 400, 640)
PROBABILITY_SHAPE = (16, 4, 400, 640)
OUTPUT_NAMES = ("labels", "class_probability")
DEFAULT_MODEL = Path("models/ritnet-b16-fp32-uncertainty.onnx")
MODES = ("baseline", "iobinding", "cudagraph")


def _preload_runtime_dlls(ort: Any) -> None:
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload(directory="")
    except TypeError:
        preload()


def _create_strict_session(model: Path, device_id: int, *, cuda_graph: bool) -> Any:
    ort = _import_onnxruntime()
    _preload_runtime_dlls(ort)
    available = list(ort.get_available_providers())
    if CUDA_PROVIDER not in available:
        raise RuntimeError(
            f"{CUDA_PROVIDER} unavailable; available providers={available}"
        )

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    provider_options = {
        "device_id": str(int(device_id)),
        "use_tf32": "0",
    }
    if cuda_graph:
        provider_options["enable_cuda_graph"] = "1"

    session = ort.InferenceSession(
        str(model),
        sess_options=options,
        providers=[(CUDA_PROVIDER, provider_options)],
    )
    session.disable_fallback()
    active = list(session.get_providers())
    if not active or active[0] != CUDA_PROVIDER:
        raise RuntimeError(
            "CUDAExecutionProvider did not become primary; "
            f"active providers={active}"
        )

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise RuntimeError(f"expected one RITnet input, got {len(inputs)}")
    if tuple(inputs[0].shape) != INPUT_SHAPE or inputs[0].type != "tensor(float)":
        raise RuntimeError(
            f"unexpected RITnet input contract: {inputs[0].type} {inputs[0].shape}"
        )
    output_by_name = {item.name: item for item in outputs}
    labels = output_by_name.get("labels")
    probability = output_by_name.get("class_probability")
    if labels is None or probability is None:
        raise RuntimeError("RITnet model is missing labels/class_probability outputs")
    if tuple(labels.shape) != LABELS_SHAPE or labels.type != "tensor(uint8)":
        raise RuntimeError(f"unexpected labels contract: {labels.type} {labels.shape}")
    if (
        tuple(probability.shape) != PROBABILITY_SHAPE
        or probability.type != "tensor(float)"
    ):
        raise RuntimeError(
            "unexpected class_probability contract: "
            f"{probability.type} {probability.shape}"
        )
    return session


def _input_pool(count: int, seed: int) -> list[np.ndarray]:
    if count < 1:
        raise ValueError("pool batch count must be >=1")
    rng = np.random.default_rng(int(seed))
    pool: list[np.ndarray] = []
    for _ in range(int(count)):
        tensor = rng.uniform(-1.0, 1.0, size=INPUT_SHAPE).astype(np.float32)
        pool.append(np.ascontiguousarray(tensor))
    return pool


def _run_baseline(session: Any, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = session.run(list(OUTPUT_NAMES), {session.get_inputs()[0].name: tensor})
    if len(raw) != 2:
        raise RuntimeError(f"baseline returned {len(raw)} outputs")
    labels = np.asarray(raw[0])
    probability = np.asarray(raw[1])
    if labels.shape != LABELS_SHAPE or labels.dtype != np.uint8:
        raise RuntimeError(f"baseline labels mismatch: {labels.shape} {labels.dtype}")
    if probability.shape != PROBABILITY_SHAPE or probability.dtype != np.float32:
        raise RuntimeError(
            f"baseline probability mismatch: {probability.shape} {probability.dtype}"
        )
    return labels, probability


class _BoundRunner:
    def __init__(self, session: Any, device_id: int, *, cuda_graph: bool) -> None:
        ort = _import_onnxruntime()
        self.session = session
        self.cuda_graph = bool(cuda_graph)
        self.input_name = session.get_inputs()[0].name
        self.input_value = ort.OrtValue.ortvalue_from_shape_and_type(
            list(INPUT_SHAPE), np.float32, "cuda", int(device_id)
        )
        self.labels_value = ort.OrtValue.ortvalue_from_shape_and_type(
            list(LABELS_SHAPE), np.uint8, "cuda", int(device_id)
        )
        self.probability_value = ort.OrtValue.ortvalue_from_shape_and_type(
            list(PROBABILITY_SHAPE), np.float32, "cuda", int(device_id)
        )
        self.binding = session.io_binding()
        self.binding.bind_ortvalue_input(self.input_name, self.input_value)
        self.binding.bind_ortvalue_output("labels", self.labels_value)
        self.binding.bind_ortvalue_output("class_probability", self.probability_value)
        self.run_options = None
        if self.cuda_graph:
            self.run_options = ort.RunOptions()
            self.run_options.add_run_config_entry("gpu_graph_id", "0")

    def run(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.input_value.update_inplace(tensor)
        if self.run_options is None:
            self.session.run_with_iobinding(self.binding)
        else:
            self.session.run_with_iobinding(self.binding, self.run_options)
        # .numpy() is deliberately inside the measured call. Production needs
        # labels and probabilities on CPU for Topology geometry/soft summaries.
        labels = self.labels_value.numpy()
        probability = self.probability_value.numpy()
        return labels, probability


def _parity(
    actual: tuple[np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray],
    *,
    probability_atol: float,
) -> dict[str, Any]:
    actual_labels, actual_probability = actual
    expected_labels, expected_probability = expected
    labels_equal = bool(np.array_equal(actual_labels, expected_labels))
    diff = np.abs(
        np.asarray(actual_probability, dtype=np.float32)
        - np.asarray(expected_probability, dtype=np.float32)
    )
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    probability_ok = bool(max_abs <= float(probability_atol))
    return {
        "labels_exact": labels_equal,
        "probability_max_abs_diff": max_abs,
        "probability_mean_abs_diff": mean_abs,
        "probability_atol": float(probability_atol),
        "probability_within_atol": probability_ok,
        "pass": bool(labels_equal and probability_ok),
    }


def _stats(samples_ms: list[float], batches: int) -> dict[str, Any]:
    if not samples_ms:
        raise ValueError("no timing samples")
    total_ms = float(sum(samples_ms))
    batch_median = float(statistics.median(samples_ms))
    batch_p95 = float(np.percentile(np.asarray(samples_ms, dtype=np.float64), 95))
    eyes = int(batches) * FIXED_BATCH_SIZE
    return {
        "timed_batches": int(batches),
        "timed_eyes": eyes,
        "total_ms": total_ms,
        "mean_ms_per_batch": float(total_ms / batches),
        "median_ms_per_batch": batch_median,
        "p95_ms_per_batch": batch_p95,
        "eyes_per_second": float(eyes / (total_ms / 1000.0)),
    }


def _benchmark_mode(
    mode: str,
    *,
    model: Path,
    device_id: int,
    pool: list[np.ndarray],
    references: list[tuple[np.ndarray, np.ndarray]],
    warmup: int,
    batches: int,
    probability_atol: float,
) -> dict[str, Any]:
    cuda_graph = mode == "cudagraph"
    session = _create_strict_session(model, device_id, cuda_graph=cuda_graph)
    runner: Any
    if mode == "baseline":
        runner = lambda tensor: _run_baseline(session, tensor)
    else:
        bound = _BoundRunner(session, device_id, cuda_graph=cuda_graph)
        runner = bound.run

    parity_rows: list[dict[str, Any]] = []
    for index, tensor in enumerate(pool):
        parity_rows.append(
            _parity(
                runner(tensor),
                references[index],
                probability_atol=probability_atol,
            )
        )
    parity_pass = all(row["pass"] for row in parity_rows)
    parity_summary = {
        "pass": bool(parity_pass),
        "pool_batches_checked": len(parity_rows),
        "labels_exact_all": bool(all(row["labels_exact"] for row in parity_rows)),
        "probability_max_abs_diff": float(
            max(row["probability_max_abs_diff"] for row in parity_rows)
        ),
        "probability_mean_abs_diff_max": float(
            max(row["probability_mean_abs_diff"] for row in parity_rows)
        ),
        "probability_atol": float(probability_atol),
    }
    if not parity_pass:
        return {
            "mode": mode,
            "status": "parity_failed",
            "parity": parity_summary,
        }

    for index in range(int(warmup)):
        runner(pool[index % len(pool)])

    samples_ms: list[float] = []
    for index in range(int(batches)):
        tensor = pool[index % len(pool)]
        started = time.perf_counter()
        runner(tensor)
        samples_ms.append((time.perf_counter() - started) * 1000.0)

    result = {
        "mode": mode,
        "status": "pass",
        "parity": parity_summary,
        "timing": _stats(samples_ms, batches),
    }
    del runner
    del session
    gc.collect()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--pool-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--probability-atol", type=float, default=1e-6)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = args.model.resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    if args.batches < 1 or args.warmup < 0:
        raise ValueError("--batches must be >=1 and --warmup must be >=0")
    device_id = parse_device_id(args.device)
    pool = _input_pool(args.pool_batches, args.seed)

    # Freeze references from the exact ordinary session.run path used by current
    # production before testing alternative transport paths.
    baseline_reference_session = _create_strict_session(
        model, device_id, cuda_graph=False
    )
    references = [_run_baseline(baseline_reference_session, tensor) for tensor in pool]
    del baseline_reference_session
    gc.collect()

    results: list[dict[str, Any]] = []
    for mode in args.modes:
        try:
            result = _benchmark_mode(
                mode,
                model=model,
                device_id=device_id,
                pool=pool,
                references=references,
                warmup=args.warmup,
                batches=args.batches,
                probability_atol=args.probability_atol,
            )
        except Exception as exc:
            result = {
                "mode": mode,
                "status": "failed_runtime",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        results.append(result)

    baseline_rate = None
    for result in results:
        if result.get("mode") == "baseline" and result.get("status") == "pass":
            baseline_rate = float(result["timing"]["eyes_per_second"])
            break
    if baseline_rate:
        for result in results:
            if result.get("status") == "pass":
                rate = float(result["timing"]["eyes_per_second"])
                result["speedup_vs_baseline"] = float(rate / baseline_rate)

    payload = {
        "benchmark": "ritnet-cuda-transport-v1",
        "model": str(model),
        "device_id": int(device_id),
        "fixed_batch_size": FIXED_BATCH_SIZE,
        "precision": "fp32",
        "use_tf32": False,
        "output_contract": list(OUTPUT_NAMES),
        "timed_batches_per_mode": int(args.batches),
        "timed_eyes_per_mode": int(args.batches) * FIXED_BATCH_SIZE,
        "warmup_batches_per_mode": int(args.warmup),
        "pool_batches": int(args.pool_batches),
        "results": results,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")

    failures = [result for result in results if result.get("status") != "pass"]
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
