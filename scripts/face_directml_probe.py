from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_batch_sizes(value: str) -> list[int]:
    values = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n < 1:
            raise argparse.ArgumentTypeError("batch sizes must be positive integers")
        values.append(n)
    if not values:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    return list(dict.fromkeys(values))


def _np_dtype(ort_type: str) -> np.dtype:
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
        "tensor(bool)": np.bool_,
    }
    try:
        return np.dtype(mapping[ort_type])
    except KeyError as exc:
        raise RuntimeError(f"Unsupported ONNX input type for probe: {ort_type}") from exc


def _resolve_shape(shape: list[Any], batch_size: int) -> tuple[int, ...]:
    resolved: list[int] = []
    for i, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        elif i == 0:
            resolved.append(batch_size)
        else:
            raise RuntimeError(
                f"Probe only supports dynamic batch dimension; unresolved dim at axis {i}: {shape}"
            )
    return tuple(resolved)


def _make_input(shape: tuple[int, ...], dtype: np.dtype, mode: str, seed: int) -> np.ndarray:
    if mode == "zeros":
        return np.zeros(shape, dtype=dtype)
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.floating):
        return rng.standard_normal(shape).astype(dtype, copy=False)
    if np.issubdtype(dtype, np.integer):
        return rng.integers(0, 4, size=shape, dtype=dtype)
    if dtype == np.dtype(np.bool_):
        return (rng.random(shape) > 0.5).astype(dtype)
    raise RuntimeError(f"Unsupported probe dtype: {dtype}")


def _profile_provider_counts(profile_path: Path) -> dict[str, Any]:
    try:
        events = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"profile_parse_error": f"{type(exc).__name__}: {exc}"}

    provider_counts: Counter[str] = Counter()
    op_counts_by_provider: dict[str, Counter[str]] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        args = event.get("args") or {}
        if not isinstance(args, dict):
            continue
        provider = args.get("provider")
        if not provider:
            continue
        provider = str(provider)
        provider_counts[provider] += 1
        op_name = args.get("op_name") or args.get("op_type") or event.get("name") or "unknown"
        op_counts_by_provider.setdefault(provider, Counter())[str(op_name)] += 1

    cpu_kernel_events = int(provider_counts.get("CPUExecutionProvider", 0))
    dml_kernel_events = int(provider_counts.get("DmlExecutionProvider", 0))
    return {
        "kernel_event_count_by_provider": dict(provider_counts),
        "top_ops_by_provider": {
            provider: counts.most_common(20) for provider, counts in op_counts_by_provider.items()
        },
        "cpu_kernel_events": cpu_kernel_events,
        "dml_kernel_events": dml_kernel_events,
        "cpu_fallback_observed": cpu_kernel_events > 0,
    }


def _create_session(model_path: Path, batch_size: int, device_id: int, enable_profile: bool):
    import onnxruntime as ort

    available = ort.get_available_providers()
    if "DmlExecutionProvider" not in available:
        raise RuntimeError(
            "DmlExecutionProvider is unavailable. Install onnxruntime-directml only "
            "and confirm the AMD/DirectX 12 device is visible before benchmarking. "
            f"Available providers: {available}"
        )

    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if enable_profile:
        so.enable_profiling = True
        so.profile_file_prefix = str(model_path.parent / f"ort_profile_{model_path.stem}_b{batch_size}")
    if hasattr(so, "add_free_dimension_override_by_name"):
        try:
            so.add_free_dimension_override_by_name("batch", int(batch_size))
        except Exception:
            pass

    dml_provider: Any
    if device_id == 0:
        dml_provider = "DmlExecutionProvider"
    else:
        dml_provider = ("DmlExecutionProvider", {"device_id": str(device_id)})

    # CPU remains registered second so unsupported graph nodes can still be surfaced
    # explicitly in profiling. However, disable the Python InferenceSession wrapper's
    # provider-level fallback: if DML session creation itself fails, we want the error
    # rather than a silently recreated CPU-only session.
    session = ort.InferenceSession(
        str(model_path),
        sess_options=so,
        providers=[dml_provider, "CPUExecutionProvider"],
        enable_fallback=0,
    )
    session.disable_fallback()
    return ort, session


def _benchmark_model(
    model_path: Path,
    batch_size: int,
    warmup: int,
    iterations: int,
    device_id: int,
    input_mode: str,
    seed: int,
) -> dict[str, Any]:
    ort, session = _create_session(model_path, batch_size, device_id, enable_profile=True)
    inputs_meta = session.get_inputs()
    feeds: dict[str, np.ndarray] = {}
    input_specs = []
    for index, meta in enumerate(inputs_meta):
        shape = _resolve_shape(list(meta.shape), batch_size)
        dtype = _np_dtype(meta.type)
        feeds[meta.name] = _make_input(shape, dtype, input_mode, seed + index)
        input_specs.append({"name": meta.name, "shape": list(meta.shape), "resolved_shape": list(shape), "type": meta.type})

    for _ in range(warmup):
        session.run(None, feeds)

    t0 = time.perf_counter()
    output_values = None
    for _ in range(iterations):
        output_values = session.run(None, feeds)
    elapsed = time.perf_counter() - t0

    profile_file = Path(session.end_profiling())
    profile_summary = _profile_provider_counts(profile_file)
    outputs_meta = session.get_outputs()
    output_specs = []
    if output_values is None:
        output_values = []
    for meta, value in zip(outputs_meta, output_values):
        arr = np.asarray(value)
        output_specs.append(
            {
                "name": meta.name,
                "declared_shape": list(meta.shape),
                "actual_shape": list(arr.shape),
                "type": meta.type,
                "finite_fraction": float(np.isfinite(arr).mean()) if np.issubdtype(arr.dtype, np.number) and arr.size else None,
            }
        )

    processed = batch_size * iterations
    return {
        "batch_size": batch_size,
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "timed_sec": elapsed,
        "images_per_sec_model_core": processed / elapsed if elapsed > 0 else None,
        "latency_ms_per_batch": elapsed * 1000.0 / iterations if iterations else None,
        "latency_ms_per_image": elapsed * 1000.0 / processed if processed else None,
        "session_providers": session.get_providers(),
        "session_provider_options": session.get_provider_options(),
        "python_wrapper_fallback_enabled": False,
        "provider_list_semantics": "session.get_providers() lists registered providers; use profile kernel counts as execution evidence.",
        "inputs": input_specs,
        "outputs": output_specs,
        "profile_file": str(profile_file),
        "profile": profile_summary,
        "onnxruntime_version": ort.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gate-1 ONNX Runtime DirectML probe for RGB Face candidate models. "
            "This is a model-core/provider/fallback smoke benchmark, not the final 300-frame parity benchmark."
        )
    )
    parser.add_argument("--model", action="append", required=True, help="ONNX model path; repeat for multiple models")
    parser.add_argument("--output", required=True, help="JSON manifest output path")
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=[1, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--input-mode", choices=["random", "zeros"], default="random")
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be >=0 and iterations must be >=1")

    models = [Path(p).resolve() for p in args.model]
    for path in models:
        if not path.exists():
            raise FileNotFoundError(path)

    import onnxruntime as ort

    summary: dict[str, Any] = {
        "schema_version": "rgb-face-directml-probe-v0.2",
        "scope": "gate1_model_core_provider_fallback_smoke_only",
        "warning": "Synthetic inputs are used here. Do not compare these speeds to the saved 300-frame CPU reference or use them to freeze the Face backend.",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "device_id": args.device_id,
        },
        "settings": {
            "batch_sizes": args.batch_sizes,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "input_mode": args.input_mode,
            "seed": args.seed,
            "enable_mem_pattern": False,
            "execution_mode": "ORT_SEQUENTIAL",
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "python_wrapper_fallback_enabled": False,
        },
        "models": [],
    }

    for model_path in models:
        model_record: dict[str, Any] = {
            "path": str(model_path),
            "sha256": _sha256(model_path),
            "size_bytes": model_path.stat().st_size,
            "batch_results": [],
        }
        for batch_size in args.batch_sizes:
            print(f"[directml] {model_path.name}: batch={batch_size}")
            try:
                result = _benchmark_model(
                    model_path=model_path,
                    batch_size=batch_size,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device_id=args.device_id,
                    input_mode=args.input_mode,
                    seed=args.seed,
                )
                result["status"] = "ok"
            except Exception as exc:
                result = {
                    "batch_size": batch_size,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            model_record["batch_results"].append(result)
        summary["models"].append(model_record)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
