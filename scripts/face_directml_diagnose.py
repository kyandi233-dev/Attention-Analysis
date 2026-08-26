from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


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
    if ort_type not in mapping:
        raise RuntimeError(f"Unsupported ONNX input type: {ort_type}")
    return np.dtype(mapping[ort_type])


def _resolve_shape(shape: list[Any], batch_size: int) -> tuple[int, ...]:
    resolved: list[int] = []
    for i, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        elif i == 0:
            resolved.append(batch_size)
        else:
            raise RuntimeError(f"Unresolved non-batch dimension at axis {i}: {shape}")
    return tuple(resolved)


def _make_feeds(session: Any, batch_size: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feeds: dict[str, np.ndarray] = {}
    for meta in session.get_inputs():
        shape = _resolve_shape(list(meta.shape), batch_size)
        dtype = _np_dtype(meta.type)
        if np.issubdtype(dtype, np.floating):
            value = rng.standard_normal(shape).astype(dtype, copy=False)
        elif np.issubdtype(dtype, np.integer):
            value = rng.integers(0, 4, size=shape, dtype=dtype)
        elif dtype == np.dtype(np.bool_):
            value = (rng.random(shape) > 0.5).astype(dtype)
        else:
            raise RuntimeError(f"Unsupported dtype: {dtype}")
        feeds[meta.name] = value
    return feeds


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
            provider: counts.most_common(30) for provider, counts in op_counts_by_provider.items()
        },
        "cpu_kernel_events": cpu_kernel_events,
        "dml_kernel_events": dml_kernel_events,
        "cpu_execution_observed": cpu_kernel_events > 0,
    }


def _session_options(
    ort: Any,
    batch_size: int,
    *,
    strict_dml: bool,
    verbose: bool,
    profile_prefix: Path,
) -> Any:
    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_profiling = True
    so.profile_file_prefix = str(profile_prefix)
    if hasattr(so, "add_free_dimension_override_by_name"):
        try:
            so.add_free_dimension_override_by_name("batch", int(batch_size))
        except Exception:
            pass
    if strict_dml:
        # ORT core-layer switch: fail graph assignment if unsupported nodes require CPU execution.
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if verbose:
        so.log_severity_level = 0
        so.log_verbosity_level = 1
    return so


def _try_session(
    model_path: Path,
    batch_size: int,
    strict_dml: bool,
    verbose: bool,
    seed: int,
    profile_prefix: Path,
) -> dict[str, Any]:
    import onnxruntime as ort

    providers = ["DmlExecutionProvider"] if strict_dml else ["DmlExecutionProvider", "CPUExecutionProvider"]
    record: dict[str, Any] = {
        "mode": "strict_dml_no_cpu_fallback" if strict_dml else "dml_with_cpu_fallback",
        "requested_providers": providers,
        "python_wrapper_fallback_enabled": not strict_dml,
        "cpu_ep_graph_fallback_disabled": strict_dml,
        "provider_list_semantics": (
            "session.get_providers() is the registered-provider list, not proof of which provider executed nodes; "
            "use the ORT profile counts below as execution evidence."
        ),
    }
    session = None
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=_session_options(
                ort,
                batch_size,
                strict_dml=strict_dml,
                verbose=verbose,
                profile_prefix=profile_prefix,
            ),
            providers=providers,
            enable_fallback=0 if strict_dml else 1,
        )
        if strict_dml:
            session.disable_fallback()

        record["session_providers"] = session.get_providers()
        record["session_provider_options"] = session.get_provider_options()
        feeds = _make_feeds(session, batch_size, seed)
        outputs = session.run(None, feeds)
        record["status"] = "ok"
        record["outputs"] = [
            {
                "name": meta.name,
                "shape": list(np.asarray(value).shape),
                "finite_fraction": float(np.isfinite(np.asarray(value)).mean())
                if np.issubdtype(np.asarray(value).dtype, np.number) and np.asarray(value).size
                else None,
            }
            for meta, value in zip(session.get_outputs(), outputs)
        ]
        profile_file = Path(session.end_profiling())
        record["profile_file"] = str(profile_file)
        record["profile"] = _profile_provider_counts(profile_file)
    except Exception as exc:
        record["status"] = "error"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
        record["error_repr"] = repr(exc)
        if session is not None:
            try:
                profile_file = Path(session.end_profiling())
                record["profile_file"] = str(profile_file)
                record["profile"] = _profile_provider_counts(profile_file)
            except Exception as profile_exc:
                record["profile_end_error"] = f"{type(profile_exc).__name__}: {profile_exc}"
    return record


def _onnx_inventory(model_path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    op_counts: Counter[str] = Counter()
    domain_op_counts: Counter[str] = Counter()
    for node in model.graph.node:
        op_counts[node.op_type] += 1
        domain = node.domain or "ai.onnx"
        domain_op_counts[f"{domain}::{node.op_type}"] += 1
    return {
        "ir_version": model.ir_version,
        "opsets": [{"domain": x.domain or "ai.onnx", "version": x.version} for x in model.opset_import],
        "node_count": len(model.graph.node),
        "op_type_counts": dict(op_counts.most_common()),
        "domain_op_type_counts": dict(domain_op_counts.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose an ONNX model's actual DirectML execution. Runs a normal DML+CPU session and a strict "
            "DML session with both ORT graph-node CPU fallback and Python wrapper fallback disabled, then profiles "
            "which provider actually executed kernels."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose ONNX Runtime native logging in the terminal")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch-size must be >= 1")

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    import onnxruntime as ort

    available = ort.get_available_providers()
    if "DmlExecutionProvider" not in available:
        raise RuntimeError(f"DmlExecutionProvider unavailable: {available}")

    result = {
        "schema_version": "rgb-face-directml-diagnostic-v0.3",
        "purpose": "Identify actual provider execution before changing export/model structure.",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "onnxruntime": ort.__version__,
            "available_providers": available,
        },
        "model": str(model_path),
        "batch_size": args.batch_size,
        "onnx_inventory": _onnx_inventory(model_path),
        "fallback_allowed": _try_session(
            model_path,
            args.batch_size,
            False,
            args.verbose,
            args.seed,
            output.parent / f"ort_diag_fallback_{model_path.stem}_b{args.batch_size}",
        ),
        "strict_dml": _try_session(
            model_path,
            args.batch_size,
            True,
            args.verbose,
            args.seed,
            output.parent / f"ort_diag_strict_{model_path.stem}_b{args.batch_size}",
        ),
        "interpretation": (
            "Do not infer execution from session.get_providers(): it reports registered providers. "
            "Use profile.dml_kernel_events and profile.cpu_kernel_events. If strict mode succeeds with DML kernels > 0 "
            "and CPU kernels = 0, the graph executed without observed CPU kernels. If strict mode errors, use the error/log. "
            "If normal mode executes CPU while strict mode executes DML, investigate provider partition/session-option behavior "
            "before changing the scientific model."
        ),
    }

    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
