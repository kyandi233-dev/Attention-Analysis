from __future__ import annotations

"""Schema-safe real-300 Face parity wrapper.

This v0.3 entrypoint preserves the v0.2 implementation as historical provenance
while fixing two validation-only issues discovered on the first real run:

1. LibreFace CPU and DirectML AU parquet files use different column names for
   the same ordered outputs. v0.2 relied on pandas merge suffixes that are only
   applied to overlapping names, causing KeyError for CPU-only names such as
   ``au_1_intensity``.
2. pandas Spearman correlation delegates to SciPy. The DirectML runtime does not
   need SciPy for inference, so v0.3 computes Spearman as Pearson correlation of
   average ranks using pandas + NumPy only.

No inference, model export, or saved parquet is modified by this wrapper.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import face_real_parity_v02 as base


def _metric_no_scipy(a: Any, b: Any) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "max_abs": None,
            "pearson_r": None,
            "spearman_rho": None,
        }

    a = a[mask]
    b = b[mask]
    delta = a - b
    abs_delta = np.abs(delta)
    pearson = None
    spearman = None

    if len(a) >= 2 and np.std(a) > 0 and np.std(b) > 0:
        pearson = float(np.corrcoef(a, b)[0, 1])
        ar = pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)
        br = pd.Series(b).rank(method="average").to_numpy(dtype=np.float64)
        if np.std(ar) > 0 and np.std(br) > 0:
            spearman = float(np.corrcoef(ar, br)[0, 1])

    return {
        "n": int(len(a)),
        "mae": float(abs_delta.mean()),
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "max_abs": float(abs_delta.max()),
        "pearson_r": pearson,
        "spearman_rho": spearman,
    }


def _group_safe(cpu: pd.DataFrame, dml: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    common = [c for c in cols if c in cpu.columns and c in dml.columns]
    if not common:
        return {"columns": [], "aggregate": _metric_no_scipy([], []), "per_column": {}}
    per = {
        c: _metric_no_scipy(base._numeric(cpu[c]), base._numeric(dml[c]))
        for c in common
    }
    return {
        "columns": common,
        "aggregate": _metric_no_scipy(
            np.concatenate([base._numeric(cpu[c]) for c in common]),
            np.concatenate([base._numeric(dml[c]) for c in common]),
        ),
        "per_column": per,
    }


def _ordered_numeric_safe(cpu_path: Path, dml_path: Path, prefix: str) -> dict[str, Any]:
    cpu = pd.read_parquet(cpu_path)
    dml = pd.read_parquet(dml_path)

    cpu_num = [
        c for c in cpu.columns
        if c != "benchmark_index" and pd.api.types.is_numeric_dtype(cpu[c])
    ]
    dml_num = [
        c for c in dml.columns
        if c != "benchmark_index" and pd.api.types.is_numeric_dtype(dml[c])
    ]

    # Prefer exact semantic names when the two schemas genuinely share them.
    shared = [c for c in cpu_num if c in dml_num]
    if shared:
        cpu_view = cpu[["benchmark_index", *shared]].copy()
        dml_view = dml[["benchmark_index", *shared]].copy()
        merged = cpu_view.merge(
            dml_view,
            on="benchmark_index",
            how="inner",
            suffixes=("__cpu", "__dml"),
            validate="one_to_one",
        )
        a = pd.DataFrame({c: merged[f"{c}__cpu"] for c in shared})
        b = pd.DataFrame({c: merged[f"{c}__dml"] for c in shared})
        result = _group_safe(a, b, shared)
        result["schema_alignment"] = "shared_numeric_column_names"
        result["cpu_columns"] = shared
        result["dml_columns"] = shared
        return result

    # LibreFace AU reference and DML outputs intentionally use different labels
    # but preserve the documented model-output order. Normalize each side to a
    # canonical temporary schema BEFORE merge; do not rely on merge suffixes for
    # non-overlapping source names.
    if len(cpu_num) != len(dml_num):
        return {
            "columns": [],
            "aggregate": _metric_no_scipy([], []),
            "per_column": {},
            "error": f"ordered numeric width mismatch CPU={len(cpu_num)} DML={len(dml_num)}",
            "cpu_columns": cpu_num,
            "dml_columns": dml_num,
        }

    canonical = [f"{prefix}_{i}" for i in range(len(cpu_num))]
    cpu_view = cpu[["benchmark_index", *cpu_num]].copy()
    dml_view = dml[["benchmark_index", *dml_num]].copy()
    cpu_view = cpu_view.rename(columns={src: dst for src, dst in zip(cpu_num, canonical)})
    dml_view = dml_view.rename(columns={src: dst for src, dst in zip(dml_num, canonical)})

    merged = cpu_view.merge(
        dml_view,
        on="benchmark_index",
        how="inner",
        suffixes=("__cpu", "__dml"),
        validate="one_to_one",
    )
    a = pd.DataFrame({c: merged[f"{c}__cpu"] for c in canonical})
    b = pd.DataFrame({c: merged[f"{c}__dml"] for c in canonical})
    result = _group_safe(a, b, canonical)
    result["schema_alignment"] = "ordered_numeric_columns_normalized_before_merge"
    result["cpu_columns"] = cpu_num
    result["dml_columns"] = dml_num
    result["ordered_column_map"] = [
        {"canonical": canonical[i], "cpu": cpu_num[i], "dml": dml_num[i]}
        for i in range(len(canonical))
    ]
    return result


# Patch only validation helpers; all candidate-specific matching and retention
# logic continues to come from the frozen v0.2 implementation.
base._metric = _metric_no_scipy
base._group = _group_safe
base._ordered_numeric = _ordered_numeric_safe


def main() -> None:
    # Reuse the tested v0.2 CLI and candidate logic, then rewrite the schema
    # marker in the produced JSON so downstream records can distinguish the fix.
    import sys

    base.main()

    output = None
    for i, token in enumerate(sys.argv[:-1]):
        if token == "--output":
            output = Path(sys.argv[i + 1]).resolve()
            break
    if output and output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        payload["schema_version"] = "rgb-face-real300-parity-v0.3"
        payload["parity_fix"] = (
            "v0.3 normalizes ordered LibreFace CPU/DML numeric schemas before merge "
            "and computes Spearman without SciPy; no inference outputs were changed."
        )
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
