from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.rgb.paths import RGBOutputLayout


LIBREFACE_QC_SCHEMA_VERSION = "rgb-face-benchmark-libreface-qc-v0.1"


def _component_summary(table: pd.DataFrame) -> dict[str, object]:
    if table.empty:
        return {
            "rows": 0,
            "columns": [],
            "column_count": 0,
            "row_any_valid_fraction": None,
            "cell_valid_fraction": None,
            "constant_numeric_columns": [],
        }
    columns = [str(c) for c in table.columns if c != "benchmark_index"]
    subset = table[columns] if columns else pd.DataFrame(index=table.index)
    valid = subset.notna() if columns else pd.DataFrame(index=table.index)
    constant_numeric: list[str] = []
    for col in columns:
        values = pd.to_numeric(table[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not values.empty and values.nunique(dropna=True) <= 1:
            constant_numeric.append(col)
    return {
        "rows": int(len(table)),
        "columns": columns,
        "column_count": len(columns),
        "row_any_valid_fraction": float(valid.any(axis=1).mean()) if columns and len(table) else None,
        "cell_valid_fraction": float(valid.to_numpy().mean()) if columns and subset.size else None,
        "constant_numeric_columns": constant_numeric,
    }


def summarize_libreface_benchmark(
    sample: pd.DataFrame,
    alignment: pd.DataFrame,
    components: dict[str, pd.DataFrame],
) -> tuple[dict[str, object], pd.DataFrame]:
    if sample.empty or "benchmark_index" not in sample.columns or "image_path" not in sample.columns:
        raise ValueError("Benchmark frame manifest is invalid")
    if alignment.empty or "benchmark_index" not in alignment.columns or "alignment_success" not in alignment.columns:
        raise ValueError("LibreFace alignment output is invalid")

    per_image = sample.merge(alignment, on="benchmark_index", how="left", validate="one_to_one")
    per_image["alignment_success"] = per_image["alignment_success"].fillna(False).astype(bool)

    phase_summary: dict[str, object] = {}
    if "phase" in per_image.columns:
        for phase, group in per_image.groupby("phase", dropna=False):
            phase_summary[str(phase)] = {
                "input_images": int(len(group)),
                "aligned_faces": int(group["alignment_success"].sum()),
                "alignment_failures": int((~group["alignment_success"]).sum()),
                "alignment_valid_fraction": float(group["alignment_success"].mean()),
            }

    failed = per_image[~per_image["alignment_success"]].copy()
    failed_records = []
    for row in failed.itertuples(index=False):
        failed_records.append({
            "benchmark_index": int(row.benchmark_index),
            "image_path": str(row.image_path),
            "phase": str(row.phase) if hasattr(row, "phase") else None,
            "alignment_error": None if pd.isna(getattr(row, "alignment_error", None)) else str(getattr(row, "alignment_error")),
        })

    aligned_indices = set(
        pd.to_numeric(
            alignment.loc[alignment["alignment_success"].fillna(False).astype(bool), "benchmark_index"],
            errors="coerce",
        ).dropna().astype(int).tolist()
    )

    component_summaries: dict[str, object] = {}
    for name, table in components.items():
        item = _component_summary(table)
        if "benchmark_index" in table.columns:
            component_indices = set(
                pd.to_numeric(table["benchmark_index"], errors="coerce").dropna().astype(int).tolist()
            )
            item["benchmark_index_unique"] = int(len(component_indices))
            item["covers_all_aligned_indices"] = component_indices == aligned_indices
            item["missing_aligned_indices"] = sorted(aligned_indices - component_indices)[:50]
            item["unexpected_indices"] = sorted(component_indices - aligned_indices)[:50]
        component_summaries[name] = item

    headpose_fraction = None
    landmarks_fraction = None
    if "headpose_json" in per_image.columns:
        headpose_fraction = float(per_image.loc[per_image["alignment_success"], "headpose_json"].notna().mean()) if per_image["alignment_success"].any() else None
    if "landmarks_json" in per_image.columns:
        landmarks_fraction = float(per_image.loc[per_image["alignment_success"], "landmarks_json"].notna().mean()) if per_image["alignment_success"].any() else None

    summary = {
        "schema_version": LIBREFACE_QC_SCHEMA_VERSION,
        "expected_input_images": int(len(sample)),
        "aligned_faces": int(per_image["alignment_success"].sum()),
        "alignment_failures": int((~per_image["alignment_success"]).sum()),
        "alignment_valid_fraction": float(per_image["alignment_success"].mean()),
        "phase": phase_summary,
        "alignment_outputs": {
            "headpose_json_valid_fraction_among_aligned": headpose_fraction,
            "landmarks_json_valid_fraction_among_aligned": landmarks_fraction,
        },
        "components": component_summaries,
        "failed_inputs": failed_records,
        "interpretation": (
            "This first-round QC checks sparse shared benchmark frames, alignment coverage and native component output completeness. "
            "It does not establish temporal smoothness or scientific accuracy. LibreFace timing from a manifest with alignment_reused=true "
            "excludes first-pass face-alignment cost and must not be compared as total end-to-end wall time against Py-Feat."
        ),
    }
    return summary, per_image


def run_libreface_benchmark_qc(config: Config, subject: str) -> dict[str, object]:
    layout = RGBOutputLayout.from_config(config)
    root = layout.test_dir() / "face-benchmark" / subject
    frame_manifests = sorted(root.glob("*_face-benchmark_frames.csv"))
    if len(frame_manifests) != 1:
        raise RuntimeError(f"Expected one benchmark frame manifest in {root}, found {len(frame_manifests)}")

    alignment_path = root / "libreface_alignment.parquet"
    if not alignment_path.exists():
        raise FileNotFoundError(f"LibreFace alignment output not found: {alignment_path}")

    component_paths = {
        "au_detection": root / "libreface_au_detection.parquet",
        "au_intensity": root / "libreface_au_intensity.parquet",
        "expression": root / "libreface_expression.parquet",
        "gaze": root / "libreface_gaze.parquet",
    }
    missing = [str(path) for path in component_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"LibreFace component outputs missing: {missing}")

    sample = pd.read_csv(frame_manifests[0])
    alignment = pd.read_parquet(alignment_path)
    components = {name: pd.read_parquet(path) for name, path in component_paths.items()}
    summary, per_image = summarize_libreface_benchmark(sample, alignment, components)
    summary["subject"] = subject
    summary["frame_manifest"] = str(frame_manifests[0])
    summary["alignment_parquet"] = str(alignment_path)

    manifest_path = root / "libreface_benchmark_manifest.json"
    if manifest_path.exists():
        try:
            run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            run_manifest = None
        if isinstance(run_manifest, dict):
            summary["benchmark_runtime_manifest"] = {
                "alignment_reused": run_manifest.get("alignment_reused"),
                "timing_sec": run_manifest.get("timing_sec"),
                "input_images_per_sec_total_reported": run_manifest.get("input_images_per_sec_total"),
                "runtime": run_manifest.get("runtime"),
            }

    per_image_path = root / "libreface_qc_per_image.csv"
    json_path = root / "libreface_qc.json"
    per_image.to_csv(per_image_path, index=False, encoding="utf-8-sig")
    summary["per_image_output"] = str(per_image_path)
    summary["output"] = str(json_path)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
