from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.rgb.paths import RGBOutputLayout


FACE_CONTINUOUS_QC_SCHEMA = "rgb-face-continuous-qc-v0.1"
PYFEAT_AU_RE = re.compile(r"AU(\d+)$")


def _safe_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _temporal_metrics(table: pd.DataFrame, columns: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    dt = pd.to_numeric(table.get("dt_ms"), errors="coerce") / 1000.0
    gap = table.get("temporal_gap", pd.Series(False, index=table.index)).fillna(False).astype(bool)
    for col in columns:
        if col not in table.columns:
            continue
        x = pd.to_numeric(table[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        prev = x.shift(1)
        pair = x.notna() & prev.notna() & (~gap) & dt.notna() & (dt > 0)
        diffs = (x - prev).where(pair).dropna()
        rates = ((x - prev).abs() / dt).where(pair).dropna()
        lag1 = None
        if int(pair.sum()) >= 3:
            lag1 = _safe_float(x[pair].corr(prev[pair]))
        result[col] = {
            "valid_fraction": float(x.notna().mean()) if len(x) else None,
            "unique_values": int(x.dropna().nunique()),
            "std": _safe_float(x.std()),
            "lag1_correlation": lag1,
            "median_abs_step": _safe_float(diffs.abs().median()),
            "p95_abs_step": _safe_float(diffs.abs().quantile(0.95)) if not diffs.empty else None,
            "max_abs_step": _safe_float(diffs.abs().max()) if not diffs.empty else None,
            "median_abs_rate_per_sec": _safe_float(rates.median()) if not rates.empty else None,
            "valid_step_pairs": int(pair.sum()),
        }
    return result


def _parse_headpose(value) -> tuple[float | None, float | None, float | None]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None, None
    obj = value
    if isinstance(value, str):
        try:
            obj = json.loads(value)
        except Exception:
            obj = value
    if isinstance(obj, dict):
        return _safe_float(obj.get("pitch")), _safe_float(obj.get("yaw")), _safe_float(obj.get("roll"))
    text = str(obj)
    found = {}
    for key in ("pitch", "yaw", "roll"):
        match = re.search(rf"{key}\s*:\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        found[key] = float(match.group(1)) if match else None
    return found["pitch"], found["yaw"], found["roll"]


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _prepare_pyfeat(sample: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    input_col = next((c for c in ("input", "image_path", "file", "filename") if c in raw.columns), None)
    if input_col is None:
        raise ValueError("Py-Feat raw output has no recognizable input-path column")
    work = raw.copy()
    work["_image_name"] = work[input_col].astype(str).map(lambda p: Path(p).name)
    score = pd.to_numeric(work.get("FaceScore"), errors="coerce")
    work["_score_sort"] = score.fillna(-np.inf)
    counts = work.groupby("_image_name").size().rename("face_count")
    primary = work.sort_values(["_image_name", "_score_sort"], ascending=[True, False]).groupby("_image_name", as_index=False).first()

    out = sample.copy()
    out["_image_name"] = out["image_path"].astype(str).map(lambda p: Path(p).name)
    out = out.merge(counts, left_on="_image_name", right_index=True, how="left")
    out["face_count"] = out["face_count"].fillna(0).astype(int)
    keep = [c for c in primary.columns if c not in {"_score_sort"}]
    out = out.merge(primary[keep], on="_image_name", how="left", suffixes=("", "__pyfeat"))

    au_cols = [c for c in raw.columns if PYFEAT_AU_RE.fullmatch(str(c))]
    gaze_cols = [c for c in ("gaze_pitch", "gaze_yaw", "gaze_angle") if c in out.columns]
    head_cols = [c for c in ("Pitch", "Roll", "Yaw") if c in out.columns]
    emotion_cols = [c for c in ("Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger") if c in out.columns]
    summary = {
        "images_with_face": int((out["face_count"] > 0).sum()),
        "images_without_face": int((out["face_count"] == 0).sum()),
        "images_with_multiple_faces": int((out["face_count"] > 1).sum()),
        "coverage_fraction": float((out["face_count"] > 0).mean()),
        "au": _temporal_metrics(out, au_cols),
        "gaze": _temporal_metrics(out, gaze_cols),
        "head_pose": _temporal_metrics(out, head_cols),
        "emotion": _temporal_metrics(out, emotion_cols),
    }
    return out.drop(columns=["_image_name"], errors="ignore"), summary


def _prepare_libreface(
    sample: pd.DataFrame,
    alignment: pd.DataFrame,
    au_detection: pd.DataFrame,
    au_intensity: pd.DataFrame,
    expression: pd.DataFrame,
    gaze: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    out = sample.merge(alignment, on="benchmark_index", how="left", validate="one_to_one")
    out["alignment_success"] = out["alignment_success"].fillna(False).astype(bool)
    for prefix, comp in (("det", au_detection), ("int", au_intensity), ("expr", expression), ("gaze", gaze)):
        ren = {c: f"{prefix}__{c}" for c in comp.columns if c != "benchmark_index"}
        out = out.merge(comp.rename(columns=ren), on="benchmark_index", how="left")

    parsed = out.get("headpose_json", pd.Series(index=out.index, dtype=object)).map(_parse_headpose)
    out["lf_pitch"] = [v[0] for v in parsed]
    out["lf_yaw"] = [v[1] for v in parsed]
    out["lf_roll"] = [v[2] for v in parsed]

    int_cols = [c for c in out.columns if c.startswith("int__au_") and c.endswith("_intensity")]
    det_cols = [c for c in out.columns if c.startswith("det__au_")]
    gaze_cols = [c for c in ("gaze__gaze_pitch", "gaze__gaze_yaw") if c in out.columns]
    expr_cols = [c for c in out.columns if c.startswith("expr__")]

    expression_change_fraction = None
    if expr_cols:
        label = out[expr_cols[0]].astype("string")
        valid_pair = label.notna() & label.shift(1).notna() & (~out.get("temporal_gap", False).fillna(False).astype(bool))
        if valid_pair.any():
            expression_change_fraction = float((label[valid_pair] != label.shift(1)[valid_pair]).mean())

    summary = {
        "aligned_faces": int(out["alignment_success"].sum()),
        "alignment_failures": int((~out["alignment_success"]).sum()),
        "coverage_fraction": float(out["alignment_success"].mean()),
        "au_intensity": _temporal_metrics(out, int_cols),
        "au_detection": _temporal_metrics(out, det_cols),
        "gaze": _temporal_metrics(out, gaze_cols),
        "head_pose": _temporal_metrics(out, ["lf_pitch", "lf_roll", "lf_yaw"]),
        "expression_change_fraction": expression_change_fraction,
    }
    return out, summary


def _rank_corr(x: pd.Series, y: pd.Series) -> float | None:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    mask = a.notna() & b.notna()
    if int(mask.sum()) < 5 or a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return None
    return _safe_float(a[mask].rank().corr(b[mask].rank()))


def _agreement(pyfeat: pd.DataFrame, libreface: pd.DataFrame) -> dict[str, object]:
    merged = pyfeat.merge(
        libreface[[c for c in libreface.columns if c == "benchmark_index" or c.startswith("int__") or c.startswith("gaze__") or c.startswith("lf_")]],
        on="benchmark_index",
        how="inner",
    )
    au: dict[str, object] = {}
    for col in [c for c in pyfeat.columns if PYFEAT_AU_RE.fullmatch(str(c))]:
        number = int(PYFEAT_AU_RE.fullmatch(str(col)).group(1))
        lf_col = f"int__au_{number}_intensity"
        if lf_col in merged.columns:
            au[col] = {
                "libreface_column": lf_col,
                "spearman_rank_correlation": _rank_corr(merged[col], merged[lf_col]),
            }
    gaze = {}
    for py_col, lf_col in (("gaze_pitch", "gaze__gaze_pitch"), ("gaze_yaw", "gaze__gaze_yaw")):
        if py_col in merged.columns and lf_col in merged.columns:
            gaze[py_col] = {"libreface_column": lf_col, "spearman_rank_correlation": _rank_corr(merged[py_col], merged[lf_col])}
    pose = {}
    for py_col, lf_col in (("Pitch", "lf_pitch"), ("Roll", "lf_roll"), ("Yaw", "lf_yaw")):
        if py_col in merged.columns and lf_col in merged.columns:
            pose[py_col] = {"libreface_column": lf_col, "spearman_rank_correlation": _rank_corr(merged[py_col], merged[lf_col])}
    return {
        "note": "Cross-model correlations are descriptive agreement only, not accuracy; models may use different scales and preprocessing.",
        "common_au_intensity": au,
        "gaze": gaze,
        "head_pose": pose,
    }


def run_face_continuous_qc(config: Config, subject: str) -> dict[str, object]:
    layout = RGBOutputLayout.from_config(config)
    root = layout.test_dir() / "face-continuous" / subject
    manifests = sorted(root.glob("*_face-continuous_frames.csv"))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one continuous frame manifest in {root}, found {len(manifests)}")
    sample = pd.read_csv(manifests[0])
    if sample.empty or "benchmark_index" not in sample.columns:
        raise ValueError("Continuous frame manifest is invalid")

    required = {
        "pyfeat": root / "pyfeat_raw.parquet",
        "alignment": root / "libreface_alignment.parquet",
        "au_detection": root / "libreface_au_detection.parquet",
        "au_intensity": root / "libreface_au_intensity.parquet",
        "expression": root / "libreface_expression.parquet",
        "gaze": root / "libreface_gaze.parquet",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Continuous Face candidate outputs missing: {missing}")

    py_raw = pd.read_parquet(required["pyfeat"])
    py_table, py_summary = _prepare_pyfeat(sample, py_raw)
    lf_table, lf_summary = _prepare_libreface(
        sample,
        pd.read_parquet(required["alignment"]),
        pd.read_parquet(required["au_detection"]),
        pd.read_parquet(required["au_intensity"]),
        pd.read_parquet(required["expression"]),
        pd.read_parquet(required["gaze"]),
    )

    py_runtime = _load_json(root / "pyfeat_benchmark_manifest.json")
    lf_runtime = _load_json(root / "libreface_benchmark_manifest.json")
    lf_end_to_end = None
    if lf_runtime is not None:
        lf_end_to_end = lf_runtime.get("alignment_reused") is False

    summary = {
        "schema_version": FACE_CONTINUOUS_QC_SCHEMA,
        "subject": subject,
        "frame_manifest": str(manifests[0]),
        "sample_rows": int(len(sample)),
        "sample_median_dt_ms": _safe_float(pd.to_numeric(sample.get("dt_ms"), errors="coerce").median()),
        "sample_temporal_gap_rows": int(sample.get("temporal_gap", pd.Series(False, index=sample.index)).fillna(False).astype(bool).sum()),
        "pyfeat": py_summary,
        "libreface": lf_summary,
        "cross_model_agreement": _agreement(py_table, lf_table),
        "runtime": {
            "pyfeat": py_runtime,
            "libreface": lf_runtime,
            "libreface_runtime_is_fresh_end_to_end": lf_end_to_end,
        },
        "interpretation": (
            "Temporal metrics quantify continuity/jitter and output variability on one contiguous window. "
            "They do not establish scientific accuracy without ground truth. Cross-model correlations are descriptive only."
        ),
    }

    py_out = root / "pyfeat_continuous_primary.csv"
    lf_out = root / "libreface_continuous_merged.csv"
    json_out = root / "face_continuous_qc.json"
    py_table.to_csv(py_out, index=False, encoding="utf-8-sig")
    lf_table.to_csv(lf_out, index=False, encoding="utf-8-sig")
    summary["pyfeat_primary_output"] = str(py_out)
    summary["libreface_merged_output"] = str(lf_out)
    summary["output"] = str(json_out)
    json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
