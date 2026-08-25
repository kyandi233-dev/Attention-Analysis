from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PYFEAT_AUS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11", "AU12",
    "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU43",
]
PYFEAT_EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
BLENDSHAPES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft",
    "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]


def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def _metric(a: Any, b: Any) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return {"n": 0, "mae": None, "rmse": None, "max_abs": None, "pearson_r": None, "spearman_rho": None}
    a, b = a[mask], b[mask]
    delta = a - b
    abs_delta = np.abs(delta)
    pearson = None
    spearman = None
    if len(a) >= 2 and np.std(a) > 0 and np.std(b) > 0:
        pearson = float(np.corrcoef(a, b)[0, 1])
        spearman = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    return {
        "n": int(len(a)),
        "mae": float(abs_delta.mean()),
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "max_abs": float(abs_delta.max()),
        "pearson_r": pearson,
        "spearman_rho": spearman,
    }


def _group(cpu: pd.DataFrame, dml: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    common = [c for c in cols if c in cpu.columns and c in dml.columns]
    if not common:
        return {"columns": [], "aggregate": _metric([], []), "per_column": {}}
    per = {c: _metric(_numeric(cpu[c]), _numeric(dml[c])) for c in common}
    return {
        "columns": common,
        "aggregate": _metric(
            np.concatenate([_numeric(cpu[c]) for c in common]),
            np.concatenate([_numeric(dml[c]) for c in common]),
        ),
        "per_column": per,
    }


def _iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = [float(x) for x in a]
    bx1, by1, bw, bh = [float(x) for x in b]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0 else float("nan")


def _find_manifest(root: Path) -> Path:
    candidates = list(dict.fromkeys(
        sorted(root.glob("*_face-continuous_frames.csv"))
        + sorted(root.glob("*_face-benchmark_frames.csv"))
    ))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one shared frame manifest in {root}, found {len(candidates)}")
    return candidates[0]


def _cpu_pyfeat_index(cpu: pd.DataFrame, manifest: Path) -> pd.DataFrame:
    frames = pd.read_csv(manifest)
    out = cpu.copy()
    if "frame" in out.columns:
        f = pd.to_numeric(out["frame"], errors="coerce")
        if f.notna().all():
            lookup = dict(enumerate(frames["benchmark_index"].astype(int).tolist()))
            out["benchmark_index"] = f.astype(int).map(lookup)
            if out["benchmark_index"].notna().all():
                out["benchmark_index"] = out["benchmark_index"].astype(int)
                return out
    if "input" in out.columns:
        lookup = {
            str(Path(str(p)).resolve()): int(i)
            for p, i in zip(frames["image_path"], frames["benchmark_index"])
        }
        mapped = [lookup.get(str(Path(str(x)).resolve())) for x in out["input"]]
        if all(x is not None for x in mapped):
            out["benchmark_index"] = np.asarray(mapped, dtype=int)
            return out
    raise RuntimeError("Could not map saved CPU Py-Feat rows to benchmark_index")


def _valid_box_rows(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"]
    if not set(cols).issubset(df.columns):
        return df.iloc[0:0]
    values = df[cols].apply(pd.to_numeric, errors="coerce")
    return df[np.isfinite(values).all(axis=1)].copy()


def _match_pyfeat(cpu: pd.DataFrame, dml: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    box_cols = ["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"]
    cpu_rows: list[pd.Series] = []
    dml_rows: list[pd.Series] = []
    ious: list[float] = []
    all_indices = sorted(set(cpu["benchmark_index"].dropna().astype(int)) | set(dml["benchmark_index"].dropna().astype(int)))
    for idx in all_indices:
        c = _valid_box_rows(cpu[cpu["benchmark_index"] == idx])
        g0 = dml[dml["benchmark_index"] == idx].copy()
        if "detected" in g0.columns:
            g0 = g0[g0["detected"].fillna(False).astype(bool)]
        g = _valid_box_rows(g0)
        if c.empty or g.empty:
            continue
        available = list(g.index)
        for _, crow in c.iterrows():
            if not available:
                break
            ca = np.asarray([crow[x] for x in box_cols], dtype=np.float64)
            scores = [
                _iou_xywh(ca, np.asarray([g.loc[gi, x] for x in box_cols], dtype=np.float64))
                for gi in available
            ]
            if not np.isfinite(scores).any():
                continue
            best_pos = int(np.nanargmax(scores))
            gi = available.pop(best_pos)
            cpu_rows.append(crow)
            dml_rows.append(g.loc[gi])
            ious.append(float(scores[best_pos]))
    return pd.DataFrame(cpu_rows).reset_index(drop=True), pd.DataFrame(dml_rows).reset_index(drop=True), ious


def _top_class_agreement(cpu: pd.DataFrame, dml: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    common = [c for c in cols if c in cpu.columns and c in dml.columns]
    if len(common) < 2 or len(cpu) != len(dml):
        return {"n": 0, "agreement": None, "columns": common}
    ca = cpu[common].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    da = dml[common].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    mask = np.isfinite(ca).all(axis=1) & np.isfinite(da).all(axis=1)
    if not mask.any():
        return {"n": 0, "agreement": None, "columns": common}
    return {
        "n": int(mask.sum()),
        "agreement": float((np.argmax(ca[mask], axis=1) == np.argmax(da[mask], axis=1)).mean()),
        "columns": common,
    }


def parity_pyfeat(root: Path, dml_dir: Path) -> dict[str, Any]:
    manifest = _find_manifest(root)
    cpu = _cpu_pyfeat_index(pd.read_parquet(root / "pyfeat_raw.parquet"), manifest)
    dml = pd.read_parquet(dml_dir / "pyfeat_dml_raw.parquet")
    matched_cpu, matched_dml, ious = _match_pyfeat(cpu, dml)

    cpu_counts = _valid_box_rows(cpu).groupby("benchmark_index").size()
    dml_valid = dml[dml.get("detected", pd.Series(False, index=dml.index)).fillna(False).astype(bool)]
    dml_counts = _valid_box_rows(dml_valid).groupby("benchmark_index").size()
    all_idx = sorted(set(cpu["benchmark_index"].astype(int)) | set(dml["benchmark_index"].astype(int)))
    count_mismatch = [
        int(i) for i in all_idx if int(cpu_counts.get(i, 0)) != int(dml_counts.get(i, 0))
    ]

    landmarks68 = [f"x_{i}" for i in range(68)] + [f"y_{i}" for i in range(68)]
    mesh = [f"mesh_{axis}_{i}" for axis in "xyz" for i in range(478)]

    return {
        "candidate": "pyfeat_detectorv2_scientific_core",
        "cpu_reference_rows": int(len(cpu)),
        "dml_rows": int(len(dml)),
        "matched_face_rows": int(len(matched_cpu)),
        "coverage": {
            "cpu_detected_frames": int(len(cpu_counts)),
            "dml_detected_frames": int(len(dml_counts)),
            "face_count_mismatch_frames": count_mismatch,
            "face_count_mismatch_n": int(len(count_mismatch)),
        },
        "bbox_iou": {
            "n": int(len(ious)),
            "mean": float(np.nanmean(ious)) if ious else None,
            "median": float(np.nanmedian(ious)) if ious else None,
            "min": float(np.nanmin(ious)) if ious else None,
        },
        "groups": {
            "face_score": _group(matched_cpu, matched_dml, ["FaceScore"]),
            "landmarks68": _group(matched_cpu, matched_dml, landmarks68),
            "au20": _group(matched_cpu, matched_dml, PYFEAT_AUS),
            "emotion7": _group(matched_cpu, matched_dml, PYFEAT_EMOTIONS),
            "valence_arousal": _group(matched_cpu, matched_dml, ["valence", "arousal"]),
            "pose6d": _group(matched_cpu, matched_dml, ["Pitch", "Roll", "Yaw", "X", "Y", "Z"]),
            "gaze": _group(matched_cpu, matched_dml, ["gaze_pitch", "gaze_yaw", "gaze_angle"]),
            "mesh478_original_frame": _group(matched_cpu, matched_dml, mesh),
            "blendshapes52": _group(matched_cpu, matched_dml, BLENDSHAPES),
        },
        "emotion_top_class": _top_class_agreement(matched_cpu, matched_dml, PYFEAT_EMOTIONS),
        "retention_not_parity_tested": [
            "raw RetinaFace bbox/5-point landmarks: retained only in DML layer because saved Detectorv2 CPU Fex does not expose them",
            "raw gaze/pose and normalized mesh: retained only in DML layer; CPU Fex stores canonical/postprocessed forms",
            "identity: intentionally excluded from DirectML scientific-core route",
        ],
    }


def _ordered_numeric(cpu_path: Path, dml_path: Path, prefix: str) -> dict[str, Any]:
    cpu = pd.read_parquet(cpu_path)
    dml = pd.read_parquet(dml_path)
    merged = cpu.merge(dml, on="benchmark_index", how="inner", suffixes=("__cpu", "__dml"))
    direct = [c for c in cpu.columns if c != "benchmark_index" and c in dml.columns]
    numeric_direct = [
        c for c in direct
        if pd.api.types.is_numeric_dtype(cpu[c]) or pd.api.types.is_numeric_dtype(dml[c])
    ]
    if numeric_direct:
        a = pd.DataFrame({c: merged[f"{c}__cpu"] for c in numeric_direct})
        b = pd.DataFrame({c: merged[f"{c}__dml"] for c in numeric_direct})
        return _group(a, b, numeric_direct)

    cpu_num = [c for c in cpu.columns if c != "benchmark_index" and pd.api.types.is_numeric_dtype(cpu[c])]
    dml_num = [c for c in dml.columns if c != "benchmark_index" and pd.api.types.is_numeric_dtype(dml[c])]
    if len(cpu_num) != len(dml_num):
        return {
            "columns": [],
            "aggregate": _metric([], []),
            "per_column": {},
            "error": f"ordered numeric width mismatch CPU={len(cpu_num)} DML={len(dml_num)}",
            "cpu_columns": cpu_num,
            "dml_columns": dml_num,
        }
    a = pd.DataFrame({f"{prefix}_{i}": merged[f"{cpu_num[i]}__cpu"] for i in range(len(cpu_num))})
    b = pd.DataFrame({f"{prefix}_{i}": merged[f"{dml_num[i]}__dml"] for i in range(len(dml_num))})
    return _group(a, b, list(a.columns))


def _flatten_json_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_json_numeric(value[key], child))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            child = f"{prefix}[{i}]"
            out.update(_flatten_json_numeric(item, child))
    elif isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
        out[prefix or "value"] = float(value)
    return out


def _json_column_metric(cpu: pd.DataFrame, dml: pd.DataFrame, col: str) -> dict[str, Any]:
    merged = cpu[["benchmark_index", col]].merge(
        dml[["benchmark_index", col]], on="benchmark_index", how="inner", suffixes=("__cpu", "__dml")
    )
    pairs: dict[str, list[list[float]]] = {}
    rows_used = 0
    for row in merged.itertuples(index=False):
        ca, da = getattr(row, f"{col}__cpu"), getattr(row, f"{col}__dml")
        if ca is None or da is None or pd.isna(ca) or pd.isna(da):
            continue
        try:
            cf = _flatten_json_numeric(json.loads(str(ca)))
            df = _flatten_json_numeric(json.loads(str(da)))
        except Exception:
            continue
        common = sorted(set(cf) & set(df))
        if not common:
            continue
        rows_used += 1
        for key in common:
            slot = pairs.setdefault(key, [[], []])
            slot[0].append(cf[key])
            slot[1].append(df[key])
    if not pairs:
        return {"rows_used": rows_used, "numeric_paths": 0, "aggregate": _metric([], []), "per_path": {}}
    per = {key: _metric(vals[0], vals[1]) for key, vals in pairs.items()}
    return {
        "rows_used": rows_used,
        "numeric_paths": len(pairs),
        "aggregate": _metric(
            np.concatenate([np.asarray(v[0], dtype=float) for v in pairs.values()]),
            np.concatenate([np.asarray(v[1], dtype=float) for v in pairs.values()]),
        ),
        "per_path": per,
    }


def _expression_libreface(cpu_path: Path, dml_path: Path) -> dict[str, Any]:
    cpu = pd.read_parquet(cpu_path)
    dml = pd.read_parquet(dml_path)
    merged = cpu.merge(dml, on="benchmark_index", how="inner", suffixes=("__cpu", "__dml"))
    string_cols = [
        c for c in cpu.columns
        if c != "benchmark_index" and (cpu[c].dtype == object or pd.api.types.is_string_dtype(cpu[c]))
    ]
    for col in string_cols:
        ccol = f"{col}__cpu" if col in dml.columns else col
        if ccol in merged.columns and "expression_label" in merged.columns:
            a = merged[ccol].astype(str)
            b = merged["expression_label"].astype(str)
            return {"n": int(len(merged)), "cpu_label_column": col, "label_agreement": float((a == b).mean())}
    return {
        "n": int(len(merged)),
        "cpu_columns": list(cpu.columns),
        "dml_columns": list(dml.columns),
        "label_agreement": None,
        "note": "No CPU string label column was automatically identifiable; inspect columns if expression parity needs a schema adapter.",
    }


def parity_libreface(root: Path, prep_dir: Path, dml_dir: Path) -> dict[str, Any]:
    cpu_align = pd.read_parquet(root / "libreface_alignment.parquet")
    new_align = pd.read_parquet(prep_dir / "libreface_dml_alignment.parquet")
    common = cpu_align[["benchmark_index", "alignment_success"]].merge(
        new_align[["benchmark_index", "alignment_success"]],
        on="benchmark_index", how="inner", suffixes=("__cpu", "__dml"),
    )
    flag_agreement = float((
        common["alignment_success__cpu"].fillna(False).astype(bool)
        == common["alignment_success__dml"].fillna(False).astype(bool)
    ).mean()) if len(common) else None

    return {
        "candidate": "libreface2",
        "alignment": {
            "cpu_success": int(cpu_align["alignment_success"].fillna(False).astype(bool).sum()),
            "fresh_run_success": int(new_align["alignment_success"].fillna(False).astype(bool).sum()),
            "success_flag_agreement": flag_agreement,
            "headpose_numeric_json": _json_column_metric(cpu_align, new_align, "headpose_json"),
            "landmarks_numeric_json": _json_column_metric(cpu_align, new_align, "landmarks_json"),
        },
        "groups": {
            "au_intensity": _ordered_numeric(
                root / "libreface_au_intensity.parquet",
                dml_dir / "libreface_dml_au_intensity.parquet",
                "au_int",
            ),
            "au_detection": _ordered_numeric(
                root / "libreface_au_detection.parquet",
                dml_dir / "libreface_dml_au_detection.parquet",
                "au_det",
            ),
            "gaze": _ordered_numeric(
                root / "libreface_gaze.parquet",
                dml_dir / "libreface_dml_gaze.parquet",
                "gaze",
            ),
        },
        "expression": _expression_libreface(
            root / "libreface_expression.parquet",
            dml_dir / "libreface_dml_expression.parquet",
        ),
        "retention_not_parity_tested": [
            "raw AU probabilities are newly retained in the DirectML validation layer; saved CPU benchmark kept only LibreFace's derived outputs",
            "1404 MediaPipe gaze features are retained in prep output for provenance/reuse",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retention-aware CPU-reference parity for real 300-frame DirectML Face validation")
    parser.add_argument("--candidate", choices=["pyfeat", "libreface"], required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--dml-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prep-dir", help="Required for LibreFace fresh alignment/headpose/landmarks parity")
    args = parser.parse_args()

    root = Path(args.benchmark_dir).resolve()
    dml_dir = Path(args.dml_dir).resolve()
    if args.candidate == "pyfeat":
        result = parity_pyfeat(root, dml_dir)
    else:
        if not args.prep_dir:
            raise ValueError("--prep-dir is required for LibreFace")
        result = parity_libreface(root, Path(args.prep_dir).resolve(), dml_dir)

    result["schema_version"] = "rgb-face-real300-parity-v0.2"
    result["interpretation"] = (
        "Parity and speed are separate decision dimensions. This report does not freeze the Face backend; "
        "large scientific-output drift with strong bbox/alignment parity should first be checked for CPU preprocessing/interpolation differences."
    )
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
