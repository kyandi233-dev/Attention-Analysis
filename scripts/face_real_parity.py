from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _num(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def _metric(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return {"n": 0, "mae": None, "max_abs": None, "pearson_r": None, "spearman_rho": None}
    a, b = a[mask], b[mask]
    diff = np.abs(a - b)
    pearson = None
    spearman = None
    if len(a) >= 2 and np.std(a) > 0 and np.std(b) > 0:
        pearson = float(np.corrcoef(a, b)[0, 1])
        spearman = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    return {
        "n": int(len(a)),
        "mae": float(diff.mean()),
        "max_abs": float(diff.max()),
        "pearson_r": pearson,
        "spearman_rho": spearman,
    }


def _group_metric(cpu: pd.DataFrame, dml: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    cols = [c for c in columns if c in cpu.columns and c in dml.columns]
    if not cols:
        return {"columns": [], "aggregate": _metric([], []), "per_column": {}}
    per = {c: _metric(_num(cpu[c]).to_numpy(), _num(dml[c]).to_numpy()) for c in cols}
    agg = _metric(
        np.concatenate([_num(cpu[c]).to_numpy() for c in cols]),
        np.concatenate([_num(dml[c]).to_numpy() for c in cols]),
    )
    return {"columns": cols, "aggregate": agg, "per_column": per}


def _iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(aw, 0) * max(ah, 0) + max(bw, 0) * max(bh, 0) - inter
    return inter / union if union > 0 else float("nan")


def _pyfeat_cpu_with_index(cpu: pd.DataFrame, frame_manifest: Path) -> pd.DataFrame:
    frames = pd.read_csv(frame_manifest)
    mapping = {i: int(v) for i, v in enumerate(frames["benchmark_index"].tolist())}
    out = cpu.copy()
    if "frame" in out.columns:
        vals = pd.to_numeric(out["frame"], errors="coerce")
        if vals.notna().all():
            out["benchmark_index"] = vals.astype(int).map(mapping).fillna(vals.astype(int)).astype(int)
            return out
    if "input" in out.columns:
        by_path = {
            str(Path(p).resolve()): int(i)
            for p, i in zip(frames["image_path"], frames["benchmark_index"])
        }
        out["benchmark_index"] = [
            by_path.get(str(Path(str(p)).resolve()), np.nan) for p in out["input"]
        ]
        if out["benchmark_index"].notna().all():
            out["benchmark_index"] = out["benchmark_index"].astype(int)
            return out
    raise RuntimeError("Could not map CPU Py-Feat rows to benchmark_index")


def _match_pyfeat(cpu: pd.DataFrame, dml: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    box_cols = ["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"]
    cpu_rows, dml_rows, ious = [], [], []
    for idx in sorted(set(cpu["benchmark_index"]) | set(dml["benchmark_index"])):
        c = cpu[cpu["benchmark_index"] == idx].copy()
        g = dml[(dml["benchmark_index"] == idx) & dml.get("detected", True)].copy()
        if c.empty or g.empty:
            continue
        if set(box_cols).issubset(c.columns):
            c = c[np.isfinite(c[box_cols].apply(pd.to_numeric, errors="coerce")).all(axis=1)]
        if c.empty:
            continue
        available = list(g.index)
        for _, crow in c.iterrows():
            if not available:
                break
            ca = np.array([float(crow[x]) for x in box_cols])
            scores = []
            for gi in available:
                grow = g.loc[gi]
                gb = np.array([float(grow[x]) for x in box_cols])
                scores.append(_iou_xywh(ca, gb))
            best_pos = int(np.nanargmax(scores))
            gi = available.pop(best_pos)
            cpu_rows.append(crow)
            dml_rows.append(g.loc[gi])
            ious.append(float(scores[best_pos]))
    return pd.DataFrame(cpu_rows).reset_index(drop=True), pd.DataFrame(dml_rows).reset_index(drop=True), ious


def parity_pyfeat(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.benchmark_dir).resolve()
    dml_dir = Path(args.dml_dir).resolve()
    manifests = list(dict.fromkeys(
        sorted(root.glob("*_face-continuous_frames.csv")) + sorted(root.glob("*_face-benchmark_frames.csv"))
    ))
    if len(manifests) != 1:
        raise RuntimeError("Expected one frame manifest")
    cpu = _pyfeat_cpu_with_index(pd.read_parquet(root / "pyfeat_raw.parquet"), manifests[0])
    dml = pd.read_parquet(dml_dir / "pyfeat_dml_raw.parquet")
    c, g, ious = _match_pyfeat(cpu, dml)

    aus = [
        "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11", "AU12",
        "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU43",
    ]
    emotions = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
    pose = ["Pitch", "Roll", "Yaw", "X", "Y", "Z"]
    gaze = ["gaze_pitch", "gaze_yaw", "gaze_angle"]
    va = ["valence", "arousal"]
    mesh = [f"mesh_{axis}_{i}" for axis in "xyz" for i in range(478)]
    blend_candidates = [x for x in g.columns if x in cpu.columns and x not in aus + emotions + pose + gaze + va + mesh]
    blend = [
        x for x in blend_candidates
        if not x.startswith(("Face", "Identity", "x_", "y_"))
        and x not in {"input", "frame", "benchmark_index", "detected", "face_rank", "FrameHeight", "FrameWidth"}
    ]

    cpu_detected_frames = int(cpu.groupby("benchmark_index").apply(
        lambda z: np.isfinite(z[["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"]].apply(
            pd.to_numeric, errors="coerce"
        )).all(axis=1).any()
    ).sum())
    dml_detected_frames = int(dml.groupby("benchmark_index")["detected"].any().sum())

    return {
        "candidate": "pyfeat_detectorv2_scientific_core",
        "cpu_rows": int(len(cpu)),
        "dml_rows": int(len(dml)),
        "matched_face_rows": int(len(c)),
        "cpu_detected_frames": cpu_detected_frames,
        "dml_detected_frames": dml_detected_frames,
        "bbox_iou": {
            "n": len(ious),
            "mean": float(np.nanmean(ious)) if ious else None,
            "median": float(np.nanmedian(ious)) if ious else None,
            "min": float(np.nanmin(ious)) if ious else None,
        },
        "groups": {
            "au": _group_metric(c, g, aus),
            "emotion": _group_metric(c, g, emotions),
            "valence_arousal": _group_metric(c, g, va),
            "pose": _group_metric(c, g, pose),
            "gaze": _group_metric(c, g, gaze),
            "mesh478": _group_metric(c, g, mesh),
            "blendshapes_common": _group_metric(c, g, blend),
        },
        "notes": [
            "Identity is excluded by the DirectML scientific-core decision and is not part of parity acceptance.",
            "Face rows are matched within benchmark_index by greedy highest facebox IoU; multi-face rows are not silently discarded.",
        ],
    }


def _ordered_numeric_compare(cpu_path: Path, dml_path: Path, prefix: str) -> dict[str, Any]:
    c = pd.read_parquet(cpu_path)
    g = pd.read_parquet(dml_path)
    merged = c.merge(g, on="benchmark_index", how="inner", suffixes=("__cpu", "__dml"))
    direct = [x for x in c.columns if x != "benchmark_index" and x in g.columns]
    if direct:
        a = pd.DataFrame({x: merged[f"{x}__cpu"] for x in direct})
        b = pd.DataFrame({x: merged[f"{x}__dml"] for x in direct})
        return _group_metric(a, b, direct)
    cc = [x for x in c.columns if x != "benchmark_index" and pd.api.types.is_numeric_dtype(c[x])]
    gg = [x for x in g.columns if x != "benchmark_index" and pd.api.types.is_numeric_dtype(g[x])]
    if len(cc) != len(gg):
        return {
            "columns": [],
            "aggregate": _metric([], []),
            "per_column": {},
            "error": f"ordered numeric width mismatch CPU={len(cc)} DML={len(gg)}",
        }
    a = pd.DataFrame({f"{prefix}_{i}": merged[f"{cc[i]}__cpu"] for i in range(len(cc))})
    b = pd.DataFrame({f"{prefix}_{i}": merged[f"{gg[i]}__dml"] for i in range(len(gg))})
    return _group_metric(a, b, list(a.columns))


def parity_libreface(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.benchmark_dir).resolve()
    d = Path(args.dml_dir).resolve()
    prep = Path(args.prep_dir).resolve()

    alignment_cpu = pd.read_parquet(root / "libreface_alignment.parquet")
    alignment_new = pd.read_parquet(prep / "libreface_dml_alignment.parquet")
    common = alignment_cpu[["benchmark_index", "alignment_success"]].merge(
        alignment_new[["benchmark_index", "alignment_success"]],
        on="benchmark_index",
        suffixes=("__cpu", "__dml"),
    )
    agreement = float((
        common["alignment_success__cpu"].astype(bool)
        == common["alignment_success__dml"].astype(bool)
    ).mean()) if len(common) else None

    result = {
        "candidate": "libreface2",
        "alignment": {
            "cpu_success": int(alignment_cpu["alignment_success"].fillna(False).astype(bool).sum()),
            "dml_run_fresh_success": int(alignment_new["alignment_success"].fillna(False).astype(bool).sum()),
            "success_flag_agreement": agreement,
        },
        "groups": {
            "au_intensity": _ordered_numeric_compare(
                root / "libreface_au_intensity.parquet",
                d / "libreface_dml_au_intensity.parquet",
                "au_int",
            ),
            "au_detection": _ordered_numeric_compare(
                root / "libreface_au_detection.parquet",
                d / "libreface_dml_au_detection.parquet",
                "au_det",
            ),
            "gaze": _ordered_numeric_compare(
                root / "libreface_gaze.parquet",
                d / "libreface_dml_gaze.parquet",
                "gaze",
            ),
        },
    }

    ce = pd.read_parquet(root / "libreface_expression.parquet")
    ge = pd.read_parquet(d / "libreface_dml_expression.parquet")
    em = ce.merge(ge, on="benchmark_index", how="inner", suffixes=("__cpu", "__dml"))
    cpu_label_cols = [
        c for c in ce.columns
        if c != "benchmark_index" and (ce[c].dtype == object or pd.api.types.is_string_dtype(ce[c]))
    ]
    if cpu_label_cols:
        col = cpu_label_cols[0]
        result["expression"] = {
            "n": int(len(em)),
            "label_agreement": float((
                em[f"{col}__cpu"].astype(str) == em["expression_label"].astype(str)
            ).mean()),
        }
    else:
        result["expression"] = {
            "n": int(len(em)),
            "label_agreement": None,
            "note": "CPU expression table had no string label column; inspect columns if score-level parity is needed.",
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-reference parity summary for real 300-frame DirectML Face candidates"
    )
    parser.add_argument("--candidate", choices=["pyfeat", "libreface"], required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--dml-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prep-dir", help="Required for LibreFace fresh-alignment parity")
    args = parser.parse_args()

    if args.candidate == "pyfeat":
        result = parity_pyfeat(args)
    else:
        if not args.prep_dir:
            raise ValueError("--prep-dir required for libreface")
        result = parity_libreface(args)
    result["schema_version"] = "rgb-face-real300-parity-v0.1"

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
