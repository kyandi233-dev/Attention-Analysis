from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "rgb-face-tracking-eyelid-derived-v0.1"
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
RIGHT_IRIS_RING = [469, 470, 471, 472]
LEFT_IRIS_RING = [474, 475, 476, 477]


def _numeric(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _bbox(row: pd.Series) -> np.ndarray | None:
    vals = np.asarray([
        _numeric(row.get("FaceRectX")),
        _numeric(row.get("FaceRectY")),
        _numeric(row.get("FaceRectWidth")),
        _numeric(row.get("FaceRectHeight")),
    ], dtype=float)
    if not np.isfinite(vals).all() or vals[2] <= 0 or vals[3] <= 0:
        return None
    return vals


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _track_similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    iou = _iou(a, b)
    ac = np.array([a[0] + a[2] / 2.0, a[1] + a[3] / 2.0])
    bc = np.array([b[0] + b[2] / 2.0, b[1] + b[3] / 2.0])
    scale = max(math.sqrt(a[2] * a[3]), 1.0)
    center_dist_norm = float(np.linalg.norm(ac - bc) / scale)
    scale_log = float(abs(math.log(max(b[2] * b[3], 1e-9) / max(a[2] * a[3], 1e-9))))
    score = 0.70 * iou + 0.20 * math.exp(-center_dist_norm) + 0.10 * math.exp(-scale_log)
    return score, iou, center_dist_norm, scale_log


def assign_tracks(
    faces: pd.DataFrame,
    *,
    max_gap_ms: float = 2000.0,
    min_iou: float = 0.05,
    max_center_dist_norm: float = 0.75,
    max_scale_log: float = 0.80,
) -> pd.DataFrame:
    out = faces.copy()
    out["face_track_id"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    out["track_match_score"] = np.nan
    out["track_match_iou"] = np.nan
    out["track_center_dist_norm"] = np.nan
    out["track_scale_log"] = np.nan

    time_col = "unix_ms" if "unix_ms" in out.columns else "benchmark_index"
    frame_col = "video_frame_position" if "video_frame_position" in out.columns else "benchmark_index"
    active: dict[int, dict[str, Any]] = {}
    next_track = 0

    for _, frame in out.sort_values([time_col, "face_rank" if "face_rank" in out.columns else frame_col]).groupby(frame_col, sort=True):
        t = float(pd.to_numeric(frame[time_col], errors="coerce").iloc[0])
        if not math.isfinite(t):
            t = float(pd.to_numeric(frame[frame_col], errors="coerce").iloc[0])
        candidates: list[tuple[float, int, int, float, float, float]] = []
        valid_indices: list[int] = []
        for idx, row in frame.iterrows():
            box = _bbox(row)
            if box is None:
                continue
            valid_indices.append(idx)
            for track_id, state in active.items():
                gap = t - float(state["time"])
                if gap < 0 or gap > max_gap_ms:
                    continue
                score, iou, center_norm, scale_log = _track_similarity(np.asarray(state["bbox"], dtype=float), box)
                if scale_log > max_scale_log:
                    continue
                if iou < min_iou and center_norm > max_center_dist_norm:
                    continue
                candidates.append((score, track_id, idx, iou, center_norm, scale_log))

        used_tracks: set[int] = set()
        used_rows: set[int] = set()
        for score, track_id, idx, iou, center_norm, scale_log in sorted(candidates, key=lambda x: x[0], reverse=True):
            if track_id in used_tracks or idx in used_rows:
                continue
            out.at[idx, "face_track_id"] = track_id
            out.at[idx, "track_match_score"] = score
            out.at[idx, "track_match_iou"] = iou
            out.at[idx, "track_center_dist_norm"] = center_norm
            out.at[idx, "track_scale_log"] = scale_log
            used_tracks.add(track_id)
            used_rows.add(idx)

        for idx in valid_indices:
            if idx in used_rows:
                continue
            out.at[idx, "face_track_id"] = next_track
            used_tracks.add(next_track)
            next_track += 1

        for idx in valid_indices:
            track_id = int(out.at[idx, "face_track_id"])
            active[track_id] = {"bbox": _bbox(out.loc[idx]), "time": t}
        active = {
            track_id: state
            for track_id, state in active.items()
            if t - float(state["time"]) <= max_gap_ms
        }
    return out


def choose_primary_track(tracked: pd.DataFrame) -> tuple[int | None, dict[str, Any]]:
    valid = tracked[tracked["face_track_id"].notna()].copy()
    if valid.empty:
        return None, {"reason": "no_valid_tracks"}
    if "phase" in valid.columns:
        task = valid[valid["phase"].isin(["block1", "block2"])]
        scope = task if not task.empty else valid
        scope_name = "block1_block2" if not task.empty else "all_available"
    else:
        scope = valid
        scope_name = "all_available"

    frame_col = "video_frame_position" if "video_frame_position" in scope.columns else "benchmark_index"
    stats: list[dict[str, Any]] = []
    for track_id, group in scope.groupby("face_track_id"):
        area = pd.to_numeric(group.get("FaceRectWidth"), errors="coerce") * pd.to_numeric(group.get("FaceRectHeight"), errors="coerce")
        face_score = pd.to_numeric(group.get("FaceScore"), errors="coerce")
        stats.append(
            {
                "track_id": int(track_id),
                "frames": int(group[frame_col].nunique()),
                "median_area": float(area.median()) if area.notna().any() else None,
                "median_face_score": float(face_score.median()) if face_score.notna().any() else None,
            }
        )
    stats.sort(
        key=lambda s: (
            s["frames"],
            -1.0 if s["median_face_score"] is None else s["median_face_score"],
            -1.0 if s["median_area"] is None else s["median_area"],
        ),
        reverse=True,
    )
    chosen = stats[0]["track_id"]
    return int(chosen), {"scope": scope_name, "track_stats": stats, "rule": "max task-frame occupancy; tie by median FaceScore then median bbox area"}


def _point(row: pd.Series, idx: int) -> np.ndarray:
    return np.asarray([_numeric(row.get(f"mesh_x_{idx}")), _numeric(row.get(f"mesh_y_{idx}"))], dtype=float)


def _dist(row: pd.Series, a: int, b: int) -> float:
    pa, pb = _point(row, a), _point(row, b)
    if not np.isfinite(pa).all() or not np.isfinite(pb).all():
        return float("nan")
    return float(np.linalg.norm(pa - pb))


def _ear(row: pd.Series, indices: list[int]) -> float:
    p1, p2, p3, p4, p5, p6 = indices
    h = _dist(row, p1, p4)
    v1 = _dist(row, p2, p6)
    v2 = _dist(row, p3, p5)
    if not all(math.isfinite(x) for x in (h, v1, v2)) or h <= 0:
        return float("nan")
    return float((v1 + v2) / (2.0 * h))


def _aperture(row: pd.Series, indices: list[int]) -> float:
    _, p2, p3, _, p5, p6 = indices
    d1 = _dist(row, p2, p6)
    d2 = _dist(row, p3, p5)
    if not all(math.isfinite(x) for x in (d1, d2)):
        return float("nan")
    return float((d1 + d2) / 2.0)


def _iris_diameter(row: pd.Series, ring: list[int]) -> float:
    pts = [_point(row, idx) for idx in ring]
    if not all(np.isfinite(p).all() for p in pts):
        return float("nan")
    distances = [float(np.linalg.norm(pts[i] - pts[j])) for i in range(len(pts)) for j in range(i + 1, len(pts))]
    return max(distances) if distances else float("nan")


def _open_reference(values: pd.Series, baseline_mask: pd.Series, *, min_baseline_n: int = 30) -> tuple[float | None, str, int]:
    numeric = pd.to_numeric(values, errors="coerce")
    baseline = numeric[baseline_mask & numeric.notna()]
    if len(baseline) >= min_baseline_n:
        source = baseline
        label = "baseline_top30_median"
    else:
        source = numeric[numeric.notna()]
        label = "all_valid_top30_median_fallback"
    if source.empty:
        return None, label, 0
    q70 = float(source.quantile(0.70))
    top = source[source >= q70]
    ref = float(top.median()) if not top.empty else float(source.median())
    return ref, label, int(len(source))


def derive_eye_features(tracked: pd.DataFrame, primary_track: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame_col = "video_frame_position" if "video_frame_position" in tracked.columns else "benchmark_index"
    identity_cols = [
        c for c in [
            "subject", "benchmark_index", "video_frame_position", "capture_frame_idx", "unix_ms",
            "target_unix_ms", "sample_error_ms", "dt_ms", "phase", "block", "trial_num",
            "condition", "cycle", "stimulus", "is_nogo", "error", "probe", "behavior_state",
            "dryrun_window", "temporal_gap", "capture_gap_before",
        ] if c in tracked.columns
    ]
    frames = tracked.sort_values(frame_col).drop_duplicates(frame_col)[identity_cols].copy()
    if primary_track is None:
        for col in ["primary_face_present", "ear_left", "ear_right", "ear_mean", "aperture_iris_left", "aperture_iris_right"]:
            frames[col] = False if col == "primary_face_present" else np.nan
        return frames, {"primary_track": None, "eye_valid_fraction": 0.0}

    primary = tracked[tracked["face_track_id"] == primary_track].copy()
    primary["primary_face_present"] = True
    primary["ear_right"] = primary.apply(lambda row: _ear(row, RIGHT_EYE_EAR), axis=1)
    primary["ear_left"] = primary.apply(lambda row: _ear(row, LEFT_EYE_EAR), axis=1)
    primary["ear_mean"] = primary[["ear_left", "ear_right"]].mean(axis=1)
    primary["eye_aperture_px_right"] = primary.apply(lambda row: _aperture(row, RIGHT_EYE_EAR), axis=1)
    primary["eye_aperture_px_left"] = primary.apply(lambda row: _aperture(row, LEFT_EYE_EAR), axis=1)
    primary["iris_diameter_px_right"] = primary.apply(lambda row: _iris_diameter(row, RIGHT_IRIS_RING), axis=1)
    primary["iris_diameter_px_left"] = primary.apply(lambda row: _iris_diameter(row, LEFT_IRIS_RING), axis=1)
    primary["aperture_iris_right"] = primary["eye_aperture_px_right"] / primary["iris_diameter_px_right"]
    primary["aperture_iris_left"] = primary["eye_aperture_px_left"] / primary["iris_diameter_px_left"]
    primary["aperture_iris_mean"] = primary[["aperture_iris_left", "aperture_iris_right"]].mean(axis=1)
    primary["native_eyeBlinkLeft"] = pd.to_numeric(primary.get("eyeBlinkLeft"), errors="coerce")
    primary["native_eyeBlinkRight"] = pd.to_numeric(primary.get("eyeBlinkRight"), errors="coerce")
    primary["native_eyeBlink_mean"] = primary[["native_eyeBlinkLeft", "native_eyeBlinkRight"]].mean(axis=1)
    primary["eye_geometry_valid"] = np.isfinite(primary[["ear_left", "ear_right", "aperture_iris_left", "aperture_iris_right"]]).all(axis=1)

    baseline_mask = primary["phase"].eq("baseline") if "phase" in primary.columns else pd.Series(False, index=primary.index)
    refs: dict[str, Any] = {}
    for side in ["left", "right"]:
        ref, source, n = _open_reference(primary[f"aperture_iris_{side}"], baseline_mask)
        refs[side] = {"open_reference": ref, "source": source, "source_n": n}
        if ref is not None and ref > 0:
            primary[f"eye_openness_norm_{side}"] = primary[f"aperture_iris_{side}"] / ref
            primary[f"closure_fraction_{side}"] = (1.0 - primary[f"eye_openness_norm_{side}"].clip(upper=1.0)).clip(lower=0.0, upper=1.0)
            primary[f"closure80_proxy_{side}"] = primary[f"eye_openness_norm_{side}"] <= 0.20
        else:
            primary[f"eye_openness_norm_{side}"] = np.nan
            primary[f"closure_fraction_{side}"] = np.nan
            primary[f"closure80_proxy_{side}"] = pd.NA
    primary["eye_openness_norm_mean"] = primary[["eye_openness_norm_left", "eye_openness_norm_right"]].mean(axis=1)
    primary["closure_fraction_mean"] = primary[["closure_fraction_left", "closure_fraction_right"]].mean(axis=1)

    derived_cols = [
        frame_col, "face_track_id", "primary_face_present", "FaceScore", "ear_left", "ear_right", "ear_mean",
        "eye_aperture_px_left", "eye_aperture_px_right", "iris_diameter_px_left", "iris_diameter_px_right",
        "aperture_iris_left", "aperture_iris_right", "aperture_iris_mean", "native_eyeBlinkLeft",
        "native_eyeBlinkRight", "native_eyeBlink_mean", "eye_geometry_valid", "eye_openness_norm_left",
        "eye_openness_norm_right", "eye_openness_norm_mean", "closure_fraction_left", "closure_fraction_right",
        "closure_fraction_mean", "closure80_proxy_left", "closure80_proxy_right",
    ]
    derived_cols = [c for c in derived_cols if c in primary.columns]
    merged = frames.merge(primary[derived_cols], on=frame_col, how="left", validate="one_to_one")
    merged["primary_face_present"] = merged["primary_face_present"].fillna(False).astype(bool)
    if "eye_geometry_valid" in merged.columns:
        merged["eye_geometry_valid"] = merged["eye_geometry_valid"].fillna(False).astype(bool)

    valid_fraction = float(merged.get("eye_geometry_valid", pd.Series(False, index=merged.index)).mean()) if len(merged) else 0.0
    summary = {
        "primary_track": int(primary_track),
        "primary_face_present_fraction": float(merged["primary_face_present"].mean()) if len(merged) else 0.0,
        "eye_valid_fraction": valid_fraction,
        "open_references": refs,
        "ear_mean_quantiles": {
            str(q): float(pd.to_numeric(merged["ear_mean"], errors="coerce").quantile(q))
            for q in [0.01, 0.05, 0.10, 0.50, 0.90, 0.99]
            if pd.to_numeric(merged["ear_mean"], errors="coerce").notna().any()
        },
        "native_eyeBlink_mean_quantiles": {
            str(q): float(pd.to_numeric(merged["native_eyeBlink_mean"], errors="coerce").quantile(q))
            for q in [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
            if pd.to_numeric(merged["native_eyeBlink_mean"], errors="coerce").notna().any()
        },
        "notes": [
            "EAR and aperture/iris ratio are derived from retained 478-point mesh and are recalculable without rerunning Py-Feat.",
            "closure80_proxy is a provisional subject-normalized eyelid-openness proxy, not a claim of validated classical PERCLOS.",
            "Blink event thresholds are intentionally not frozen in v0.1; inspect representative-subject distributions first.",
        ],
    }
    return merged, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign Face tracks/primary face and derive eyelid signals from saved Py-Feat raw output")
    parser.add_argument("--raw", required=True, help="Py-Feat raw parquet")
    parser.add_argument("--frame-manifest", help="Optional CSV to merge benchmark_index -> timestamps/phase/context")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-track-gap-ms", type=float, default=2000.0)
    args = parser.parse_args()

    raw_path = Path(args.raw).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(raw_path)
    if args.frame_manifest:
        manifest = pd.read_csv(Path(args.frame_manifest).resolve())
        keep = [c for c in manifest.columns if c == "benchmark_index" or c not in raw.columns]
        raw = raw.merge(manifest[keep], on="benchmark_index", how="left", validate="many_to_one")

    detected = raw.copy()
    if "detected" in detected.columns:
        detected = detected[detected["detected"].fillna(False).astype(bool)].copy()
    detected = detected[_bbox_mask(detected)]
    tracked = assign_tracks(detected, max_gap_ms=args.max_track_gap_ms)
    primary_track, primary_info = choose_primary_track(tracked)
    tracked["primary_face"] = tracked["face_track_id"].eq(primary_track) if primary_track is not None else False
    eye, eye_summary = derive_eye_features(tracked, primary_track)

    tracked_path = out_dir / "face_tracks.parquet"
    eye_path = out_dir / "eye_features.parquet"
    summary_path = out_dir / "tracking_eyelid_summary.json"
    tracked.to_parquet(tracked_path, index=False, engine="pyarrow", compression="zstd")
    eye.to_parquet(eye_path, index=False, engine="pyarrow", compression="zstd")

    frame_col = "video_frame_position" if "video_frame_position" in tracked.columns else "benchmark_index"
    multi_face_frames = int((tracked.groupby(frame_col).size() > 1).sum()) if not tracked.empty else 0
    summary = {
        "schema_version": SCHEMA_VERSION,
        "raw_input": str(raw_path),
        "frame_manifest": str(Path(args.frame_manifest).resolve()) if args.frame_manifest else None,
        "tracking": {
            "detected_face_rows": int(len(tracked)),
            "track_count": int(tracked["face_track_id"].nunique()) if not tracked.empty else 0,
            "multi_face_frames": multi_face_frames,
            "max_track_gap_ms": float(args.max_track_gap_ms),
            "primary_track": primary_track,
            "primary_selection": primary_info,
        },
        "eyelid": eye_summary,
        "outputs": {"tracks": str(tracked_path), "eye_features": str(eye_path)},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _bbox_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(False, index=df.index)
    cols = ["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"]
    if not set(cols).issubset(df.columns):
        return pd.Series(False, index=df.index)
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    return np.isfinite(vals).all(axis=1) & (vals["FaceRectWidth"] > 0) & (vals["FaceRectHeight"] > 0)


if __name__ == "__main__":
    main()
