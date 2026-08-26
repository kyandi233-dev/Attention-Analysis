from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import face_derive_tracking_eyelid as v01


SCHEMA_VERSION = "rgb-face-tracking-eyelid-derived-v0.2"


def _frame_col(df: pd.DataFrame) -> str:
    return "video_frame_position" if "video_frame_position" in df.columns else "benchmark_index"


def _track_stats(group: pd.DataFrame) -> list[dict[str, Any]]:
    frame_col = _frame_col(group)
    stats: list[dict[str, Any]] = []
    for track_id, track in group.groupby("face_track_id"):
        width = pd.to_numeric(track.get("FaceRectWidth"), errors="coerce")
        height = pd.to_numeric(track.get("FaceRectHeight"), errors="coerce")
        x = pd.to_numeric(track.get("FaceRectX"), errors="coerce")
        y = pd.to_numeric(track.get("FaceRectY"), errors="coerce")
        frame_w = pd.to_numeric(track.get("FrameWidth"), errors="coerce")
        frame_h = pd.to_numeric(track.get("FrameHeight"), errors="coerce")
        score = pd.to_numeric(track.get("FaceScore"), errors="coerce")
        area = width * height
        cx_norm = (x + width / 2.0) / frame_w.replace(0, np.nan)
        cy_norm = (y + height / 2.0) / frame_h.replace(0, np.nan)
        area_norm = area / (frame_w * frame_h).replace(0, np.nan)
        stats.append(
            {
                "track_id": int(track_id),
                "frames": int(track[frame_col].nunique()),
                "median_face_score": float(score.median()) if score.notna().any() else None,
                "median_area": float(area.median()) if area.notna().any() else None,
                "median_center_x_norm": float(cx_norm.median()) if cx_norm.notna().any() else None,
                "median_center_y_norm": float(cy_norm.median()) if cy_norm.notna().any() else None,
                "median_area_norm": float(area_norm.median()) if area_norm.notna().any() else None,
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
    return stats


def assign_tracks_window_aware(
    detected: pd.DataFrame,
    *,
    max_gap_ms: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "dryrun_window" not in detected.columns or detected["dryrun_window"].dropna().nunique() <= 1:
        tracked = v01.assign_tracks(detected, max_gap_ms=max_gap_ms)
        tracked["face_track_local_id"] = tracked["face_track_id"]
        return tracked, {
            "mode": "continuous",
            "reason": "no_multiple_dryrun_windows",
        }

    ordered_windows = (
        detected.groupby("dryrun_window", sort=False)["benchmark_index"]
        .min()
        .sort_values()
        .index.tolist()
    )
    pieces: list[pd.DataFrame] = []
    offset = 0
    window_info: list[dict[str, Any]] = []
    for window in ordered_windows:
        part = detected[detected["dryrun_window"] == window].copy()
        local = v01.assign_tracks(part, max_gap_ms=max_gap_ms)
        local_ids = local["face_track_id"].astype("Int64")
        local["face_track_local_id"] = local_ids
        if local_ids.notna().any():
            local["face_track_id"] = (local_ids + offset).astype("Int64")
            local_track_count = int(local_ids.nunique())
            offset += int(local_ids.max()) + 1
        else:
            local_track_count = 0
        pieces.append(local)
        window_info.append(
            {
                "window": str(window),
                "phase": str(part["phase"].dropna().iloc[0]) if "phase" in part.columns and part["phase"].notna().any() else None,
                "frames": int(part[_frame_col(part)].nunique()),
                "tracks": local_track_count,
            }
        )
    tracked = pd.concat(pieces, ignore_index=True)
    return tracked, {
        "mode": "dryrun_window_aware",
        "rule": "reset temporal tracker at each intentionally discontinuous dry-run window; keep globally unique segment IDs",
        "windows": window_info,
    }


def choose_primary_segments(tracked: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = tracked.copy()
    out["primary_face"] = False

    if out.empty:
        return out, {"mode": "none", "reason": "no_valid_tracks", "segments": []}

    if "dryrun_window" not in out.columns or out["dryrun_window"].dropna().nunique() <= 1:
        primary_track, info = v01.choose_primary_track(out)
        if primary_track is not None:
            out["primary_face"] = out["face_track_id"].eq(primary_track)
        return out, {
            "mode": "continuous_single_track",
            "primary_track": primary_track,
            "legacy_selection": info,
            "segments": ([{"window": None, "track_id": int(primary_track)}] if primary_track is not None else []),
        }

    ordered_windows = (
        out.groupby("dryrun_window", sort=False)["benchmark_index"]
        .min()
        .sort_values()
        .index.tolist()
    )
    segments: list[dict[str, Any]] = []
    for window in ordered_windows:
        group = out[out["dryrun_window"] == window]
        stats = _track_stats(group)
        if not stats:
            continue
        chosen = stats[0]
        track_id = int(chosen["track_id"])
        mask = out["dryrun_window"].eq(window) & out["face_track_id"].eq(track_id)
        out.loc[mask, "primary_face"] = True
        window_frames = int(group[_frame_col(group)].nunique())
        segments.append(
            {
                "window": str(window),
                "phase": str(group["phase"].dropna().iloc[0]) if "phase" in group.columns and group["phase"].notna().any() else None,
                "primary_track_id": track_id,
                "primary_frames": int(chosen["frames"]),
                "window_detected_frames": window_frames,
                "primary_window_coverage": float(chosen["frames"] / window_frames) if window_frames else 0.0,
                "selected_track": chosen,
                "all_track_stats": stats,
            }
        )
    return out, {
        "mode": "dryrun_window_primary_segments",
        "rule": "within each intentionally discontinuous window choose max frame occupancy; tie by median FaceScore then median bbox area",
        "segments": segments,
        "note": "These are dry-run segment IDs, not a claim that temporal tracking bridged the omitted intervals. Formal full-video analysis remains continuous tracking.",
    }


def derive_eye_from_primary_segments(tracked: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if tracked.empty or not tracked["primary_face"].fillna(False).any():
        return v01.derive_eye_features(tracked, None)

    tmp = tracked.copy()
    tmp["face_track_segment_id"] = tmp["face_track_id"]
    primary_mask = tmp["primary_face"].fillna(False).astype(bool)
    original_ids = pd.to_numeric(tmp["face_track_id"], errors="coerce").fillna(-1).astype(int)
    tmp["face_track_id"] = np.where(primary_mask, 0, original_ids + 1_000_000)

    eye, summary = v01.derive_eye_features(tmp, 0)
    frame_col = _frame_col(tmp)
    segment_map = (
        tmp.loc[primary_mask, [frame_col, "face_track_segment_id", "dryrun_window"]]
        .drop_duplicates(frame_col)
        .rename(columns={"face_track_segment_id": "primary_face_segment_id"})
    )
    eye = eye.merge(segment_map, on=frame_col, how="left", validate="one_to_one")
    summary["primary_track"] = None
    summary["primary_mode"] = "window_primary_segments"
    summary["primary_segment_count"] = int(segment_map["primary_face_segment_id"].nunique()) if not segment_map.empty else 0
    summary["notes"] = list(summary.get("notes", [])) + [
        "Dry-run windows are intentionally discontinuous; eyelid features use one dominant primary-face segment per window rather than one global temporal track ID.",
        "Baseline normalization should therefore use the actual baseline windows when sufficient valid baseline frames exist.",
    ]
    return eye, summary


def _bbox_mask(df: pd.DataFrame) -> pd.Series:
    return v01._bbox_mask(df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Window-aware Face tracking/primary selection and eyelid derivation for formal dry-run"
    )
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

    tracked, tracking_mode = assign_tracks_window_aware(
        detected,
        max_gap_ms=float(args.max_track_gap_ms),
    )
    tracked, primary_info = choose_primary_segments(tracked)
    eye, eye_summary = derive_eye_from_primary_segments(tracked)

    tracked_path = out_dir / "face_tracks.parquet"
    eye_path = out_dir / "eye_features.parquet"
    summary_path = out_dir / "tracking_eyelid_summary.json"
    tracked.to_parquet(tracked_path, index=False, engine="pyarrow", compression="zstd")
    eye.to_parquet(eye_path, index=False, engine="pyarrow", compression="zstd")

    frame_col = _frame_col(tracked)
    multi_face_frames = int((tracked.groupby(frame_col).size() > 1).sum()) if not tracked.empty else 0
    primary_frames = int(tracked.loc[tracked["primary_face"].fillna(False), frame_col].nunique()) if not tracked.empty else 0
    total_frames = int(raw[frame_col].nunique()) if frame_col in raw.columns else 0

    summary = {
        "schema_version": SCHEMA_VERSION,
        "raw_input": str(raw_path),
        "frame_manifest": str(Path(args.frame_manifest).resolve()) if args.frame_manifest else None,
        "tracking": {
            "detected_face_rows": int(len(tracked)),
            "track_count": int(tracked["face_track_id"].nunique()) if not tracked.empty else 0,
            "multi_face_frames": multi_face_frames,
            "max_track_gap_ms": float(args.max_track_gap_ms),
            "tracking_mode": tracking_mode,
            "primary_selection": primary_info,
            "primary_frames": primary_frames,
            "total_input_frames": total_frames,
            "primary_frame_coverage": float(primary_frames / total_frames) if total_frames else 0.0,
        },
        "eyelid": eye_summary,
        "outputs": {"tracks": str(tracked_path), "eye_features": str(eye_path)},
        "notes": [
            "v0.2 fixes the dry-run-only artifact where one global track ID could not bridge intentionally omitted time windows.",
            "Formal continuous full-video tracking semantics are unchanged: a continuous source uses the legacy single continuous temporal tracker.",
            "Blink-event thresholds and perclos80_proxy remain provisional until sub-031/sub-033 visual and distribution QC are complete.",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
