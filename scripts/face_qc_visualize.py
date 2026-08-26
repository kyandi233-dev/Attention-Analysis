from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS = [474, 475, 476, 477]


def _num(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _frame_col(df: pd.DataFrame) -> str:
    return "video_frame_position" if "video_frame_position" in df.columns else "benchmark_index"


def _point(row: pd.Series, idx: int) -> tuple[int, int] | None:
    x = _num(row.get(f"mesh_x_{idx}"))
    y = _num(row.get(f"mesh_y_{idx}"))
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return int(round(x)), int(round(y))


def _draw_points(frame: np.ndarray, row: pd.Series, indices: list[int], color: tuple[int, int, int]) -> None:
    pts: list[tuple[int, int]] = []
    for idx in indices:
        pt = _point(row, idx)
        if pt is not None:
            pts.append(pt)
            cv2.circle(frame, pt, 2, color, -1, cv2.LINE_AA)
    if len(pts) >= 2:
        poly = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [poly], True, color, 1, cv2.LINE_AA)


def _draw_bbox(frame: np.ndarray, row: pd.Series, *, primary: bool) -> None:
    x = _num(row.get("FaceRectX"))
    y = _num(row.get("FaceRectY"))
    w = _num(row.get("FaceRectWidth"))
    h = _num(row.get("FaceRectHeight"))
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return
    color = (60, 220, 60) if primary else (0, 165, 255)
    p1 = (int(round(x)), int(round(y)))
    p2 = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(frame, p1, p2, color, 2, cv2.LINE_AA)
    track = row.get("face_track_id")
    score = _num(row.get("FaceScore"))
    label = f"{'PRIMARY' if primary else 'OTHER'} track={track}"
    if math.isfinite(score):
        label += f" score={score:.3f}"
    cv2.putText(frame, label, (p1[0], max(20, p1[1] - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def _top_emotion(row: pd.Series) -> str | None:
    names = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
    vals = [(_num(row.get(name)), name) for name in names]
    vals = [(value, name) for value, name in vals if math.isfinite(value)]
    if not vals:
        return None
    value, name = max(vals)
    return f"{name}:{value:.2f}"


def _annotate(
    frame: np.ndarray,
    faces: pd.DataFrame,
    eye_row: pd.Series | None,
) -> np.ndarray:
    out = frame.copy()
    if faces.empty:
        return out
    primary_rows = faces[faces.get("primary_face", False).fillna(False).astype(bool)] if "primary_face" in faces.columns else faces.iloc[0:0]
    for _, row in faces.iterrows():
        is_primary = bool(row.get("primary_face", False))
        _draw_bbox(out, row, primary=is_primary)
    primary = primary_rows.iloc[0] if not primary_rows.empty else None
    if primary is not None:
        _draw_points(out, primary, RIGHT_EYE, (255, 255, 0))
        _draw_points(out, primary, LEFT_EYE, (255, 255, 0))
        _draw_points(out, primary, RIGHT_IRIS, (255, 0, 255))
        _draw_points(out, primary, LEFT_IRIS, (255, 0, 255))

    lines: list[str] = []
    first = faces.iloc[0]
    window = first.get("dryrun_window")
    phase = first.get("phase")
    unix_ms = first.get("unix_ms")
    lines.append(f"window={window} phase={phase} unix_ms={unix_ms} faces={len(faces)}")
    if eye_row is not None:
        ear = _num(eye_row.get("ear_mean"))
        blink = _num(eye_row.get("native_eyeBlink_mean"))
        openness = _num(eye_row.get("eye_openness_norm_mean"))
        aperture = _num(eye_row.get("aperture_iris_mean"))
        lines.append(
            f"EAR={ear:.3f} blink={blink:.3f} openness={openness:.3f} aperture/iris={aperture:.3f}"
        )
    if primary is not None:
        pitch = _num(primary.get("Pitch"))
        yaw = _num(primary.get("Yaw"))
        gaze_pitch = _num(primary.get("gaze_pitch"))
        gaze_yaw = _num(primary.get("gaze_yaw"))
        emotion = _top_emotion(primary)
        lines.append(
            f"pose(pitch,yaw)=({pitch:.2f},{yaw:.2f}) gaze=({gaze_pitch:.2f},{gaze_yaw:.2f}) emotion={emotion}"
        )

    y = 24
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
        y += 22
    return out


class _VideoReader:
    def __init__(self, video: Path, seek_threshold: int = 120) -> None:
        self.cap = cv2.VideoCapture(str(video))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source video: {video}")
        self.last: int | None = None
        self.seek_threshold = int(seek_threshold)

    def read(self, target: int) -> np.ndarray:
        if self.last is None or target <= self.last or target - self.last > self.seek_threshold:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(target))
        else:
            for _ in range(target - self.last - 1):
                if not self.cap.grab():
                    raise RuntimeError(f"Failed to advance video before frame {target}")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to decode video frame {target}")
        self.last = target
        return frame

    def close(self) -> None:
        self.cap.release()


def _select_even(values: list[int], n: int) -> list[int]:
    values = sorted(dict.fromkeys(int(v) for v in values))
    if len(values) <= n:
        return values
    idx = np.linspace(0, len(values) - 1, n).round().astype(int)
    return [values[i] for i in idx]


def _clip_positions(center_pos: int, manifest: pd.DataFrame, *, half_seconds: float, fps: float) -> list[int]:
    frame_col = _frame_col(manifest)
    row = manifest[manifest[frame_col] == center_pos]
    if row.empty:
        return [center_pos]
    window = row.iloc[0].get("dryrun_window")
    same = manifest[manifest["dryrun_window"].eq(window)].sort_values("benchmark_index") if "dryrun_window" in manifest.columns else manifest.sort_values("benchmark_index")
    pos_list = same[frame_col].astype(int).tolist()
    center_idx = min(range(len(pos_list)), key=lambda i: abs(pos_list[i] - center_pos))
    radius = int(round(half_seconds * fps))
    lo = max(0, center_idx - radius)
    hi = min(len(pos_list), center_idx + radius + 1)
    return pos_list[lo:hi]


def _write_clip(
    output: Path,
    positions: list[int],
    *,
    reader: _VideoReader,
    tracks_by_frame: dict[int, pd.DataFrame],
    eye_by_frame: dict[int, pd.Series],
    fps: float,
) -> None:
    if not positions:
        return
    first = reader.read(int(positions[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create QC video: {output}")
    try:
        writer.write(_annotate(first, tracks_by_frame.get(int(positions[0]), pd.DataFrame()), eye_by_frame.get(int(positions[0]))))
        for pos in positions[1:]:
            frame = reader.read(int(pos))
            writer.write(_annotate(frame, tracks_by_frame.get(int(pos), pd.DataFrame()), eye_by_frame.get(int(pos))))
    finally:
        writer.release()


def _write_contact_sheet(
    output: Path,
    positions: list[int],
    *,
    reader: _VideoReader,
    tracks_by_frame: dict[int, pd.DataFrame],
    eye_by_frame: dict[int, pd.Series],
    tile_width: int = 480,
    columns: int = 4,
) -> None:
    tiles: list[np.ndarray] = []
    for pos in positions:
        frame = reader.read(int(pos))
        annotated = _annotate(frame, tracks_by_frame.get(int(pos), pd.DataFrame()), eye_by_frame.get(int(pos)))
        scale = tile_width / annotated.shape[1]
        tile = cv2.resize(annotated, (tile_width, int(round(annotated.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"frame={pos}", (8, tile.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(tile, f"frame={pos}", (8, tile.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    tile_h = max(t.shape[0] for t in tiles)
    rows = int(math.ceil(len(tiles) / columns))
    canvas = np.zeros((rows * tile_h, columns * tile_width, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, columns)
        canvas[r * tile_h:r * tile_h + tile.shape[0], c * tile_width:(c + 1) * tile_width] = tile
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Failed to write contact sheet: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Face tracking/eyelid QC images and short clips from saved raw/derived outputs")
    parser.add_argument("--tracks", required=True, help="Window-aware face_tracks.parquet")
    parser.add_argument("--eye", required=True, help="eye_features.parquet")
    parser.add_argument("--sample-manifest", required=True, help="*_face-dryrun_manifest.json")
    parser.add_argument("--frame-manifest", required=True, help="*_face-dryrun_frames.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    tracks = pd.read_parquet(Path(args.tracks).resolve())
    eye = pd.read_parquet(Path(args.eye).resolve())
    frame_manifest = pd.read_csv(Path(args.frame_manifest).resolve())
    sample_meta = json.loads(Path(args.sample_manifest).resolve().read_text(encoding="utf-8"))
    source_video = Path(str(sample_meta["source_video"])).expanduser().resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_col = _frame_col(tracks)
    tracks_by_frame = {int(k): g.copy() for k, g in tracks.groupby(frame_col, sort=False)}
    eye_frame_col = _frame_col(eye)
    eye_by_frame = {int(row[eye_frame_col]): row for _, row in eye.iterrows()}

    counts = tracks.groupby(frame_col).size()
    multiface = counts[counts > 1].index.astype(int).tolist()
    multiface_contact = _select_even(multiface, 4)

    eye_valid = eye[pd.to_numeric(eye.get("ear_mean"), errors="coerce").notna()].copy()
    low_ear = eye_valid.nsmallest(4, "ear_mean")[eye_frame_col].astype(int).tolist() if not eye_valid.empty else []
    blink_numeric = pd.to_numeric(eye_valid.get("native_eyeBlink_mean"), errors="coerce")
    high_blink = (
        eye_valid.assign(_blink=blink_numeric).nlargest(4, "_blink")[eye_frame_col].astype(int).tolist()
        if blink_numeric.notna().any() else []
    )
    contact_positions = []
    for value in multiface_contact + low_ear + high_blink:
        if int(value) not in contact_positions:
            contact_positions.append(int(value))
    contact_positions = contact_positions[:12]

    min_ear_pos = int(eye_valid.nsmallest(1, "ear_mean").iloc[0][eye_frame_col]) if not eye_valid.empty else None
    multiface_center = int(np.median(multiface)) if multiface else None

    reader = _VideoReader(source_video)
    try:
        contact_path = out_dir / "face_qc_contact_sheet.jpg"
        _write_contact_sheet(
            contact_path,
            contact_positions,
            reader=reader,
            tracks_by_frame=tracks_by_frame,
            eye_by_frame=eye_by_frame,
        )

        outputs: dict[str, Any] = {"contact_sheet": str(contact_path) if contact_path.exists() else None}
        if multiface_center is not None:
            positions = _clip_positions(multiface_center, frame_manifest, half_seconds=5.0, fps=float(args.fps))
            path = out_dir / "face_qc_multiface_clip.mp4"
            _write_clip(path, positions, reader=reader, tracks_by_frame=tracks_by_frame, eye_by_frame=eye_by_frame, fps=float(args.fps))
            outputs["multiface_clip"] = str(path)
            outputs["multiface_clip_center_frame"] = multiface_center
        if min_ear_pos is not None:
            positions = _clip_positions(min_ear_pos, frame_manifest, half_seconds=3.0, fps=float(args.fps))
            path = out_dir / "face_qc_blink_extreme_clip.mp4"
            _write_clip(path, positions, reader=reader, tracks_by_frame=tracks_by_frame, eye_by_frame=eye_by_frame, fps=float(args.fps))
            outputs["blink_extreme_clip"] = str(path)
            outputs["blink_extreme_center_frame"] = min_ear_pos
    finally:
        reader.close()

    summary = {
        "schema_version": "rgb-face-qc-visualization-v0.1",
        "source_video": str(source_video),
        "tracks": str(Path(args.tracks).resolve()),
        "eye": str(Path(args.eye).resolve()),
        "sampled_fps": float(args.fps),
        "multi_face_frames": len(multiface),
        "contact_positions": contact_positions,
        "lowest_ear_frames": low_ear,
        "highest_native_eyeBlink_frames": high_blink,
        "outputs": outputs,
        "notes": [
            "QC rendering reads saved tracking/eyelid outputs plus the original AVI; it does not rerun Py-Feat.",
            "Primary boxes are green; other detected faces are orange. Eye and iris topology is drawn only for the selected primary face.",
            "The multiface clip is centered on the median multiface sampled frame; the blink clip is centered on the minimum-EAR sampled frame.",
        ],
    }
    summary_path = out_dir / "face_qc_visualization_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
