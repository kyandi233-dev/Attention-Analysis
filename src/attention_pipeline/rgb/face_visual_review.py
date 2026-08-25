from __future__ import annotations

import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from attention_pipeline.config import Config
from attention_pipeline.rgb.paths import RGBOutputLayout


FACE_VISUAL_REVIEW_SCHEMA = "rgb-face-visual-review-v0.2"
PYFEAT_AU_RE = re.compile(r"AU(\d+)$")
LIBREFACE_AU_RE = re.compile(r"(?:au_intensity__|int__)?au_(\d+)_intensity$")

# Shared MediaPipe topology subsets. Both candidates expose a MediaPipe-style
# 478-point face mesh, so the review should compare like with like rather than
# showing Py-Feat's 68-point compatibility subset against LibreFace's mesh.
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159,
             160, 161, 246, 33]
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386,
            387, 388, 466, 263]
RIGHT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
LEFT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270,
              269, 267, 0, 37, 39, 40, 185, 61]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310,
              311, 312, 13, 82, 81, 80, 191, 78]
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
NOSE_WINGS = [98, 97, 2, 326, 327]
RIGHT_IRIS = [468, 469, 470, 471, 472, 468]
LEFT_IRIS = [473, 474, 475, 476, 477, 473]
KEY_CONTOURS = [FACE_OVAL, RIGHT_EYE, LEFT_EYE, RIGHT_BROW, LEFT_BROW,
                LIPS_OUTER, LIPS_INNER, NOSE_BRIDGE, NOSE_WINGS]

AU_CN = {
    1: "内眉上扬", 2: "外眉上扬", 4: "眉毛下压", 5: "上眼睑抬高",
    6: "面颊抬高", 7: "眼睑收紧", 9: "皱鼻", 10: "上唇抬高",
    11: "鼻唇沟加深", 12: "嘴角上提", 14: "酒窝/嘴角收紧",
    15: "嘴角下拉", 17: "下巴抬高", 20: "嘴唇横向拉伸",
    23: "嘴唇收紧", 24: "嘴唇压紧", 25: "嘴唇分开",
    26: "下颌下降", 28: "嘴唇内吸", 43: "闭眼",
}

PYFEAT_EMOTION_CN = {
    "Neutral": "中性", "Happy": "快乐", "Sad": "悲伤", "Surprise": "惊讶",
    "Fear": "恐惧", "Disgust": "厌恶", "Anger": "愤怒",
}
LIBREFACE_EXPRESSION_CN = {
    "Neutral": "中性", "Happiness": "快乐", "Sadness": "悲伤",
    "Surprise": "惊讶", "Fear": "恐惧", "Disgust": "厌恶",
    "Anger": "愤怒", "Contempt": "蔑视",
}


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _fmt(value: float | None, digits: int = 2) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _primary_pyfeat(raw: pd.DataFrame) -> pd.DataFrame:
    input_col = next((c for c in ("input", "image_path", "file", "filename") if c in raw.columns), None)
    if input_col is None:
        raise ValueError("Py-Feat raw output has no recognizable input-path column")
    work = raw.copy()
    work["_image_name"] = work[input_col].astype(str).map(lambda p: Path(p).name)
    if "FaceScore" in work.columns:
        work["_score"] = pd.to_numeric(work["FaceScore"], errors="coerce").fillna(-np.inf)
        work = work.sort_values(["_image_name", "_score"], ascending=[True, False])
    return work.groupby("_image_name", as_index=False).first()


def _parse_pyfeat_mesh(row: pd.Series, width: int, height: int) -> list[tuple[int, int] | None]:
    xy: list[tuple[float, float] | None] = []
    numeric: list[tuple[float, float]] = []
    for i in range(478):
        x = _safe_float(row.get(f"mesh_x_{i}"))
        y = _safe_float(row.get(f"mesh_y_{i}"))
        point = None if x is None or y is None else (x, y)
        xy.append(point)
        if point is not None:
            numeric.append(point)
    if not numeric:
        return []

    xs = np.asarray([p[0] for p in numeric], dtype=float)
    ys = np.asarray([p[1] for p in numeric], dtype=float)
    # Detectorv2 normally writes mesh coordinates back in original-frame pixels.
    # Keep a defensive normalized-coordinate fallback for compatibility.
    normalized = (
        np.nanpercentile(np.abs(xs), 95) <= 2.0
        and np.nanpercentile(np.abs(ys), 95) <= 2.0
    )
    out: list[tuple[int, int] | None] = []
    for point in xy:
        if point is None:
            out.append(None)
            continue
        x, y = point
        if normalized:
            x *= width
            y *= height
        out.append((int(round(x)), int(round(y))))
    return out


def _parse_libreface_landmarks(value, width: int, height: int) -> list[tuple[int, int] | None]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        obj = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []
    points: list[tuple[int, int] | None] = []
    i = 0
    while f"lm_mp_{i}_x" in obj or f"lm_mp_{i}_y" in obj:
        x = _safe_float(obj.get(f"lm_mp_{i}_x"))
        y = _safe_float(obj.get(f"lm_mp_{i}_y"))
        if x is None or y is None:
            points.append(None)
        else:
            points.append((int(round(x * width)), int(round(y * height))))
        i += 1
    return points


def _parse_headpose(value) -> tuple[float | None, float | None, float | None]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None, None
    try:
        obj = json.loads(value) if isinstance(value, str) else value
    except Exception:
        obj = value
    if isinstance(obj, dict):
        return _safe_float(obj.get("pitch")), _safe_float(obj.get("yaw")), _safe_float(obj.get("roll"))
    text = str(obj)
    vals = []
    for key in ("pitch", "yaw", "roll"):
        match = re.search(rf"{key}\s*:\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        vals.append(float(match.group(1)) if match else None)
    return vals[0], vals[1], vals[2]


def _draw_contour(frame: np.ndarray, points: list[tuple[int, int] | None], indices: list[int], color, thickness=1) -> None:
    for a, b in zip(indices[:-1], indices[1:]):
        if a >= len(points) or b >= len(points):
            continue
        pa, pb = points[a], points[b]
        if pa is not None and pb is not None:
            cv2.line(frame, pa, pb, color, thickness, cv2.LINE_AA)


def _draw_shared_mesh(frame: np.ndarray, points: list[tuple[int, int] | None]) -> None:
    if not points:
        return
    for contour in KEY_CONTOURS:
        _draw_contour(frame, points, contour, (255, 220, 80), 1)
    # Iris is intentionally distinguished from eyelid/eye-corner geometry.
    _draw_contour(frame, points, RIGHT_IRIS, (255, 80, 255), 2)
    _draw_contour(frame, points, LEFT_IRIS, (255, 80, 255), 2)
    for idx in range(468, min(478, len(points))):
        if points[idx] is not None:
            cv2.circle(frame, points[idx], 2, (255, 80, 255), -1, cv2.LINE_AA)


def _point_or_center(points: list[tuple[int, int] | None], indices: list[int], fallback: tuple[int, int]) -> tuple[int, int]:
    selected = [points[i] for i in indices if i < len(points) and points[i] is not None]
    if not selected:
        return fallback
    return (
        int(round(float(np.mean([p[0] for p in selected])))),
        int(round(float(np.mean([p[1] for p in selected])))),
    )


def _draw_gaze(frame: np.ndarray, center: tuple[int, int], pitch_rad: float | None, yaw_rad: float | None, scale: int = 90) -> None:
    """Qualitative gaze direction cue, not a calibrated screen-point estimate."""
    if pitch_rad is None or yaw_rad is None:
        return
    dx = int(round(math.sin(yaw_rad) * scale))
    dy = int(round(-math.sin(pitch_rad) * scale))
    cv2.arrowedLine(frame, center, (center[0] + dx, center[1] + dy),
                    (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.2)


def _rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=float)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _draw_head_axes(frame: np.ndarray, center: tuple[int, int], pitch_rad: float | None,
                    yaw_rad: float | None, roll_rad: float | None, scale: int = 65) -> None:
    """Projected head-pose orientation cue. Axes are qualitative, not calibrated 3-D."""
    if pitch_rad is None or yaw_rad is None or roll_rad is None:
        return
    r = _rotation_matrix(pitch_rad, yaw_rad, roll_rad)
    # BGR: X red, Y green, Z blue.
    for axis, color, label in ((0, (0, 0, 255), "X"),
                               (1, (0, 255, 0), "Y"),
                               (2, (255, 0, 0), "Z")):
        dx = int(round(r[0, axis] * scale))
        dy = int(round(-r[1, axis] * scale))
        endpoint = (center[0] + dx, center[1] + dy)
        cv2.line(frame, center, endpoint, color, 3, cv2.LINE_AA)
        cv2.putText(frame, label, endpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _load_cn_font(size: int):
    try:
        from PIL import ImageFont
    except Exception:
        return None
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return None


def _put_panel(frame: np.ndarray, title: str, lines: list[str]) -> None:
    box_w = min(frame.shape[1] - 16, 720)
    line_h = 24
    box_h = min(frame.shape[0] - 16, 38 + line_h * len(lines))
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    title_font = _load_cn_font(20)
    body_font = _load_cn_font(16)
    if title_font is not None and body_font is not None:
        try:
            from PIL import Image, ImageDraw
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)
            draw.text((16, 13), title, font=title_font, fill=(255, 255, 255))
            y = 39
            for line in lines:
                draw.text((16, y), line, font=body_font, fill=(255, 255, 255))
                y += line_h
            frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            return
        except Exception:
            pass

    # Safe fallback when Pillow/CJK system fonts are unavailable.
    cv2.putText(frame, title.encode("ascii", "ignore").decode() or "Face review",
                (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    y = 52
    for line in lines:
        safe = line.encode("ascii", "ignore").decode()
        cv2.putText(frame, safe, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 23


def _top_pyfeat_aus(row: pd.Series, n: int = 3) -> list[str]:
    values = []
    for col in row.index:
        match = PYFEAT_AU_RE.fullmatch(str(col))
        if not match:
            continue
        value = _safe_float(row.get(col))
        if value is None:
            continue
        number = int(match.group(1))
        values.append((value, number))
    values.sort(reverse=True)
    return [f"AU{number:02d} {AU_CN.get(number, '')} = {value:.3f}" for value, number in values[:n]]


def _top_libreface_aus(row: pd.Series, n: int = 3) -> list[str]:
    values = []
    for col in row.index:
        match = LIBREFACE_AU_RE.fullmatch(str(col))
        if not match:
            continue
        value = _safe_float(row.get(col))
        if value is None:
            continue
        number = int(match.group(1))
        values.append((value, number))
    values.sort(reverse=True)
    return [f"AU{number:02d} {AU_CN.get(number, '')} = {value:.3f}" for value, number in values[:n]]


def _pyfeat_emotion(row: pd.Series) -> str:
    candidates = []
    for name, cn in PYFEAT_EMOTION_CN.items():
        value = _safe_float(row.get(name))
        if value is not None:
            candidates.append((value, name, cn))
    if not candidates:
        return "表情分类：NA"
    value, name, cn = max(candidates)
    return f"表情分类：{cn} ({name}) {value:.3f}"


def _libreface_expression(row: pd.Series) -> str:
    value = row.get("expression__facial_expression")
    if value is None or pd.isna(value):
        value = row.get("expr__facial_expression")
    if value is None or pd.isna(value):
        return "表情分类：NA"
    label = str(value)
    return f"表情分类：{LIBREFACE_EXPRESSION_CN.get(label, label)} ({label})"


def _draw_pyfeat(frame: np.ndarray, row: pd.Series) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    mesh = _parse_pyfeat_mesh(row, w, h)
    _draw_shared_mesh(out, mesh)

    x = _safe_float(row.get("FaceRectX"))
    y = _safe_float(row.get("FaceRectY"))
    bw = _safe_float(row.get("FaceRectWidth"))
    bh = _safe_float(row.get("FaceRectHeight"))
    if None not in (x, y, bw, bh):
        cv2.rectangle(out, (int(round(x)), int(round(y))),
                      (int(round(x + bw)), int(round(y + bh))), (0, 180, 0), 2)
        fallback_center = (int(round(x + bw / 2)), int(round(y + bh / 2)))
    else:
        fallback_center = (w // 2, h // 2)

    gaze_origin = _point_or_center(mesh, [33, 133, 362, 263], fallback_center)
    pose_origin = _point_or_center(mesh, [1], fallback_center)
    gp = _safe_float(row.get("gaze_pitch"))
    gy = _safe_float(row.get("gaze_yaw"))
    pitch = _safe_float(row.get("Pitch"))
    roll = _safe_float(row.get("Roll"))
    yaw = _safe_float(row.get("Yaw"))
    _draw_gaze(out, gaze_origin, gp, gy)
    _draw_head_axes(out, pose_origin, pitch, yaw, roll)

    face_score = _safe_float(row.get("FaceScore"))
    lines = [
        "黄色箭头=注视方向｜红X/绿Y/蓝Z=头姿方向",
        f"人脸检测置信度：{_fmt(face_score, 4)}",
        f"头姿 P点头 / R歪头 / Y转头：{_fmt(pitch, 3)} / {_fmt(roll, 3)} / {_fmt(yaw, 3)} rad",
        f"注视 pitch / yaw：{_fmt(gp, 3)} / {_fmt(gy, 3)} rad",
        _pyfeat_emotion(row),
        "主要AU（模型概率 0-1）：",
    ] + _top_pyfeat_aus(row, 3)
    _put_panel(out, "Py-Feat Detectorv2", lines)
    return out


def _draw_libreface(frame: np.ndarray, row: pd.Series) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    mesh = _parse_libreface_landmarks(row.get("landmarks_json"), w, h)
    _draw_shared_mesh(out, mesh)

    fallback_center = (w // 2, h // 2)
    gaze_origin = _point_or_center(mesh, [33, 133, 362, 263], fallback_center)
    pose_origin = _point_or_center(mesh, [1], fallback_center)

    gp_deg = _safe_float(row.get("gaze__gaze_pitch"))
    gy_deg = _safe_float(row.get("gaze__gaze_yaw"))
    pitch_deg, yaw_deg, roll_deg = _parse_headpose(row.get("headpose_json"))
    _draw_gaze(
        out,
        gaze_origin,
        math.radians(gp_deg) if gp_deg is not None else None,
        math.radians(gy_deg) if gy_deg is not None else None,
    )
    _draw_head_axes(
        out,
        pose_origin,
        math.radians(pitch_deg) if pitch_deg is not None else None,
        math.radians(yaw_deg) if yaw_deg is not None else None,
        math.radians(roll_deg) if roll_deg is not None else None,
    )

    aligned = bool(row.get("alignment_success", False))
    lines = [
        "黄色箭头=注视方向｜红X/绿Y/蓝Z=头姿方向",
        f"人脸对齐：{'成功' if aligned else '失败'}",
        f"头姿 P点头 / R歪头 / Y转头：{_fmt(pitch_deg)} / {_fmt(roll_deg)} / {_fmt(yaw_deg)} deg",
        f"注视 pitch / yaw：{_fmt(gp_deg)} / {_fmt(gy_deg)} deg",
        _libreface_expression(row),
        "主要AU强度（0-5）：",
    ] + _top_libreface_aus(row, 3)
    _put_panel(out, "LibreFace 2.0", lines)
    return out


def run_face_visual_review(config: Config, subject: str) -> dict[str, object]:
    layout = RGBOutputLayout.from_config(config)
    root = layout.test_dir() / "face-continuous" / subject
    manifests = sorted(root.glob("*_face-continuous_frames.csv"))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one continuous frame manifest in {root}, found {len(manifests)}")
    sample = pd.read_csv(manifests[0])
    py_path = root / "pyfeat_raw.parquet"
    lf_path = root / "libreface_raw.parquet"
    if not py_path.exists() or not lf_path.exists():
        raise FileNotFoundError("Both pyfeat_raw.parquet and libreface_raw.parquet are required")

    py = _primary_pyfeat(pd.read_parquet(py_path))
    py["_image_name"] = py["_image_name"].astype(str)
    lf = pd.read_parquet(lf_path)

    work = sample.copy()
    work["_image_name"] = work["image_path"].astype(str).map(lambda p: Path(p).name)
    work = work.merge(py, on="_image_name", how="left", suffixes=("", "__pyfeat"))
    lf_keep = [c for c in lf.columns if c == "benchmark_index" or c not in work.columns]
    work = work.merge(lf[lf_keep], on="benchmark_index", how="left")

    first = cv2.imread(str(work.iloc[0]["image_path"]))
    if first is None:
        raise RuntimeError("Could not read first continuous review image")
    h, w = first.shape[:2]
    fps = 10.0
    if "dt_ms" in sample.columns:
        med = pd.to_numeric(sample["dt_ms"], errors="coerce").dropna().median()
        if pd.notna(med) and med > 0:
            fps = 1000.0 / float(med)

    # v0.2 intentionally writes a new file, preserving the earlier v0.1 review.
    output = root / f"{subject}_face-visual-review-v2.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output}")
    frames_written = 0
    try:
        for row in work.itertuples(index=False):
            s = pd.Series(row._asdict())
            image = cv2.imread(str(s["image_path"]))
            if image is None:
                continue
            left = _draw_pyfeat(image, s)
            right = _draw_libreface(image, s)
            writer.write(np.concatenate([left, right], axis=1))
            frames_written += 1
    finally:
        writer.release()

    summary = {
        "schema_version": FACE_VISUAL_REVIEW_SCHEMA,
        "subject": subject,
        "source_manifest": str(manifests[0]),
        "frames_requested": int(len(work)),
        "frames_written": int(frames_written),
        "video_fps": float(fps),
        "output": str(output),
        "shared_overlay": "Both panels use the same MediaPipe-style key contours; iris points 468-477 are highlighted separately when available.",
        "gaze_overlay": "Yellow arrow = qualitative gaze direction cue; not a calibrated screen-point estimate.",
        "head_pose_overlay": "Red X / green Y / blue Z = projected head-pose orientation cue; qualitative, not calibrated 3-D.",
        "left_panel": "Py-Feat: 478 mesh key contours, iris, gaze, head-pose axes, top 3 AU probabilities with Chinese descriptions, top 7-class emotion probability.",
        "right_panel": "LibreFace: MediaPipe key contours, iris, gaze, head-pose axes, top 3 AU intensities with Chinese descriptions, categorical expression and face-alignment status.",
        "warning": "This is model-prediction visualization, not human ground-truth annotation. Py-Feat AU values [0,1] and LibreFace AU intensity [0,5] are different output scales and must not be directly compared numerically.",
    }
    manifest = root / f"{subject}_face-visual-review-v2_manifest.json"
    manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["manifest"] = str(manifest)
    return summary
