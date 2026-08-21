"""NIR 六算法单帧检测适配器（在 venv-pupil 环境运行）。

对一批 320×160 灰度 ROI 图，跑 PyPupilEXT 六种瞳孔算法 × {raw, CLAHE}，
输出 detections.csv 长表，供主环境 `nir benchmark` / `nir evaluate` 评估。

用法：
    venv-pupil/Scripts/python.exe scripts/nir_detect_batch.py \
        --manifest <sample_id,roi_path csv> \
        --out <detections.csv> \
        [--algorithms ElSe,ExCuSe,PuRe,PuReST,Starburst,Swirski2D] \
        [--preprocessing raw,clahe] [--limit N]

约定（与 PyPupilEXT 0.0.1 实测一致）：
- 椭圆 axis 为全长直径；angle（度）直接配 cv2.ellipse(center, (a/2,b/2), angle) 绘制。
- 未检出用 pupil.valid(0.0) 判定，不用 center 判断。
- PuReST 是状态化追踪器，独立帧之间必须每帧 reset()。
- import pypupilext 会向 stdout 打一行噪声；本脚本只写 CSV，parent 解析文件不受影响。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ALGORITHMS = ["ElSe", "ExCuSe", "PuRe", "PuReST", "Starburst", "Swirski2D"]
DEFAULT_PREPROCESSING = ["raw", "clahe"]
CLAHE_CLIP = 2.0
CLAHE_GRID = 8
# 光度合理性门控：inside=0.6× 缩小椭圆，ring=1.4× 放大椭圆减去 inside；面积<此值视为无效
PHOTO_INSIDE_SCALE = 0.6
PHOTO_RING_SCALE = 1.4
PHOTO_MIN_AREA = 5

OUTPUT_HEADER = [
    "sample_id", "algorithm", "preprocessing", "config", "returned",
    "center_x", "center_y",
    "raw_axis_w_px", "raw_axis_h_px", "raw_angle_deg",
    "major_diameter", "minor_diameter", "major_angle_deg", "angle_deg",
    "confidence", "outline_confidence", "runtime_ms",
    "photometric_contrast", "error",
]


def canonicalize_axes(axis_w: float, axis_h: float, angle_deg: float) -> tuple[float, float, float]:
    """Return major/minor diameters and the major-axis angle in [0, 180)."""
    w, h, angle = float(axis_w), float(axis_h), float(angle_deg)
    if w >= h:
        return w, h, angle % 180.0
    return h, w, (angle + 90.0) % 180.0


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append({
                "sample_id": str(raw["sample_id"]).strip(),
                "roi_path": str(raw["roi_path"]).strip(),
            })
    return rows


def load_gray(path: Path) -> np.ndarray:
    # cv2.imread 无法读取含中文路径，用 np.fromfile + imdecode 规避
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"无法读取 ROI 图: {path}")
    return np.ascontiguousarray(image)


def _apply_params(detector, algorithm: str, params: dict) -> None:
    """按参数 dict 设置检测器属性；Swirski2D 的属性在 detector.params 上。"""
    for key, value in params.items():
        target = detector.params if algorithm == "Swirski2D" else detector
        setattr(target, key, value)


def compute_photometric_contrast(
    gray: np.ndarray, center, axis_w: float, axis_h: float, angle_deg: float
) -> float:
    """光度合理性：(环均值 − 内均值)/255。inside=0.6×、ring=1.4×；面积<5px 或轴过小 → NaN。"""
    height, width = gray.shape
    cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
    inside_axes = (int(round(axis_w * PHOTO_INSIDE_SCALE / 2)), int(round(axis_h * PHOTO_INSIDE_SCALE / 2)))
    ring_axes = (int(round(axis_w * PHOTO_RING_SCALE / 2)), int(round(axis_h * PHOTO_RING_SCALE / 2)))
    if min(inside_axes) < 1 or min(ring_axes) < 1:
        return float("nan")
    inside_mask = np.zeros((height, width), dtype=np.uint8)
    ring_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(inside_mask, (cx, cy), inside_axes, int(round(angle_deg)) % 360, 0, 360, 255, -1)
    cv2.ellipse(ring_mask, (cx, cy), ring_axes, int(round(angle_deg)) % 360, 0, 360, 255, -1)
    ring = ring_mask & ~inside_mask
    if cv2.countNonZero(inside_mask) < PHOTO_MIN_AREA or cv2.countNonZero(ring) < PHOTO_MIN_AREA:
        return float("nan")
    inside_mean = float(gray[inside_mask > 0].mean())
    ring_mean = float(gray[ring > 0].mean())
    return (ring_mean - inside_mean) / 255.0


def run_detection(algorithm: str, image: np.ndarray, params: dict | None = None) -> dict:
    import pypupilext
    detector = getattr(pypupilext, algorithm)()
    if hasattr(detector, "reset"):
        detector.reset()  # 独立帧：PuReST 清状态，防止串帧跟踪
    if params:
        _apply_params(detector, algorithm, params)
    start = time.perf_counter()
    pupil = detector.runWithConfidence(image)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    base = {
        "confidence": float(pupil.confidence),
        "outline_confidence": float(pupil.outline_confidence),
        "runtime_ms": elapsed_ms,
    }
    if not pupil.valid(0.0):
        return {
            "returned": 0, "center_x": np.nan, "center_y": np.nan,
            "raw_axis_w_px": np.nan, "raw_axis_h_px": np.nan, "raw_angle_deg": np.nan,
            "major_diameter": np.nan, "minor_diameter": np.nan,
            "major_angle_deg": np.nan, "angle_deg": np.nan,
            "photometric_contrast": np.nan,
            **base,
        }
    center = pupil.center
    raw_axis_w, raw_axis_h = map(float, pupil.size)
    raw_angle = float(pupil.angle)
    major, minor, major_angle = canonicalize_axes(raw_axis_w, raw_axis_h, raw_angle)
    return {
        "returned": 1,
        "center_x": float(center[0]), "center_y": float(center[1]),
        "raw_axis_w_px": raw_axis_w, "raw_axis_h_px": raw_axis_h,
        "raw_angle_deg": raw_angle,
        "major_diameter": major, "minor_diameter": minor,
        "major_angle_deg": major_angle, "angle_deg": major_angle,
        "photometric_contrast": compute_photometric_contrast(
            image, center, raw_axis_w, raw_axis_h, raw_angle
        ),
        **base,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="NIR 六算法单帧检测适配器")
    parser.add_argument("--manifest", required=True, help="CSV: sample_id,roi_path")
    parser.add_argument("--out", required=True, help="输出 detections.csv 路径")
    parser.add_argument("--algorithms", default=",".join(DEFAULT_ALGORITHMS))
    parser.add_argument("--preprocessing", default=",".join(DEFAULT_PREPROCESSING))
    parser.add_argument("--params", default="{}", help='JSON: {algorithm: {属性: 值}}（参数调优用）')
    parser.add_argument("--config-name", default="default", help="参数集合名，写入 config 列（sweep 区分用）")
    parser.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 个样本（冒烟）")
    args = parser.parse_args(argv)

    algorithms = [name for name in args.algorithms.split(",") if name]
    preprocessing = [name for name in args.preprocessing.split(",") if name]
    params_by_algorithm = json.loads(args.params) if args.params.strip() else {}
    samples = load_manifest(Path(args.manifest))
    if args.limit > 0:
        samples = samples[: args.limit]

    # 提前 import，把 pypupilext 的 stdout 噪声留在 CSV 写入之前
    import pypupilext  # noqa: F401

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_HEADER)
        writer.writeheader()
        done = 0
        for sample in samples:
            try:
                raw = load_gray(Path(sample["roi_path"]))
            except Exception as exc:
                for algorithm in algorithms:
                    for prep in preprocessing:
                        writer.writerow({
                            "sample_id": sample["sample_id"], "algorithm": algorithm,
                            "preprocessing": prep, "config": args.config_name, "returned": 0,
                            "photometric_contrast": np.nan, "error": f"load_failed:{exc}",
                        })
                done += 1
                print(f"[progress] {done}/{len(samples)}", file=sys.stderr)
                continue
            clahe_image = (
                cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_GRID, CLAHE_GRID)).apply(raw)
                if "clahe" in preprocessing else None
            )
            for prep in preprocessing:
                image = raw if prep == "raw" else clahe_image
                for algorithm in algorithms:
                    row = {
                        "sample_id": sample["sample_id"], "algorithm": algorithm,
                        "preprocessing": prep, "config": args.config_name,
                    }
                    try:
                        row.update(run_detection(algorithm, image, params_by_algorithm.get(algorithm)))
                        row["error"] = ""
                    except Exception as exc:
                        row.update({
                            "returned": 0, "center_x": np.nan, "center_y": np.nan,
                            "raw_axis_w_px": np.nan, "raw_axis_h_px": np.nan, "raw_angle_deg": np.nan,
                            "major_diameter": np.nan, "minor_diameter": np.nan,
                            "major_angle_deg": np.nan, "angle_deg": np.nan,
                            "confidence": np.nan, "outline_confidence": np.nan, "runtime_ms": np.nan,
                            "photometric_contrast": np.nan,
                        })
                        row["error"] = f"{type(exc).__name__}:{exc}"
                    writer.writerow(row)
            done += 1
            print(f"[progress] {done}/{len(samples)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
