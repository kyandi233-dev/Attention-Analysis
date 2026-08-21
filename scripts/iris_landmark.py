"""MediaPipe Iris Landmark 虹膜关键点封装（主环境 mediapipe，延迟导入）。

> 08-16（Asia/Shanghai）｜face_landmarker.task 输出 478 点（468 面部 + 10 iris），
> 取每眼 iris 5 点（468:473 右 / 473:478 左）用 minEnclosingCircle 拟合虹膜圆。
> ⚠️ 输出是【虹膜】中心/直径，非瞳孔（瞳孔在虹膜内，直径更小）；且依赖完整人脸检测，
> 特写画面（双眼/鼻梁，无完整脸）会失败——仅全脸 ROI 适用，如实标注不夸大。

输入：整帧图片目录（含完整人脸）。
输出：虹膜圆 CSV（中心/直径，kind=iris 标注）。

用法：
    $env:PYTHONPATH='src'
    & 'D:/Code/python/python.exe' scripts/iris_landmark.py --image_dir <dir> --out <csv>
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from attention_pipeline.nir.sequence import FaceLandmarkerSession


def iris_circles(points_xy: np.ndarray) -> dict | None:
    """从 478 点 landmark 提取双眼 iris 圆。返回 {eye: {center, diameter}} 或 None。

    points_xy 长度必须 ≥478（含 iris 点）；左右用虹膜中心 x 排序（x 小=eye_right，
    画面左=被摄者右眼，与 EYE_KEYS/EYE_CORNERS 约定一致），不依赖 landmark 索引左右约定。
    """
    if points_xy.ndim != 2 or points_xy.shape[0] < 478:
        return None
    iris = points_xy[468:478]  # 10 点：右眼 5 + 左眼 5
    if iris.shape[0] != 10 or not np.isfinite(iris).all():
        return None
    right5 = iris[0:5]
    left5 = iris[5:10]
    circles = {}
    for name, pts in (("eye_right", right5), ("eye_left", left5)):
        (cx, cy), radius = cv2.minEnclosingCircle(pts.astype(np.float32))
        circles[name] = {"center": (float(cx), float(cy)), "diameter": 2.0 * float(radius)}
    # 按 x 排序：x 小=eye_right
    if circles["eye_right"]["center"][0] > circles["eye_left"]["center"][0]:
        circles["eye_right"], circles["eye_left"] = circles["eye_left"], circles["eye_right"]
    return circles


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MediaPipe Iris Landmark 虹膜圆（吃整帧图片目录）")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=None, help="face_landmarker.task 路径（缺省读 config）")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    if args.model is None:
        script_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(script_dir.parent / "src"))
        from attention_pipeline.config import load_config
        config = load_config(str(script_dir.parent / "configs" / "formal.yaml"))
        model_path = str(config.path_value("face_landmarker_model"))
    else:
        model_path = args.model

    session = FaceLandmarkerSession(Path(model_path), confidence=args.confidence)
    files = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")) +
                   glob.glob(os.path.join(args.image_dir, "*.png")) +
                   glob.glob(os.path.join(args.image_dir, "*.bmp")))
    if args.limit:
        files = files[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["sample_id", "eye", "algorithm", "found", "center_x", "center_y",
            "major_diameter", "minor_diameter", "angle_deg", "confidence", "kind"]
    try:
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for i, fp in enumerate(files):
                sid = os.path.splitext(os.path.basename(fp))[0]
                frame = cv2.imread(fp, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                pts = session.detect(frame)
                circles = iris_circles(pts) if pts is not None else None
                for eye in ("eye_right", "eye_left"):
                    row = {"sample_id": sid, "eye": eye, "algorithm": "IrisLandmark",
                           "kind": "iris", "found": False}
                    if circles and circles[eye] is not None:
                        cx, cy = circles[eye]["center"]
                        d = circles[eye]["diameter"]
                        row.update({
                            "found": True, "center_x": cx, "center_y": cy,
                            "major_diameter": d, "minor_diameter": d,
                            "angle_deg": 0.0, "confidence": 1.0,
                        })
                    writer.writerow(row)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(files)}", flush=True)
    finally:
        session.close()
    print(f"wrote {out_path} ({len(files)} images × 2 eyes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
