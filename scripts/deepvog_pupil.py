"""DeepVOG 瞳孔分割封装（2D master 版，延迟导入 keras/tensorflow）。

> 08-16（Asia/Shanghai）｜DeepVOG = UNet(10×10 大核) 输出瞳孔概率图 → 椭圆拟合。
> 仅用 2D 分割，不做 3D 眼球/视线（那需相机焦距/传感器标定，超出瞳孔直径需求）。
> 依赖 standalone keras + tensorflow（两环境均未装）→ 延迟导入，部署时另装（见 SETUP.md）。

输入：单眼 ROI 灰度图（任意 W×H），resize 到 240×320 训练域（宽320 高240）。
输出：瞳孔椭圆 CSV（对齐 RITnet infer 的最小列）。

用法（部署时装好 keras 后）：
    & 'D:/Code/python/python.exe' scripts/deepvog_pupil.py --roi_dir <dir> --out <csv>
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np


def load_model(model_path: str):
    """延迟导入 keras + DeepVOG 模型。model_path 指向 DeepVOG_weights.h5。

    权重与模型代码同目录（models/DeepVOG-master/deepvog/model/），load_DeepVOG()
    用 __file__ 定位权重，无需显式传权重路径。此处仅需把 deepvog 包根加入 sys.path。
    """
    h5 = Path(model_path).resolve()
    deepvog_root = h5.parent.parent.parent  # .../deepvog/model/DeepVOG_weights.h5 -> .../DeepVOG-master
    if str(deepvog_root) not in sys.path:
        sys.path.insert(0, str(deepvog_root))
    from deepvog.model.DeepVOG_model import load_DeepVOG
    return load_DeepVOG()


def preprocess(roi_gray: np.ndarray) -> np.ndarray:
    """roi_gray (H,W) uint8 -> (1,240,320,3) float [0,1]（对齐 DeepVOG _preprocess_image）。"""
    import cv2
    img = roi_gray.astype(np.float32) / 255.0
    img = cv2.resize(img, (320, 240))  # 宽320 高240（DeepVOG 固定训练域）
    return np.repeat(img[..., None], 3, axis=-1)[None, ...]


def infer(model, roi_gray: np.ndarray) -> np.ndarray:
    """返回瞳孔概率图 (240,320)（softmax 第 1 通道 = pupil）。"""
    prob = model.predict(preprocess(roi_gray), verbose=0)[0, :, :, 1]
    return prob


def pupil_ellipse(prob_map: np.ndarray, roi_w: int, roi_h: int) -> dict | None:
    """瞳孔概率图 → 阈值 → 缩回 ROI 坐标 → 最大连通域 → 椭圆。"""
    import cv2
    pupil = (prob_map > 0.5).astype(np.uint8)
    mask = cv2.resize(pupil, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5 or cv2.contourArea(c) < 5:
        return None
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(c)
    conf = float(prob_map[pupil.astype(bool)].mean()) if pupil.any() else 0.0
    return dict(
        found=True, center_x=float(cx), center_y=float(cy),
        major_diameter=float(MA), minor_diameter=float(ma),
        angle_deg=float(ang), confidence=conf,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DeepVOG 瞳孔分割（吃单眼 ROI 目录）")
    parser.add_argument("--roi_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=None, help="DeepVOG_weights.h5 路径（缺省读 config）")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    if args.model is None:
        script_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(script_dir.parent / "src"))
        from attention_pipeline.config import load_config
        config = load_config(str(script_dir.parent / "configs" / "formal.yaml"))
        model_path = str(config.path_value("deepvog_weights"))
    else:
        model_path = args.model

    model = load_model(model_path)
    files = sorted(glob.glob(os.path.join(args.roi_dir, "*_eye_*.png")))
    if args.limit:
        files = files[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    cols = ["sample_id", "algorithm", "found", "center_x", "center_y",
            "major_diameter", "minor_diameter", "angle_deg", "confidence"]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for i, fp in enumerate(files):
            sid = os.path.basename(fp)[: -len(".png")]
            roi = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if roi is None:
                writer.writerow({"sample_id": sid, "algorithm": "DeepVOG", "found": False})
                continue
            h, w = roi.shape[:2]
            prob = infer(model, roi)
            e = pupil_ellipse(prob, w, h)
            row = {"sample_id": sid, "algorithm": "DeepVOG"}
            row.update(e or {"found": False})
            writer.writerow(row)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(files)}", flush=True)
    print(f"wrote {out_path} ({len(files)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
