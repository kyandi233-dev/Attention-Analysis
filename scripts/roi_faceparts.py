"""正式实验 ROI 提取——YOLO face-parts 后端（主环境，ultralytics）。

> 08-16（Asia/Shanghai）｜ignaciohrdz/yolo-face-parts-detector：直接检测 eye bbox，
> 不依赖完整人脸，专为 close-up/遮挡/缺脸设计（此前 MediaPipe/YuNet/YOLO-face 全 0/60）。
> 类别映射（EDA/prepare_full_dataset.py:501）：0=eye, 1=nose, 2=mouth, 3=eyebrow。
> 单 eye 类不区分左右，按框中心 x 排序：图像左=subject 右眼（与 EYE_CORNERS 一致）。

用法：
    $env:PYTHONPATH='src'
    & 'D:/Code/python/python.exe' scripts/roi_faceparts.py --subject sub-011 --conf 0.3 --imgsz 1280

产出同 roi_mediapipe：artifacts/formal-validate/<subject>/

注意：
  - ultralytics 推理默认 letterbox 到 imgsz×imgsz；1080p 特写 letterbox 后眼会压缩，
    检出率对 imgsz 敏感，实测可试 640/1280。
  - agnostic_nms=True 保留作者设定；会让 eye(0) 与 eyebrow(3) 跨类抑制。
  - faceparts 的 eye 框宽 ≈ 眼角跨距近似，corner_span 语义沿用 roi_yunet
    （ROI 源图宽 = eye 框宽 / corner_span）。
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import roi_common


def _make_model(model_path):
    """延迟导入 ultralytics（测试可 monkeypatch 此钩子，无需真装 ultralytics）。"""
    from ultralytics import YOLO
    return YOLO(str(model_path))


class FacePartsRoi:
    base_name = "faceparts"

    def __init__(self, model_path, out_w: int, out_h: int, corner_span: float = 0.5,
                 conf: float = 0.25, imgsz: int = 640):
        started = time.perf_counter()
        self._model = _make_model(model_path)
        self.model_load_ms = (time.perf_counter() - started) * 1000.0
        self.last_timing = {}
        self.out_w, self.out_h = out_w, out_h
        self.corner_span = float(corner_span)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        # 类别映射必须与库定义一致（0=eye），否则左右/类别判定都会错
        names = self._model.names
        if names.get(0) != "eye":
            raise ValueError(f"faceparts 模型类别映射异常（期望 0=eye，实际 {names}）")
        arch = "n"
        stem = str(model_path).lower()
        for candidate in ("x", "l", "m", "s", "n"):
            if f"yolov8{candidate}" in stem:
                arch = candidate
                break
        self.name = f"{self.base_name}-{arch}"

    def eyes(self, frame: np.ndarray) -> dict[str, dict] | None:
        started = time.perf_counter()
        result = self._model(
            frame, conf=self.conf, verbose=False, agnostic_nms=True, imgsz=self.imgsz
        )[0]
        inference_ms = (time.perf_counter() - started) * 1000.0
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            self.last_timing = {"inference_ms": inference_ms, "crop_normalize_ms": 0.0}
            return None
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        eye_idx = [i for i in range(len(xyxy)) if clss[i] == 0]
        if len(eye_idx) < 2:
            self.last_timing = {"inference_ms": inference_ms, "crop_normalize_ms": 0.0}
            return None
        # 取 conf 最高的两个 eye 框（>2 框时丢弃低置信误检），再按 x 中心排序分左右
        eye_boxes = sorted(((xyxy[i], confs[i]) for i in eye_idx), key=lambda t: -t[1])[:2]
        eye_boxes.sort(key=lambda t: (t[0][0] + t[0][2]) / 2.0)  # x 小=eye_right
        crop_started = time.perf_counter()
        out = {}
        for eye, (box, _conf) in zip(roi_common.EYE_KEYS, eye_boxes):
            x0, y0, x1, y1 = box
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            eye_width = x1 - x0
            box_w = max(8, int(round(eye_width / self.corner_span)))
            box_h = max(8, int(round(box_w * self.out_h / self.out_w)))
            roi = roi_common.crop_resize_gray_payload(
                frame,
                (int(cx) - box_w // 2, int(cy) - box_h // 2,
                 int(cx) + box_w // 2, int(cy) + box_h // 2),
                self.out_w, self.out_h,
                reference_points=np.asarray([[cx, cy]], dtype=float),
                reference_kind="faceparts_eye_bbox",
            )
            if roi is not None:
                out[eye] = roi
        self.last_timing = {
            "inference_ms": inference_ms,
            "crop_normalize_ms": (time.perf_counter() - crop_started) * 1000.0,
        }
        return out or None

    def close(self):
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="正式实验 ROI 提取：YOLO face-parts 后端")
    roi_common.add_common_args(parser)
    parser.add_argument("--faceparts-model", default=None)
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO 检测置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理输入尺寸（letterbox 到 imgsz×imgsz）")
    args = parser.parse_args(argv)
    model_path = roi_common.resolve_common_args(args, "faceparts_model", "faceparts_model")
    provider = FacePartsRoi(model_path, args.roi_w, args.roi_h, args.corner_span,
                            conf=args.conf, imgsz=args.imgsz)
    return roi_common.finish(args, provider)


if __name__ == "__main__":
    raise SystemExit(main())
