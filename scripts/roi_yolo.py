"""正式实验 ROI 提取——YOLO-face 后端（主环境，onnxruntime 零装包）。

> 08-16（Asia/Shanghai）｜本仓库 lindevs 版输出 [1,5,8400]（仅 bbox4+conf，无关键点），
> 眼点用【人脸框内比例】估计：图像左=subject 右（与 EYE_CORNERS 一致）。
> 若换带 5 关键点的模型（C>=15）自动用关键点。onnxruntime 支持中文路径，无需 ascii 处理。

用法：
    $env:PYTHONPATH='src'
    & 'D:/Code/python/python.exe' scripts/roi_yolo.py --subject sub-013

产出同 roi_mediapipe：artifacts/formal-validate/<subject>/
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import roi_common


class YoloRoi:
    base_name = "yolo"

    def __init__(self, model_path, out_w: int, out_h: int, corner_span: float = 0.5, estimated_eye_width_ratio: float = 0.30, score_threshold: float = 0.5):
        import onnxruntime as ort
        started = time.perf_counter()
        self._sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.model_load_ms = (time.perf_counter() - started) * 1000.0
        self.last_timing = {}
        self.out_w, self.out_h = out_w, out_h
        self.corner_span = float(corner_span)
        self.estimated_eye_width_ratio = float(estimated_eye_width_ratio)
        self.score_threshold = float(score_threshold)
        self.name = f"{self.base_name}-span{int(round(self.corner_span * 1000)):03d}"
        inp = self._sess.get_inputs()[0]
        self._in_name = inp.name
        self._in_h, self._in_w = int(inp.shape[2]), int(inp.shape[3])

    def _preprocess(self, frame):
        img = cv2_resize(frame, (self._in_w, self._in_h)).astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))[None, ...]

    def _parse(self, out, frame_shape):
        h, w = frame_shape[:2]
        arr = np.asarray(out[0])
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
            arr = arr.T  # -> [N, C]
        c = arr.shape[1] if arr.ndim == 2 else 0
        s = np.array([w / self._in_w, h / self._in_h])
        if c >= 15:  # 带 5 关键点头：bbox4 + kp10 + cls
            idx = int(np.argmax(arr[:, -1]))
            if float(arr[idx, -1]) < self.score_threshold:
                raise ValueError("yolo_face_below_threshold")
            cx, cy, bw, bh = arr[idx, :4]
            box = [(cx - bw / 2) * s[0], (cy - bh / 2) * s[1], (cx + bw / 2) * s[0], (cy + bh / 2) * s[1]]
            return box, arr[idx, 4:14].reshape(5, 2) * s
        if c == 5:  # 纯检测头（本仓库 lindevs 版）
            idx = int(np.argmax(arr[:, 4]))
            if float(arr[idx, 4]) < self.score_threshold:
                raise ValueError("yolo_face_below_threshold")
            cx, cy, bw, bh = arr[idx, :4]
            x0, y0 = (cx - bw / 2) * s[0], (cy - bh / 2) * s[1]
            x1, y1 = (cx + bw / 2) * s[0], (cy + bh / 2) * s[1]
            return [x0, y0, x1, y1], None
        raise ValueError(f"未识别的 YOLO-face 输出 shape {arr.shape}，需按实际模型调 _parse")

    def eyes(self, frame):
        started = time.perf_counter()
        box, kp = self._parse(self._sess.run(None, {self._in_name: self._preprocess(frame)}), frame.shape)
        inference_ms = (time.perf_counter() - started) * 1000.0
        crop_started = time.perf_counter()
        if kp is None:  # 无关键点 → 框内比例估计（图像左=subject 右）
            x0, y0, x1, y1 = box
            right_center = (x0 + 0.38 * (x1 - x0), y0 + 0.38 * (y1 - y0))
            left_center = (x0 + 0.62 * (x1 - x0), y0 + 0.38 * (y1 - y0))
        else:
            right_center, left_center = kp[0], kp[1]
        eye_dist = float(np.linalg.norm(np.array(left_center) - np.array(right_center)))
        estimated_eye_width = max(eye_dist * self.estimated_eye_width_ratio, 8.0)
        box_w = max(8, int(round(estimated_eye_width / self.corner_span)))
        box_h = max(8, int(round(box_w * self.out_h / self.out_w)))
        out = {}
        for eye, center in (("eye_right", right_center), ("eye_left", left_center)):
            cx, cy = int(center[0]), int(center[1])
            roi = roi_common.crop_resize_gray_payload(frame, (cx - box_w // 2, cy - box_h // 2,
                                                      cx + box_w // 2, cy + box_h // 2), self.out_w, self.out_h,
                                                      reference_points=np.asarray([center], dtype=float),
                                                      reference_kind="yolo_eye_center_or_bbox_estimate")
            if roi is not None:
                out[eye] = roi
        self.last_timing = {
            "inference_ms": inference_ms,
            "crop_normalize_ms": (time.perf_counter() - crop_started) * 1000.0,
        }
        return out or None

    def close(self):
        pass


def cv2_resize(img, size):
    import cv2
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="正式实验 ROI 提取：YOLO-face 后端")
    roi_common.add_common_args(parser)
    parser.add_argument("--yolo-model", default=None)
    args = parser.parse_args(argv)
    model_path = roi_common.resolve_common_args(args, "yolo_model", "yolo_model")
    provider = YoloRoi(model_path, args.roi_w, args.roi_h, args.corner_span)
    return roi_common.finish(args, provider)


if __name__ == "__main__":
    raise SystemExit(main())





