"""正式实验 ROI 提取——YuNet 后端（主环境，OpenCV 内置 FaceDetectorYN）。

> 08-16（Asia/Shanghai）｜人脸框 + 5 关键点 → 以眼中心裁固定 ROI。
> 关键点顺序为【图像】右/左：kp[0]=图像右（=subject 左眼）、kp[1]=图像左（=subject 右眼），
> 映射到 EYE_KEYS 时已对调；ROI 样本图里可人眼核对左右。cv2.dnn 不支持中文路径 → ascii_model_path。

用法：
    $env:PYTHONPATH='src'
    & 'D:/Code/python/python.exe' scripts/roi_yunet.py --subject sub-013

产出同 roi_mediapipe：artifacts/formal-validate/<subject>/
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import roi_common
from roi_common import ascii_model_path


class YuNetRoi:
    base_name = "yunet"

    def __init__(self, model_path, out_w: int, out_h: int, corner_span: float = 0.5, estimated_eye_width_ratio: float = 0.30):
        started = time.perf_counter()
        self._det = cv2.FaceDetectorYN.create(ascii_model_path(model_path), "", (0, 0), 0.5, 0.3, 500)
        self.model_load_ms = (time.perf_counter() - started) * 1000.0
        self.last_timing = {}
        self.out_w, self.out_h = out_w, out_h
        self.corner_span = float(corner_span)
        self.estimated_eye_width_ratio = float(estimated_eye_width_ratio)
        self.name = f"{self.base_name}-span{int(round(self.corner_span * 1000)):03d}"
        self._diag = False

    def eyes(self, frame: np.ndarray) -> dict[str, np.ndarray] | None:
        h, w = frame.shape[:2]
        started = time.perf_counter()
        self._det.setInputSize((w, h))
        ok, faces = self._det.detect(frame)
        inference_ms = (time.perf_counter() - started) * 1000.0
        if not ok or faces is None or len(faces) == 0:
            self.last_timing = {"inference_ms": inference_ms, "crop_normalize_ms": 0.0}
            return None
        # FaceDetectorYN returns landmarks in input-image pixel coordinates.
        # Multiplying by image width/height again sends them outside the frame.
        kp = faces[0, 4:14].reshape(5, 2).astype(float)
        if (
            not np.isfinite(kp).all()
            or (kp[:, 0] < 0).any()
            or (kp[:, 0] >= w).any()
            or (kp[:, 1] < 0).any()
            or (kp[:, 1] >= h).any()
        ):
            raise ValueError("yunet_keypoints_out_of_bounds")
        img_right, img_left = kp[0], kp[1]  # 图像右=subject 左；图像左=subject 右
        if not self._diag:
            self._diag = True
            print(f"[yunet] 诊断：图像右眼{kp[0].round(1)} 图像左眼{kp[1].round(1)} "
                  f"人脸框{faces[0, :4].round(1)}（ROI 样本图核对左右）", flush=True)
        eye_dist = float(np.linalg.norm(img_left - img_right))
        estimated_eye_width = max(eye_dist * self.estimated_eye_width_ratio, 8.0)
        box_w = max(8, int(round(estimated_eye_width / self.corner_span)))
        box_h = max(8, int(round(box_w * self.out_h / self.out_w)))
        crop_started = time.perf_counter()
        out = {}
        for eye, center in (("eye_right", img_left), ("eye_left", img_right)):
            cx, cy = int(center[0]), int(center[1])
            roi = roi_common.crop_resize_gray_payload(frame, (cx - box_w // 2, cy - box_h // 2,
                                                      cx + box_w // 2, cy + box_h // 2), self.out_w, self.out_h,
                                                      reference_points=np.asarray([center], dtype=float),
                                                      reference_kind="yunet_eye_center_estimated_scale")
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
    parser = argparse.ArgumentParser(description="正式实验 ROI 提取：YuNet 后端")
    roi_common.add_common_args(parser)
    parser.add_argument("--yunet-model", default=None)
    args = parser.parse_args(argv)
    model_path = roi_common.resolve_common_args(args, "yunet_model", "yunet_model")
    provider = YuNetRoi(model_path, args.roi_w, args.roi_h, args.corner_span)
    return roi_common.finish(args, provider)


if __name__ == "__main__":
    raise SystemExit(main())







