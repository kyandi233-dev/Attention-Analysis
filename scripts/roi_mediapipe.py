"""正式实验 ROI 提取——mediapipe 后端（主环境）。

> 08-16（Asia/Shanghai）｜复用 FaceLandmarkerSession；正式双眼特写上完整人脸landmark为0/60，仅保留候选实现
> 眼角 landmark → normalized_eye_roi 320×160，口径最接近阶段5 序列。

用法：
    $env:PYTHONPATH='src'
    & 'D:/Code/python/python.exe' scripts/roi_mediapipe.py --subject sub-013
    & 'D:/Code/python/python.exe' scripts/roi_mediapipe.py --subject sub-013 --n-segments 5 --frames-per-seg 120

产出：artifacts/formal-validate/<subject>/（segments PNG + manifest.csv + detections_sequence.csv + summary.csv + roi_samples/）
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import roi_common
from attention_pipeline.nir.roi import EYE_CORNERS, normalized_eye_roi
from attention_pipeline.nir.sequence import FaceLandmarkerSession, ear_for_eyes


class MediaPipeRoi:
    base_name = "mediapipe"

    def __init__(self, model_path, out_w: int, out_h: int, corner_span: float = 0.5,
                 confidence: float = 0.5, clahe: bool = False):
        started = time.perf_counter()
        self._session = FaceLandmarkerSession(Path(model_path), confidence=confidence)  # 内部处理中文路径
        self.model_load_ms = (time.perf_counter() - started) * 1000.0
        self.last_timing = {}
        self.out_w, self.out_h = out_w, out_h
        self.corner_span = float(corner_span)
        self.confidence = float(confidence)
        self.clahe = bool(clahe)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None
        tag = f"span{int(round(self.corner_span * 1000)):03d}c{int(round(self.confidence * 100)):03d}"
        if clahe:
            tag += "clahe"
        self.name = f"{self.base_name}-{tag}"

    def eyes(self, frame: np.ndarray) -> dict[str, np.ndarray] | None:
        started = time.perf_counter()
        pts = self._session.detect(frame)
        inference_ms = (time.perf_counter() - started) * 1000.0
        if pts is None:
            self.last_timing = {"inference_ms": inference_ms, "crop_normalize_ms": 0.0}
            return None
        crop_started = time.perf_counter()
        out = {}
        ears = ear_for_eyes(pts)
        for eye in roi_common.EYE_KEYS:
            roi, affine, _dist = normalized_eye_roi(
                frame, pts, EYE_CORNERS[eye], (self.out_w, self.out_h), self.corner_span
            )
            if roi is None or affine is None:
                continue
            gray = np.ascontiguousarray(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
            if self._clahe is not None:
                gray = self._clahe.apply(gray)  # Heimerl 思路：对比度增强
            canthi = pts[list(EYE_CORNERS[eye])]
            payload = roi_common.roi_payload(
                gray, affine, canthi_source=canthi, reference_kind="real_canthi",
                reference_points_source=canthi,
            )
            payload["ear"] = float(ears[eye])
            out[eye] = payload
        self.last_timing = {
            "inference_ms": inference_ms,
            "crop_normalize_ms": (time.perf_counter() - crop_started) * 1000.0,
        }
        return out or None

    def close(self):
        self._session.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="正式实验 ROI 提取：mediapipe 后端")
    roi_common.add_common_args(parser)
    parser.add_argument("--face-model", default=None)
    parser.add_argument("--confidence", type=float, default=0.5, help="MediaPipe 检测/存在/跟踪置信度阈值")
    parser.add_argument("--clahe", action="store_true", help="ROI 对比度增强（Heimerl 思路）")
    args = parser.parse_args(argv)
    model_path = roi_common.resolve_common_args(args, "face_landmarker_model", "face_model")
    provider = MediaPipeRoi(model_path, args.roi_w, args.roi_h, args.corner_span,
                            confidence=args.confidence, clahe=args.clahe)
    return roi_common.finish(args, provider)


if __name__ == "__main__":
    raise SystemExit(main())








