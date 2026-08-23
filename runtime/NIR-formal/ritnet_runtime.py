"""Minimal in-process RITnet inference for the portable GPU trial package."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch


class RitnetRuntime:
    def __init__(self, package_root: Path, weights: Path, input_size: tuple[int, int] = (640, 400),
                 device: str = "0"):
        module_dir = package_root / "ritnet"
        sys.path.insert(0, str(module_dir))
        from densenet import DenseNet2D

        requested = str(device).strip().lower()
        if requested == "cpu" or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif requested.isdigit():
            self.device = torch.device(f"cuda:{requested}")
        else:
            self.device = torch.device(requested)
        self.input_size = input_size
        self.model = DenseNet2D(dropout=True, prob=0.2)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.gamma_table = (255.0 * (np.linspace(0, 1, 256) ** 0.8)).astype(np.uint8)
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def infer(self, roi_gray: np.ndarray) -> dict:
        roi_h, roi_w = roi_gray.shape[:2]
        image = cv2.resize(roi_gray, self.input_size)
        image = cv2.LUT(image, self.gamma_table)
        image = self.clahe.apply(image)
        tensor = torch.from_numpy(image).float().to(self.device)
        tensor = ((tensor / 255.0 - 0.5) / 0.5)[None, None]
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        pred = logits[0].argmax(0).cpu().numpy()
        pupil_prob = probs[0, 3].cpu().numpy()
        pupil = (pred == 3).astype(np.uint8)
        mask = cv2.resize(pupil, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"found": False}
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5 or cv2.contourArea(contour) < 5:
            return {"found": False}
        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        confidence = float(pupil_prob[pupil.astype(bool)].mean()) if pupil.any() else 0.0
        area = float(cv2.contourArea(contour))
        return {
            "found": True,
            "center_x": float(cx),
            "center_y": float(cy),
            "axis_a": float(axis_a),
            "axis_b": float(axis_b),
            "angle_deg": float(angle),
            "mask_area": area,
            "equiv_diameter": float(2 * np.sqrt(area / np.pi)),
            "pupil_confidence": confidence,
        }
