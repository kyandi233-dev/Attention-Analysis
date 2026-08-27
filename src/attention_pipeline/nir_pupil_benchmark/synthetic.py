"""Synthetic eye-crop generator for smoke tests.

Produces a grayscale eye-like crop with a dark pupil ellipse on a textured
iris, a bright specular glint, and mild eyelid shading, so classical edge- and
contrast-based detectors have a realistic signal to work with. The known ground
truth ellipse is returned alongside the image for verification.
"""
from __future__ import annotations

import numpy as np


def make_synthetic_eye(
    width: int = 424,
    height: int = 187,
    pupil_center: tuple[float, float] | None = None,
    major: float = 14.0,
    minor: float = 8.0,
    angle_deg: float = 12.0,
    iris_level: int = 130,
    pupil_level: int = 28,
    noise: float = 6.0,
    seed: int = 12345,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return (grayscale crop, ground-truth ellipse dict)."""
    import cv2

    width = int(width)
    height = int(height)
    if pupil_center is None:
        pupil_center = (width * 0.5, height * 0.5)
    cx, cy = float(pupil_center[0]), float(pupil_center[1])

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    # radial iris gradient centred on the pupil
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    iris = iris_level + 35 * np.exp(-((dist - 40) / 90.0) ** 2).astype(float)
    image = np.clip(iris, 0, 255).astype(np.uint8)

    # pupil ellipse (full axes: major x minor)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(cx), int(cy)),
        (int(major / 2), int(minor / 2)),
        float(angle_deg),
        0,
        360,
        255,
        -1,
    )
    image[mask > 0] = int(pupil_level)

    # specular glint near the pupil centre
    cv2.circle(image, (int(cx + 1), int(cy - 1)), 2, 190, -1)

    # upper-eyelid soft shading
    lid = np.clip(50 * np.exp(-((yy - 12) / 26.0) ** 2).astype(float), 0, 255).astype(np.uint8)
    image = np.clip(image.astype(np.int16) + lid, 0, 255).astype(np.uint8)

    if noise > 0:
        image = np.clip(
            image.astype(np.int16) + rng.normal(0, noise, image.shape), 0, 255
        ).astype(np.uint8)

    truth = {
        "center_x": cx,
        "center_y": cy,
        "major_axis": float(major),
        "minor_axis": float(minor),
        "angle_deg": float(angle_deg),
        "diameter_geom": float(np.sqrt(major * minor)),
    }
    return image, truth


def write_smoke_manifest(
    out_dir: str,
    *,
    n_frames: int = 4,
    width: int = 424,
    height: int = 187,
    seed: int = 7,
) -> list[dict]:
    """Generate n synthetic crops + a manifest CSV, returning the manifest rows."""
    import csv
    from pathlib import Path

    import cv2

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(seed)
    for frame_idx in range(n_frames):
        eye = "eye_left" if frame_idx % 2 == 0 else "eye_right"
        cx = width * 0.5 + float(rng.normal(0, 4))
        cy = height * 0.5 + float(rng.normal(0, 3))
        major = 14.0 + float(rng.uniform(-2, 2))
        minor = 8.0 + float(rng.uniform(-1.5, 1.5))
        angle = 12.0 + float(rng.uniform(-6, 6))
        image, _truth = make_synthetic_eye(
            width, height, (cx, cy), major, minor, angle, seed=seed * 1000 + frame_idx
        )
        name = f"smoke_{frame_idx:03d}_{eye}.png"
        cv2.imwrite(str(out / name), image)
        rows.append(
            {
                "subject": "sub-999",
                "phase": "block1",
                "frame_idx": frame_idx,
                "eye": eye,
                "sample_role": "smoke",
                "crop_path": name,
                "bbox_x1": 0,
                "bbox_y1": 0,
                "bbox_x2": width,
                "bbox_y2": height,
                "truth_center_x": cx,
                "truth_center_y": cy,
                "truth_major_axis": major,
                "truth_minor_axis": minor,
                "truth_angle_deg": angle,
            }
        )
    manifest = out / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows
