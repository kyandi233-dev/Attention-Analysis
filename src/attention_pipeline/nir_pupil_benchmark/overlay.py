"""Ellipse overlay / montage helpers for manual QC.

Overlays are a human-inspection aid, never a correctness metric. Detected
ellipses are drawn from the unified schema columns; the source crop is loaded
from disk via ``crop_path``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def draw_detection(
    image: np.ndarray,
    row: dict,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw the ellipse stored in a unified row onto a BGR copy of the image."""
    import cv2

    out = image
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    else:
        out = out.copy()
    try:
        cx = float(row["center_x"])
        cy = float(row["center_y"])
        major = float(row["major_axis"])
        minor = float(row["minor_axis"])
        angle = float(row["angle_deg"])
    except (KeyError, TypeError, ValueError):
        return out
    if not np.isfinite([cx, cy, major, minor]).all() or major <= 0 or minor <= 0:
        return out
    cv2.ellipse(
        out,
        (int(cx), int(cy)),
        (int(major / 2), int(minor / 2)),
        angle,
        0,
        360,
        color,
        thickness,
    )
    cv2.circle(out, (int(cx), int(cy)), 2, color, -1)
    return out


def write_algorithm_montage(
    frame: pd.DataFrame,
    crop_root: str | Path,
    out_dir: str | Path,
    *,
    algorithms=None,
    max_frames: int = 24,
    palette=None,
) -> list[Path]:
    """Write one montage PNG per algorithm with the detected ellipse overlaid.

    Returns the list of written paths. Only rows with a valid crop and either a
    returned ellipse or a failure are drawn (failed rows still appear labelled).
    """
    crop_root = Path(crop_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    algorithms = list(algorithms) if algorithms else sorted(frame["algorithm"].unique())
    palette = palette or [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (200, 120, 40),
    ]
    written: list[Path] = []
    for algorithm in algorithms:
        sub = frame[frame["algorithm"] == algorithm].head(max_frames)
        if sub.empty:
            continue
        images = []
        labels = []
        for row in sub.to_dict("records"):
            crop_path = row.get("crop_path")
            image = None
            if crop_path:
                path = crop_root / str(crop_path)
                if path.exists():
                    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                image = np.zeros((200, 320, 3), dtype=np.uint8)
            overlay = draw_detection(image, row)
            returned = bool(row.get("algorithm_returned"))
            sane = bool(row.get("geometry_sane"))
            status = "RET" if returned else "MISS"
            if returned and sane:
                status = "OK"
            elif returned and not sane:
                status = "GEOM?"
            failure = row.get("failure")
            label = f"{status}"
            if failure:
                label += f" {str(failure)[:22]}"
            labels.append(label)
            images.append(overlay)
        cols = min(6, len(images))
        rows = int(np.ceil(len(images) / cols))
        grid = np.zeros((rows * 200, cols * 320, 3), dtype=np.uint8)
        for i, (image, label) in enumerate(zip(images, labels)):
            r, c = divmod(i, cols)
            resized = cv2.resize(image, (320, 200), interpolation=cv2.INTER_AREA)
            cv2.putText(resized, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 4)
            cv2.putText(resized, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 200, 255), 1)
            grid[r * 200:(r + 1) * 200, c * 320:(c + 1) * 320] = resized
        path = out_dir / f"overlay_{algorithm}.png"
        cv2.imwrite(str(path), grid)
        written.append(path)
    return written
