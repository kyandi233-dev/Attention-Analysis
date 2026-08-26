from __future__ import annotations

import cv2
import numpy as np

import face_qc_visualize as v01


FULL_MESH_POINTS = list(range(478))
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
ORIGINAL_ANNOTATE = v01._annotate


def _draw_full_mesh(frame: np.ndarray, row) -> None:
    # Full 478-point diagnostic overlay. Every retained mesh point is shown so
    # whole-face drift is visible; feature contours are added for orientation.
    for idx in FULL_MESH_POINTS:
        pt = v01._point(row, idx)
        if pt is not None:
            cv2.circle(frame, pt, 1, (145, 145, 145), -1, cv2.LINE_AA)

    for indices, closed in [
        (FACE_OVAL, True),
        (LIPS_OUTER, True),
        (NOSE_BRIDGE, False),
    ]:
        pts = [v01._point(row, idx) for idx in indices]
        pts = [pt for pt in pts if pt is not None]
        if len(pts) >= 2:
            poly = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [poly], closed, (190, 190, 190), 1, cv2.LINE_AA)


def _annotate_full(frame: np.ndarray, faces, eye_row):
    base = frame.copy()
    if not faces.empty and "primary_face" in faces.columns:
        primary_rows = faces[faces["primary_face"].fillna(False).astype(bool)]
        if not primary_rows.empty:
            _draw_full_mesh(base, primary_rows.iloc[0])
    # Reuse v0.1 overlay so eye/iris landmarks, bbox, tracking labels and
    # numeric diagnostics remain identical; those high-value eye points are
    # drawn on top of the neutral full-face mesh.
    return ORIGINAL_ANNOTATE(base, faces, eye_row)


def main() -> None:
    v01._annotate = _annotate_full
    v01.main()


if __name__ == "__main__":
    main()
