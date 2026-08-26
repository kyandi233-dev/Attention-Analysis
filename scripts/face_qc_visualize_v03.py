from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd

import face_qc_visualize as v01
import face_qc_visualize_v02 as v02


TEXT_COLOR = (20, 20, 20)


def _annotate_full_black(frame: np.ndarray, faces: pd.DataFrame, eye_row: pd.Series | None) -> np.ndarray:
    out = frame.copy()
    if faces.empty:
        return out

    primary_rows = (
        faces[faces["primary_face"].fillna(False).astype(bool)]
        if "primary_face" in faces.columns
        else faces.iloc[0:0]
    )
    primary = primary_rows.iloc[0] if not primary_rows.empty else None

    if primary is not None:
        v02._draw_full_mesh(out, primary)

    for _, row in faces.iterrows():
        v01._draw_bbox(out, row, primary=bool(row.get("primary_face", False)))

    if primary is not None:
        v01._draw_points(out, primary, v01.RIGHT_EYE, (255, 255, 0))
        v01._draw_points(out, primary, v01.LEFT_EYE, (255, 255, 0))
        v01._draw_points(out, primary, v01.RIGHT_IRIS, (255, 0, 255))
        v01._draw_points(out, primary, v01.LEFT_IRIS, (255, 0, 255))

    lines: list[str] = []
    first = faces.iloc[0]
    lines.append(
        f"window={first.get('dryrun_window')} phase={first.get('phase')} "
        f"unix_ms={first.get('unix_ms')} faces={len(faces)}"
    )

    if eye_row is not None:
        ear = v01._num(eye_row.get("ear_mean"))
        blink = v01._num(eye_row.get("native_eyeBlink_mean"))
        openness = v01._num(eye_row.get("eye_openness_norm_mean"))
        aperture = v01._num(eye_row.get("aperture_iris_mean"))
        lines.append(
            f"EAR={ear:.3f} blink={blink:.3f} openness={openness:.3f} aperture/iris={aperture:.3f}"
        )

    if primary is not None:
        pitch = v01._num(primary.get("Pitch"))
        yaw = v01._num(primary.get("Yaw"))
        gaze_pitch = v01._num(primary.get("gaze_pitch"))
        gaze_yaw = v01._num(primary.get("gaze_yaw"))
        emotion = v01._top_emotion(primary)
        lines.append(
            f"pose(pitch,yaw)=({pitch:.2f},{yaw:.2f}) "
            f"gaze=({gaze_pitch:.2f},{gaze_yaw:.2f}) emotion={emotion}"
        )

    y = 24
    for line in lines:
        cv2.putText(
            out,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        y += 22
    return out


def _write_contact_sheet_black(
    output,
    positions,
    *,
    reader,
    tracks_by_frame,
    eye_by_frame,
    tile_width: int = 480,
    columns: int = 4,
) -> None:
    tiles: list[np.ndarray] = []
    for pos in positions:
        frame = reader.read(int(pos))
        annotated = _annotate_full_black(
            frame,
            tracks_by_frame.get(int(pos), pd.DataFrame()),
            eye_by_frame.get(int(pos)),
        )
        scale = tile_width / annotated.shape[1]
        tile = cv2.resize(
            annotated,
            (tile_width, int(round(annotated.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        cv2.putText(
            tile,
            f"frame={pos}",
            (8, tile.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        return
    tile_h = max(t.shape[0] for t in tiles)
    rows = int(math.ceil(len(tiles) / columns))
    canvas = np.zeros((rows * tile_h, columns * tile_width, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, columns)
        canvas[
            r * tile_h:r * tile_h + tile.shape[0],
            c * tile_width:(c + 1) * tile_width,
        ] = tile
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Failed to write contact sheet: {output}")


def main() -> None:
    # v0.3 is the recommended QC renderer: full 478-point mesh plus single-layer
    # black diagnostic text. v0.1/v0.2 remain unchanged for provenance.
    v01._annotate = _annotate_full_black
    v01._write_contact_sheet = _write_contact_sheet_black
    v01.main()


if __name__ == "__main__":
    main()
