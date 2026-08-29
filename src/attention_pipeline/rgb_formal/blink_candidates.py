from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)

BASE_FACE_COLUMNS = {
    "subject", "unix_ms", "video_frame_position", "capture_frame_idx", "phase", "block",
    "trial_num", "cycle_num", "behavior_state", "is_probe", "probe_onset_time",
    "temporal_gap", "capture_gap_before", "dt_ms", "face_count", "face_track_id",
    "primary_face", "is_primary_face", "face_rank", "detected", "face_valid",
}
EYE_MESH_COLUMNS = {
    f"mesh_{axis}_{index}"
    for index in set(RIGHT_EYE + LEFT_EYE)
    for axis in ("x", "y")
}
FACE_PROJECTION_COLUMNS = BASE_FACE_COLUMNS | EYE_MESH_COLUMNS


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - CI/formal env includes Parquet support
        raise RuntimeError("pyarrow is required for projected Face Parquet reads") from exc
    return list(pq.ParquetFile(path).schema.names)


def read_face_projection(path: str | Path) -> pd.DataFrame:
    """Read only the Face columns required by the active Blink candidate contract."""
    source = Path(path)
    available = set(_parquet_columns(source))
    selected = sorted(FACE_PROJECTION_COLUMNS & available)
    if "unix_ms" not in selected:
        raise ValueError("face raw missing unix_ms")
    # Never fall back to an unprojected read. Missing optional columns are handled downstream.
    return pd.read_parquet(source, columns=selected)


def _ear(table: pd.DataFrame, points: Iterable[int]) -> pd.Series:
    p = list(points)
    coords: dict[int, tuple[pd.Series, pd.Series]] = {}
    for idx in p:
        coords[idx] = (
            pd.to_numeric(table[f"mesh_x_{idx}"], errors="coerce"),
            pd.to_numeric(table[f"mesh_y_{idx}"], errors="coerce"),
        )

    def dist(a: int, b: int) -> pd.Series:
        ax, ay = coords[a]
        bx, by = coords[b]
        return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    denom = 2.0 * dist(p[0], p[3]).replace(0, np.nan)
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / denom


def _primary_face_projection(face: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select only demonstrably unambiguous primary-face rows.

    ``face_rank == 0`` is diagnostic only and is never sufficient to establish a primary face.
    """
    if face.empty:
        return pd.DataFrame(), {"primary_face_status": "not_estimable", "primary_face_reason": "face_source_empty"}
    if "unix_ms" not in face.columns:
        raise ValueError("face raw missing unix_ms")

    primary_col = next((c for c in ("primary_face", "is_primary_face") if c in face.columns), None)
    out_rows: list[pd.Series] = []
    ambiguous_frames = 0
    rank_only_frames = 0
    multi_face_frames = 0
    explicit_primary_frames = 0
    singleton_frames = 0

    for _, group in face.groupby("unix_ms", sort=True, dropna=False):
        frame = group.copy()
        if "face_count" in frame.columns:
            counts = pd.to_numeric(frame["face_count"], errors="coerce").dropna()
            face_count = int(counts.max()) if len(counts) else len(frame)
        else:
            face_count = len(frame)
        if face_count > 1 or len(frame) > 1:
            multi_face_frames += 1

        selected: pd.Series | None = None
        selection_source = ""
        if primary_col is not None:
            mask = frame[primary_col].fillna(False).astype(bool)
            if int(mask.sum()) == 1:
                selected = frame.loc[mask].iloc[0].copy()
                selection_source = "explicit_unique_primary"
                explicit_primary_frames += 1
        if selected is None and len(frame) == 1 and face_count <= 1:
            selected = frame.iloc[0].copy()
            selection_source = "unambiguous_single_face"
            singleton_frames += 1

        if selected is None:
            ambiguous_frames += 1
            if "face_rank" in frame.columns and pd.to_numeric(frame["face_rank"], errors="coerce").eq(0).any():
                rank_only_frames += 1
            selected = frame.iloc[0].copy()
            selected["primary_face_reliable"] = False
            selected["primary_face_selection_source"] = "ambiguous_no_reliable_primary"
            selected["multi_face_ambiguous"] = True
        else:
            detected_ok = True
            if "detected" in selected.index:
                detected_ok = bool(selected.get("detected", False))
            if "face_valid" in selected.index:
                detected_ok = detected_ok and bool(selected.get("face_valid", False))
            selected["primary_face_reliable"] = bool(detected_ok)
            selected["primary_face_selection_source"] = selection_source if detected_ok else "face_detection_or_validity_false"
            selected["multi_face_ambiguous"] = False
        selected["observed_face_count"] = max(face_count, len(frame))
        out_rows.append(selected)

    out = pd.DataFrame(out_rows).sort_values("unix_ms").reset_index(drop=True)
    return out, {
        "primary_face_status": "generated",
        "ambiguous_primary_frames": ambiguous_frames,
        "multi_face_frames": multi_face_frames,
        "rank0_only_rejected_frames": rank_only_frames,
        "explicit_primary_frames": explicit_primary_frames,
        "unambiguous_single_face_frames": singleton_frames,
    }


def _reference(
    ear: pd.Series,
    phase: pd.Series,
    *,
    preferred_phase: str,
    minimum_valid_frames: int,
) -> tuple[float, str, int]:
    values = pd.to_numeric(ear, errors="coerce")
    valid = values[np.isfinite(values) & (values > 0)]
    preferred = valid[phase.reindex(valid.index).astype(str).eq(preferred_phase)]
    if len(preferred) >= minimum_valid_frames:
        source = preferred
        label = f"{preferred_phase}_top30_median"
    elif len(valid) >= minimum_valid_frames:
        source = valid
        label = "all_valid_top30_median_fallback_not_resting_baseline"
    else:
        return np.nan, "not_estimable_insufficient_open_reference", int(len(valid))
    threshold = source.quantile(0.70)
    top = source[source >= threshold]
    return float(top.median()), label, int(len(source))


def _event_table(
    frame: pd.DataFrame,
    *,
    min_ms: float,
    max_ms: float,
    gap_reset_ms: float,
) -> tuple[pd.Series, pd.DataFrame, float | None]:
    times = pd.to_numeric(frame["unix_ms"], errors="coerce")
    dt = times.diff()
    positive = dt[(dt > 0) & (dt <= gap_reset_ms)]
    nominal = float(positive.median()) if len(positive) else None

    closed = frame["blink_closed_bilateral_candidate"].fillna(False).astype(bool).to_numpy()
    breaks = frame["blink_segment_break"].fillna(True).astype(bool).to_numpy()
    t = times.to_numpy(float)
    ids = np.full(len(frame), np.nan)
    events: list[dict[str, Any]] = []
    eid = 0
    start: int | None = None

    def finish(end: int) -> None:
        nonlocal start, eid
        if start is None or end < start:
            start = None
            return
        if nominal is None or not np.isfinite(t[start]) or not np.isfinite(t[end]):
            start = None
            return
        duration = float((t[end] - t[start]) + nominal)
        if min_ms <= duration <= max_ms:
            eid += 1
            ids[start : end + 1] = eid
            events.append(
                {
                    "blink_event_id": eid,
                    "start_unix_ms": float(t[start]),
                    "end_unix_ms": float(t[end]),
                    "duration_ms": duration,
                    "frame_n": int(end - start + 1),
                    "event_type": "algorithm_defined_blink_candidate",
                }
            )
        start = None

    for i in range(len(frame)):
        if breaks[i]:
            if start is not None:
                finish(i - 1)
        if closed[i]:
            if start is None:
                start = i
        elif start is not None:
            finish(i - 1)
    if start is not None:
        finish(len(frame) - 1)

    return pd.Series(ids, index=frame.index, dtype="Float64"), pd.DataFrame(events), nominal


def derive_blink_candidates(
    face: pd.DataFrame,
    *,
    preferred_phase: str = "baseline",
    minimum_valid_frames: int = 30,
    relative_openness_threshold: float = 0.20,
    minimum_closed_duration_ms: float = 50.0,
    maximum_closed_duration_ms: float = 1000.0,
    gap_reset_ms: float = 250.0,
    maximum_bilateral_relative_difference: float = 0.35,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected, primary_status = _primary_face_projection(face)
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame(), {**primary_status, "blink_status": "not_estimable"}

    mesh_ok = all(f"mesh_{axis}_{idx}" in selected.columns for idx in set(RIGHT_EYE + LEFT_EYE) for axis in ("x", "y"))
    if not mesh_ok:
        return selected, pd.DataFrame(), {
            **primary_status,
            "blink_status": "not_estimable",
            "blink_reason": "eye_mesh_columns_missing",
        }

    out = selected[[c for c in selected.columns if c in FACE_PROJECTION_COLUMNS or c.startswith("primary_") or c in {"observed_face_count", "multi_face_ambiguous"}]].copy()
    out["ear_right"] = _ear(selected, RIGHT_EYE)
    out["ear_left"] = _ear(selected, LEFT_EYE)
    phase = out.get("phase", pd.Series("", index=out.index))
    left_ref, left_source, left_n = _reference(
        out["ear_left"], phase, preferred_phase=preferred_phase, minimum_valid_frames=minimum_valid_frames
    )
    right_ref, right_source, right_n = _reference(
        out["ear_right"], phase, preferred_phase=preferred_phase, minimum_valid_frames=minimum_valid_frames
    )
    out["left_eye_open_reference"] = left_ref
    out["right_eye_open_reference"] = right_ref
    out["left_eye_reference_source"] = left_source
    out["right_eye_reference_source"] = right_source
    out["left_eye_observable"] = out["ear_left"].notna() & bool(np.isfinite(left_ref) and left_ref > 0)
    out["right_eye_observable"] = out["ear_right"].notna() & bool(np.isfinite(right_ref) and right_ref > 0)
    out["left_eye_openness_norm"] = out["ear_left"] / left_ref if np.isfinite(left_ref) and left_ref > 0 else np.nan
    out["right_eye_openness_norm"] = out["ear_right"] / right_ref if np.isfinite(right_ref) and right_ref > 0 else np.nan
    denom = out[["left_eye_openness_norm", "right_eye_openness_norm"]].abs().median(axis=1).replace(0, np.nan)
    out["bilateral_openness_relative_difference"] = (
        out["left_eye_openness_norm"] - out["right_eye_openness_norm"]
    ).abs() / denom
    out["bilateral_eye_consistent"] = out["bilateral_openness_relative_difference"].le(maximum_bilateral_relative_difference)
    out["blink_bilateral_observable"] = (
        out["primary_face_reliable"].fillna(False).astype(bool)
        & out["left_eye_observable"].fillna(False).astype(bool)
        & out["right_eye_observable"].fillna(False).astype(bool)
    )
    out["blink_closed_bilateral_candidate"] = (
        out["blink_bilateral_observable"]
        & out["left_eye_openness_norm"].le(relative_openness_threshold)
        & out["right_eye_openness_norm"].le(relative_openness_threshold)
    )

    times = pd.to_numeric(out["unix_ms"], errors="coerce")
    gap_from_time = times.diff().gt(gap_reset_ms) | times.diff().le(0)
    source_gap = pd.Series(False, index=out.index)
    for col in ("temporal_gap", "capture_gap_before"):
        if col in out.columns:
            source_gap |= out[col].fillna(False).astype(bool)

    if "face_track_id" in out.columns:
        tracks = out["face_track_id"].astype("string")
        track_reset = tracks.notna() & tracks.shift(1).notna() & tracks.ne(tracks.shift(1))
    else:
        track_reset = pd.Series(False, index=out.index)
    unreliable = ~out["blink_bilateral_observable"].fillna(False).astype(bool)
    out["primary_face_track_reset"] = track_reset
    out["blink_segment_break"] = gap_from_time.fillna(True) | source_gap | track_reset | unreliable
    out.loc[out.index[0], "blink_segment_break"] = True

    out["blink_event_id"], events, nominal = _event_table(
        out,
        min_ms=minimum_closed_duration_ms,
        max_ms=maximum_closed_duration_ms,
        gap_reset_ms=gap_reset_ms,
    )
    fallback = left_source.endswith("fallback_not_resting_baseline") or right_source.endswith("fallback_not_resting_baseline")
    blink_estimable = bool(out["blink_bilateral_observable"].any() and nominal is not None)
    return out.reset_index(drop=True), events, {
        **primary_status,
        "blink_status": "generated" if blink_estimable else "not_estimable",
        "blink_reason": "" if blink_estimable else "bilateral_eye_or_time_interval_not_observable",
        "event_definition": "algorithm_defined_blink_candidate",
        "nominal_frame_interval_ms": nominal,
        "left_reference_source": left_source,
        "right_reference_source": right_source,
        "left_reference_n": left_n,
        "right_reference_n": right_n,
        "reference_fallback_not_resting_baseline": bool(fallback),
        "left_eye_observable_rows": int(out["left_eye_observable"].sum()),
        "right_eye_observable_rows": int(out["right_eye_observable"].sum()),
        "bilateral_observable_rows": int(out["blink_bilateral_observable"].sum()),
        "bilateral_inconsistent_rows": int((out["blink_bilateral_observable"] & ~out["bilateral_eye_consistent"]).sum()),
        "gap_break_rows": int((gap_from_time.fillna(False) | source_gap).sum()),
        "track_reset_rows": int(track_reset.sum()),
        "blink_event_candidate_n": int(len(events)),
        "manual_visual_validation_performed": False,
    }
