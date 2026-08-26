from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Ellipse:
    """Ellipse in native crop pixels; axis_a/axis_b are full axis lengths."""

    cx: float
    cy: float
    axis_a: float
    axis_b: float
    angle_deg: float = 0.0

    def is_finite_positive(self) -> bool:
        values = np.asarray(
            [self.cx, self.cy, self.axis_a, self.axis_b, self.angle_deg], dtype=float
        )
        return bool(
            np.isfinite(values).all()
            and float(self.axis_a) > 0
            and float(self.axis_b) > 0
        )

    @property
    def diameter_geom(self) -> float:
        if not self.is_finite_positive():
            return float("nan")
        return sqrt(float(self.axis_a) * float(self.axis_b))


def normalize_subject(value: str | int) -> str:
    raw = str(value).strip().lower().replace("_", "-")
    if raw.startswith("sub-"):
        suffix = raw[4:]
    elif raw.startswith("sub"):
        suffix = raw[3:].lstrip("-")
    else:
        suffix = raw
    if not suffix.isdigit():
        raise ValueError(f"invalid subject identifier: {value!r}")
    return f"sub-{int(suffix):03d}"


def normalize_phase(value: Any) -> str:
    raw = str(value).strip().lower().replace("_", "")
    return {
        "block1": "block1",
        "b1": "block1",
        "1": "block1",
        "block2": "block2",
        "b2": "block2",
        "2": "block2",
    }.get(raw, str(value).strip().lower())


def resolve_column(
    columns: Iterable[str],
    aliases: Sequence[str],
    *,
    required: bool = True,
    label: str = "column",
) -> str | None:
    available = {str(column).lower(): str(column) for column in columns}
    for alias in aliases:
        found = available.get(str(alias).lower())
        if found is not None:
            return found
    if required:
        raise KeyError(f"{label}: none of these aliases exist: {', '.join(aliases)}")
    return None


def _uniform_pick(values: Sequence[int], n: int) -> list[int]:
    values = sorted(set(int(value) for value in values))
    if n <= 0 or not values:
        return []
    if len(values) <= n:
        return values
    positions = np.rint(np.linspace(0, len(values) - 1, num=n)).astype(int)
    result: list[int] = []
    for position in positions:
        value = values[int(position)]
        if value not in result:
            result.append(value)
    if len(result) < n:
        used = set(result)
        result.extend(value for value in values if value not in used)
    return sorted(result[:n])


def deterministic_frame_sample(
    frame: pd.DataFrame,
    *,
    frame_col: str,
    phase_col: str,
    block_uniform_n: int,
) -> pd.DataFrame:
    """Uniformly sample each formal block before any pupil detector is run."""
    work = frame[[frame_col, phase_col]].copy()
    work["frame_idx"] = pd.to_numeric(work[frame_col], errors="coerce")
    work = work[work["frame_idx"].notna()].copy()
    work["frame_idx"] = work["frame_idx"].astype("int64")
    work["phase"] = work[phase_col].map(normalize_phase)
    work = work[work["phase"].isin(["block1", "block2"])]
    work = work.drop_duplicates(["phase", "frame_idx"]).sort_values(["phase", "frame_idx"])

    rows: list[dict[str, Any]] = []
    for phase in ("block1", "block2"):
        values = work.loc[work["phase"] == phase, "frame_idx"].tolist()
        for frame_idx in _uniform_pick(values, int(block_uniform_n)):
            rows.append({"phase": phase, "frame_idx": int(frame_idx), "sample_role": f"{phase}_uniform"})
    return pd.DataFrame(rows, columns=["phase", "frame_idx", "sample_role"])


def choose_continuous_window(
    frame: pd.DataFrame,
    *,
    frame_col: str,
    phase_col: str,
    n_frames: int,
    preferred_phase: str = "block1",
) -> pd.DataFrame:
    """Pick a consecutive frame-index run nearest the middle of the preferred block."""
    if n_frames <= 0:
        return pd.DataFrame(columns=["phase", "frame_idx", "sample_role"])
    work = frame[[frame_col, phase_col]].copy()
    work["frame_idx"] = pd.to_numeric(work[frame_col], errors="coerce")
    work = work[work["frame_idx"].notna()].copy()
    work["frame_idx"] = work["frame_idx"].astype("int64")
    work["phase"] = work[phase_col].map(normalize_phase)
    work = work.drop_duplicates(["phase", "frame_idx"]).sort_values(["phase", "frame_idx"])

    phases = [preferred_phase] + [phase for phase in ("block1", "block2") if phase != preferred_phase]
    for phase in phases:
        values = work.loc[work["phase"] == phase, "frame_idx"].to_numpy(dtype=np.int64)
        if len(values) < n_frames:
            continue
        runs = np.split(values, np.flatnonzero(np.diff(values) != 1) + 1)
        runs = [run for run in runs if len(run) >= n_frames]
        if not runs:
            continue
        midpoint = float(np.median(values))
        options: list[tuple[float, int, np.ndarray]] = []
        for run in runs:
            maximum_start = len(run) - n_frames
            ideal = int(np.clip(round(midpoint - run[0] - (n_frames - 1) / 2), 0, maximum_start))
            selected = run[ideal : ideal + n_frames]
            center = float((selected[0] + selected[-1]) / 2.0)
            options.append((abs(center - midpoint), int(selected[0]), selected))
        selected = min(options, key=lambda item: (item[0], item[1]))[2]
        return pd.DataFrame(
            {"phase": phase, "frame_idx": selected.astype("int64"), "sample_role": "temporal"}
        )
    return pd.DataFrame(columns=["phase", "frame_idx", "sample_role"])


def merge_sample_sets(*frames: pd.DataFrame) -> pd.DataFrame:
    roles: dict[tuple[str, int], set[str]] = {}
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for row in frame.itertuples(index=False):
            key = (str(row.phase), int(row.frame_idx))
            roles.setdefault(key, set()).update(str(row.sample_role).split(";"))
    return pd.DataFrame(
        [
            {"phase": phase, "frame_idx": frame_idx, "sample_role": ";".join(sorted(role_set))}
            for (phase, frame_idx), role_set in sorted(roles.items())
        ],
        columns=["phase", "frame_idx", "sample_role"],
    )


def ellipse_geometry_plausible(
    ellipse: Ellipse | None,
    width: float,
    height: float,
    *,
    center_margin_fraction: float = 0.05,
) -> bool:
    """Weak sanity check only; it must not be interpreted as pupil ground truth."""
    if ellipse is None or not ellipse.is_finite_positive():
        return False
    width, height = float(width), float(height)
    if width <= 0 or height <= 0:
        return False
    margin_x = width * float(center_margin_fraction)
    margin_y = height * float(center_margin_fraction)
    if not (-margin_x <= ellipse.cx <= width + margin_x):
        return False
    if not (-margin_y <= ellipse.cy <= height + margin_y):
        return False
    return bool(ellipse.axis_a <= width * 1.25 and ellipse.axis_b <= height * 1.25)


def center_distance(a: Ellipse | None, b: Ellipse | None) -> float:
    if a is None or b is None or not a.is_finite_positive() or not b.is_finite_positive():
        return float("nan")
    return float(np.hypot(a.cx - b.cx, a.cy - b.cy))


def safe_ratio(numerator: float, denominator: float) -> float:
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite([numerator, denominator]).all() or denominator <= 0:
        return float("nan")
    return numerator / denominator


def _object_value(obj: Any, names: Sequence[str]) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            value = getattr(obj, name)
            return value() if callable(value) and name in {"majorAxis", "minorAxis", "diameter"} else value
    return None


def _pair(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)
    if hasattr(value, "width") and hasattr(value, "height"):
        return float(value.width), float(value.height)
    try:
        values = list(value)
    except TypeError:
        return None
    if len(values) < 2:
        return None
    return float(values[0]), float(values[1])


def parse_pypupil_result(result: Any) -> dict[str, Any]:
    """Normalize a PyPupilEXT Pupil object without importing its binary extension."""
    if result is None:
        return {
            "ellipse": None,
            "returned": False,
            "confidence": np.nan,
            "outline_confidence": np.nan,
        }
    center = _pair(_object_value(result, ["center", "Center"]))
    size = _pair(_object_value(result, ["size", "Size"]))
    angle = _object_value(result, ["angle", "Angle"])

    ellipse = None
    if center is not None and size is not None:
        candidate = Ellipse(
            cx=float(center[0]),
            cy=float(center[1]),
            axis_a=float(size[0]),
            axis_b=float(size[1]),
            angle_deg=float(angle) if angle is not None else 0.0,
        )
        if candidate.is_finite_positive():
            ellipse = candidate

    def numeric(value: Any) -> float:
        try:
            output = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return output if np.isfinite(output) else float("nan")

    return {
        "ellipse": ellipse,
        "returned": bool(ellipse is not None),
        "confidence": numeric(_object_value(result, ["confidence", "Confidence"])),
        "outline_confidence": numeric(
            _object_value(result, ["outline_confidence", "outlineConfidence", "OutlineConfidence"])
        ),
    }


def temporal_stability_table(detections: pd.DataFrame) -> pd.DataFrame:
    """Frame-to-frame stability for rows belonging to the dedicated temporal sample."""
    if detections.empty:
        return pd.DataFrame()
    work = detections[detections["sample_role"].astype(str).str.contains("temporal", regex=False)].copy()
    work = work[work["geometry_plausible"].fillna(False).astype(bool)]
    rows: list[dict[str, Any]] = []
    keys = ["subject", "eye", "algorithm", "phase"]
    for key, group in work.groupby(keys, dropna=False):
        group = group.sort_values("frame_idx")
        if len(group) < 2:
            continue
        frame_diff = group["frame_idx"].diff()
        consecutive = frame_diff.eq(1)
        dx = group["center_x"].diff()[consecutive]
        dy = group["center_y"].diff()[consecutive]
        center_step = np.hypot(dx, dy)
        diameter_step = group["diameter_geom"].diff().abs()[consecutive]
        denom = group["diameter_geom"].shift(1)[consecutive]
        diameter_rel_step = diameter_step / denom.replace(0, np.nan)
        rows.append(
            {
                **dict(zip(keys, key if isinstance(key, tuple) else (key,))),
                "n_plausible": int(len(group)),
                "n_consecutive_pairs": int(consecutive.sum()),
                "center_step_median_px": float(center_step.median()) if len(center_step) else np.nan,
                "center_step_p95_px": float(center_step.quantile(0.95)) if len(center_step) else np.nan,
                "diameter_step_median_px": float(diameter_step.median()) if len(diameter_step) else np.nan,
                "diameter_rel_step_median": float(diameter_rel_step.median()) if len(diameter_rel_step) else np.nan,
                "diameter_rel_step_p95": float(diameter_rel_step.quantile(0.95)) if len(diameter_rel_step) else np.nan,
            }
        )
    return pd.DataFrame(rows)
