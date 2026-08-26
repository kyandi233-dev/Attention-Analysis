from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, sqrt
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Ellipse:
    """Ellipse in pixel coordinates; axis_a/axis_b are full axis lengths."""

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
    return {"block1": "block1", "b1": "block1", "1": "block1",
            "block2": "block2", "b2": "block2", "2": "block2"}.get(
        raw, str(value).strip().lower()
    )


def resolve_column(
    columns: Iterable[str],
    aliases: Sequence[str],
    *,
    explicit: str | None = None,
    required: bool = True,
    label: str = "column",
) -> str | None:
    available = {str(column).lower(): str(column) for column in columns}
    if explicit:
        found = available.get(str(explicit).lower())
        if found is None:
            raise KeyError(f"{label}: configured column {explicit!r} not found")
        return found
    for alias in aliases:
        found = available.get(str(alias).lower())
        if found is not None:
            return found
    if required:
        raise KeyError(f"{label}: none of these aliases exist: {', '.join(aliases)}")
    return None


def ellipse_shape_matrix(ellipse: Ellipse) -> np.ndarray:
    if not ellipse.is_finite_positive():
        raise ValueError("ellipse must have finite positive axes")
    theta = np.deg2rad(float(ellipse.angle_deg))
    c, s = float(np.cos(theta)), float(np.sin(theta))
    rotation = np.asarray([[c, -s], [s, c]], dtype=float)
    radii_squared = np.diag(
        [(float(ellipse.axis_a) / 2.0) ** 2, (float(ellipse.axis_b) / 2.0) ** 2]
    )
    return rotation @ radii_squared @ rotation.T


def transform_ellipse_anisotropic(
    ellipse: Ellipse,
    *,
    scale_x: float,
    scale_y: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> Ellipse:
    """Map a rotated ellipse through non-uniform x/y scaling and translation."""
    sx, sy = float(scale_x), float(scale_y)
    if not (np.isfinite([sx, sy]).all() and sx > 0 and sy > 0):
        raise ValueError("scale_x/scale_y must be finite and positive")
    scale = np.diag([sx, sy])
    center = scale @ np.asarray([ellipse.cx, ellipse.cy], dtype=float)
    center += np.asarray([offset_x, offset_y], dtype=float)
    shape = scale @ ellipse_shape_matrix(ellipse) @ scale.T
    eigenvalues, eigenvectors = np.linalg.eigh(shape)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if np.any(eigenvalues <= 0) or not np.isfinite(eigenvalues).all():
        raise ValueError("mapped ellipse has invalid eigenvalues")
    axes = 2.0 * np.sqrt(eigenvalues)
    vector = eigenvectors[:, 0]
    angle = degrees(atan2(float(vector[1]), float(vector[0]))) % 180.0
    return Ellipse(
        cx=float(center[0]),
        cy=float(center[1]),
        axis_a=float(axes[0]),
        axis_b=float(axes[1]),
        angle_deg=float(angle),
    )


def shift_ellipse(ellipse: Ellipse, offset_x: float, offset_y: float) -> Ellipse:
    return Ellipse(
        cx=float(ellipse.cx) + float(offset_x),
        cy=float(ellipse.cy) + float(offset_y),
        axis_a=float(ellipse.axis_a),
        axis_b=float(ellipse.axis_b),
        angle_deg=float(ellipse.angle_deg),
    )


def ellipse_geometry_plausible(
    ellipse: Ellipse | None,
    width: float,
    height: float,
    *,
    center_margin_fraction: float = 0.15,
    max_axis_fraction: float = 1.5,
) -> bool:
    """Weak geometry sanity check only; this is not a pupil-validity threshold."""
    if ellipse is None or not ellipse.is_finite_positive():
        return False
    width, height = float(width), float(height)
    if width <= 0 or height <= 0:
        return False
    margin_x = width * center_margin_fraction
    margin_y = height * center_margin_fraction
    if not (
        -margin_x <= ellipse.cx <= width + margin_x
        and -margin_y <= ellipse.cy <= height + margin_y
    ):
        return False
    max_axis = max(width, height) * max_axis_fraction
    return bool(ellipse.axis_a <= max_axis and ellipse.axis_b <= max_axis)


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


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def _uniform_pick(values: Sequence[int], n: int) -> list[int]:
    values = sorted(set(int(value) for value in values))
    if n <= 0 or not values:
        return []
    if len(values) <= n:
        return values
    positions = np.linspace(0, len(values) - 1, num=n)
    indices = list(dict.fromkeys(np.rint(positions).astype(int).tolist()))
    picked = [values[index] for index in indices]
    if len(picked) < n:
        used = set(picked)
        picked.extend(value for value in values if value not in used)
    return sorted(picked[:n])


def deterministic_benchmark_sample(
    frame: pd.DataFrame,
    *,
    frame_col: str,
    phase_col: str,
    block_uniform_n: int,
    high_quality_n: int,
    difficult_n: int,
    confidence_col: str | None = None,
    primary_valid_col: str | None = None,
    strict_valid_col: str | None = None,
    difficult_flag_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Choose the sparse benchmark frames before any candidate algorithm runs."""
    work = frame.copy()
    work["frame_idx_norm"] = pd.to_numeric(work[frame_col], errors="coerce")
    work = work[work["frame_idx_norm"].notna()].copy()
    work["frame_idx_norm"] = work["frame_idx_norm"].astype("int64")
    work["phase_norm"] = work[phase_col].map(normalize_phase)
    work["confidence_norm"] = (
        pd.to_numeric(work[confidence_col], errors="coerce")
        if confidence_col and confidence_col in work
        else np.nan
    )
    work["primary_norm"] = (
        _as_bool(work[primary_valid_col])
        if primary_valid_col and primary_valid_col in work
        else False
    )
    work["strict_norm"] = (
        _as_bool(work[strict_valid_col])
        if strict_valid_col and strict_valid_col in work
        else False
    )
    difficult = pd.Series(False, index=work.index)
    for column in difficult_flag_cols:
        if column in work:
            difficult = difficult | _as_bool(work[column])
    work["difficult_norm"] = difficult
    work["restored_norm"] = work["primary_norm"] & ~work["strict_norm"]

    quality = work.groupby(["phase_norm", "frame_idx_norm"], as_index=False).agg(
        confidence=("confidence_norm", "max"),
        primary=("primary_norm", "max"),
        strict=("strict_norm", "max"),
        restored=("restored_norm", "max"),
        difficult_flag=("difficult_norm", "max"),
    )
    roles: dict[tuple[str, int], set[str]] = {}

    def add(phase: str, frame_idx: int, role: str) -> None:
        roles.setdefault((str(phase), int(frame_idx)), set()).add(role)

    for phase in ("block1", "block2"):
        candidates = quality.loc[quality["phase_norm"] == phase, "frame_idx_norm"].tolist()
        for frame_idx in _uniform_pick(candidates, int(block_uniform_n)):
            add(phase, frame_idx, f"{phase}_uniform")

    high = quality[quality["primary"].astype(bool)].copy()
    if high.empty:
        high = quality.copy()
    high["confidence_sort"] = pd.to_numeric(high["confidence"], errors="coerce").fillna(-np.inf)
    high = high.sort_values(
        ["confidence_sort", "phase_norm", "frame_idx_norm"],
        ascending=[False, True, True],
        kind="stable",
    )
    for _, row in high.head(max(0, int(high_quality_n))).iterrows():
        add(str(row["phase_norm"]), int(row["frame_idx_norm"]), "high_quality")

    hard = quality.copy()
    hard["difficulty_rank"] = (
        hard["restored"].astype(int) * 4
        + hard["difficult_flag"].astype(int) * 2
        + (~hard["primary"].astype(bool)).astype(int)
    )
    hard["confidence_sort"] = pd.to_numeric(hard["confidence"], errors="coerce").fillna(np.inf)
    hard = hard.sort_values(
        ["difficulty_rank", "confidence_sort", "phase_norm", "frame_idx_norm"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    for _, row in hard.head(max(0, int(difficult_n))).iterrows():
        add(str(row["phase_norm"]), int(row["frame_idx_norm"]), "difficult")

    requested = max(0, int(block_uniform_n)) * 2 + max(0, int(high_quality_n)) + max(0, int(difficult_n))
    target = min(len(quality), requested)
    if len(roles) < target:
        remaining = [
            (str(row["phase_norm"]), int(row["frame_idx_norm"]))
            for _, row in quality.sort_values(["phase_norm", "frame_idx_norm"]).iterrows()
            if (str(row["phase_norm"]), int(row["frame_idx_norm"])) not in roles
        ]
        need = target - len(roles)
        positions = _uniform_pick(list(range(len(remaining))), need)
        for position in positions:
            phase, frame_idx = remaining[position]
            add(phase, frame_idx, "fill")

    return pd.DataFrame(
        [
            {"phase": phase, "frame_idx": frame_idx, "sample_role": ";".join(sorted(role_set))}
            for (phase, frame_idx), role_set in sorted(roles.items())
        ],
        columns=["phase", "frame_idx", "sample_role"],
    )


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
        return pd.DataFrame(columns=["phase", "frame_idx"])
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
        eligible = [run for run in runs if len(run) >= n_frames]
        if not eligible:
            continue
        midpoint = float(np.median(values))
        options: list[tuple[float, int, np.ndarray]] = []
        for run in eligible:
            maximum_start = len(run) - n_frames
            ideal = int(np.clip(round(midpoint - run[0] - (n_frames - 1) / 2), 0, maximum_start))
            selected = run[ideal : ideal + n_frames]
            center = float((selected[0] + selected[-1]) / 2.0)
            options.append((abs(center - midpoint), int(selected[0]), selected))
        selected = min(options, key=lambda item: (item[0], item[1]))[2]
        return pd.DataFrame({"phase": phase, "frame_idx": selected.astype("int64")})
    return pd.DataFrame(columns=["phase", "frame_idx"])


def _object_value(obj: Any, names: Sequence[str]) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            value = getattr(obj, name)
            if callable(value) and name in {"valid", "isValid"}:
                return value()
            return value
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
    """Normalize PyPupilEXT result objects without importing the Windows extension."""
    if result is None:
        return {"ellipse": None, "algorithm_valid": False, "confidence": np.nan, "outline_confidence": np.nan}
    center = _pair(_object_value(result, ["center", "Center"]))
    size = _pair(_object_value(result, ["size", "Size"]))
    angle = _object_value(result, ["angle", "Angle"])
    ellipse = None
    if center is not None and size is not None:
        candidate = Ellipse(
            cx=float(center[0]), cy=float(center[1]),
            axis_a=float(size[0]), axis_b=float(size[1]),
            angle_deg=float(angle) if angle is not None else 0.0,
        )
        if candidate.is_finite_positive():
            ellipse = candidate
    valid = _object_value(result, ["valid", "isValid"])

    def numeric(value: Any) -> float:
        try:
            output = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return output if np.isfinite(output) else float("nan")

    return {
        "ellipse": ellipse,
        "algorithm_valid": bool(ellipse is not None if valid is None else valid),
        "confidence": numeric(_object_value(result, ["confidence", "Confidence"])),
        "outline_confidence": numeric(
            _object_value(result, ["outline_confidence", "outlineConfidence", "OutlineConfidence"])
        ),
    }
