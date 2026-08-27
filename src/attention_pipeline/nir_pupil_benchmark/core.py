from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# Official PyPupilEXT NO_CONFIDENCE sentinel (Pupil.h).
OFFICIAL_CONFIDENCE_THRESHOLD = -1.0

# geometry_sane defaults (crop-pixel semantics, tuned to ~424x187 tight crops).
GEOMETRY_CENTER_MARGIN_FRACTION = 0.05
GEOMETRY_MAX_AXIS_FRACTION = 0.65
GEOMETRY_MIN_AXIS = 2.0
GEOMETRY_MIN_ASPECT = 0.2


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
    def major_axis(self) -> float:
        return max(float(self.axis_a), float(self.axis_b))

    @property
    def minor_axis(self) -> float:
        return min(float(self.axis_a), float(self.axis_b))

    @property
    def diameter_geom(self) -> float:
        if not self.is_finite_positive():
            return float("nan")
        return sqrt(float(self.axis_a) * float(self.axis_b))

    @property
    def area(self) -> float:
        if not self.is_finite_positive():
            return float("nan")
        return pi * float(self.axis_a) * float(self.axis_b) / 4.0


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


def geometry_sane(
    ellipse: Ellipse | None,
    width: float,
    height: float,
    *,
    center_margin_fraction: float = GEOMETRY_CENTER_MARGIN_FRACTION,
    max_axis_fraction: float = GEOMETRY_MAX_AXIS_FRACTION,
    min_axis: float = GEOMETRY_MIN_AXIS,
    min_aspect: float = GEOMETRY_MIN_ASPECT,
) -> bool:
    """Stricter geometric sanity for a detection candidate.

    This is a quality gate on *geometry alone*; it says nothing about whether
    the detection is a credible pupil. It is deliberately independent of the
    official valid()/confidence semantics so that confidence-less algorithms
    (ElSe/ExCuSe/Swirski2D/Starburst) can still be gated consistently.
    """
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
    major = ellipse.major_axis
    minor = ellipse.minor_axis
    if major > float(max_axis_fraction) * min(width, height):
        return False
    if minor < float(min_axis):
        return False
    if major <= 0 or minor / major < float(min_aspect):
        return False
    return True


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


def _numeric(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output if np.isfinite(output) else float("nan")


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


def _ellipse_from_pair(
    center, size, angle_deg, *, angle_reference: str = "size0"
) -> Ellipse | None:
    if center is None or size is None:
        return None
    major = max(float(size[0]), float(size[1]))
    minor = min(float(size[0]), float(size[1]))
    if major <= 0 or minor <= 0:
        return None
    angle = _numeric(angle_deg)
    if angle_reference == "size0" and float(size[1]) > float(size[0]):
        # OpenCV RotatedRect.angle describes the orientation of size.width.
        # Once axes are canonicalized to (major, minor), rotate by 90 degrees
        # when the original second axis was the major one.
        angle = (angle + 90.0) % 180.0
    elif np.isfinite(angle):
        angle = angle % 180.0
    candidate = Ellipse(
        cx=float(center[0]),
        cy=float(center[1]),
        axis_a=major,
        axis_b=minor,
        angle_deg=angle,
    )
    return candidate if candidate.is_finite_positive() else None


def normalize_result(result: Any) -> dict[str, Any] | None:
    """Extract common detection fields from a raw result object.

    Handles two official shapes:
    - PyPupilEXT ``Pupil`` (attribute-based; center/size are (x,y)/(w,h) tuples,
      full axis lengths; ``valid()`` and ``hasOutline()`` available).
    - pupil-detectors ``Detector2D.detect()`` dict (``ellipse.center/axes/angle``,
      ``confidence``; axes are (minor, major) full axis lengths).

    Returns None when the result carries no usable detection shape.
    """
    if result is None:
        return None

    # --- Pupil Labs 2D dict shape ---
    if isinstance(result, Mapping) and "ellipse" in result:
        ellipse = result.get("ellipse") or {}
        center = _pair(ellipse.get("center"))
        axes = _pair(ellipse.get("axes"))
        angle = _numeric(ellipse.get("angle"))
        confidence = _numeric(result.get("confidence"))
        has_outline = bool(axes is not None and axes[0] > 0 and axes[1] > 0)
        return {
            "center": center,
            "size": axes,  # (minor, major) full axes, ordered
            "angle_deg": angle,
            "native_confidence": confidence,
            "outline_confidence": float("nan"),
            "has_outline": has_outline,
            "official_valid": bool(confidence > 0) if np.isfinite(confidence) else None,
            # pupil-detectors already converts its fitted ellipse angle to the
            # major-axis convention while exporting axes=(minor, major).
            "angle_reference": "major",
        }

    # --- PyPupilEXT Pupil shape ---
    center = _pair(_object_value(result, ["center", "Center"]))
    size = _pair(_object_value(result, ["size", "Size"]))
    angle = _object_value(result, ["angle", "Angle"])
    has_outline_attr = getattr(result, "hasOutline", None)
    if callable(has_outline_attr):
        try:
            has_outline = bool(has_outline_attr())
        except Exception:
            has_outline = bool(size is not None and size[0] > 0 and size[1] > 0)
    else:
        has_outline = bool(size is not None and size[0] > 0 and size[1] > 0)

    official_valid = None
    valid_attr = getattr(result, "valid", None)
    if callable(valid_attr):
        # pybind11 binding does not expose the C++ default; the official default
        # threshold is NO_CONFIDENCE (-1.0), so pass it explicitly.
        try:
            official_valid = bool(valid_attr(OFFICIAL_CONFIDENCE_THRESHOLD))
        except TypeError:
            try:
                official_valid = bool(valid_attr())
            except Exception:
                official_valid = None
        except Exception:
            official_valid = None

    return {
        "center": center,
        "size": size,  # (w, h) full axis lengths, possibly unordered
        "angle_deg": _numeric(angle),
        "native_confidence": _numeric(_object_value(result, ["confidence", "Confidence"])),
        "outline_confidence": _numeric(
            _object_value(result, ["outline_confidence", "outlineConfidence", "OutlineConfidence"])
        ),
        "has_outline": has_outline,
        "official_valid": official_valid,
        "angle_reference": "size0",
    }


def parse_pupil_result(
    result: Any,
    *,
    width: float | None = None,
    height: float | None = None,
) -> dict[str, Any]:
    """Normalize a raw algorithm result with the three-layer semantics kept apart.

    - ``algorithm_returned``: the run completed and the object carries an
      ellipse outline (``hasOutline()`` / axes > 0). Explicitly NOT the same as
      "a credible pupil was detected".
    - ``official_valid``: strict official ``Pupil.valid(-1.0)`` when available;
      for confidence-less algorithms (ElSe/ExCuSe/Swirski2D/Starburst) this is
      False by official semantics; for Pupil Labs 2D it is confidence > 0.
    - ``geometry_sane``: pure geometric gate; None when crop size is unknown.

    ``width``/``height`` are the crop size in pixels; when given, ``geometry_sane``
    is evaluated. ``result`` may be None (no object produced).
    """
    empty = {
        "ellipse": None,
        "algorithm_returned": False,
        "official_valid": False,
        "geometry_sane": False if (width is not None and height is not None) else None,
        "center_x": float("nan"), "center_y": float("nan"),
        "major_axis": float("nan"), "minor_axis": float("nan"),
        "angle_deg": float("nan"), "diameter_geom": float("nan"), "area": float("nan"),
        "native_confidence": float("nan"), "outline_confidence": float("nan"),
    }
    normalized = normalize_result(result)
    if normalized is None:
        return empty

    ellipse = None
    algorithm_returned = False
    if normalized["has_outline"] and normalized["center"] is not None and normalized["size"] is not None:
        ellipse = _ellipse_from_pair(
            normalized["center"], normalized["size"], normalized["angle_deg"],
            angle_reference=normalized.get("angle_reference", "size0"),
        )
        algorithm_returned = ellipse is not None

    official_valid = bool(normalized["official_valid"]) if normalized["official_valid"] is not None else False

    sane: bool | None
    if width is not None and height is not None:
        sane = geometry_sane(ellipse, width, height)
    else:
        sane = None

    return {
        "ellipse": ellipse,
        "algorithm_returned": algorithm_returned,
        "official_valid": official_valid,
        "geometry_sane": sane,
        "center_x": ellipse.cx if ellipse else float("nan"),
        "center_y": ellipse.cy if ellipse else float("nan"),
        "major_axis": ellipse.major_axis if ellipse else float("nan"),
        "minor_axis": ellipse.minor_axis if ellipse else float("nan"),
        "angle_deg": ellipse.angle_deg if ellipse else float("nan"),
        "diameter_geom": ellipse.diameter_geom if ellipse else float("nan"),
        "area": ellipse.area if ellipse else float("nan"),
        "native_confidence": normalized["native_confidence"],
        "outline_confidence": normalized["outline_confidence"],
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
