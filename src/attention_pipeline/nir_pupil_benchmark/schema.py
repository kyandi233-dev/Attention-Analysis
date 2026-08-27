"""Unified benchmark schema and per-algorithm metadata.

The seven algorithms share one output schema (RESULT_COLUMNS). Per-algorithm
facts (native confidence, statefulness, internal resize, diameter override
support) are recorded here so provenance never has to be re-derived.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Canonical algorithm names. Pupil Labs 2D is the standalone
# pupil-detectors Detector2D; the other six come from PyPupilEXT.
ALGORITHMS: tuple[str, ...] = (
    "PuRe",
    "PuReST",
    "PupilLabs2D",
    "ElSe",
    "ExCuSe",
    "Swirski2D",
    "Starburst",
)

# Unified output row, one per (frame, eye, algorithm).
RESULT_COLUMNS: tuple[str, ...] = (
    "subject",
    "phase",
    "frame_idx",
    "eye",
    "sample_role",
    "algorithm",
    "algorithm_returned",
    "official_valid",
    "geometry_sane",
    "center_x",
    "center_y",
    "major_axis",
    "minor_axis",
    "angle_deg",
    "diameter_geom",
    "area",
    "runtime_ms",
    "native_confidence",
    "outline_confidence",
    "confidence_runtime_ms",
    "failure",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "input_width",
    "input_height",
    "params_provenance",
)


@dataclass(frozen=True)
class AlgorithmSpec:
    """Static facts about one algorithm, from official source audit (docs 030)."""

    name: str
    package: str                      # "pypupilext" | "pupil_detectors"
    class_name: str
    has_native_confidence: bool
    has_state: bool                   # cross-frame state in the detector object
    supports_diameter_override: bool  # PuRe/PuReST: set min/maxPupilDiameterMM
    internal_resize_note: str
    default_params: dict = field(default_factory=dict)


ALGORITHM_SPECS: dict[str, AlgorithmSpec] = {
    "PuRe": AlgorithmSpec(
        name="PuRe",
        package="pypupilext",
        class_name="PuRe",
        has_native_confidence=True,
        has_state=False,
        supports_diameter_override=True,
        internal_resize_note=(
            "downscales to baseSize(320,240), maps result back to crop coords; "
            "mm->px range is estimated from crop diagonal assuming full face"
        ),
        default_params={
            "meanCanthiDistanceMM": 27.6,
            "maxPupilDiameterMM": 8.0,
            "minPupilDiameterMM": 2.0,
            "baseSize": (320, 240),
        },
    ),
    "PuReST": AlgorithmSpec(
        name="PuReST",
        package="pypupilext",
        class_name="PuReST",
        has_native_confidence=True,
        has_state=True,
        supports_diameter_override=True,
        internal_resize_note=(
            "same resize as PuRe; stateful (previousPupil decides full run vs "
            "tracking); reset() between independent frames"
        ),
        default_params={
            "meanCanthiDistanceMM": 27.6,
            "maxPupilDiameterMM": 8.0,
            "minPupilDiameterMM": 2.0,
            "baseSize": (320, 240),
        },
    ),
    "PupilLabs2D": AlgorithmSpec(
        name="PupilLabs2D",
        package="pupil_detectors",
        class_name="Detector2D",
        has_native_confidence=True,
        has_state=True,
        supports_diameter_override=False,
        internal_resize_note=(
            "coarse localization downsamples x2 when roi area > 320x240; final "
            "fit is full-res in the roi; defaults assume full eye image scale"
        ),
        default_params={
            "coarse_detection": True,
            "coarse_filter_min": 128,
            "coarse_filter_max": 280,
            "intensity_range": 23,
            "blur_size": 5,
            "canny_treshold": 160,
            "canny_ration": 2,
            "canny_aperture": 5,
            "pupil_size_max": 100,
            "pupil_size_min": 10,
            "strong_perimeter_ratio_range_min": 0.8,
            "strong_perimeter_ratio_range_max": 1.1,
            "strong_area_ratio_range_min": 0.6,
            "strong_area_ratio_range_max": 1.1,
            "contour_size_min": 5,
            "ellipse_roundness_ratio": 0.1,
            "initial_ellipse_fit_treshhold": 1.8,
            "final_perimeter_ratio_range_min": 0.6,
            "final_perimeter_ratio_range_max": 1.2,
            "ellipse_true_support_min_dist": 2.5,
            "support_pixel_ratio_exponent": 2.0,
        },
    ),
    "ElSe": AlgorithmSpec(
        name="ElSe",
        package="pypupilext",
        class_name="ElSe",
        has_native_confidence=False,
        has_state=False,
        supports_diameter_override=False,
        internal_resize_note="downscales only when max(rows,cols) > 640; 424x187 crop: no resize",
        default_params={"minAreaRatio": 0.005, "maxAreaRatio": 0.2},
    ),
    "ExCuSe": AlgorithmSpec(
        name="ExCuSe",
        package="pypupilext",
        class_name="ExCuSe",
        has_native_confidence=False,
        has_state=False,
        supports_diameter_override=False,
        internal_resize_note="downscales only when max(rows,cols) > 680; 424x187 crop: no resize",
        default_params={"max_ellipse_radi": 50, "good_ellipse_threshold": 15},
    ),
    "Swirski2D": AlgorithmSpec(
        name="Swirski2D",
        package="pypupilext",
        class_name="Swirski2D",
        has_native_confidence=False,
        has_state=False,
        supports_diameter_override=False,
        internal_resize_note=(
            "no internal resize; Radius_Min/Max directly set pupil radius search "
            "range in crop pixels (resolution sensitive) - must be frozen per scale"
        ),
        default_params={
            "Radius_Min": 40,
            "Radius_Max": 80,
            "CannyBlur": 1.6,
            "CannyThreshold1": 20,
            "CannyThreshold2": 40,
            "StarburstPoints": 0,
            "PercentageInliers": 20,
            "InlierIterations": 2,
            "ImageAwareSupport": True,
            "EarlyTerminationPercentage": 95,
            "EarlyRejection": True,
            "Seed": -1,
        },
    ),
    "Starburst": AlgorithmSpec(
        name="Starburst",
        package="pypupilext",
        class_name="Starburst",
        has_native_confidence=False,
        has_state=True,
        supports_diameter_override=False,
        internal_resize_note=(
            "no full-image resize; RANSAC uses C global rand() with no exposed "
            "seed, so runs are not strictly reproducible; stateful (startPoint)"
        ),
        default_params={
            "edge_threshold": 16,
            "rays": 18,
            "min_feature_candidates": 10,
            "corneal_reflection_ratio_to_image_size": 2,
            "crWindowSize": 301,
        },
    ),
}

# Scale rule used to freeze resolution-sensitive parameters from the tight crop
# size. Rationale (docs 030 section 6): sub-031 probe showed a pupil of roughly
# 6-16 px diameter in a ~424x187 tight crop.
SCALE_RULE = {
    "radius_min_fraction": 0.02,   # Radius_Min = max(2, round(f * min(W,H)))
    "radius_max_fraction": 0.10,   # Radius_Max = max(radius_min+4, round(f * min(W,H)))
    "seed": 0,                     # Swirski2D RANSAC fixed seed
}
