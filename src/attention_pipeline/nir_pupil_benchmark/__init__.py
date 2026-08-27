"""Native-resolution benchmark for the seven classical pupil detection algorithms.

Three-layer result semantics (docs/020-nir/030):

- ``algorithm_returned``: the algorithm produced an ellipse outline
  (``hasOutline()`` / axes > 0). Not the same as "a credible pupil was detected".
- ``official_valid``: strict official ``Pupil.valid(-1.0)`` where available;
  confidence-less algorithms report False by official semantics.
- ``geometry_sane``: pure geometric sanity gate independent of confidence.
"""
from .adapters import (
    DetectionOutput,
    UnavailableAlgorithmError,
    frozen_pupil_labs_properties,
    frozen_swirski_params,
    make_detector,
    pupil_diameter_bounds,
    run_detection,
    run_with_confidence,
)
from .incremental import EventLogger, run_incremental
from .core import (
    Ellipse,
    center_distance,
    choose_continuous_window,
    deterministic_frame_sample,
    ellipse_geometry_plausible,
    geometry_sane,
    merge_sample_sets,
    normalize_phase,
    normalize_result,
    normalize_subject,
    parse_pupil_result,
    resolve_column,
    safe_ratio,
    temporal_stability_table,
)
from .overlay import draw_detection, write_algorithm_montage
from .runner import VideoFrameSource, assemble_row, detect_crop, run_crop_list, scale_params
from .schema import ALGORITHMS, ALGORITHM_SPECS, RESULT_COLUMNS, SCALE_RULE
from .synthetic import make_synthetic_eye, write_smoke_manifest

__all__ = [
    "ALGORITHMS",
    "ALGORITHM_SPECS",
    "RESULT_COLUMNS",
    "SCALE_RULE",
    "Ellipse",
    "DetectionOutput",
    "UnavailableAlgorithmError",
    "VideoFrameSource",
    "assemble_row",
    "center_distance",
    "choose_continuous_window",
    "deterministic_frame_sample",
    "detect_crop",
    "draw_detection",
    "ellipse_geometry_plausible",
    "frozen_pupil_labs_properties",
    "frozen_swirski_params",
    "geometry_sane",
    "make_detector",
    "make_synthetic_eye",
    "merge_sample_sets",
    "normalize_phase",
    "normalize_result",
    "normalize_subject",
    "parse_pupil_result",
    "pupil_diameter_bounds",
    "resolve_column",
    "run_crop_list",
    "run_detection",
    "run_with_confidence",
    "safe_ratio",
    "scale_params",
    "temporal_stability_table",
    "write_algorithm_montage",
    "write_smoke_manifest",
]
