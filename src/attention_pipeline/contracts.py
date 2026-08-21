from __future__ import annotations

from enum import Enum


class NIRFrameStatus(str, Enum):
    NO_FACE = "no_face"
    ROI_MISSING = "roi_missing"
    PUPIL_INVISIBLE = "pupil_invisible"
    DETECTOR_MISSING = "detector_missing"
    DETECTOR_REJECTED = "detector_rejected"
    OBSERVED = "observed"
    INTERPOLATED = "interpolated"


class BehaviorWindowStatus(str, Enum):
    INSUFFICIENT_RT = "insufficient_rt"
    INSUFFICIENT_NOGO = "insufficient_nogo"
    RESPONSE_ONLY = "response_only"
    FULL_EVIDENCE = "full_evidence"


OUTPUT_DIRS = (
    "000-reports",
    "010-nir",
    "020-rgb",
    "030-cross-modal",
    "040-behavior",
    "090-manifests",
)

EYE_RIGHT = "eye_right"  # subject anatomical right = image left
EYE_LEFT = "eye_left"    # subject anatomical left = image right
EYES = (EYE_RIGHT, EYE_LEFT)

PROBE_LABELS = {
    1: "完全专注",
    2: "关注实验但未聚焦任务",
    3: "任务无关思维",
    4: "大脑空白",
}

