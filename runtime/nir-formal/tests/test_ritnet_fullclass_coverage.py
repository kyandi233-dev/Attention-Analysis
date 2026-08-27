from __future__ import annotations

import pytest

from ritnet_fullclass_coverage import build_fixed_qc_anchor_keys, build_frame_coverage
from ritnet_fullclass_schema import FRAME_COVERAGE_FIELDS, FRAME_COVERAGE_SCHEMA_VERSION


def frame(index, status="observed", selected=2, phase="block1", segment=1):
    return {
        "subject": "sub-031",
        "phase": phase,
        "phase_segment": segment,
        "frame_idx": index,
        "video_time_ms": index * 1000,
        "unix_ms": 100000 + index * 1000,
        "phase_time_ms": index * 1000,
        "status": status,
        "raw_detection_count": selected,
        "selected_eye_count": selected,
    }


def eye(index, which, phase="block1", segment=1):
    return {
        "phase": phase,
        "phase_segment": segment,
        "frame_idx": index,
        "eye": which,
    }


def final_eye(index, which, status="success", reason=None, phase="block1", segment=1):
    return {
        "phase": phase,
        "phase_segment": segment,
        "frame_idx": index,
        "eye": which,
        "ritnet_status": status,
        "ritnet_failure_reason": reason,
    }


def test_fixed_qc_anchors_come_from_full_frame_timeline_not_eye_rows():
    frames = [frame(i, selected=0 if i == 30 else 2) for i in range(0, 61)]
    anchors = build_fixed_qc_anchor_keys(frames, interval_sec=30)
    assert ("block1", 1, 0) in anchors
    assert ("block1", 1, 30) in anchors
    assert ("block1", 1, 60) in anchors


def test_coverage_retains_yolo_miss_frame_with_no_eye_rows():
    frames = [frame(0), frame(1, status="no_eye", selected=0), frame(2)]
    source_eyes = [
        eye(0, "frame_left"), eye(0, "frame_right"),
        eye(2, "frame_left"), eye(2, "frame_right"),
    ]
    finals = [
        final_eye(0, "frame_left"), final_eye(0, "frame_right"),
        final_eye(2, "frame_left"), final_eye(2, "frame_right"),
    ]
    coverage = build_frame_coverage(
        subject="sub-031",
        source_frames=frames,
        source_eye_rows=source_eyes,
        final_eye_rows=finals,
        fixed_anchor_keys={("block1", 1, 1)},
    )
    assert len(coverage) == 3
    assert tuple(coverage[1]) == FRAME_COVERAGE_FIELDS
    assert coverage[1]["frame_coverage_schema_version"] == FRAME_COVERAGE_SCHEMA_VERSION
    assert coverage[1]["coverage_status"] == "yolo_no_eye"
    assert coverage[1]["ritnet_success_eye_count"] == 0
    assert coverage[1]["fixed_qc_anchor"] is True


def test_coverage_distinguishes_single_eye_and_failed_eye():
    frames = [frame(0, selected=1), frame(1, selected=2)]
    source_eyes = [
        eye(0, "frame_left"),
        eye(1, "frame_left"), eye(1, "frame_right"),
    ]
    finals = [
        final_eye(0, "frame_left"),
        final_eye(1, "frame_left"),
        final_eye(1, "frame_right", status="failed", reason="roi_invalid:test"),
    ]
    coverage = build_frame_coverage(
        subject="sub-031",
        source_frames=frames,
        source_eye_rows=source_eyes,
        final_eye_rows=finals,
        fixed_anchor_keys=set(),
    )
    assert coverage[0]["coverage_status"] == "single_eye_success"
    assert coverage[0]["right_ritnet_status"] == "not_detected"
    assert coverage[1]["coverage_status"] == "single_eye_success"
    assert coverage[1]["right_ritnet_status"] == "failed"
    assert coverage[1]["right_failure_reason"] == "roi_invalid:test"


def test_final_decode_failure_and_roi_invalid_have_explicit_frame_statuses():
    frames = [frame(3, selected=2), frame(4, selected=2)]
    source_eyes = [
        eye(3, "frame_left"), eye(3, "frame_right"),
        eye(4, "frame_left"), eye(4, "frame_right"),
    ]
    finals = [
        final_eye(
            3,
            "frame_left",
            status="failed",
            reason="source_video_decode_failed:target_frame=3:first_failed_frame=3",
        ),
        final_eye(
            3,
            "frame_right",
            status="failed",
            reason="source_video_decode_failed:target_frame=3:first_failed_frame=3",
        ),
        final_eye(4, "frame_left", status="failed", reason="roi_invalid:ValueError:left"),
        final_eye(4, "frame_right", status="failed", reason="roi_invalid:ValueError:right"),
    ]
    coverage = build_frame_coverage(
        subject="sub-031",
        source_frames=frames,
        source_eye_rows=source_eyes,
        final_eye_rows=finals,
        fixed_anchor_keys=set(),
    )
    assert coverage[0]["coverage_status"] == "final_video_decode_failed"
    assert coverage[0]["ritnet_success_eye_count"] == 0
    assert coverage[1]["coverage_status"] == "roi_invalid"
    assert coverage[1]["ritnet_success_eye_count"] == 0


def test_video_read_failure_has_priority_in_coverage_status():
    frames = [frame(5, status="video_read_failed", selected=0)]
    coverage = build_frame_coverage(
        subject="sub-031",
        source_frames=frames,
        source_eye_rows=[],
        final_eye_rows=[],
        fixed_anchor_keys=set(),
    )
    assert coverage[0]["coverage_status"] == "video_read_failed"


def test_frames_selected_eye_count_must_match_eyes_csv_rows():
    frames = [frame(10, selected=2)]
    source_eyes = [eye(10, "frame_left")]
    with pytest.raises(ValueError, match="selected_eye_count does not match"):
        build_frame_coverage(
            subject="sub-031",
            source_frames=frames,
            source_eye_rows=source_eyes,
            final_eye_rows=[final_eye(10, "frame_left")],
            fixed_anchor_keys=set(),
        )


def test_final_eye_cannot_exist_without_matching_source_eye():
    frames = [frame(10, selected=1)]
    source_eyes = [eye(10, "frame_left")]
    finals = [final_eye(10, "frame_left"), final_eye(10, "frame_right")]
    with pytest.raises(ValueError, match="without matching source eye rows"):
        build_frame_coverage(
            subject="sub-031",
            source_frames=frames,
            source_eye_rows=source_eyes,
            final_eye_rows=finals,
            fixed_anchor_keys=set(),
        )
