from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.nir_pupil_benchmark.formal import (
    ProductionRun,
    build_sample_plan,
    discover_production_run,
    map_ritnet_to_source,
    transform_ellipse_affine,
    validate_result_contract,
)
from attention_pipeline.nir_pupil_benchmark.schema import RESULT_COLUMNS


def _completion(directory: Path, subject: str, video: Path, *, status: str, run_id: str):
    directory.mkdir(parents=True)
    (directory / "completion.json").write_text(
        json.dumps(
            {
                "status": status,
                "subject": subject,
                "video": str(video),
                "run_id": run_id,
            }
        ),
        encoding="utf-8",
    )
    if status == "complete":
        (directory / "eyes.csv").write_text("subject\n", encoding="utf-8")
        (directory / "run_manifest.json").write_text("{}", encoding="utf-8")


def test_discover_production_run_prefers_complete_b8_and_ignores_running(tmp_path):
    video = tmp_path / "source.avi"
    video.write_bytes(b"video")
    _completion(
        tmp_path / "sub-035_formal_v3.1.3_yolo_b16_fp32",
        "sub-035", video, status="running", run_id="old",
    )
    chosen = tmp_path / "sub-035_formal_v3.1.3_yolo-b8_ritnet-b16_fp32"
    _completion(chosen, "sub-035", video, status="complete", run_id="new")
    run = discover_production_run(tmp_path, "35")
    assert run.path == chosen
    assert run.run_id == "new"


def test_discover_production_run_rejects_ambiguous_complete_preferred(tmp_path):
    video = tmp_path / "source.avi"
    video.write_bytes(b"video")
    for suffix in ("a", "b"):
        _completion(
            tmp_path / f"sub-031_formal_yolo-b8_ritnet-b16_{suffix}",
            "sub-031", video, status="complete", run_id=suffix,
        )
    with pytest.raises(RuntimeError, match="ambiguous complete"):
        discover_production_run(tmp_path, "sub-031")


def test_transform_ellipse_affine_identity_and_isotropic():
    identity = transform_ellipse_affine(10, 20, 8, 4, 30, scale_x=1, scale_y=1)
    assert identity["center_x"] == pytest.approx(10)
    assert identity["center_y"] == pytest.approx(20)
    assert identity["major_axis"] == pytest.approx(8)
    assert identity["minor_axis"] == pytest.approx(4)
    assert identity["angle_deg"] == pytest.approx(30)

    scaled = transform_ellipse_affine(
        10, 20, 8, 4, 30,
        scale_x=2, scale_y=2, translate_x=100, translate_y=200,
    )
    assert scaled["center_x"] == pytest.approx(120)
    assert scaled["center_y"] == pytest.approx(240)
    assert scaled["major_axis"] == pytest.approx(16)
    assert scaled["minor_axis"] == pytest.approx(8)
    assert scaled["diameter_geom"] == pytest.approx(2 * math.sqrt(8 * 4))


def test_transform_ellipse_affine_anisotropic_preserves_area_scale():
    mapped = transform_ellipse_affine(0, 0, 12, 6, 37, scale_x=3, scale_y=2)
    # Geometric-mean full diameter scales by sqrt(det(S)).
    assert mapped["diameter_geom"] == pytest.approx(math.sqrt(12 * 6) * math.sqrt(6))
    assert mapped["major_axis"] >= mapped["minor_axis"] > 0


def test_map_ritnet_to_source_uses_roi_not_tight_bbox():
    row = {
        "ritnet_found": True,
        "roi_x1": 100, "roi_y1": 200, "roi_x2": 740, "roi_y2": 520,
        "pupil_center_x": 160, "pupil_center_y": 80,
        "pupil_axis_a": 10, "pupil_axis_b": 20, "pupil_angle_deg": 0,
    }
    mapped = map_ritnet_to_source(row, (320, 160))
    assert mapped["ritnet_found"] is True
    assert mapped["ritnet_source_center_x"] == pytest.approx(420)
    assert mapped["ritnet_source_center_y"] == pytest.approx(360)
    assert mapped["ritnet_source_major_axis"] == pytest.approx(40)
    assert mapped["ritnet_source_minor_axis"] == pytest.approx(20)


def _synthetic_eyes(n_per_block: int = 40) -> pd.DataFrame:
    rows = []
    for phase, offset in (("block1", 100), ("block2", 1000)):
        for local in range(n_per_block):
            frame_idx = offset + local
            for eye in ("eye_left", "eye_right"):
                difficult = local in {1, 2, 3, 4}
                rows.append(
                    {
                        "subject": "sub-031",
                        "phase": phase,
                        "phase_segment": 1,
                        "frame_idx": frame_idx,
                        "eye": eye,
                        "status": "ritnet_missing" if difficult and local == 1 else "observed",
                        "frame_status": "two_eyes",
                        "anchor_yolo_confidence": 0.99 - local / 1000,
                        "bbox_x1": 10.2, "bbox_y1": 5.2,
                        "bbox_x2": 50.8, "bbox_y2": 25.8,
                        "roi_x1": 0, "roi_y1": 0, "roi_x2": 64, "roi_y2": 32,
                        "roi_clipped": difficult and local == 2,
                        "ritnet_found": not (difficult and local == 1),
                        "pupil_center_x": 160,
                        "pupil_center_y": 80,
                        "pupil_axis_a": 10,
                        "pupil_axis_b": 20,
                        "pupil_angle_deg": 0,
                        "pupil_confidence": 0.05 if difficult else 0.95 - local / 1000,
                    }
                )
    return pd.DataFrame(rows)


def test_build_sample_plan_is_deterministic_disjoint_and_consecutive():
    eyes = _synthetic_eyes()
    kwargs = dict(
        block_uniform_n=3,
        ritnet_high_quality_n=2,
        ritnet_difficult_n=2,
        temporal_n=5,
        temporal_preferred_phase="block1",
    )
    tight_a, temporal_a = build_sample_plan(eyes, **kwargs)
    tight_b, temporal_b = build_sample_plan(eyes, **kwargs)
    pd.testing.assert_frame_equal(tight_a, tight_b)
    pd.testing.assert_frame_equal(temporal_a, temporal_b)
    assert len(tight_a) == 10  # 3+3 block + 2 high + 2 difficult, disjoint
    assert tight_a["sample_role"].str.contains("block1_uniform", regex=False).sum() == 3
    assert tight_a["sample_role"].str.contains("block2_uniform", regex=False).sum() == 3
    assert tight_a["sample_role"].str.contains("ritnet_high_quality", regex=False).sum() == 2
    assert tight_a["sample_role"].str.contains("ritnet_difficult", regex=False).sum() == 2
    assert len(temporal_a) == 5
    assert np.diff(temporal_a["frame_idx"]).tolist() == [1, 1, 1, 1]


def test_result_contract_rejects_equal_row_count_with_wrong_identity_key():
    identity = {
        "subject": "sub-031",
        "phase": "block1",
        "phase_segment": 1,
        "frame_idx": 100,
        "eye": "eye_left",
        "input_kind": "production_tight_bbox",
        "sequence_id": "",
        "input_status": "ready",
    }
    manifest = pd.DataFrame([identity])
    result = {column: None for column in RESULT_COLUMNS}
    result.update(identity)
    result.update({"algorithm": "PuRe", "frame_idx": 101})
    results = pd.DataFrame([result], columns=RESULT_COLUMNS)
    with pytest.raises(AssertionError, match="formal result contract failed"):
        validate_result_contract(manifest, results, ["PuRe"])
