from __future__ import annotations

import pandas as pd

from attention_pipeline.nir_pipeline_validation.supplementary_figures import (
    supplementary02_probe_objective_behavior,
    supplementary03_probe_response_times,
)


def test_probe_objective_behavior_supplement_exports(tmp_path):
    frame = pd.DataFrame(
        {
            "subject": ["sub-031", "sub-032"] * 4,
            "block_num": [1, 1, 2, 2] * 2,
            "probe_response": [1, 2, 1, 2] * 2,
            "window_name": ["pre_10s"] * 4 + ["pre_20s"] * 4,
            "n_go": [8] * 8,
            "n_nogo": [2] * 8,
            "n_commission": [0, 1, 0, 1, 0, 1, 1, 0],
            "n_omission": [0, 0, 1, 0, 0, 1, 0, 0],
            "n_ambiguous_omission": [0] * 8,
            "n_anticipatory_candidate": [0, 1, 0, 1, 1, 0, 1, 0],
            "go_rt_cv": [0.10, 0.15, 0.12, 0.18, 0.11, 0.16, 0.13, 0.17],
        }
    )
    outputs = supplementary02_probe_objective_behavior(
        frame,
        base=tmp_path / "S02",
        formats=["png"],
        raster_dpi=100,
    )
    assert outputs == [str(tmp_path / "S02.png")]
    assert (tmp_path / "S02.png").is_file()


def test_probe_response_time_supplement_exports(tmp_path):
    frame = pd.DataFrame(
        {
            "subject": ["sub-031", "sub-032", "sub-031", "sub-032"],
            "block_num": [1, 1, 2, 2],
            "probe_response": [1, 2, 1, 2],
            "probe_rt": [500, 600, 550, 620],
            "probe_vigilance_rt": [650, 700, 680, 720],
        }
    )
    outputs = supplementary03_probe_response_times(
        frame,
        base=tmp_path / "S03",
        formats=["png"],
        raster_dpi=100,
    )
    assert outputs == [str(tmp_path / "S03.png")]
    assert (tmp_path / "S03.png").is_file()
