from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.nir_formal_analysis.probe_contract import (
    _strict_preprobe_trials,
    _visual_exposure,
    _visual_lookup,
)


def _trials() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "block_num": [1, 1, 1, 1],
            "trial_num": [8, 9, 10, 11],
            "absolute_onset_time": [7000.0, 8000.0, 9000.0, 11000.0],
            "stimulus_name": ["a", "b", "ANCHOR", "future"],
            "stimulus_size": [80, 80, 80, 80],
            "is_no_go": [0, 0, 0, 0],
            "correct": [1, 1, 1, 1],
            "commission": [0, 0, 0, 0],
            "omission": [0, 0, 0, 0],
            "rt": [400.0, 420.0, 440.0, 460.0],
            "prestimulus_press_flag": [False] * 4,
            "ambiguous_omission_flag": [False] * 4,
            "anticipatory_candidate_flag": [False] * 4,
        }
    )


def test_temporal_rule_alone_would_leak_anchor_but_strict_contract_excludes_it() -> None:
    strict, anchor_under_old_rule, old_n = _strict_preprobe_trials(
        _trials(),
        block_num=1,
        probe_trial_num=10,
        probe_onset_ms=10000.0,
        start_ms=6000.0,
        end_ms=10000.0,
    )
    assert anchor_under_old_rule is True
    assert old_n == 3
    assert strict["trial_num"].tolist() == [8, 9]
    assert "ANCHOR" not in set(strict["stimulus_name"])


def test_probe_visual_exposure_uses_only_strict_preprobe_trials() -> None:
    strict, _, _ = _strict_preprobe_trials(
        _trials(),
        block_num=1,
        probe_trial_num=10,
        probe_onset_ms=10000.0,
        start_ms=6000.0,
        end_ms=10000.0,
    )
    visual = pd.DataFrame(
        {
            "stimulus_name": ["a", "b", "ANCHOR", "future"],
            "stimulus_size_pct": [80, 80, 80, 80],
            "central_rel_lum_mean": [0.2, 0.4, 9.0, 99.0],
            "central_rms_contrast": [0.1, 0.3, 9.0, 99.0],
        }
    )
    lookup, metrics = _visual_lookup(visual)
    exposure = _visual_exposure(strict, lookup, metrics)
    assert exposure["visual_trial_n"] == 2
    assert exposure["visual_joined_trial_n"] == 2
    assert exposure["visual_trial_coverage"] == 1.0
    # Decimal means are floating-point values; test the scientific value rather
    # than binary representation equality.
    assert np.isclose(exposure["central_rel_lum_mean__preprobe_mean"], 0.3)
    assert np.isclose(exposure["central_rms_contrast__preprobe_mean"], 0.2)
