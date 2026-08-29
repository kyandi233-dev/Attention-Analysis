from __future__ import annotations

import numpy as np
import pandas as pd

from attention_pipeline.nir_formal_analysis.scientific_models import (
    add_pupil_within_between,
    fit_trial_reference_models,
)


def test_reference_pupil_within_between_decomposition() -> None:
    frame = pd.DataFrame(
        {
            "analysis_group_token": ["p1", "p1", "p2", "p2"],
            "pupil_median": [1.0, 3.0, 10.0, 14.0],
        }
    )
    out = add_pupil_within_between(frame)
    assert out.loc[0, "pupil_between"] == 2.0
    assert out.loc[1, "pupil_between"] == 2.0
    assert out.loc[2, "pupil_between"] == 12.0
    assert np.isclose(out.groupby("analysis_group_token")["pupil_within"].sum().to_numpy(), 0.0).all()


def test_trial_reference_models_write_not_estimable_instead_of_empty_silence() -> None:
    frame = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s2", "s2"],
            "analysis_group_token": ["p1", "p1", "p2", "p2"],
            "is_no_go": [0, 1, 0, 1],
            "correct": [1, 1, 1, 0],
            "omission": [0, 0, 1, 0],
            "commission": [0, 0, 0, 1],
            "rt": [400.0, np.nan, 500.0, np.nan],
            "pupil_within": [-0.2, 0.2, -0.3, 0.3],
            "pupil_between": [1.0, 1.0, 2.0, 2.0],
            "time_in_block_z": [0.0, 0.1, 0.0, 0.1],
            "pupil_valid_fraction": [1.0] * 4,
            "internal_coverage_fraction": [1.0] * 4,
        }
    )
    results, failures = fit_trial_reference_models(
        frame,
        min_participant_groups=6,
        min_rows=24,
    )
    assert results.empty
    assert not failures.empty
    assert failures["status"].eq("not_estimable").all()
    assert set(failures["outcome"]) == {"rt", "omission", "commission"}
    assert set(failures["adjusted"]) == {False, True}


def test_go_omission_and_nogo_commission_are_never_collapsed() -> None:
    frame = pd.DataFrame(
        {
            "session_id": ["s1"] * 8,
            "analysis_group_token": ["p1"] * 8,
            "is_no_go": [0, 0, 0, 0, 1, 1, 1, 1],
            "correct": [1, 1, 0, 1, 1, 0, 1, 0],
            "omission": [0, 0, 1, 0, 0, 0, 0, 0],
            "commission": [0, 0, 0, 0, 0, 1, 0, 1],
            "rt": [400.0, 420.0, np.nan, 430.0, np.nan, np.nan, np.nan, np.nan],
            "pupil_within": np.linspace(-1, 1, 8),
            "pupil_between": [2.0] * 8,
        }
    )
    _, failures = fit_trial_reference_models(frame, min_participant_groups=99, min_rows=999)
    labels = set(failures["model_name"])
    assert any(name.startswith("go_omission__") for name in labels)
    assert any(name.startswith("nogo_commission__") for name in labels)
    assert not any("correct_combined" in name for name in labels)
