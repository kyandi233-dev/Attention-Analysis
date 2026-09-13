"""Contract tests for the descriptive probability diagnostics.

The synthetic fixture is built so the participant-macro AUROC and Brier score are exactly
computable by hand, which pins the aggregation convention (probe-equal inside a
participant, then participant-equal) rather than merely checking that a number appears.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.supervised_learning.probability_diagnostics import (
    ProbabilityDiagnosticsError,
    build_probability_diagnostics,
    calibration_model,
    diagnostic_summary,
    participant_auroc_values,
)

# P-0: labels [0,0,1,1] with probabilities [0.1,0.2,0.8,0.9] -> AUROC 1.0, Brier 0.025
# P-1: labels [0,1,0,1] with probabilities [0.2,0.8,0.3,0.7] -> AUROC 1.0, Brier 0.065
# P-2: single class [1,1,1,1] -> AUROC not estimable (excluded), Brier 0.075
# The Brier score stays defined for single-class participants, so it averages over all
# three: (0.025 + 0.065 + 0.075) / 3 = 0.055. AUROC averages over two.
LABELS = [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]
PROBS = [0.1, 0.2, 0.8, 0.9, 0.2, 0.8, 0.3, 0.7, 0.9, 0.8, 0.7, 0.6]
PARTICIPANTS = ["P-0"] * 4 + ["P-1"] * 4 + ["P-2"] * 4


def _predictions(model_id: str = "m1", n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": [model_id] * n,
            "analysis_set_id": ["AS.test"] * n,
            "membership_type": ["included_missing_aware"] * n,
            "participant_group_id": PARTICIPANTS[:n],
            "q1_binary": LABELS[:n],
            "p_q1_equals_1": PROBS[:n],
            "model_failed": [False] * n,
        }
    )


def _small_options() -> dict:
    return {"replicates": 200, "seed": 20260830, "n_bins": 4}


def test_participant_macro_auroc_and_brier_match_hand_computation() -> None:
    diagnostics, _ = build_probability_diagnostics(_predictions(), **_small_options())
    assert len(diagnostics) == 1
    row = diagnostics.iloc[0]

    assert row["model_id"] == "m1"
    assert row["n_participants"] == 3
    assert row["n_probes"] == 12
    assert row["positive_rate"] == pytest.approx(8 / 12)

    assert row["participant_macro_auroc"] == pytest.approx(1.0)
    assert row["n_participants_auroc_estimable"] == 2
    assert row["n_participants_auroc_single_class"] == 1
    assert row["participant_macro_brier"] == pytest.approx(0.055)


def test_auroc_excludes_single_class_participants_without_imputation() -> None:
    frame = _predictions()
    values, n_estimable, n_single = participant_auroc_values(frame)
    assert n_estimable == 2
    assert n_single == 1
    assert len(values) == 2
    assert np.isfinite(values).all()


def test_diagnostics_are_declared_non_selective() -> None:
    diagnostics, _ = build_probability_diagnostics(_predictions(), **_small_options())
    row = diagnostics.iloc[0]
    assert bool(row["used_for_model_or_c_selection"]) is False
    assert bool(row["used_for_feature_selection"]) is False
    assert row["role"] == "supplementary_descriptive_diagnostic"
    assert bool(row["prediction_model_retrained_in_bootstrap"]) is False
    assert bool(row["calibration_model_refitted_in_bootstrap"]) is True

    summary = diagnostic_summary(diagnostics)
    assert summary["used_for_model_or_c_selection"] is False
    assert summary["used_for_feature_selection"] is False
    assert "participant_macro_auroc" in summary["metrics"]
    assert summary["n_diagnostic_rows"] == 1
    assert summary["n_unique_models"] == 1


def test_interval_brackets_the_point_estimate() -> None:
    diagnostics, _ = build_probability_diagnostics(_predictions(), **_small_options())
    row = diagnostics.iloc[0]
    assert row["participant_macro_auroc_ci_lower"] <= row["participant_macro_auroc"]
    assert row["participant_macro_auroc"] <= row["participant_macro_auroc_ci_upper"]
    assert row["participant_macro_brier_ci_lower"] <= row["participant_macro_brier"]
    assert row["participant_macro_brier"] <= row["participant_macro_brier_ci_upper"]


def test_calibration_bins_cover_every_probe_exactly_once() -> None:
    diagnostics, bins = build_probability_diagnostics(_predictions(), **_small_options())
    assert len(diagnostics) == 1
    assert int(bins["n_probes"].sum()) == 12
    assert bins["bin_index"].is_monotonic_increasing
    assert (bins["bin_lower"] <= bins["bin_upper"]).all()
    assert bins["observed_positive_rate"].between(0.0, 1.0).all()
    assert bins["mean_predicted_probability"].between(0.0, 1.0).all()


def test_failed_and_nonfinite_rows_are_dropped() -> None:
    frame = _predictions()
    frame.loc[0, "model_failed"] = True
    frame.loc[1, "p_q1_equals_1"] = np.nan
    frame.loc[2, "q1_binary"] = np.nan
    diagnostics, _ = build_probability_diagnostics(frame, **_small_options())
    assert int(diagnostics.iloc[0]["n_probes"]) == 9


def test_single_class_only_input_reports_not_estimable() -> None:
    frame = _predictions()
    frame["q1_binary"] = 1
    diagnostics, _ = build_probability_diagnostics(frame, **_small_options())
    row = diagnostics.iloc[0]
    assert row["n_participants_auroc_estimable"] == 0
    assert row["n_participants_auroc_single_class"] == 3
    assert math.isnan(float(row["participant_macro_auroc"]))
    # Brier stays defined even when AUROC is not.
    assert np.isfinite(float(row["participant_macro_brier"]))
    assert row["calibration_status"] == "not_estimable_single_class"


def test_calibration_model_recovers_a_known_miscalibration() -> None:
    """An over-confident predictor must show a calibration slope well below 1.

    Labels must be generated *stochastically* from a true probability: a deterministic
    label rule would make the calibration model perfectly separated and the slope would
    explode instead of shrinking.
    """
    rng = np.random.default_rng(7)
    n = 4000
    true_logit = rng.normal(0.0, 1.0, size=n)
    true_probability = 1.0 / (1.0 + np.exp(-true_logit))
    labels = rng.binomial(1, true_probability)
    # Deliberately over-confident: the log-odds are stretched by a factor of 2.5, so an
    # ideally calibrated re-fit should recover a slope near 1 / 2.5.
    probabilities = 1.0 / (1.0 + np.exp(-2.5 * true_logit))
    frame = pd.DataFrame(
        {
            "participant_group_id": [f"P-{i % 40}" for i in range(n)],
            "q1_binary": labels,
            "p_q1_equals_1": probabilities,
        }
    )
    intercept, slope, status = calibration_model(frame)
    assert status == "estimable"
    assert slope < 0.6
    assert np.isfinite(intercept)


def test_calibration_slope_guard_flags_low_discrimination() -> None:
    """A near-chance score must not yield an interpretable calibration slope.

    This mirrors a real finding: the frozen sensor-only models return slopes between -0.2 and
    -43.9 with intervals up to 72 units wide, which is numerical instability rather than
    evidence of inverted probabilities.
    """
    from attention_pipeline.supervised_learning.probability_diagnostics import (
        CALIBRATION_SLOPE_NOT_REPORTABLE_LOW_DISCRIMINATION,
        CALIBRATION_SLOPE_REPORTABLE,
    )

    # Well-discriminating fixture: AUROC CI lies above 0.5, so the slope is reportable.
    good, _ = build_probability_diagnostics(_predictions(), **_small_options())
    assert good.iloc[0]["calibration_slope_reporting"] == CALIBRATION_SLOPE_REPORTABLE
    assert "discrimination is established" in good.iloc[0]["calibration_slope_reporting_note"]

    # Near-chance fixture: probabilities are independent of the labels.
    rng = np.random.default_rng(11)
    n_participants = 40
    rows = []
    for index in range(n_participants):
        for _ in range(20):
            rows.append(
                {
                    "model_id": "m_chance",
                    "run_id": "r1",
                    "analysis_set_id": "AS.test",
                    "membership_type": "included_missing_aware",
                    "participant_group_id": f"P-{index}",
                    "q1_binary": int(rng.integers(0, 2)),
                    "p_q1_equals_1": float(rng.uniform(0.05, 0.95)),
                    "model_failed": False,
                }
            )
    chance, _ = build_probability_diagnostics(pd.DataFrame(rows), **_small_options())
    row = chance.iloc[0]
    assert row["calibration_slope_reporting"] == (
        CALIBRATION_SLOPE_NOT_REPORTABLE_LOW_DISCRIMINATION
    )
    assert "must not be read" in row["calibration_slope_reporting_note"]
    # The raw estimate is retained regardless of the flag.
    assert np.isfinite(float(row["calibration_slope"]))
    assert int(row["calibration_n_valid_replicates"]) > 0

    summary = diagnostic_summary(chance)
    assert summary["n_calibration_slope_not_reportable_low_discrimination"] == 1
    assert summary["n_calibration_slope_reportable"] == 0


def test_deterministic_for_a_fixed_seed() -> None:
    first, first_bins = build_probability_diagnostics(_predictions(), **_small_options())
    second, second_bins = build_probability_diagnostics(_predictions(), **_small_options())
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_bins, second_bins)


def test_multiple_models_are_reported_separately() -> None:
    combined = pd.concat([_predictions("m1"), _predictions("m2")], ignore_index=True)
    diagnostics, _ = build_probability_diagnostics(combined, **_small_options())
    assert sorted(diagnostics["model_id"]) == ["m1", "m2"]


def test_same_analysis_set_in_two_runs_stays_separate() -> None:
    """Two runs sharing (model_id, analysis_set_id, membership_type) must not be merged.

    This is a real hazard, not a hypothetical: ``smoke_go_omission__aware`` writes the same
    ``analysis_set_id`` and ``model_id`` as the formal go_omission run.
    """
    formal = _predictions("m1")
    formal["run_id"] = ["formal_run"] * len(formal)
    smoke = _predictions("m1", n=4)
    smoke["run_id"] = ["smoke_run"] * len(smoke)

    diagnostics, _ = build_probability_diagnostics(
        pd.concat([formal, smoke], ignore_index=True), **_small_options()
    )
    assert len(diagnostics) == 2
    assert sorted(diagnostics["run_id"]) == ["formal_run", "smoke_run"]
    by_run = diagnostics.set_index("run_id")
    assert int(by_run.loc["formal_run", "n_probes"]) == 12
    assert int(by_run.loc["smoke_run", "n_probes"]) == 4
    assert diagnostic_summary(diagnostics)["n_runs"] == 2


def test_cli_run_glob_excludes_smoke_directories(tmp_path) -> None:
    """The CLI default must not pick up exploratory smoke run directories."""
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "build_probability_diagnostics.py"
    spec = importlib.util.spec_from_file_location("build_probability_diagnostics", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    formal = tmp_path / "AS.full__included_missing_aware"
    smoke = tmp_path / "smoke_go_omission__aware"
    for directory in (formal, smoke):
        directory.mkdir(parents=True)
        _predictions("m1").to_csv(directory / "probe_predictions.csv", index=False)

    parser = module.build_parser()
    args = parser.parse_args(["--run-root", str(tmp_path), "--output-root", str(tmp_path / "out")])
    assert args.run_glob == "*__included_missing_aware"

    matched = sorted(p.parent.name for p in tmp_path.glob(f"{args.run_glob}/probe_predictions.csv"))
    assert matched == ["AS.full__included_missing_aware"]


def test_empty_or_unusable_input_fails_closed() -> None:
    with pytest.raises(ProbabilityDiagnosticsError, match="missing required columns"):
        build_probability_diagnostics(pd.DataFrame({"model_id": ["m1"]}))

    unusable = _predictions()
    unusable["model_failed"] = True
    with pytest.raises(ProbabilityDiagnosticsError, match="no usable"):
        build_probability_diagnostics(unusable)
