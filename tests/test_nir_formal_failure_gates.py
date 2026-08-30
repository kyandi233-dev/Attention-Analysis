import numpy as np
import pandas as pd

from attention_pipeline.nir_pipeline_validation.scientific_contract import (
    audit_model_result,
    report_admission,
)


class _NonConverged:
    converged = False
    params = pd.Series([1.0])
    bse = pd.Series([0.1])


class _NonFinite:
    converged = True
    params = pd.Series([np.nan])
    bse = pd.Series([0.1])


class _SingularRandomEffect:
    converged = True
    params = pd.Series([1.0])
    bse = pd.Series([0.1])
    cov_re = np.array([[0.0]])


def test_failed_models_are_rejected_with_explicit_reason():
    assert audit_model_result(_NonConverged(), model_name="m").failure_type == "not_converged"
    assert audit_model_result(_NonFinite(), model_name="m").failure_type == "nonfinite_estimates"
    assert audit_model_result(_SingularRandomEffect(), model_name="m").failure_type == "singular_random_effects"


def test_report_gate_does_not_treat_failure_table_as_success():
    windows = pd.DataFrame({
        "session_id": ["s1", "s1"],
        "analysis_group_token": ["g1", "g1"],
        "block_num": [1, 1],
        "track": ["left_primary", "right_primary"],
        "go_omission_target": [0.0, 0.0],
        "nogo_commission_target": [np.nan, np.nan],
    })
    probes = pd.DataFrame({"probe_response": [1], "probe_vigilance": [2]})
    failures = pd.DataFrame({"status": ["not_estimable"], "failure_type": ["singular_random_effects"]})
    result = report_admission(
        figure_names=[f"Figure{i:02d}" for i in range(1, 11)],
        trial_windows=windows,
        probe_windows=probes,
        model_failures=failures,
        failure_tables_written=True,
        topology={
            "n_sessions": 44,
            "n_analysis_groups": 38,
            "n_repeated_participant_groups": 6,
            "max_sessions_per_participant": 2,
            "group_size_distribution": {"1": 32, "2": 6},
        },
    )
    assert result["gates"]["model_failure_gate_active"]
    assert result["scientific_inference_authorized"] is False
