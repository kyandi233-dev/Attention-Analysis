from __future__ import annotations

import pandas as pd

from .science_v3 import (
    BehaviorScienceConfig,
    MODEL_FAILURE_COLUMNS,
    fit_q1_nominal,
    fit_q2_ordinal,
)

# ``omission_rate`` in the historical canonical layer is the raw task-program
# omission rate.  Only the two additional decomposition endpoints need a shim;
# the raw endpoint is already covered by the canonical Q1/Q2 model pass.
EXTRA_OMISSION_PROBE_PREDICTORS = (
    "clean_go_omission_rate",
    "timing_ambiguous_go_omission_rate",
)


def _shim_predictor(frame: pd.DataFrame, predictor: str) -> pd.DataFrame:
    required = {
        "q1_nominal_4class",
        "q2_ordinal_4level",
        "repeat_participant_id",
        "session_id",
        predictor,
    }
    missing = required - set(frame.columns)
    if missing:
        return pd.DataFrame()
    out = frame[list(required)].copy()
    out["omission_rate"] = pd.to_numeric(out[predictor], errors="coerce")
    return out.drop(columns=[predictor])


def _relabel(frame: pd.DataFrame, predictor: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame.copy() if frame is not None else pd.DataFrame()
    out = frame.copy()
    if "predictor" in out.columns:
        out["predictor"] = predictor
    if "model_name" in out.columns:
        out["model_name"] = out["model_name"].astype(str).str.replace(
            "omission_rate", predictor, regex=False
        )
    out["endpoint_role"] = "prespecified_formal_omission_endpoint"
    return out


def fit_omission_probe_models(
    primary_probe: pd.DataFrame,
    cfg: BehaviorScienceConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the existing Q1/Q2 model families for clean/ambiguous omission.

    No new statistical family is introduced.  Each extra predictor is presented
    to the existing canonical model as a temporary ``omission_rate`` column, then
    outputs/failures are relabeled to the actual formal endpoint name.
    """
    cfg = cfg or BehaviorScienceConfig()
    q1_rows: list[pd.DataFrame] = []
    q2_rows: list[pd.DataFrame] = []
    failure_rows: list[pd.DataFrame] = []

    for predictor in EXTRA_OMISSION_PROBE_PREDICTORS:
        shim = _shim_predictor(primary_probe, predictor)
        if shim.empty:
            failure_rows.append(pd.DataFrame([{
                "model_name": f"Q1_Q2_{predictor}",
                "model_family": "probe_model_contract",
                "outcome": "Q1/Q2",
                "predictor": predictor,
                "status": "not_estimable",
                "n_rows": 0,
                "participant_group_n": 0,
                "session_n": 0,
                "reason": "required probe columns or formal omission endpoint missing",
                "endpoint_role": "prespecified_formal_omission_endpoint",
            }]))
            continue
        q1, q1_fail = fit_q1_nominal(shim, cfg)
        q2, q2_fail = fit_q2_ordinal(shim, cfg)
        q1_rows.append(_relabel(q1, predictor))
        q2_rows.append(_relabel(q2, predictor))
        failure_rows.append(_relabel(q1_fail, predictor))
        failure_rows.append(_relabel(q2_fail, predictor))

    q1_out = pd.concat([x for x in q1_rows if x is not None and not x.empty], ignore_index=True, sort=False) if any(x is not None and not x.empty for x in q1_rows) else pd.DataFrame()
    q2_out = pd.concat([x for x in q2_rows if x is not None and not x.empty], ignore_index=True, sort=False) if any(x is not None and not x.empty for x in q2_rows) else pd.DataFrame()
    nonempty_fail = [x for x in failure_rows if x is not None and not x.empty]
    failures = pd.concat(nonempty_fail, ignore_index=True, sort=False) if nonempty_fail else pd.DataFrame(columns=MODEL_FAILURE_COLUMNS)
    for column in MODEL_FAILURE_COLUMNS:
        if column not in failures.columns:
            failures[column] = pd.NA
    return q1_out, q2_out, failures
