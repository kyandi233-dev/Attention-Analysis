from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta

from ..config import Config
from ..contracts import BehaviorWindowStatus


def _jeffreys(commissions: int, opportunities: int) -> tuple[float, float, float]:
    alpha = commissions + 0.5
    beta_value = opportunities - commissions + 0.5
    mean = alpha / (alpha + beta_value)
    low, high = beta.ppf([0.025, 0.975], alpha, beta_value)
    return float(mean), float(low), float(high)


def summarize_window(block: pd.DataFrame, end_ms: float, duration_sec: int, nogo_n: int) -> dict:
    start_ms = end_ms - duration_sec * 1000
    time_rows = block.loc[
        block["absolute_onset_time"].ge(start_ms)
        & block["absolute_onset_time"].le(end_ms)
    ]
    responded_go = time_rows.loc[time_rows["is_no_go"].eq(0) & time_rows["rt"].notna()]
    go_rows = time_rows.loc[time_rows["is_no_go"].eq(0)]
    time_nogo_rows = time_rows.loc[time_rows["is_no_go"].eq(1)]
    available_nogo = block.loc[
        block["is_no_go"].eq(1) & block["absolute_onset_time"].le(end_ms)
    ].sort_values("absolute_onset_time")
    recent_nogo = available_nogo.tail(nogo_n)
    actual_nogo = len(recent_nogo)
    commissions = int(recent_nogo["commission"].eq(1).sum())
    posterior_mean, posterior_low, posterior_high = _jeffreys(commissions, actual_nogo)
    omissions = int(go_rows["omission"].eq(1).sum()) if "omission" in go_rows else 0
    omission_mean, omission_low, omission_high = _jeffreys(omissions, len(go_rows))
    time_commissions = int(time_nogo_rows["commission"].eq(1).sum()) if "commission" in time_nogo_rows else 0
    time_commission_mean, time_commission_low, time_commission_high = _jeffreys(time_commissions, len(time_nogo_rows))
    if len(responded_go) == 0:
        status = BehaviorWindowStatus.INSUFFICIENT_RT.value
    elif actual_nogo == 0:
        status = BehaviorWindowStatus.RESPONSE_ONLY.value
    elif actual_nogo < nogo_n:
        status = BehaviorWindowStatus.INSUFFICIENT_NOGO.value
    else:
        status = BehaviorWindowStatus.FULL_EVIDENCE.value
    actual_span = (
        float(recent_nogo["absolute_onset_time"].max() - recent_nogo["absolute_onset_time"].min()) / 1000
        if actual_nogo >= 2 else 0.0
    )
    evidence_age = (
        float(end_ms - recent_nogo["absolute_onset_time"].max()) / 1000
        if actual_nogo else np.nan
    )
    block_first_ms = float(block["absolute_onset_time"].min())
    actual_start_ms = max(start_ms, block_first_ms)
    actual_coverage_sec = max(0.0, (end_ms - actual_start_ms) / 1000)
    short_rt_count = int(responded_go["rt"].lt(150).sum())
    return {
        "window_start_ms": start_ms,
        "window_end_ms": end_ms,
        "time_window_sec": duration_sec,
        "nogo_window_target": nogo_n,
        "window_status": status,
        "window_actual_start_ms": actual_start_ms,
        "window_actual_coverage_sec": actual_coverage_sec,
        "time_window_is_partial": actual_coverage_sec + 1e-9 < duration_sec,
        "trial_count": int(len(time_rows)),
        "go_opportunities": int(len(go_rows)),
        "go_rt_count": int(len(responded_go)),
        "go_rt_median_ms": float(responded_go["rt"].median()) if len(responded_go) else np.nan,
        "go_rt_iqr_ms": float(responded_go["rt"].quantile(0.75) - responded_go["rt"].quantile(0.25)) if len(responded_go) else np.nan,
        "go_rt_lt_150_count": short_rt_count,
        "go_rt_lt_150_rate": short_rt_count / len(go_rows) if len(go_rows) else np.nan,
        "go_omissions": omissions,
        "omission_jeffreys_mean": omission_mean,
        "omission_jeffreys_ci95_low": omission_low,
        "omission_jeffreys_ci95_high": omission_high,
        "time_nogo_opportunities": int(len(time_nogo_rows)),
        "time_nogo_commissions": time_commissions,
        "time_commission_jeffreys_mean": time_commission_mean,
        "time_commission_jeffreys_ci95_low": time_commission_low,
        "time_commission_jeffreys_ci95_high": time_commission_high,
        "nogo_opportunities_actual": int(actual_nogo),
        "nogo_commissions": commissions,
        "nogo_actual_span_sec": actual_span,
        "nogo_evidence_age_sec": evidence_age,
        "commission_jeffreys_mean": posterior_mean,
        "commission_jeffreys_ci95_low": posterior_low,
        "commission_jeffreys_ci95_high": posterior_high,
    }


def rolling_evidence(config: Config, trials: pd.DataFrame) -> pd.DataFrame:
    behavior = config.section("behavior")
    rows = []
    for (subject_id, block_num, condition), block in trials.groupby(
        ["subject_id", "block_num", "condition"], sort=True
    ):
        block = block.sort_values("absolute_onset_time")
        first = float(block["absolute_onset_time"].min())
        last = float(block["absolute_onset_time"].max())
        endpoints = np.arange(first, last + 1, behavior["step_sec"] * 1000)
        for end_ms in endpoints:
            for duration in behavior["time_windows_sec"]:
                for nogo_n in behavior["nogo_opportunity_windows"]:
                    row = summarize_window(block, end_ms, duration, nogo_n)
                    row.update({"subject_id": subject_id, "block_num": int(block_num), "condition": condition})
                    rows.append(row)
    return pd.DataFrame(rows)


def probe_evidence(config: Config, trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_, block_num, condition), block in trials.groupby(
        ["subject_id", "block_num", "condition"], sort=True
    ):
        probes = block.loc[block["is_probe"].eq(1) & block["probe_onset_time"].notna()]
        for _, probe in probes.iterrows():
            for duration in config.section("behavior")["probe_pre_windows_sec"]:
                # Probe sensitivity uses all requested No-Go opportunity definitions.
                for nogo_n in config.section("behavior")["nogo_opportunity_windows"]:
                    row = summarize_window(block, float(probe["probe_onset_time"]), duration, nogo_n)
                    row.update({
                        "subject_id": probe["subject_id"],
                        "block_num": int(block_num),
                        "condition": condition,
                        "probe_after_trial": int(probe["trial_num"]),
                        "probe_response": int(probe["probe_response"]),
                        "probe_state_label": probe["probe_state_label"],
                    })
                    rows.append(row)
    return pd.DataFrame(rows)


def cohort_rolling_evidence(config: Config, trials: pd.DataFrame) -> pd.DataFrame:
    """Build rolling evidence for a cohort DataFrame with a canonical ``subject`` column."""
    rows = []
    behavior = config.section("behavior")
    for (subject, block_num, condition), block in trials.groupby(
        ["subject", "block_num", "condition"], sort=True
    ):
        block = block.sort_values("absolute_onset_time")
        first = float(block["absolute_onset_time"].min())
        last = float(block["absolute_onset_time"].max())
        endpoints = np.arange(first, last + 1, behavior["step_sec"] * 1000)
        for end_ms in endpoints:
            for duration in behavior["time_windows_sec"]:
                for nogo_n in behavior["nogo_opportunity_windows"]:
                    row = summarize_window(block, float(end_ms), int(duration), int(nogo_n))
                    row.update({"subject": subject, "block_num": int(block_num), "condition": condition})
                    rows.append(row)
    return pd.DataFrame(rows)


def cohort_probe_evidence(config: Config, trials: pd.DataFrame) -> pd.DataFrame:
    """Build pre-probe evidence without merging the four nominal probe states."""
    rows = []
    behavior = config.section("behavior")
    for (subject, block_num, condition), block in trials.groupby(
        ["subject", "block_num", "condition"], sort=True
    ):
        block = block.sort_values("absolute_onset_time")
        probes = block.loc[block["is_probe"].eq(1) & block["probe_onset_time"].notna()]
        for _, probe in probes.iterrows():
            probe_number = config.section("protocol")["probe_after_trials"].index(int(probe["trial_num"])) + 1
            for duration in behavior["probe_pre_windows_sec"]:
                for nogo_n in behavior["nogo_opportunity_windows"]:
                    row = summarize_window(block, float(probe["probe_onset_time"]), int(duration), int(nogo_n))
                    row.update({
                        "subject": subject,
                        "block_num": int(block_num),
                        "condition": condition,
                        "probe_number_in_block": probe_number,
                        "probe_after_trial": int(probe["trial_num"]),
                        "probe_onset_time": int(probe["probe_onset_time"]),
                        "probe_response": int(probe["probe_response"]),
                        "probe_state_label": probe["probe_state_label"],
                    })
                    rows.append(row)
    return pd.DataFrame(rows)
