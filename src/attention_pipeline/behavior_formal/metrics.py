"""Metrics for the final two-block formal SART behavior analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..config import Config


def _corrected_rate(successes: int, opportunities: int) -> float:
    if opportunities <= 0:
        return float("nan")
    return (successes + 0.5) / (opportunities + 1.0)


def sdt_measures(hit_rate: float, fa_rate: float) -> dict[str, float]:
    if not (np.isfinite(hit_rate) and np.isfinite(fa_rate)):
        return {"dprime": np.nan, "c": np.nan, "beta": np.nan}
    z_h = stats.norm.ppf(hit_rate)
    z_f = stats.norm.ppf(fa_rate)
    dprime = z_h - z_f
    c = -(z_h + z_f) / 2.0
    denom = stats.norm.pdf(z_f)
    beta = stats.norm.pdf(z_h) / denom if denom > 0 else np.nan
    return {"dprime": float(dprime), "c": float(c), "beta": float(beta)}


def _exgauss_negll(params, data: np.ndarray) -> float:
    mu, sigma, tau = params
    if sigma <= 1e-6 or tau <= 1e-6:
        return 1e12
    x = (data - mu) / sigma
    cdf_term = stats.norm.cdf(x - sigma / tau)
    logp = -np.log(tau) + (mu - data) / tau + sigma**2 / (2 * tau**2) + np.log(np.clip(cdf_term, 1e-300, None))
    return float(-np.sum(logp))


def fit_exgaussian(rt: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(rt, errors="coerce").dropna().to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return {"exg_mu": np.nan, "exg_sigma": np.nan, "exg_tau": np.nan, "exg_n": len(x)}
    mu0 = float(np.mean(x))
    sigma0 = max(float(np.std(x, ddof=1)), 1.0)
    starts = [(mu0, sigma0, sigma0 * 0.5), (float(np.median(x)), sigma0, sigma0 * 0.5)]
    best = None
    best_value = np.inf
    for start in starts:
        result = optimize.minimize(_exgauss_negll, start, args=(x,), method="Nelder-Mead", options={"maxiter": 4000})
        if result.success and result.fun < best_value:
            best = result.x
            best_value = float(result.fun)
    if best is None:
        return {"exg_mu": np.nan, "exg_sigma": np.nan, "exg_tau": np.nan, "exg_n": len(x)}
    return {"exg_mu": float(best[0]), "exg_sigma": float(best[1]), "exg_tau": float(best[2]), "exg_n": int(len(x))}


def formal_block_metrics(config: Config, trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    min_n = int(config.section("behavior").get("rt_cv_min_n", 20))
    for (subject, block_num), block in trials.groupby(["subject", "block_num"], sort=True):
        go = block.loc[block["is_no_go"].eq(0)]
        nogo = block.loc[block["is_no_go"].eq(1)]
        hits = int(go["correct"].eq(1).sum())
        false_alarms = int(nogo["commission"].eq(1).sum())
        hit_rate = _corrected_rate(hits, len(go))
        fa_rate = _corrected_rate(false_alarms, len(nogo))
        sdt = sdt_measures(hit_rate, fa_rate)
        go_rt = go["go_rt_valid"].dropna()
        n_rt = int(len(go_rt))
        rt_mean = float(go_rt.mean()) if n_rt else np.nan
        rt_sd = float(go_rt.std(ddof=1)) if n_rt > 1 else np.nan
        exg = fit_exgaussian(go_rt)
        rows.append({
            "subject": subject,
            "block_num": int(block_num),
            "condition": str(block["condition"].iloc[0]),
            "trials": int(len(block)),
            "go_opportunities": int(len(go)),
            "nogo_opportunities": int(len(nogo)),
            "correct_go_hits": hits,
            "nogo_commissions": false_alarms,
            "hit_rate_loglinear": hit_rate,
            "false_alarm_rate_loglinear": fa_rate,
            "dprime_loglinear": sdt["dprime"],
            "c": sdt["c"],
            "beta": sdt["beta"],
            "omission_rate": float(go["omission"].mean()) if len(go) else np.nan,
            "commission_rate": float(nogo["commission"].mean()) if len(nogo) else np.nan,
            "go_rt_count": n_rt,
            "go_rt_median_ms": float(go_rt.median()) if n_rt else np.nan,
            "go_rt_mean_ms": rt_mean,
            "go_rt_sd_ms": rt_sd,
            "rt_cv": (rt_sd / rt_mean) if n_rt >= min_n and np.isfinite(rt_sd) and rt_mean > 0 else np.nan,
            "go_rt_lt_150_rate": float(go_rt.lt(150).mean()) if n_rt else np.nan,
            **exg,
        })
    return pd.DataFrame(rows)


def cycle_bin_metrics(config: Config, trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    min_n = int(config.section("behavior").get("rt_cv_min_n", 20))
    for (subject, block_num, cycle_bin), grp in trials.dropna(subset=["cycle_bin"]).groupby(["subject", "block_num", "cycle_bin"], sort=True):
        go = grp.loc[grp["is_no_go"].eq(0)]
        nogo = grp.loc[grp["is_no_go"].eq(1)]
        go_rt = go["go_rt_valid"].dropna()
        rt_mean = float(go_rt.mean()) if len(go_rt) else np.nan
        rt_sd = float(go_rt.std(ddof=1)) if len(go_rt) > 1 else np.nan
        rows.append({
            "subject": subject,
            "block_num": int(block_num),
            "cycle_bin": int(cycle_bin),
            "go_opportunities": int(len(go)),
            "nogo_opportunities": int(len(nogo)),
            "commission_rate": float(nogo["commission"].mean()) if len(nogo) else np.nan,
            "omission_rate": float(go["omission"].mean()) if len(go) else np.nan,
            "go_rt_count": int(len(go_rt)),
            "go_rt_median_ms": float(go_rt.median()) if len(go_rt) else np.nan,
            "rt_cv": (rt_sd / rt_mean) if len(go_rt) >= min_n and np.isfinite(rt_sd) and rt_mean > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def probe_behaviour_link(config: Config, trials: pd.DataFrame) -> pd.DataFrame:
    """One row per probe with behavior immediately preceding that probe."""
    preceding_n = int(config.section("behavior").get("probe_preceding_go_trials", 8))
    rows = []
    for (subject, block_num), block in trials.groupby(["subject", "block_num"], sort=True):
        block = block.sort_values("trial_num").reset_index(drop=True)
        probe_rows = block.index[block["is_probe"].eq(1) & block["probe_response"].notna()]
        for probe_order, idx in enumerate(probe_rows, start=1):
            p = block.loc[idx]
            preceding = block.loc[(block.index < idx) & block["is_no_go"].eq(0) & block["correct"].eq(1), "go_rt_valid"].dropna().tail(preceding_n)
            mean = float(preceding.mean()) if len(preceding) else np.nan
            sd = float(preceding.std(ddof=1)) if len(preceding) > 1 else np.nan
            rows.append({
                "subject": subject,
                "block_num": int(block_num),
                "probe_number_in_block": int(probe_order),
                "probe_after_trial": int(p["trial_num"]),
                "probe_response": int(p["probe_response"]),
                "probe_state_label": p.get("probe_state_label"),
                "probe_vigilance": int(p["probe_vigilance"]) if pd.notna(p["probe_vigilance"]) else np.nan,
                "probe_vigilance_label": p.get("probe_vigilance_label"),
                "pre_go_n": int(len(preceding)),
                "pre_go_rt_median_ms": float(preceding.median()) if len(preceding) else np.nan,
                "pre_go_rt_mean_ms": mean,
                "pre_go_rt_cv": (sd / mean) if np.isfinite(sd) and mean > 0 else np.nan,
            })
    return pd.DataFrame(rows)
