"""Statistics for the final FocusWave v3.1.3 two-block formal behavior analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..config import Config

MAIN_METRICS = {
    "commission_rate": "No-Go commission rate",
    "omission_rate": "Go omission rate",
    "dprime_loglinear": "d-prime",
    "c": "criterion c",
    "beta": "beta",
    "go_rt_median_ms": "Go RT median (ms)",
    "rt_cv": "RT CV",
    "exg_tau": "ex-Gaussian tau (ms)",
}


def holm(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.zeros(len(p_values), dtype=float)
    running = 0.0
    n = len(p_values)
    for rank, idx in enumerate(order):
        value = min(1.0, float(p_values[idx]) * (n - rank))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def _bootstrap_mean_ci(delta: np.ndarray, seed: int, iterations: int) -> tuple[float, float]:
    if len(delta) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.empty(iterations, dtype=float)
    for i in range(iterations):
        boots[i] = float(np.mean(rng.choice(delta, len(delta), replace=True)))
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def paired_block_effects(config: Config, blocks: pd.DataFrame) -> pd.DataFrame:
    """Primary BB inference: paired B2-B1 comparisons at subject level."""
    stats_cfg = config.section("stats")
    seed = int(stats_cfg.get("seed", 20260824))
    iterations = int(stats_cfg.get("bootstrap_iterations", 20000))
    requested = stats_cfg.get("main_metrics", list(MAIN_METRICS))
    rows = []
    raw_p = []
    for metric in requested:
        if metric not in blocks.columns:
            continue
        wide = blocks.pivot_table(index="subject", columns="block_num", values=metric)
        if not {1, 2}.issubset(wide.columns):
            continue
        pair = wide[[1, 2]].dropna()
        if len(pair) < 2:
            continue
        delta = (pair[2] - pair[1]).to_numpy(dtype=float)
        if np.allclose(delta, 0):
            p, statistic = 1.0, 0.0
        else:
            test = sps.wilcoxon(delta)
            statistic, p = float(test.statistic), float(test.pvalue)
        ci_low, ci_high = _bootstrap_mean_ci(delta, seed, iterations)
        sd_delta = float(np.std(delta, ddof=1)) if len(delta) > 1 else np.nan
        dz = float(np.mean(delta) / sd_delta) if np.isfinite(sd_delta) and sd_delta > 0 else np.nan
        raw_p.append(p)
        rows.append({
            "metric": metric,
            "metric_label": MAIN_METRICS.get(metric, metric),
            "n": int(len(pair)),
            "B1_mean": float(pair[1].mean()),
            "B2_mean": float(pair[2].mean()),
            "B2_minus_B1_mean": float(np.mean(delta)),
            "B2_minus_B1_median": float(np.median(delta)),
            "B2_minus_B1_ci95_low": ci_low,
            "B2_minus_B1_ci95_high": ci_high,
            "cohen_dz": dz,
            "wilcoxon_statistic": statistic,
            "wilcoxon_p": p,
        })
    for row, value in zip(rows, holm(raw_p)):
        row["wilcoxon_p_holm"] = value
    return pd.DataFrame(rows)


def block_bin_anova(bins: pd.DataFrame, dv: str) -> dict:
    try:
        from statsmodels.stats.anova import AnovaRM
        data = bins[["subject", "block_num", "cycle_bin", dv]].dropna().copy()
        counts = data.groupby(["subject", "block_num", "cycle_bin"]).size()
        if counts.empty or not counts.eq(1).all():
            return {"status": "unbalanced"}
        result = AnovaRM(data, depvar=dv, subject="subject", within=["block_num", "cycle_bin"]).fit().anova_table
        return {"status": "ok", "effects": {str(idx): {"F": float(row["F Value"]), "p": float(row["Pr > F"])} for idx, row in result.iterrows()}}
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}


def rt_drift_mixedlm(trials: pd.DataFrame) -> dict:
    data = trials.loc[trials["go_rt_valid"].notna()].copy()
    if len(data) < 50 or data["subject"].nunique() < 2:
        return {"status": "insufficient", "n_trials": int(len(data))}
    try:
        from statsmodels.formula.api import mixedlm
        model = mixedlm("go_rt_valid ~ C(block_num) * cycle_num", data, groups=data["subject"]).fit()
        return {
            "status": "ok",
            "n_trials": int(len(data)),
            "n_subjects": int(data["subject"].nunique()),
            "params": {str(k): float(v) for k, v in model.params.items()},
            "pvalues": {str(k): float(v) for k, v in model.pvalues.items()},
        }
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}


def pre_nogo_events(trials: pd.DataFrame, previous_go: int = 4) -> pd.DataFrame:
    records = []
    for (subject, block_num), block in trials.groupby(["subject", "block_num"], sort=True):
        block = block.sort_values("trial_num")
        baseline = float(block["go_rt_valid"].median())
        go_history: list[float] = []
        for _, row in block.iterrows():
            if int(row["is_no_go"]) == 0:
                if pd.notna(row["go_rt_valid"]):
                    go_history.append(float(row["go_rt_valid"]))
                continue
            recent = go_history[-previous_go:]
            if len(recent) < previous_go:
                continue
            for lag, rt in zip(range(-previous_go, 0), recent):
                records.append({"subject": subject, "block_num": int(block_num), "trial_num": int(row["trial_num"]), "commission": int(row["commission"]), "lag": int(lag), "rt_offset_ms": rt - baseline})
    return pd.DataFrame(records)


def pre_nogo_stats(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    subject_lag = events.groupby(["subject", "commission", "lag"], as_index=False).agg(rt_offset_ms=("rt_offset_ms", "mean"))
    rows = []
    for lag in sorted(subject_lag["lag"].unique()):
        wide = subject_lag.loc[subject_lag["lag"].eq(lag)].pivot(index="subject", columns="commission", values="rt_offset_ms")
        if not {0, 1}.issubset(wide.columns):
            continue
        pair = wide[[0, 1]].dropna()
        if len(pair) < 2:
            continue
        delta = pair[1] - pair[0]
        p = float(sps.wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
        rows.append({"lag": int(lag), "n_subjects": int(len(pair)), "commission_minus_correct_ms": float(delta.mean()), "wilcoxon_p": p})
    result = pd.DataFrame(rows)
    if len(result):
        result["wilcoxon_p_holm"] = holm(result["wilcoxon_p"].tolist())
    return result


def probe_associations(probes: pd.DataFrame) -> dict:
    out: dict = {"n_probes": int(len(probes))}
    if probes.empty:
        return out
    q2 = probes.dropna(subset=["probe_vigilance", "pre_go_rt_median_ms"])
    if len(q2) >= 5 and q2["probe_vigilance"].nunique() >= 2:
        rho, p = sps.spearmanr(q2["probe_vigilance"], q2["pre_go_rt_median_ms"])
        out["vigilance_vs_pre_rt"] = {"rho": float(rho), "p": float(p), "n": int(len(q2))}
    q1 = probes.dropna(subset=["probe_response", "pre_go_rt_median_ms"])
    groups = [grp["pre_go_rt_median_ms"].dropna().to_numpy() for _, grp in q1.groupby("probe_response") if len(grp["pre_go_rt_median_ms"].dropna()) >= 2]
    if len(groups) >= 2:
        h, p = sps.kruskal(*groups)
        out["attention_state_vs_pre_rt"] = {"H": float(h), "p": float(p), "groups": int(len(groups)), "n": int(len(q1))}
    return out
