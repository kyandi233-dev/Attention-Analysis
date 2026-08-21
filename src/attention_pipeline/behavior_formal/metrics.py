"""正式 BBB SART 指标层：block / 周期bin / 时间窗 / 探针 四层。

包含新增：RT 变异性（rt_sd / rt_cv / ex-Gaussian τ）与信号检测论反应标准 c、β。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.stats import beta as beta_dist

from ..config import Config


# ---------------------------------------------------------------- SDT 反应标准

def sdt_measures(hit_rate: float, fa_rate: float) -> dict:
    """由 loglinear 校正后的 hit/fa 率计算 d′、c、β。

    c = -(zH+zF)/2（c<0 宽松=倾向按键，c>0 保守）；β=f(zH)/f(zF) 似然比。
    """
    if not (np.isfinite(hit_rate) and np.isfinite(fa_rate)):
        return {"dprime": np.nan, "c": np.nan, "beta": np.nan}
    z_h = stats.norm.ppf(hit_rate)
    z_f = stats.norm.ppf(fa_rate)
    dprime = z_h - z_f
    c = -(z_h + z_f) / 2.0
    beta = stats.norm.pdf(z_h) / stats.norm.pdf(z_f) if stats.norm.pdf(z_f) > 0 else np.nan
    return {"dprime": dprime, "c": c, "beta": beta}


def _corrected_rate(successes: int, opportunities: int) -> float:
    if opportunities <= 0:
        return float("nan")
    return (successes + 0.5) / (opportunities + 1.0)


# ---------------------------------------------------------------- ex-Gaussian

def _exgauss_negll(params, data: np.ndarray) -> float:
    mu, sigma, tau = params
    if sigma <= 1e-6 or tau <= 1e-6:
        return 1e12
    x = (data - mu) / sigma
    # 对 cdf 与 tau 做下界裁剪，避免优化器探索无效参数时 log(0) → RuntimeWarning
    cdf_term = stats.norm.cdf(x - sigma / tau)
    logp = -np.log(np.clip(tau, 1e-6, None)) + (mu - data) / tau + sigma ** 2 / (2 * tau ** 2) + np.log(np.clip(cdf_term, 1e-300, None))
    return float(-np.sum(logp))


def fit_exgaussian(rt: pd.Series) -> dict:
    """拟合正确 Go RT 的 ex-Gaussian（μ,σ,τ），τ=慢尾均值（注意滑脱）。"""
    x = rt.dropna().to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return {"exg_mu": np.nan, "exg_sigma": np.nan, "exg_tau": np.nan, "exg_n": int(len(x))}
    mu0, sigma0, tau0 = float(np.mean(x)), float(np.std(x, ddof=1)), float(np.std(x, ddof=1)) * 0.5
    best = None
    best_val = np.inf
    for init in ((mu0, sigma0, tau0), (mu0, sigma0 * 0.8, tau0), (np.median(x), sigma0, tau0)):
        try:
            res = optimize.minimize(_exgauss_negll, init, args=(x,), method="Nelder-Mead",
                                    options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best = res.x
        except Exception:
            continue
    if best is None:
        return {"exg_mu": np.nan, "exg_sigma": np.nan, "exg_tau": np.nan, "exg_n": int(len(x))}
    return {"exg_mu": float(best[0]), "exg_sigma": float(best[1]), "exg_tau": float(best[2]), "exg_n": int(len(x))}


# ---------------------------------------------------------------- block 指标

def formal_block_metrics(config: Config, trials: pd.DataFrame) -> pd.DataFrame:
    """逐 被试×block 指标：机会数、错误率、loglinear d′/c/β、RT 中位/SD/CV、ex-Gaussian、QC。"""
    rows = []
    min_n = int(config.section("behavior")["rt_cv_min_n"])
    for (subject, block_num), block in trials.groupby(["subject", "block_num"], sort=True):
        go = block.loc[block["is_no_go"].eq(0)]
        nogo = block.loc[block["is_no_go"].eq(1)]
        hits = int(go["correct"].eq(1).sum())
        false_alarms = int(nogo["commission"].eq(1).sum())
        hit_rate = _corrected_rate(hits, len(go))
        fa_rate = _corrected_rate(false_alarms, len(nogo))
        sdt = sdt_measures(hit_rate, fa_rate)
        go_rt = go["go_rt_valid"].dropna()
        n = int(go_rt.notna().sum())
        rt_mean = float(go_rt.mean()) if n else np.nan
        rt_sd = float(go_rt.std(ddof=1)) if n > 1 else np.nan
        rt_cv = rt_sd / rt_mean if (np.isfinite(rt_sd) and rt_mean > 0) else np.nan
        exg = fit_exgaussian(go_rt)
        prestim = int(block["prestimulus_press_ms"].fillna("").astype(str).str.len().gt(0).sum())
        rows.append({
            "subject": subject,
            "block_num": int(block_num),
            "condition": block["condition"].iloc[0],
            "go_opportunities": int(len(go)),
            "correct_go_hits": hits,
            "nogo_opportunities": int(len(nogo)),
            "nogo_commissions": false_alarms,
            "hit_rate_loglinear": hit_rate,
            "false_alarm_rate_loglinear": fa_rate,
            "dprime_loglinear": sdt["dprime"],
            "c": sdt["c"],
            "beta": sdt["beta"],
            "omission_rate": float(go["omission"].mean()) if len(go) else np.nan,
            "commission_rate": float(nogo["commission"].mean()) if len(nogo) else np.nan,
            "go_rt_count": n,
            "go_rt_median_ms": float(go_rt.median()) if n else np.nan,
            "go_rt_mean_ms": rt_mean,
            "go_rt_sd_ms": rt_sd,
            "rt_cv": rt_cv if n >= min_n else np.nan,
            "go_rt_lt_150_rate": float(go_rt.lt(150).sum() / n) if n else np.nan,
            "rt_qc_lt_100_count": int(go_rt.lt(100).sum()),
            "rt_qc_gt_1150_count": int(go_rt.gt(1150).sum()),
            "prestimulus_press_count": prestim,
            "prestimulus_press_rate": prestim / len(block) if len(block) else np.nan,
            "exg_mu": exg["exg_mu"],
            "exg_sigma": exg["exg_sigma"],
            "exg_tau": exg["exg_tau"],
            "exg_n": exg["exg_n"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 周期 bin 指标

def cycle_bin_metrics(config: Config, trials: pd.DataFrame) -> pd.DataFrame:
    """逐 被试×block×cycle_bin：实际机会数与错误数、Jeffreys 率、RT 中位、rt_cv（计数门控）。"""
    min_n = int(config.section("behavior")["rt_cv_min_n"])
    rows = []
    for (subject, block_num), block in trials.groupby(["subject", "block_num"], sort=True):
        for bin_id, grp in block.groupby("cycle_bin", dropna=True, sort=True):
            go = grp.loc[grp["is_no_go"].eq(0)]
            nogo = grp.loc[grp["is_no_go"].eq(1)]
            n_comm = int(nogo["commission"].eq(1).sum())
            n_omiss = int(go["omission"].eq(1).sum())
            comm_mean, comm_lo, comm_hi = _jeffreys(n_comm, len(nogo))
            omiss_mean, omiss_lo, omiss_hi = _jeffreys(n_omiss, len(go))
            go_rt = go["go_rt_valid"].dropna()
            n_rt = int(go_rt.notna().sum())
            rt_median = float(go_rt.median()) if n_rt else np.nan
            rt_mean = float(go_rt.mean()) if n_rt else np.nan
            rt_sd = float(go_rt.std(ddof=1)) if n_rt > 1 else np.nan
            rt_cv = rt_sd / rt_mean if (np.isfinite(rt_sd) and rt_mean > 0 and n_rt >= min_n) else np.nan
            rows.append({
                "subject": subject,
                "block_num": int(block_num),
                "cycle_bin": int(bin_id),
                "go_opportunities": int(len(go)),
                "nogo_opportunities": int(len(nogo)),
                "nogo_commissions": n_comm,
                "go_omissions": n_omiss,
                "commission_jeffreys_mean": comm_mean,
                "commission_jeffreys_ci95_low": comm_lo,
                "commission_jeffreys_ci95_high": comm_hi,
                "omission_jeffreys_mean": omiss_mean,
                "omission_jeffreys_ci95_low": omiss_lo,
                "omission_jeffreys_ci95_high": omiss_hi,
                "go_rt_count": n_rt,
                "go_rt_median_ms": rt_median,
                "rt_cv": rt_cv,
                "go_rt_lt_150_rate": float(go_rt.lt(150).sum() / n_rt) if n_rt else np.nan,
                "prestimulus_press_count": int(grp["prestimulus_press_ms"].fillna("").astype(str).str.len().gt(0).sum()),
            })
    return pd.DataFrame(rows)


def _jeffreys(k: int, n: int) -> tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    a, b = k + 0.5, n - k + 0.5
    return float(a / (a + b)), float(beta_dist.ppf(0.025, a, b)), float(beta_dist.ppf(0.975, a, b))


# ---------------------------------------------------------------- 时间窗指标

def rolling_evidence_formal(config: Config, trials: pd.DataFrame) -> pd.DataFrame:
    """30/60/90/120s × 步长10s × nogo{6,8,12} 滑窗（复用 behavior.evidence.cohort_rolling_evidence）。"""
    from ..behavior.evidence import cohort_rolling_evidence
    return cohort_rolling_evidence(config, trials)


def probe_evidence_formal(config: Config, trials: pd.DataFrame) -> pd.DataFrame:
    """探针前/后窗行为 × Q1/Q2。探针编号按 per-block probe_positions 查表。"""
    from ..behavior.evidence import summarize_window
    behavior = config.section("behavior")
    positions = config.section("protocol")["probe_positions"]
    pre_durs = behavior["probe_pre_windows_sec"]
    post_durs = behavior["probe_post_windows_sec"]
    rows = []
    for (subject, block_num), block in trials.groupby(["subject", "block_num"], sort=True):
        block = block.sort_values("absolute_onset_time")
        pos_list = positions[str(block_num)] if str(block_num) in positions else positions[int(block_num)]
        probes = block.loc[block["is_probe"].eq(1) & block["probe_onset_time"].notna()].sort_values("trial_num")
        for _, probe in probes.iterrows():
            probe_num = int(pos_list.index(int(probe["trial_num"]))) + 1
            base = {
                "subject": subject,
                "block_num": int(block_num),
                "probe_number_in_block": probe_num,
                "probe_after_trial": int(probe["trial_num"]),
                "probe_onset_time": int(probe["probe_onset_time"]),
                "probe_response": int(probe["probe_response"]),
                "probe_state_label": probe["probe_state_label"],
                "probe_vigilance": int(probe["probe_vigilance"]),
                "probe_rt_ms": probe["probe_rt"] if pd.notna(probe["probe_rt"]) else np.nan,
                "probe_vigilance_rt_ms": probe["probe_vigilance_rt"] if pd.notna(probe["probe_vigilance_rt"]) else np.nan,
            }
            onset = float(probe["probe_onset_time"])
            for kind, durs in (("pre", pre_durs), ("post", post_durs)):
                for duration in durs:
                    for nogo_n in behavior["nogo_opportunity_windows"]:
                        if kind == "pre":
                            win = summarize_window(block, onset, int(duration), int(nogo_n))
                        else:
                            win = summarize_window(block, onset + duration * 1000.0, int(duration), int(nogo_n))
                        row = dict(base)
                        row.update({
                            "window_kind": kind,
                            "time_window_sec": duration,
                            "nogo_window_target": nogo_n,
                            "window_status": win["window_status"],
                            "go_rt_median_ms": win["go_rt_median_ms"],
                            "go_rt_iqr_ms": win["go_rt_iqr_ms"],
                            "go_rt_lt_150_rate": win["go_rt_lt_150_rate"],
                            "time_commission_jeffreys_mean": win["time_commission_jeffreys_mean"],
                            "time_commission_jeffreys_ci95_low": win["time_commission_jeffreys_ci95_low"],
                            "time_commission_jeffreys_ci95_high": win["time_commission_jeffreys_ci95_high"],
                            "omission_jeffreys_mean": win["omission_jeffreys_mean"],
                            "time_nogo_opportunities": win["time_nogo_opportunities"],
                            "go_rt_count": win["go_rt_count"],
                        })
                        rows.append(row)
    return pd.DataFrame(rows)


def probe_behaviour_link(trials: pd.DataFrame) -> pd.DataFrame:
    """逐探针：探针评分 + 探针前/后 8 试次内的行为（用于 Q1/Q2 关联）。"""
    out = []
    for (subject, block_num), block in trials.groupby(["subject", "block_num"], sort=True):
        block = block.sort_values("trial_num")
        probe_idx = block.index[block["is_probe"].eq(1) & block["probe_response"].notna()]
        for p_idx in probe_idx:
            p = block.loc[p_idx]
            trial_num = int(p["trial_num"])
            pre = block.loc[(block.index < p_idx) & block["is_no_go"].eq(0) & block["correct"].eq(1)]
            post = block.loc[(block.index > p_idx) & block["is_no_go"].eq(0) & block["correct"].eq(1)]
            pre_rt = pre["go_rt_valid"].dropna().tail(8)
            post_rt = post["go_rt_valid"].dropna().head(8)
            pre_comm = int(block.loc[(block.index < p_idx) & (block.index >= p_idx - 18) & block["commission"].eq(1)].shape[0])
            post_comm = int(block.loc[(block.index > p_idx) & (block.index <= p_idx + 18) & block["commission"].eq(1)].shape[0])
            out.append({
                "subject": subject,
                "block_num": int(block_num),
                "probe_after_trial": trial_num,
                "probe_response": int(p["probe_response"]),
                "probe_vigilance": int(p["probe_vigilance"]),
                "pre_go_rt_median_ms": float(pre_rt.median()) if len(pre_rt) else np.nan,
                "pre_go_rt_cv": float(pre_rt.std(ddof=1) / pre_rt.mean()) if len(pre_rt) > 1 and pre_rt.mean() > 0 else np.nan,
                "pre_go_rt_lt150_rate": float(pre_rt.lt(150).sum() / len(pre_rt)) if len(pre_rt) else np.nan,
                "pre_comm_count_18": pre_comm,
                "post_go_rt_median_ms": float(post_rt.median()) if len(post_rt) else np.nan,
                "post_go_rt_cv": float(post_rt.std(ddof=1) / post_rt.mean()) if len(post_rt) > 1 and post_rt.mean() > 0 else np.nan,
                "post_comm_count_18": post_comm,
            })
    return pd.DataFrame(out)
