"""正式 BBB SART 统计层：主效应 / 交互 / 回归 / 相关。

推断单位=被试；bootstrap seed 固定；Holm 校正；n 小，报告 p 一律配效应量与 CI。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..config import Config


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni 校正（升序 p 从小到大逐步调整）。"""
    order = np.argsort(p_values)
    n = len(p_values)
    adjusted = np.full(n, np.nan)
    for rank, idx in enumerate(order, start=1):
        adjusted[idx] = max(p_values[idx] * (n - rank + 1), adjusted[order[:rank - 1]].max() if rank > 1 else 0.0)
    return [float(min(v, 1.0)) for v in adjusted]


def _bootstrap_ci(delta: np.ndarray, rng: np.random.Generator, iters: int) -> tuple[float, float]:
    boots = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(iters)])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


# ---------------------------------------------------------------- 主效应

MAIN_METRICS = {
    "commission_rate": "No-Go 误按率",
    "omission_rate": "Go 漏按率",
    "dprime_loglinear": "d′（loglinear）",
    "c": "反应标准 c",
    "beta": "似然比 β",
    "go_rt_median_ms": "正确 Go RT 中位数(ms)",
    "rt_cv": "RT-CV",
    "exg_tau": "ex-Gaussian τ(ms)",
}


def main_effects(config: Config, blocks: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """B1/B2/B3 主效应：Friedman + Kendall's W → 配对 Wilcoxon（Holm）→ 自助法 CI。
    返回 (汇总表, {metric: 详细 dict})。"""
    rng = np.random.default_rng(int(config.section("stats")["seed"]))
    iters = int(config.section("stats")["bootstrap_iterations"])
    rows = []
    details: dict = {}
    for metric, label in MAIN_METRICS.items():
        if metric not in blocks.columns:
            continue
        wide = blocks.pivot_table(index="subject", columns="block_num", values=metric)
        wide = wide[[1, 2, 3]]
        valid = wide.dropna()
        if len(valid) < 3:
            continue
        friedman = sps.friedmanchisquare(valid[1], valid[2], valid[3])
        # Kendall's W
        mat = valid.to_numpy(dtype=float)
        n, k = mat.shape
        ranks = sps.rankdata(mat, axis=1)
        W = 12 * float(np.sum((ranks.mean(axis=0) - (k + 1) / 2) ** 2)) / (n * k * (k + 1)) if n else np.nan
        # 配对对比（B3-B1 为主报告）
        contrasts = {"B2_minus_B1": (1, 2), "B3_minus_B2": (2, 3), "B3_minus_B1": (1, 3)}
        p_raw = []
        for _, (a, b) in contrasts.items():
            delta = valid[b] - valid[a]
            if np.all(delta == 0):
                p_raw.append(1.0)
            else:
                try:
                    p_raw.append(float(sps.wilcoxon(delta).pvalue))
                except ValueError:
                    p_raw.append(1.0)
        p_adj = holm(p_raw)
        d31 = (valid[3] - valid[1]).dropna().to_numpy(dtype=float)
        ci_lo, ci_hi = _bootstrap_ci(d31, rng, iters)
        n_worse = int((d31 > 0).sum())
        n_better = int((d31 < 0).sum())
        rows.append({
            "metric": metric,
            "metric_label": label,
            "n": int(len(valid)),
            "B1_mean": float(valid[1].mean()),
            "B2_mean": float(valid[2].mean()),
            "B3_mean": float(valid[3].mean()),
            "friedman_chi2": float(friedman.statistic),
            "friedman_p": float(friedman.pvalue),
            "kendall_W": W,
            "B2-B1_wilcoxon_p": p_raw[0],
            "B2-B1_wilcoxon_p_holm": p_adj[0],
            "B3-B2_wilcoxon_p": p_raw[1],
            "B3-B2_wilcoxon_p_holm": p_adj[1],
            "B3-B1_wilcoxon_p": p_raw[2],
            "B3-B1_wilcoxon_p_holm": p_adj[2],
            "B3-B1_mean_delta": float(d31.mean()) if len(d31) else np.nan,
            "B3-B1_ci95_low": ci_lo,
            "B3-B1_ci95_high": ci_hi,
            "n_worse": n_worse,
            "n_better": n_better,
            "n_same": int((d31 == 0).sum()),
        })
        details[metric] = {
            "wide": valid, "friedman": friedman, "W": W,
            "contrasts_p_raw": p_raw, "contrasts_p_holm": p_adj, "d31": d31,
        }
    return pd.DataFrame(rows), details


def _anova_rm(df: pd.DataFrame, dv: str) -> dict:
    """2 路重复测量 AnovaRM（block, bin）；异常静默降级返回空描述。"""
    try:
        from statsmodels.stats.anova import AnovaRM
        data = df[[dv, "subject", "block_num", "cycle_bin"]].dropna().copy()
        data["subject"] = data["subject"].astype(str)
        data["block_num"] = data["block_num"].astype(int)
        data["cycle_bin"] = data["cycle_bin"].astype(int)
        aov = AnovaRM(data, depvar=dv, subject="subject",
                      within=["block_num", "cycle_bin"]).fit()
        return {k: float(v) for k, v in aov.anova_table["Pr > F"].items()} | {
            f"F_{k}": float(v) for k, v in aov.anova_table["F Value"].items()
        }
    except Exception as exc:  # 小样本/不平衡时退化为描述性
        return {"_error": str(exc)}


def interaction_analysis(config: Config, bins: pd.DataFrame, trials: pd.DataFrame | None = None) -> dict:
    """block × 周期bin 交互：AnovaRM（RT 中位 / commission Jeffreys 均值）+ MixedLM 稳健性。"""
    out = {}
    rt = bins[["subject", "block_num", "cycle_bin", "go_rt_median_ms"]].dropna()
    comm = bins[["subject", "block_num", "cycle_bin", "commission_jeffreys_mean"]].dropna()
    out["rt_median_anova"] = _anova_rm(rt, "go_rt_median_ms")
    out["commission_anova"] = _anova_rm(comm, "commission_jeffreys_mean")
    if trials is not None:
        try:
            from statsmodels.formula.api import mixedlm
            sub = trials.loc[trials["go_rt_valid"].notna()].copy()
            sub["block_num"] = sub["block_num"].astype(int)
            sub["cycle_bin"] = sub["cycle_bin"].astype(int)
            model = mixedlm("go_rt_valid ~ C(block_num)*C(cycle_bin)", sub, groups=sub["subject"]).fit()
            interact_terms = [k for k in model.params.index if ":" in k]
            out["mixedlm"] = {
                "interact_p_min": float(min(model.pvalues[t] for t in interact_terms)) if interact_terms else np.nan,
                "n_trials": int(len(sub)),
                "n_subjects": int(sub["subject"].nunique()),
            }
        except Exception as exc:
            out["mixedlm"] = {"_error": str(exc)}
    return out


def regression_rt_drift(trials: pd.DataFrame) -> pd.DataFrame:
    """每 block 的 RT 漂移 MixedLM：rt ~ cycle_num + (1|subject)；加 block×cycle 交互模型。"""
    from statsmodels.formula.api import mixedlm
    rows = []
    for block_num in (1, 2, 3):
        sub = trials.loc[trials["block_num"].eq(block_num) & trials["go_rt_valid"].notna()]
        if len(sub) < 50:
            continue
        try:
            model = mixedlm("go_rt_valid ~ cycle_num", sub, groups=sub["subject"]).fit()
            rows.append({
                "model": f"B{block_num}",
                "slope_ms_per_cycle": float(model.params["cycle_num"]),
                "se": float(model.bse["cycle_num"]),
                "z": float(model.tvalues["cycle_num"]),
                "p": float(model.pvalues["cycle_num"]),
                "n_trials": int(len(sub)),
                "n_subjects": int(sub["subject"].nunique()),
            })
        except Exception as exc:
            rows.append({"model": f"B{block_num}", "slope_ms_per_cycle": np.nan,
                         "se": np.nan, "z": np.nan, "p": np.nan,
                         "n_trials": int(len(sub)), "n_subjects": int(sub["subject"].nunique()),
                         "error": str(exc)})
    sub = trials.loc[trials["go_rt_valid"].notna()]
    try:
        model = mixedlm("go_rt_valid ~ C(block_num)*cycle_num", sub, groups=sub["subject"]).fit()
        rows.append({
            "model": "block×cycle交互",
            "slope_ms_per_cycle": float(model.params["cycle_num"]),
            "se": float(model.bse["cycle_num"]),
            "z": float(model.tvalues["cycle_num"]),
            "p": float(model.pvalues["cycle_num"]),
            "interact_p": float(model.pvalues.get("C(block_num)[T.2]:cycle_num", np.nan)),
            "n_trials": int(len(sub)),
            "n_subjects": int(sub["subject"].nunique()),
        })
    except Exception as exc:
        rows.append({"model": "block×cycle交互", "slope_ms_per_cycle": np.nan, "se": np.nan,
                     "z": np.nan, "p": np.nan, "interact_p": np.nan,
                     "n_trials": int(len(sub)), "n_subjects": int(sub["subject"].nunique()),
                     "error": str(exc)})
    return pd.DataFrame(rows)


def pre_nogo_events(trials: pd.DataFrame, previous_go: int = 4) -> pd.DataFrame:
    """No-Go 事件前 previous_go 个正确 Go 的 RT 偏移（相对 block 内正确 Go 中位数）。"""
    records = []
    for (subject, block_num), block in trials.groupby(["subject", "block_num"], sort=True):
        block = block.sort_values("trial_num")
        baseline = float(block["go_rt_valid"].median())
        history: list[tuple[int, float]] = []
        for _, row in block.iterrows():
            if row["is_no_go"] == 0:
                if pd.notna(row["go_rt_valid"]):
                    history.append((int(row["trial_num"]), float(row["go_rt_valid"])))
                continue
            recent = history[-previous_go:]
            if len(recent) < previous_go:
                continue
            event_id = f"{subject}-B{int(block_num)}-T{int(row['trial_num'])}"
            for lag, (_, value) in zip(range(-previous_go, 0), recent):
                records.append({
                    "event_id": event_id,
                    "subject": subject,
                    "block_num": int(block_num),
                    "commission": int(row["commission"]),
                    "lag": lag,
                    "rt_ms": value,
                    "rt_offset_ms": value - baseline,
                })
    return pd.DataFrame(records)


def pre_nogo_stats(events: pd.DataFrame) -> pd.DataFrame:
    """错误前 RT 偏移：被试内聚合 → 逐 lag 配对检验（commission vs 正确抑制）。"""
    subject_lag = events.groupby(["subject", "commission", "lag"], as_index=False).agg(
        rt_offset_ms=("rt_offset_ms", "mean"), events=("event_id", "nunique"))
    rows = []
    for lag in sorted(subject_lag["lag"].unique()):
        wide = subject_lag.loc[subject_lag["lag"].eq(lag)].pivot(index="subject", columns="commission", values="rt_offset_ms")
        if 0 not in wide or 1 not in wide:
            continue
        pair = wide[[0, 1]].dropna()
        if len(pair) < 2:
            continue
        delta = pair[1] - pair[0]
        try:
            p = float(sps.wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
        except ValueError:
            p = 1.0
        rows.append({
            "lag": lag,
            "n_subjects": int(len(pair)),
            "correct_inhibit_offset_ms": float(pair[0].mean()),
            "commission_offset_ms": float(pair[1].mean()),
            "commission_minus_correct_ms": float(delta.mean()),
            "wilcoxon_p": p,
            "n_commission_events": int(subject_lag.loc[subject_lag["lag"].eq(lag) & subject_lag["commission"].eq(1), "events"].sum()),
            "n_correct_events": int(subject_lag.loc[subject_lag["lag"].eq(lag) & subject_lag["commission"].eq(0), "events"].sum()),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["wilcoxon_p_holm"] = holm(df["wilcoxon_p"].tolist())
    return df


def commission_gee(trials: pd.DataFrame) -> dict:
    """事件级 GEE 二项：commission ~ lag 偏移 + 位置 + block，按被试聚类。"""
    events = pre_nogo_events(trials, previous_go=4)
    summary = (events.groupby(["event_id", "subject", "block_num", "commission"])
               .agg(lag1_offset_ms=("rt_offset_ms", lambda s: s.iloc[-1]),
                    lag2_offset_ms=("rt_offset_ms", lambda s: s.iloc[-2]))
               .reset_index())
    meta = trials[["subject", "block_num", "trial_num", "position_in_cycle"]].drop_duplicates()
    meta["event_id"] = meta["subject"] + "-B" + meta["block_num"].astype(str) + "-T" + meta["trial_num"].astype(str)
    df = summary.merge(meta[["event_id", "position_in_cycle"]], on="event_id", how="left").dropna(subset=["lag1_offset_ms", "lag2_offset_ms"])
    if len(df) < 30 or df["commission"].nunique() < 2:
        return {"status": "insufficient", "rows": len(df)}
    try:
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial
        import statsmodels.api as sm
        model = GEE.from_formula(
            "commission ~ lag1_offset_ms + lag2_offset_ms + position_in_cycle + C(block_num)",
            groups=df["subject"], data=df, family=Binomial(), cov_struct=Exchangeable())
        res = model.fit()
        return {
            "status": "ok",
            "rows": int(len(df)),
            "n_subjects": int(df["subject"].nunique()),
            "commission_events": int(df["commission"].sum()),
            "params": {k: float(v) for k, v in res.params.items()},
            "pvalues": {k: float(v) for k, v in res.pvalues.items()},
            "conf_int": {k: [float(v[0]), float(v[1])] for k, v in res.conf_int().iterrows()},
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc), "rows": int(len(df))}


def probe_association(link: pd.DataFrame) -> dict:
    """探针 ↔ 行为：Q2 被试内 Spearman→单样本 Wilcoxon；Q2 二分化；Q1 分类对比。"""
    out = {}
    # Q2（有序）→ 探针后 RT / 误按
    rows = []
    for subject, grp in link.groupby("subject"):
        sub = grp.dropna(subset=["probe_vigilance", "post_go_rt_median_ms"])
        if len(sub) >= 3 and sub["probe_vigilance"].nunique() > 1:
            rho_rt, _ = sps.spearmanr(sub["probe_vigilance"], sub["post_go_rt_median_ms"])
            rows.append({"subject": subject, "rho_rt": rho_rt, "n": len(sub)})
    out["q2_rt_spearman"] = pd.DataFrame(rows)
    if len(rows) >= 3:
        rhos = [r["rho_rt"] for r in rows if np.isfinite(r["rho_rt"])]
        out["q2_rt_wilcoxon_p"] = float(sps.wilcoxon(rhos).pvalue) if len(rhos) >= 3 and np.any(rhos != 0) else np.nan
        out["q2_rt_median_rho"] = float(np.median(rhos))
        out["q2_rt_n_subjects"] = len(rhos)
    # Q2 二分化（低 1-2 vs 高 3-4）→ 探针后行为
    d = link.dropna(subset=["probe_vigilance", "post_go_rt_median_ms"]).copy()
    d["vig_hi"] = d["probe_vigilance"].ge(3).astype(int)
    subject_means = d.groupby(["subject", "vig_hi"]).agg(
        post_rt=("post_go_rt_median_ms", "mean"),
        post_comm=("post_comm_count_18", "mean")).reset_index()
    wide_rt = subject_means.pivot(index="subject", columns="vig_hi", values="post_rt")
    if 0 in wide_rt and 1 in wide_rt:
        pair = wide_rt[[0, 1]].dropna()
        if len(pair) >= 3:
            delta = pair[1] - pair[0]
            out["q2_hi_lo_rt_delta_ms"] = float(delta.mean())
            out["q2_hi_lo_rt_wilcoxon_p"] = float(sps.wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
            out["q2_hi_lo_rt_n"] = int(len(pair))
    # Q1 分类 → 探针后行为（名义，逐类均值，不做统计推断）
    q1 = link.dropna(subset=["probe_response", "post_go_rt_median_ms"]).groupby("probe_response")["post_go_rt_median_ms"].agg(["mean", "median", "count"])
    q1_comm = link.dropna(subset=["probe_response", "post_comm_count_18"]).groupby("probe_response")["post_comm_count_18"].agg(["mean", "count"])
    out["q1_post_rt_by_category"] = q1
    out["q1_post_comm_by_category"] = q1_comm
    return out


def correlation_analysis(config: Config, blocks: pd.DataFrame) -> dict:
    """相关：SAT（d′ vs RT-CV / RT中位）、跨 block 一致性、指标相关矩阵。"""
    rng = np.random.default_rng(int(config.section("stats")["seed"]))
    iters = int(config.section("stats")["bootstrap_iterations"])
    out = {}
    # 速度-准确权衡：subject×block 水平
    sat = blocks.dropna(subset=["dprime_loglinear", "rt_cv"])
    rho_cv, p_cv = sps.spearmanr(sat["dprime_loglinear"], sat["rt_cv"])
    out["sat_dprime_rtcv"] = {"rho": float(rho_cv), "p": float(p_cv), "n": int(len(sat))}
    sat_rt = blocks.dropna(subset=["dprime_loglinear", "go_rt_median_ms"])
    rho_rt, p_rt = sps.spearmanr(sat_rt["dprime_loglinear"], sat_rt["go_rt_median_ms"])
    out["sat_dprime_rtmedian"] = {"rho": float(rho_rt), "p": float(p_rt), "n": int(len(sat_rt))}
    # 跨 block 一致性（B1 vs B3）
    consistency_rows = []
    for metric in ["commission_rate", "omission_rate", "dprime_loglinear", "c", "go_rt_median_ms", "rt_cv"]:
        wide = blocks.pivot_table(index="subject", columns="block_num", values=metric)[[1, 3]].dropna()
        if len(wide) < 3:
            continue
        rho, p = sps.spearmanr(wide[1], wide[3])
        consistency_rows.append({"metric": metric, "rho_B1_B3": float(rho), "p": float(p), "n": int(len(wide))})
    out["cross_block"] = pd.DataFrame(consistency_rows)
    # 指标相关矩阵（subject×block 水平）
    corr_cols = ["commission_rate", "omission_rate", "dprime_loglinear", "c", "go_rt_median_ms", "rt_cv", "go_rt_lt_150_rate", "prestimulus_press_rate"]
    present = [c for c in corr_cols if c in blocks.columns]
    corr = blocks[present].corr(method="spearman")
    out["corr_matrix"] = corr
    return out
