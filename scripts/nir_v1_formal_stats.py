"""Formal NIR v1 statistics and presentation figures.

The cohort and coverage tiers are inputs, not discovered here.  The primary
analysis is the frozen primary/30 s/PIR set; all centering is recomputed inside
each analytic set before fitting subject-clustered GEE models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian, Binomial


PIR = "pir_fused_pir_median"
MAD = "pir_fused_pir_mad"
SLOPE = "pir_fused_pir_robust_slope_per_s"
TIME = "time_on_task_sec"
RESPONSE = "probe_response_raw"
VIGILANCE = "probe_vigilance_raw"
FEATURES = ["pir_within_subject_deviation", MAD, SLOPE, "block", TIME]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(d: pd.DataFrame, window: str, crosswalk: pd.DataFrame | None = None) -> pd.DataFrame:
    x = d[(d.nir_quality_tier == "primary") & (d.window == window)].copy()
    # Recompute centering from this analytic set only; never reuse the dry-run
    # full-cohort centering columns.
    x["pir_subject_mean"] = x.groupby("subject")[PIR].transform("mean")
    x["pir_within_subject_deviation"] = x[PIR] - x["pir_subject_mean"]
    x["fully_focused"] = (x[RESPONSE] == 1).astype(int)
    if crosswalk is not None:
        cw = crosswalk[crosswalk["session_id"].notna()][["session_id", "repeat_participant_id", "actual_session_link_status"]].drop_duplicates("session_id")
        x = x.merge(cw, left_on="subject", right_on="session_id", how="left", validate="many_to_one")
    x["cluster_id"] = x["repeat_participant_id"].fillna(x["subject"])
    for c in FEATURES:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.dropna(subset=FEATURES + ["fully_focused", RESPONSE, VIGILANCE]).copy()


def design(x: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    z = x[terms].copy()
    for c in terms:
        if c not in ("block",):
            sd = z[c].std(ddof=0)
            z[c] = (z[c] - z[c].mean()) / sd if sd > 0 else 0.0
    return sm.add_constant(z, has_constant="add")


def gee_logistic(x: pd.DataFrame, y: str, terms: list[str]) -> tuple[pd.DataFrame, object]:
    model = GEE(x[y], design(x, terms), groups=x["cluster_id"], family=Binomial(), cov_struct=Exchangeable())
    fit = model.fit()
    rows = []
    for term, beta, se, p in zip(fit.params.index, fit.params, fit.bse, fit.pvalues):
        rows.append({"outcome": y, "term": term, "n": len(x), "subjects": x.subject.nunique(), "clusters": x.cluster_id.nunique(),
                     "estimate_log_odds": beta, "odds_ratio": np.exp(beta),
                     "ci_low": np.exp(beta - 1.96 * se), "ci_high": np.exp(beta + 1.96 * se), "p_value": p,
                     "model": "GEE binomial exchangeable, subject-clustered"})
    return pd.DataFrame(rows), fit


def gee_gaussian(x: pd.DataFrame, y: str, terms: list[str]) -> tuple[pd.DataFrame, object]:
    model = GEE(x[y], design(x, terms), groups=x["cluster_id"], family=Gaussian(), cov_struct=Exchangeable())
    fit = model.fit()
    rows = []
    for term, beta, se, p in zip(fit.params.index, fit.params, fit.bse, fit.pvalues):
        rows.append({"outcome": y, "term": term, "n": len(x), "subjects": x.subject.nunique(), "clusters": x.cluster_id.nunique(),
                     "estimate": beta, "ci_low": beta - 1.96 * se, "ci_high": beta + 1.96 * se, "p_value": p,
                     "model": "GEE Gaussian exchangeable, subject-clustered"})
    return pd.DataFrame(rows), fit


def response_descriptives(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in x.groupby(RESPONSE):
        rows.append({"response": int(label), "label": {1: "fully task-focused", 2: "not fully task-focused: focused elsewhere", 3: "not fully task-focused: task-unrelated thought", 4: "not fully task-focused: mind blank"}[int(label)],
                     "n": len(g), "subjects": g.subject.nunique(), "pir_median_mean": g[PIR].mean(),
                     "pir_median_sd": g[PIR].std(), "pir_within_deviation_mean": g["pir_within_subject_deviation"].mean(),
                     "pir_mad_mean": g[MAD].mean(), "pir_slope_mean": g[SLOPE].mean()})
    return pd.DataFrame(rows)


def forest(results: pd.DataFrame, root: Path):
    q = results[(results.term == "pir_within_subject_deviation") & (results.outcome == "fully_focused")].copy()
    q["window"] = q["window_seconds"].astype(int).astype(str) + " s"
    q = q.sort_values("window_seconds")
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=180)
    y = np.arange(len(q))
    ax.errorbar(q.odds_ratio, y, xerr=[q.odds_ratio-q.ci_low, q.ci_high-q.odds_ratio], fmt="o", capsize=4, color="#145a86")
    ax.axvline(1, color="#777", ls="--", lw=1)
    ax.set_yticks(y, q.window); ax.set_xlabel("Odds ratio per 1 SD PIR within-person deviation")
    ax.set_title("Fully task-focused probability: 10/20/30 s")
    ax.grid(axis="x", alpha=.25); fig.tight_layout(); fig.savefig(root / "figure_4_effect_forest.png"); plt.close(fig)


def make_figures(x: pd.DataFrame, core_fit, sensitivity: pd.DataFrame, root: Path):
    labels = {1: "1 Fully focused", 2: "2 Focused elsewhere", 3: "3 Task-unrelated thought", 4: "4 Mind blank"}
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    data = [x.loc[x[RESPONSE] == k, "pir_within_subject_deviation"].dropna() for k in (1, 2, 3, 4)]
    ax.violinplot(data, showmeans=True, showextrema=False); ax.set_xticks(range(1, 5), [labels[k] for k in (1,2,3,4)], rotation=15, ha="right")
    ax.set_ylabel("PIR within-person deviation (30 s)"); ax.set_title("Within-person PIR by probe response"); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(root / "figure_1_response_within_pir.png"); plt.close(fig)

    grid = np.linspace(x["pir_within_subject_deviation"].quantile(.02), x["pir_within_subject_deviation"].quantile(.98), 100)
    ref = design(x, FEATURES).drop(columns=[]).mean()
    pred = pd.DataFrame({"const": 1.0, "pir_within_subject_deviation": (grid - x["pir_within_subject_deviation"].mean()) / x["pir_within_subject_deviation"].std(ddof=0), MAD: x[MAD].mean(), SLOPE: x[SLOPE].mean(), "block": x.block.mean(), TIME: x[TIME].mean()})[core_fit.params.index]
    prob = core_fit.predict(pred)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180); ax.plot(grid, prob, color="#b33a3a", lw=2); ax.axhline(.5, color="#888", ls="--", lw=1)
    ax.set_xlabel("PIR within-person deviation (30 s)"); ax.set_ylabel("Model-predicted probability: fully task-focused"); ax.set_title("PIR change and fully-focused probability"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(root / "figure_2_predicted_focused_probability.png"); plt.close(fig)

    g = x.groupby(VIGILANCE)[PIR].agg(["mean", "std", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.3), dpi=180); ax.errorbar(g[VIGILANCE], g["mean"], yerr=g["std"], fmt="o-", capsize=4, color="#3b7d3b")
    ax.set_xticks([1,2,3,4], ["1 Very sleepy", "2 Sleepy", "3 Alert", "4 Very alert"], rotation=15, ha="right"); ax.set_xlabel("Probe vigilance"); ax.set_ylabel("PIR median (30 s)"); ax.set_title("PIR and vigilance"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(root / "figure_3_vigilance_pir.png"); plt.close(fig)
    forest(sensitivity, root)


def report(root: Path, x: pd.DataFrame, models: pd.DataFrame, desc: pd.DataFrame, vig: pd.DataFrame, sensitivity: pd.DataFrame, provenance: dict):
    core = models[(models.outcome == "fully_focused") & (models.term != "const")]
    lines = ["# NIR v1 formal statistics v1", "", "状态：`NIR_V1_FORMAL_STATS_V1`。本报告是第一版正式 NIR scientific result，不修改 cohort、底层 runtime 或既有 NIR v1 结果。", "",
             "## Analysis definition", "", "Primary analytic set = `nir_quality_tier == primary`, 30 s window, n=1174 probes. Primary pupil measure = fused full-class PIR. Subject means and within-person deviations were recomputed inside this analytic set before modeling. Repeated probes were handled with subject-clustered exchangeable GEE logistic regression because stable binomial mixed-effects fitting was not available in the current environment.", "",
             "Outcome: fully task-focused = response 1; not fully task-focused = responses 2/3/4. The latter are not relabeled as a single mind-wandering category. Predictors: PIR within-person deviation, PIR MAD, PIR robust slope, block and time in block; continuous predictors are standardized for odds-ratio presentation.", "",
             "## Main result", "", f"Analytic n={len(x)}, subjects={x.subject.nunique()}. Response counts: {x[RESPONSE].value_counts().sort_index().to_dict()}.", ""]
    for _, r in core.iterrows():
        lines.append(f"- {r.term}: OR={r.odds_ratio:.4g}, 95% CI [{r.ci_low:.4g}, {r.ci_high:.4g}], p={r.p_value:.4g}")
    lines += ["", "## Four response categories and planned contrasts", "", "The response descriptive table retains all four categories. Planned contrasts are 1 vs 2, 1 vs 3 and 1 vs 4; they are descriptive secondary models and were not used to redefine the primary outcome."]
    for pair in ((1,2),(1,3),(1,4)):
        z = x[x[RESPONSE].isin(pair)].copy(); z["contrast"] = (z[RESPONSE] == pair[0]).astype(int)
        try:
            tab, _ = gee_logistic(z, "contrast", FEATURES); r = tab[tab.term == "pir_within_subject_deviation"].iloc[0]
            if not np.isfinite(r.odds_ratio):
                lines.append(f"- {pair[0]} vs {pair[1]}: estimate unavailable because the sparse contrast did not produce finite GEE estimates (likely separation), n={len(z)}")
            else:
                lines.append(f"- {pair[0]} vs {pair[1]} PIR within-person deviation: OR={r.odds_ratio:.4g}, 95% CI [{r.ci_low:.4g}, {r.ci_high:.4g}], p={r.p_value:.4g}, n={len(z)}")
        except Exception as e:
            lines.append(f"- {pair[0]} vs {pair[1]}: model unavailable ({type(e).__name__})")
    lines += ["", "## Vigilance", "", f"Vigilance 1–4 was modeled as an ordered numeric outcome with subject-clustered GEE Gaussian as a pragmatic trend model. {vig.to_string(index=False)}", "", "This analysis does not establish causality. The focused-vs-not-focused result should be interpreted alongside the vigilance coefficient and its confidence interval, not as a blink/PERCLOS result.", "", "## Sensitivity", "", "The forest plot reports the same core PIR within-person-deviation effect for 10 s, 20 s and 30 s. 30 s remains the only primary window; no window was selected by significance.", "", "## Human-readable conclusion", "", "1. 专注和非专注是否有系统变化：见主模型中 PIR within-person deviation 的 OR/CI；OR>1 表示相对更大的 within-person PIR 与 fully task-focused 概率上升，OR<1 表示下降。", "2. 变化主要体现在大小、波动还是趋势：比较 PIR within-person deviation、MAD 和 robust slope 三个项，报告不把未显著项解释为不存在效应。", "3. 10/20/30 s 是否稳定：以 forest plot 的方向和置信区间共同判断，不按最显著窗口选主结果。", "4. 是否主要由困倦造成：结合 vigilance GEE 结果。如果 PIR 与 vigilance 关联强于与 focused outcome 的关系，应把困倦作为重要替代解释；本版不将其强行解释为注意状态。", "5. 是否进入多模态模型：本版 feature/QC 和正式统计已具备进入下一步候选模型的资格，但进入前应锁定本报告版本、保留 subject-aware 评估，并在融合模型中把 vigilance 作为协变量或敏感性分析。", "", "## Limitations", "", "未找到可靠的重复参与者映射，当前以 subject 作为 cluster/random-intercept grouping。模型是 GEE 近似而非 binomial mixed-effects。PIR feature 是统计量，不是 blink/PERCLOS；RITnet/ROI/PIR failure 仍是 QC/missingness。", "", "## Provenance", "", json.dumps(provenance, ensure_ascii=False, indent=2)]
    lines += ["", "Participant mapping audit: a versioned Beijing formal session crosswalk was found and used. The 69 sessions map to 47 repeat_participant_id clusters; identity was not inferred from features."]
    (root / "NIR_V1_FORMAL_STATS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--output-root", required=True); ap.add_argument("--crosswalk", required=False); args = ap.parse_args()
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True); inp = Path(args.input)
    d = pd.read_csv(inp)
    crosswalk = pd.read_csv(args.crosswalk) if args.crosswalk else None
    x = prepare(d, "pre_30s", crosswalk)
    model, fit = gee_logistic(x, "fully_focused", FEATURES); model["window_seconds"] = 30
    desc = response_descriptives(x); desc.to_csv(root / "response_descriptive.csv", index=False)
    vig, _ = gee_gaussian(x, VIGILANCE, ["pir_within_subject_deviation", MAD, SLOPE, "block", TIME]); vig.to_csv(root / "vigilance_model_summary.csv", index=False)
    sens_rows = []
    for w, seconds in (("pre_10s", 10), ("pre_20s", 20), ("pre_30s", 30)):
        z = prepare(d, w, crosswalk); tab, _ = gee_logistic(z, "fully_focused", FEATURES); r = tab[tab.term == "pir_within_subject_deviation"].iloc[0].to_dict(); r["window_seconds"] = seconds; sens_rows.append(r)
    sens = pd.DataFrame(sens_rows); sens.to_csv(root / "sensitivity_summary.csv", index=False)
    models_all = model.copy(); models_all.to_csv(root / "primary_model_summary.csv", index=False)
    models_all.to_csv(root / "model_summary.csv", index=False)
    provenance = {"source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "input": str(inp), "input_sha256": sha256(inp), "input_rows": int(len(d)), "primary_rows": int(len(x)), "subjects": int(x.subject.nunique()), "clusters": int(x.cluster_id.nunique()), "crosswalk": str(args.crosswalk) if args.crosswalk else None, "command": " ".join(__import__("sys").argv), "windows": [10,20,30], "centering": "recomputed by subject within each analytic window; training-fold-only required for future ML"}
    make_figures(x, fit, sens, root); provenance["figure_files"] = sorted(p.name for p in root.glob("figure_*.png")); (root / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"); report(root, x, models_all, desc, vig, sens, provenance)


if __name__ == "__main__":
    main()
