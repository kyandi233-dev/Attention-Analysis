"""Behavior science v3 contract for the formal FocusWave analysis line.

This module is the formal replacement for the historical session-as-participant
statistics. It keeps Go omission and No-Go commission separate, produces the
same canonical metrics at probe/block/session/cycle scales, preserves governed
participant clustering, and fail-closes model failures. The governed formal cohort
is 116 sessions across 61 participant groups; 10 groups have exactly two sessions.
The 149-session registration universe is provenance and does not define the downstream cohort.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, theilslopes


CANONICAL_METRICS = (
    "go_correct_rt_mean_ms",
    "go_correct_rt_median_ms",
    "go_correct_rt_sd_ms",
    "go_correct_rt_mad_ms",
    "go_correct_rt_iqr_ms",
    "go_correct_rt_cv",
    "go_correct_rt_theilsen_slope_ms_per_s",
    "omission_rate",
    "commission_rate",
    "dprime_loglinear",
    "criterion_c",
    "beta",
)
# Probe models should report the explicit formal omission names rather than the
# historical ``omission_rate`` compatibility alias.  The latter is the same raw
# program omission and would otherwise duplicate ``raw_go_omission_rate``.
PROBE_MODEL_METRICS = tuple(
    [m for m in CANONICAL_METRICS if m != "omission_rate"]
    + [
        "raw_go_omission_rate",
        "clean_go_omission_rate",
        "timing_ambiguous_go_omission_rate",
    ]
)
EXPECTED_TOPOLOGY = {
    "sessions": 44,
    "analysis_groups": 38,
    "double_session_repeat_groups": 6,
}
MODEL_FAILURE_COLUMNS = (
    "model_name", "model_family", "outcome", "predictor", "status",
    "n_rows", "participant_group_n", "session_n", "reason",
)


@dataclass(frozen=True)
class BehaviorScienceConfig:
    primary_probe_window_seconds: int = 30
    sensitivity_probe_windows_seconds: tuple[int, ...] = (10, 20, 30)
    q1_reference_category: int = 1
    min_model_rows: int = 24
    min_participant_groups: int = 6
    rt_min_ms: float = 100.0
    rt_max_ms: float | None = None
    sdt_min_go: int = 4
    sdt_min_nogo: int = 2


class BehaviorContractError(ValueError):
    pass


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _mad(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return math.nan
    med = float(x.median())
    return float(np.median(np.abs(x.to_numpy(dtype=float) - med)))


def _safe_rate(n: int, d: int) -> float:
    return float(n / d) if d else math.nan


def _sdt(go_hits: int, go_n: int, commissions: int, nogo_n: int, cfg: BehaviorScienceConfig) -> dict[str, Any]:
    if go_n < cfg.sdt_min_go or nogo_n < cfg.sdt_min_nogo:
        return {"dprime_loglinear": math.nan, "criterion_c": math.nan, "beta": math.nan,
                "sdt_status": "not_estimable_low_opportunity"}
    hit = (go_hits + 0.5) / (go_n + 1.0)
    fa = (commissions + 0.5) / (nogo_n + 1.0)
    zh, zf = float(norm.ppf(hit)), float(norm.ppf(fa))
    exponent = float(np.clip((zf * zf - zh * zh) / 2.0, -700, 700))
    return {"dprime_loglinear": zh - zf, "criterion_c": -(zh + zf) / 2.0,
            "beta": float(math.exp(exponent)), "sdt_status": "estimable"}


def aggregate_behavior_metrics(frame: pd.DataFrame, cfg: BehaviorScienceConfig | None = None) -> dict[str, Any]:
    """Canonical metrics with separate Go and No-Go denominators."""
    cfg = cfg or BehaviorScienceConfig()
    d = frame.copy()
    is_nogo = _numeric(d, "is_no_go")
    correct = _numeric(d, "correct")
    omission = _numeric(d, "omission")
    commission = _numeric(d, "commission")
    rt_all = _numeric(d, "rt")
    if "time_in_block_sec" in d:
        rt_time_all = _numeric(d, "time_in_block_sec")
    elif "trial_time_s" in d:
        rt_time_all = _numeric(d, "trial_time_s")
    else:
        onset = _numeric(d, "absolute_onset_time")
        rt_time_all = (onset - onset.min()) / 1000.0

    go = is_nogo.eq(0)
    nogo = is_nogo.eq(1)
    valid_rt = go & correct.eq(1) & rt_all.notna() & rt_all.ge(cfg.rt_min_ms)
    if cfg.rt_max_ms is not None:
        valid_rt &= rt_all.le(cfg.rt_max_ms)
    rt = rt_all.loc[valid_rt].astype(float)
    rt_time = rt_time_all.loc[valid_rt].astype(float)
    n_rt = int(len(rt))
    mean = float(rt.mean()) if n_rt else math.nan
    median = float(rt.median()) if n_rt else math.nan
    sd = float(rt.std(ddof=1)) if n_rt >= 2 else math.nan
    mad = _mad(rt)
    iqr = float(rt.quantile(.75) - rt.quantile(.25)) if n_rt else math.nan
    cv = float(sd / mean) if np.isfinite(sd) and np.isfinite(mean) and mean > 0 else math.nan
    slope = math.nan
    if n_rt >= 2 and rt_time.nunique() >= 2:
        x = rt_time - float(rt_time.min())
        slope = float(theilslopes(rt.to_numpy(dtype=float), x.to_numpy(dtype=float)).slope)

    go_n, nogo_n = int(go.sum()), int(nogo.sum())
    omission_n = int((go & omission.eq(1)).sum())
    commission_n = int((nogo & commission.eq(1)).sum())
    go_hits = int((go & correct.eq(1)).sum())
    out: dict[str, Any] = {
        "trial_opportunities": int(len(d)),
        "go_opportunities": go_n,
        "nogo_opportunities": nogo_n,
        "correct_go_rt_opportunities": n_rt,
        "go_correct_rt_mean_ms": mean,
        "go_correct_rt_median_ms": median,
        "go_correct_rt_sd_ms": sd,
        "go_correct_rt_mad_ms": mad,
        "go_correct_rt_iqr_ms": iqr,
        "go_correct_rt_cv": cv,
        "go_correct_rt_theilsen_slope_ms_per_s": slope,
        "omission_numerator": omission_n,
        "omission_denominator": go_n,
        "omission_rate": _safe_rate(omission_n, go_n),
        "commission_numerator": commission_n,
        "commission_denominator": nogo_n,
        "commission_rate": _safe_rate(commission_n, nogo_n),
        "metric_units": "RT=ms; RT slope=ms/s; error rates=proportion; SDT=dimensionless",
    }
    out.update(_sdt(go_hits, go_n, commission_n, nogo_n, cfg))
    return out


def _canonical_ids(trials: pd.DataFrame) -> pd.DataFrame:
    d = trials.copy()
    if "session_id" not in d:
        if "subject" not in d:
            raise BehaviorContractError("trial rows require session_id or subject")
        d["session_id"] = d["subject"].astype(str)
    if "repeat_participant_id" not in d:
        raise BehaviorContractError("formal behavior rows require repeat_participant_id")
    if d["repeat_participant_id"].isna().any():
        raise BehaviorContractError("repeat_participant_id contains missing values")
    if "block_id" not in d:
        if "block_num" not in d:
            raise BehaviorContractError("trial rows require block_id or block_num")
        d["block_id"] = "B" + pd.to_numeric(d["block_num"], errors="coerce").astype("Int64").astype(str)
    return d


def _aggregate_groups(d: pd.DataFrame, group_cols: Sequence[str], unit: str,
                      cfg: BehaviorScienceConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, g in d.groupby(list(group_cols), sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(group_cols, key))
        record["observation_unit"] = unit
        record.update(aggregate_behavior_metrics(g, cfg))
        rows.append(record)
    return pd.DataFrame(rows)


def build_multiscale_tables(trials: pd.DataFrame, cfg: BehaviorScienceConfig | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or BehaviorScienceConfig()
    d = _canonical_ids(trials)
    base = ["repeat_participant_id", "session_id"]
    tables = {
        "session": _aggregate_groups(d, base, "session", cfg),
        "block": _aggregate_groups(d, [*base, "block_id"], "block", cfg),
    }
    if "cycle_bin" in d and _numeric(d, "cycle_bin").notna().any():
        tables["cycle"] = _aggregate_groups(
            d.dropna(subset=["cycle_bin"]), [*base, "block_id", "cycle_bin"], "cycle_bin", cfg
        )
    else:
        tables["cycle"] = pd.DataFrame()
    return tables


def build_probe_windows(trials: pd.DataFrame, cfg: BehaviorScienceConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build strictly pre-probe windows; the anchoring trial is always excluded."""
    cfg = cfg or BehaviorScienceConfig()
    d = _canonical_ids(trials)
    if "is_probe" not in d:
        raise BehaviorContractError("probe windows require is_probe")
    if "trial_num" not in d or "absolute_onset_time" not in d:
        raise BehaviorContractError("probe windows require trial_num and absolute_onset_time")
    d["absolute_onset_time"] = _numeric(d, "absolute_onset_time")
    rows: list[dict[str, Any]] = []
    windows = tuple(sorted(set(cfg.sensitivity_probe_windows_seconds)))
    for (participant, session, block), b in d.groupby(
        ["repeat_participant_id", "session_id", "block_id"], sort=True, dropna=False
    ):
        b = b.sort_values(["trial_num", "absolute_onset_time"], kind="stable").copy()
        probes = b[b["is_probe"].fillna(0).astype(float).eq(1)]
        for probe_order, (_, p) in enumerate(probes.iterrows(), start=1):
            anchor_trial = float(p["trial_num"])
            anchor_time = float(p["absolute_onset_time"])
            probe_time_raw = pd.to_numeric(pd.Series([p.get("probe_onset_time")]), errors="coerce").iloc[0]
            probe_time = float(probe_time_raw) if np.isfinite(probe_time_raw) else anchor_time
            # Both constraints are intentional: trial_num < anchor excludes the anchoring
            # trial even when probe_onset_time occurs after that trial response.
            prior = b[(pd.to_numeric(b["trial_num"], errors="coerce") < anchor_trial)
                      & (_numeric(b, "absolute_onset_time") < probe_time)].copy()
            event_id = f"{session}|{block}|probe|{probe_order}"
            for seconds in windows:
                lower = probe_time - float(seconds) * 1000.0
                w = prior[_numeric(prior, "absolute_onset_time").ge(lower)].copy()
                record: dict[str, Any] = {
                    "repeat_participant_id": str(participant),
                    "session_id": str(session),
                    "block_id": str(block),
                    "probe_event_id": event_id,
                    "probe_order_in_block": probe_order,
                    "anchor_trial_num": int(anchor_trial),
                    "probe_time_ms": probe_time,
                    "window_seconds_nominal": int(seconds),
                    "anchor_trial_excluded": True,
                    "window_crosses_block": False,
                    "q1_nominal_4class": p.get("probe_response"),
                    "q2_ordinal_4level": p.get("probe_vigilance"),
                }
                record.update(aggregate_behavior_metrics(w, cfg))
                rows.append(record)
    sensitivity = pd.DataFrame(rows)
    if sensitivity.empty:
        return sensitivity.copy(), sensitivity
    key = ["probe_event_id", "window_seconds_nominal"]
    if sensitivity.duplicated(key).any():
        raise BehaviorContractError("duplicate probe/window key")
    sensitivity["analysis_role"] = "window_sensitivity_only"
    sensitivity["formal_independent_sample"] = False
    primary = sensitivity[
        sensitivity["window_seconds_nominal"].eq(cfg.primary_probe_window_seconds)
    ].copy()
    if primary["probe_event_id"].duplicated().any():
        raise BehaviorContractError("primary probe table must have one row per probe")
    primary["analysis_role"] = "primary_probe"
    primary["formal_independent_sample"] = True
    return primary.reset_index(drop=True), sensitivity.reset_index(drop=True)


def build_b1_b2_pairs(block_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    required = {"repeat_participant_id", "session_id", "block_id"}
    if not required.issubset(block_metrics.columns):
        raise BehaviorContractError(f"block metrics missing {sorted(required - set(block_metrics.columns))}")
    for session, s in block_metrics.groupby("session_id", sort=True):
        groups = s["repeat_participant_id"].dropna().astype(str).unique()
        if len(groups) != 1:
            failures.append({"analysis": "B1_B2", "session_id": session, "status": "not_estimable",
                             "reason": "session maps to multiple participant groups"})
            continue
        b1, b2 = s[s["block_id"].astype(str).eq("B1")], s[s["block_id"].astype(str).eq("B2")]
        if len(b1) != 1 or len(b2) != 1:
            failures.append({"analysis": "B1_B2", "session_id": session, "status": "not_estimable",
                             "reason": f"expected one B1/B2 row; got {len(b1)}/{len(b2)}"})
            continue
        for metric in CANONICAL_METRICS:
            if metric not in s:
                continue
            a = pd.to_numeric(pd.Series([b1.iloc[0][metric]]), errors="coerce").iloc[0]
            b = pd.to_numeric(pd.Series([b2.iloc[0][metric]]), errors="coerce").iloc[0]
            rows.append({"repeat_participant_id": groups[0], "session_id": session,
                         "metric": metric, "b1_value": a, "b2_value": b,
                         "b2_minus_b1": b - a if np.isfinite(a) and np.isfinite(b) else math.nan,
                         "pairing_unit": "within_session"})
    return pd.DataFrame(rows), pd.DataFrame(failures)


def validate_topology(session_metrics: pd.DataFrame,
                      expected: dict[str, int] | None = None) -> dict[str, int]:
    expected = expected or EXPECTED_TOPOLOGY
    sessions = int(session_metrics["session_id"].astype(str).nunique())
    groups = session_metrics[["repeat_participant_id", "session_id"]].drop_duplicates()
    group_n = int(groups["repeat_participant_id"].astype(str).nunique())
    sizes = groups.groupby("repeat_participant_id")["session_id"].nunique()
    repeats = int(sizes.eq(2).sum())
    observed = {"sessions": sessions, "analysis_groups": group_n,
                "double_session_repeat_groups": repeats}
    if expected and observed != expected:
        raise BehaviorContractError(f"cohort topology mismatch: observed={observed}, expected={expected}")
    return observed


def participant_disjoint_folds(frame: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Deterministic participant-exclusive folds without an sklearn dependency."""
    if "repeat_participant_id" not in frame:
        raise BehaviorContractError("repeat_participant_id is required for prediction folds")
    groups = frame.groupby("repeat_participant_id", as_index=False).size().rename(columns={"size": "n"})
    if len(groups) < 2:
        raise BehaviorContractError("at least two participant groups are required")
    k = min(max(2, int(n_splits)), len(groups))
    load = np.zeros(k, dtype=int)
    assignments: list[dict[str, Any]] = []
    for row in groups.sort_values(["n", "repeat_participant_id"], ascending=[False, True]).itertuples(index=False):
        fold = int(np.argmin(load))
        load[fold] += int(row.n)
        assignments.append({"repeat_participant_id": str(row.repeat_participant_id), "fold_id": fold})
    out = frame.merge(pd.DataFrame(assignments), on="repeat_participant_id", how="left", validate="many_to_one")
    if out.groupby("repeat_participant_id")["fold_id"].nunique().gt(1).any():
        raise AssertionError("participant leakage across folds")
    return out


def _failure(name: str, family: str, outcome: str, predictor: str, d: pd.DataFrame,
             reason: str) -> dict[str, Any]:
    return {"model_name": name, "model_family": family, "outcome": outcome,
            "predictor": predictor, "status": "not_estimable", "n_rows": int(len(d)),
            "participant_group_n": int(d.get("repeat_participant_id", pd.Series(dtype=str)).nunique()),
            "session_n": int(d.get("session_id", pd.Series(dtype=str)).nunique()), "reason": reason}


def _fit_gate(result: Any) -> str | None:
    if result is None:
        return "empty result"
    if getattr(result, "converged", True) is False:
        return "model did not converge"
    try:
        params = np.asarray(result.params, dtype=float)
        bse = np.asarray(result.bse, dtype=float)
    except Exception as exc:
        return f"invalid result arrays: {type(exc).__name__}: {exc}"
    if params.size == 0 or not np.isfinite(params).all() or not np.isfinite(bse).all():
        return "non-finite or empty parameter/SE table"
    return None


def fit_q1_nominal(primary_probe: pd.DataFrame, cfg: BehaviorScienceConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Q1 is a four-class nominal outcome; participant clustering is mandatory."""
    cfg = cfg or BehaviorScienceConfig()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for predictor in PROBE_MODEL_METRICS:
        if predictor not in primary_probe:
            continue
        d = primary_probe[["q1_nominal_4class", predictor, "repeat_participant_id", "session_id"]].copy()
        d["q1_nominal_4class"] = pd.to_numeric(d["q1_nominal_4class"], errors="coerce")
        d[predictor] = pd.to_numeric(d[predictor], errors="coerce")
        d = d.dropna()
        levels = sorted(set(d["q1_nominal_4class"].astype(int)))
        if len(d) < cfg.min_model_rows or d["repeat_participant_id"].nunique() < cfg.min_participant_groups:
            failures.append(_failure(f"Q1_{predictor}", "MNLogit_cluster_robust", "Q1", predictor, d,
                                     "insufficient rows or participant groups")); continue
        if set(levels) != {1, 2, 3, 4} or cfg.q1_reference_category not in levels:
            failures.append(_failure(f"Q1_{predictor}", "MNLogit_cluster_robust", "Q1", predictor, d,
                                     f"all Q1 categories 1-4 and reference {cfg.q1_reference_category} are required")); continue
        try:
            import statsmodels.api as sm
            ordered = [cfg.q1_reference_category] + [x for x in levels if x != cfg.q1_reference_category]
            mapping = {level: idx for idx, level in enumerate(ordered)}
            y = d["q1_nominal_4class"].astype(int).map(mapping).astype(int)
            x = d[[predictor]].astype(float)
            scale = float(x[predictor].std(ddof=0))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("predictor has zero/nonfinite variance")
            x[predictor] = (x[predictor] - float(x[predictor].mean())) / scale
            x = sm.add_constant(x, has_constant="add")
            model = sm.MNLogit(y, x)
            fit = model.fit(method="newton", maxiter=300, disp=False, cov_type="cluster",
                            cov_kwds={"groups": d["repeat_participant_id"].astype(str)})
            reason = _fit_gate(fit)
            if reason:
                failures.append(_failure(f"Q1_{predictor}", "MNLogit_cluster_robust", "Q1", predictor, d, reason)); continue
            params = pd.DataFrame(fit.params)
            bse = pd.DataFrame(fit.bse)
            for equation_index, category in enumerate(ordered[1:]):
                if equation_index not in params.columns:
                    continue
                estimate = float(params.loc[predictor, equation_index])
                se = float(bse.loc[predictor, equation_index])
                results.append({"model_name": f"Q1_{predictor}", "model_family": "MNLogit_cluster_robust",
                                "outcome": "q1_nominal_4class", "predictor": predictor,
                                "contrast_category": int(category), "reference_category": cfg.q1_reference_category,
                                "estimate_per_predictor_sd": estimate, "se": se,
                                "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se,
                                "status": "estimable", "observation_unit": "probe",
                                "participant_group_n": int(d["repeat_participant_id"].nunique()),
                                "session_n": int(d["session_id"].nunique()), "n_rows": int(len(d))})
        except Exception as exc:
            failures.append(_failure(f"Q1_{predictor}", "MNLogit_cluster_robust", "Q1", predictor, d,
                                     f"{type(exc).__name__}: {exc}"))
    return pd.DataFrame(results), pd.DataFrame(failures, columns=MODEL_FAILURE_COLUMNS)


def fit_q2_ordinal(primary_probe: pd.DataFrame, cfg: BehaviorScienceConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Q2 ordinal repeated-measure model using participant-clustered OrdinalGEE."""
    cfg = cfg or BehaviorScienceConfig()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for predictor in PROBE_MODEL_METRICS:
        if predictor not in primary_probe:
            continue
        d = primary_probe[["q2_ordinal_4level", predictor, "repeat_participant_id", "session_id"]].copy()
        d["q2_ordinal_4level"] = pd.to_numeric(d["q2_ordinal_4level"], errors="coerce")
        d[predictor] = pd.to_numeric(d[predictor], errors="coerce")
        d = d.dropna()
        if len(d) < cfg.min_model_rows or d["repeat_participant_id"].nunique() < cfg.min_participant_groups:
            failures.append(_failure(f"Q2_{predictor}", "OrdinalGEE", "Q2", predictor, d,
                                     "insufficient rows or participant groups")); continue
        if not set(d["q2_ordinal_4level"].astype(int).unique()).issubset({1, 2, 3, 4}) or d["q2_ordinal_4level"].nunique() < 3:
            failures.append(_failure(f"Q2_{predictor}", "OrdinalGEE", "Q2", predictor, d,
                                     "Q2 requires at least three observed ordered levels within 1-4")); continue
        try:
            import statsmodels.api as sm
            x = d[[predictor]].astype(float)
            scale = float(x[predictor].std(ddof=0))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("predictor has zero/nonfinite variance")
            x[predictor] = (x[predictor] - float(x[predictor].mean())) / scale
            gor = sm.cov_struct.GlobalOddsRatio("ordinal")
            model = sm.OrdinalGEE(d["q2_ordinal_4level"].astype(int), x,
                                  d["repeat_participant_id"].astype(str), cov_struct=gor)
            fit = model.fit(maxiter=100)
            reason = _fit_gate(fit)
            if reason:
                failures.append(_failure(f"Q2_{predictor}", "OrdinalGEE", "Q2", predictor, d, reason)); continue
            estimate = float(pd.Series(fit.params)[predictor])
            se = float(pd.Series(fit.bse)[predictor])
            results.append({"model_name": f"Q2_{predictor}", "model_family": "OrdinalGEE",
                            "outcome": "q2_ordinal_4level", "predictor": predictor,
                            "estimate_per_predictor_sd": estimate, "se": se,
                            "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se,
                            "status": "estimable", "observation_unit": "probe",
                            "participant_group_n": int(d["repeat_participant_id"].nunique()),
                            "session_n": int(d["session_id"].nunique()), "n_rows": int(len(d))})
        except Exception as exc:
            failures.append(_failure(f"Q2_{predictor}", "OrdinalGEE", "Q2", predictor, d,
                                     f"{type(exc).__name__}: {exc}"))
    return pd.DataFrame(results), pd.DataFrame(failures, columns=MODEL_FAILURE_COLUMNS)


def qc_denominators(trials: pd.DataFrame, primary_probe: pd.DataFrame,
                    tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    d = _canonical_ids(trials)
    return pd.DataFrame([
        {"layer": "session", "count": d["session_id"].nunique(), "unit_zh": "场次"},
        {"layer": "participant_group", "count": d["repeat_participant_id"].nunique(), "unit_zh": "匿名参与者分析组"},
        {"layer": "block", "count": len(tables["block"]), "unit_zh": "区块"},
        {"layer": "cycle", "count": len(tables.get("cycle", pd.DataFrame())), "unit_zh": "区块内时间段"},
        {"layer": "probe", "count": len(primary_probe), "unit_zh": "主探针事件"},
        {"layer": "trial", "count": len(d), "unit_zh": "试次机会"},
    ])


def write_chinese_result_summary(output: Path, topology: dict[str, int],
                                 q1_results: pd.DataFrame, q2_results: pd.DataFrame,
                                 failures: pd.DataFrame) -> None:
    lines = ["# FocusWave 行为正式分析结果说明", "",
             "本文件由行为科学 v3 正式合同生成。所有样本量均标明观察单位，重复参加场次不会被当作独立参与者。", "",
             f"当前分析队列：{topology['sessions']} 场，{topology['analysis_groups']} 个匿名参与者分析组，其中 {topology['double_session_repeat_groups']} 组包含双场重复参加；该队列不等于研究总体样本。", "",
             "## 指标口径", "",
             "Go 遗漏与 No-Go 误按使用不同机会数作为分母；RT 仅汇总正确 Go 反应。RT 输出均值、中位数、SD、MAD、IQR、CV 与 Theil–Sen 时间斜率；同时输出 d′、c 与 β。", "",
             "原始 Go omission、clean Go omission 与 timing-ambiguous Go omission 均为预先定义结局，并共享 Go 分母；clean + timing-ambiguous 必须等于 raw。clean 仅表示未检出当前定义的运动时序歧义，不等同于已证明的注意失败。", "",
             "探针主分析每个 probe 只占一行，默认 30 秒窗；10/20 秒仅用于窗口敏感性，不增加主分析样本量。探针锚定试次被严格排除。", "",
             "## Q1 / Q2", "",
             f"Q1 名义四分类模型可估计结果行数：{len(q1_results)}；Q2 有序重复测量 GEE 可估计结果行数：{len(q2_results)}。", "",
             "任何样本不足、类别缺失、未收敛或非有限参数均写入 model_failures.csv，状态为 not_estimable，不会以空表冒充成功。", "",
             f"本次失败/不可估计记录：{len(failures)}。", ""]
    output.write_text("\n".join(lines), encoding="utf-8")
