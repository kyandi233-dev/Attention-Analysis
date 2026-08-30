from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def canonical_probe_response(series: pd.Series) -> pd.Series:
    """Preserve probe_response as canonical raw categories without semantic relabeling."""
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="string")
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        integer_like = numeric[numeric_mask].map(lambda value: float(value).is_integer())
        idx_integer = integer_like[integer_like].index
        idx_float = integer_like[~integer_like].index
        result.loc[idx_integer] = numeric.loc[idx_integer].astype("Int64").astype("string")
        result.loc[idx_float] = numeric.loc[idx_float].map(lambda x: f"{float(x):g}").astype("string")
    text_mask = ~numeric_mask & series.notna()
    if text_mask.any():
        text = series.loc[text_mask].astype(str).str.strip()
        result.loc[text_mask] = text.mask(text.eq(""), pd.NA).astype("string")
    return result


def probe_event_table(probe_windows: pd.DataFrame, *, track: str) -> pd.DataFrame:
    """Return one row per probe event while checking response fields are window-invariant."""
    df = probe_windows[probe_windows["track"].astype(str).eq(track)].copy()
    if df.empty:
        return pd.DataFrame()
    keys = ["subject", "block_num", "probe_index_global"]
    response_fields = [
        column
        for column in (
            "probe_response",
            "probe_rt",
            "probe_vigilance",
            "probe_vigilance_rt",
            "probe_onset_ms",
            "probe_index_in_block",
        )
        if column in df.columns
    ]
    for column in response_fields:
        n_unique = df.groupby(keys, dropna=False)[column].nunique(dropna=False)
        if bool(n_unique.gt(1).any()):
            bad = n_unique[n_unique.gt(1)].head(5).index.tolist()
            raise ValueError(f"probe field {column} changes across windows/tracks for {bad}")
    keep = keys + response_fields
    events = df[keep].drop_duplicates(keys, keep="first").reset_index(drop=True)
    if "probe_response" in events.columns:
        events["probe_response_code"] = canonical_probe_response(events["probe_response"])
    else:
        events["probe_response_code"] = pd.Series(pd.NA, index=events.index, dtype="string")
    for column in ("probe_rt", "probe_vigilance", "probe_vigilance_rt"):
        if column in events.columns:
            events[column] = pd.to_numeric(events[column], errors="coerce")
    return events


def probe_response_subject_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    df = events.dropna(subset=["probe_response_code"]).copy()
    rows: list[dict[str, Any]] = []
    for (subject, block_num), block in df.groupby(["subject", "block_num"], sort=True):
        denominator = len(block)
        for response_code, frame in block.groupby("probe_response_code", sort=True):
            vigilance = pd.to_numeric(frame.get("probe_vigilance"), errors="coerce")
            probe_rt = pd.to_numeric(frame.get("probe_rt"), errors="coerce")
            rows.append(
                {
                    "subject": subject,
                    "block_num": int(block_num),
                    "probe_response_code": str(response_code),
                    "n_response": int(len(frame)),
                    "n_probe_with_response": int(denominator),
                    "response_fraction": float(len(frame) / denominator) if denominator else np.nan,
                    "probe_vigilance_median": float(vigilance.median()) if vigilance.notna().any() else np.nan,
                    "probe_vigilance_mean": float(vigilance.mean()) if vigilance.notna().any() else np.nan,
                    "probe_rt_median_ms": float(probe_rt.median()) if probe_rt.notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def probe_response_window_table(probe_windows: pd.DataFrame, *, track: str) -> pd.DataFrame:
    """Subject-level summaries by raw probe response and pre-probe window."""
    df = probe_windows[probe_windows["track"].astype(str).eq(track)].copy()
    if df.empty:
        return pd.DataFrame()
    df["probe_response_code"] = canonical_probe_response(df.get("probe_response", pd.Series(index=df.index, dtype=object)))
    numeric_cols = (
        "pupil_median",
        "pupil_valid_fraction",
        "internal_coverage_fraction",
        "go_rt_median_ms",
        "go_rt_mad_ms",
        "n_trials",
        "n_go",
        "n_nogo",
        "n_commission",
        "n_omission",
        "n_ambiguous_omission",
        "n_anticipatory_candidate",
        "probe_vigilance",
    )
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["commission_rate_window"] = np.where(
        pd.to_numeric(df.get("n_nogo"), errors="coerce").gt(0),
        pd.to_numeric(df.get("n_commission"), errors="coerce") / pd.to_numeric(df.get("n_nogo"), errors="coerce"),
        np.nan,
    )
    df["omission_rate_window"] = np.where(
        pd.to_numeric(df.get("n_go"), errors="coerce").gt(0),
        pd.to_numeric(df.get("n_omission"), errors="coerce") / pd.to_numeric(df.get("n_go"), errors="coerce"),
        np.nan,
    )
    df["ambiguous_omission_rate_window"] = np.where(
        pd.to_numeric(df.get("n_go"), errors="coerce").gt(0),
        pd.to_numeric(df.get("n_ambiguous_omission"), errors="coerce") / pd.to_numeric(df.get("n_go"), errors="coerce"),
        np.nan,
    )
    df["anticipatory_candidate_rate_window"] = np.where(
        pd.to_numeric(df.get("n_trials"), errors="coerce").gt(0),
        pd.to_numeric(df.get("n_anticipatory_candidate"), errors="coerce") / pd.to_numeric(df.get("n_trials"), errors="coerce"),
        np.nan,
    )

    group_cols = ["subject", "block_num", "window_name", "probe_response_code"]
    df = df.dropna(subset=["probe_response_code"])
    aggregation: dict[str, tuple[str, str]] = {}
    for column, reducer in (
        ("pupil_median", "median"),
        ("pupil_valid_fraction", "mean"),
        ("internal_coverage_fraction", "mean"),
        ("probe_vigilance", "median"),
        ("go_rt_median_ms", "median"),
        ("go_rt_mad_ms", "median"),
        ("commission_rate_window", "mean"),
        ("omission_rate_window", "mean"),
        ("ambiguous_omission_rate_window", "mean"),
        ("anticipatory_candidate_rate_window", "mean"),
    ):
        if column in df.columns:
            aggregation[column] = (column, reducer)
    if not aggregation:
        return df[group_cols].drop_duplicates().reset_index(drop=True)
    result = df.groupby(group_cols, as_index=False).agg(**aggregation)
    return result


def probe_response_vigilance_table(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "probe_vigilance" not in events.columns:
        return pd.DataFrame()
    df = events.dropna(subset=["probe_response_code", "probe_vigilance"]).copy()
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (response_code, vigilance), frame in df.groupby(
        ["probe_response_code", "probe_vigilance"], sort=True
    ):
        rows.append(
            {
                "probe_response_code": str(response_code),
                "probe_vigilance": float(vigilance),
                "n_probes": int(len(frame)),
                "n_subjects": int(frame["subject"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _model_table(model: Any, model_name: str) -> pd.DataFrame:
    params = pd.Series(model.params)
    bse = pd.Series(model.bse).reindex(params.index)
    pvalues = pd.Series(model.pvalues).reindex(params.index)
    return pd.DataFrame(
        {
            "model": model_name,
            "term": params.index.astype(str),
            "estimate": params.values,
            "se": bse.values,
            "p_value": pvalues.values,
        }
    )


def fit_probe_option_smoke_models(
    probe_windows: pd.DataFrame,
    *,
    track: str,
    window_name: str,
    min_subjects: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Smoke-test categorical probe-response interfaces without semantic relabeling."""
    df = probe_windows[
        probe_windows["track"].astype(str).eq(track)
        & probe_windows["window_name"].astype(str).eq(window_name)
    ].copy()
    df["probe_response_code"] = canonical_probe_response(df.get("probe_response", pd.Series(index=df.index, dtype=object)))
    for column in ("pupil_median", "probe_vigilance", "go_rt_median_ms"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    model_tables: list[pd.DataFrame] = []
    status: list[dict[str, Any]] = []

    def record(name: str, data: pd.DataFrame, formula: str) -> None:
        if data["subject"].nunique() < min_subjects:
            status.append({"model": name, "status": "skipped", "reason": "too_few_subjects"})
            return
        if data["probe_response_code"].nunique() < 2:
            status.append({"model": name, "status": "skipped", "reason": "single_probe_response_level"})
            return
        try:
            model = smf.mixedlm(formula, data=data, groups=data["subject"]).fit(
                reml=False, method="lbfgs", disp=False
            )
            model_tables.append(_model_table(model, name))
            status.append({"model": name, "status": "complete"})
        except Exception as exc:
            status.append(
                {
                    "model": name,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    pir = df.dropna(subset=["subject", "block_num", "probe_response_code", "pupil_median"])
    record(
        "lmm_probe_response_option_pir",
        pir,
        "pupil_median ~ C(probe_response_code) + C(block_num)",
    )

    if "probe_vigilance" in df.columns:
        vigilance = df.dropna(
            subset=["subject", "block_num", "probe_response_code", "probe_vigilance"]
        )
        record(
            "lmm_probe_response_option_vigilance",
            vigilance,
            "probe_vigilance ~ C(probe_response_code) + C(block_num)",
        )

    if "go_rt_median_ms" in df.columns:
        behavior = df.dropna(
            subset=["subject", "block_num", "probe_response_code", "go_rt_median_ms"]
        )
        record(
            "lmm_probe_response_option_preprobe_go_rt",
            behavior,
            "go_rt_median_ms ~ C(probe_response_code) + C(block_num)",
        )

    combined = (
        pd.concat(model_tables, ignore_index=True)
        if model_tables
        else pd.DataFrame(columns=["model", "term", "estimate", "se", "p_value"])
    )
    return combined, status
