from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _parse_keypresses(value: Any) -> list[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return []
    return [float(token) for token in _NUMBER_RE.findall(text)]


def _keypress_absolute_ms(value: float | None, onset_ms: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    # FocusWave formal files store Unix-ms timestamps. Keep a compatibility path
    # for future files that may store relative milliseconds instead.
    if onset_ms is not None and np.isfinite(onset_ms) and abs(value - onset_ms) <= 60000:
        return float(value)
    if abs(value) >= 1e10:
        return float(value)
    if onset_ms is None or not np.isfinite(onset_ms):
        return None
    return float(onset_ms + value)


def _prestimulus_delta_ms(value: Any, onset_ms: Any) -> float | None:
    raw = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    onset = pd.to_numeric(pd.Series([onset_ms]), errors="coerce").iloc[0]
    if pd.isna(raw):
        return None
    raw = float(raw)
    if pd.notna(onset) and (abs(raw) >= 1e10 or abs(raw - float(onset)) <= 60000):
        return raw - float(onset)
    return raw


def add_behavior_qc(trials: pd.DataFrame, *, carryover_ms: float = 200.0) -> pd.DataFrame:
    """Add non-destructive trial-level motor-timing QC fields.

    No program scoring is overwritten. Candidate thresholds are descriptive flags
    only; final exclusion thresholds remain a cohort-level decision.
    """
    df = trials.copy().sort_values(["subject", "block_num", "trial_num"]).reset_index(drop=True)

    parsed = df["raw_keypresses"].map(_parse_keypresses)
    df["n_raw_keypresses"] = parsed.map(len).astype(int)

    first_values: list[float | None] = []
    second_values: list[float | None] = []
    for keys, onset in zip(parsed, df["absolute_onset_time"], strict=False):
        onset_num = pd.to_numeric(pd.Series([onset]), errors="coerce").iloc[0]
        onset_float = None if pd.isna(onset_num) else float(onset_num)
        first_values.append(
            _keypress_absolute_ms(keys[0], onset_float) if len(keys) >= 1 else None
        )
        second_values.append(
            _keypress_absolute_ms(keys[1], onset_float) if len(keys) >= 2 else None
        )

    df["first_raw_keypress_ms"] = first_values
    df["second_raw_keypress_ms"] = second_values
    df["rt_reconstructed_ms"] = pd.to_numeric(
        df["first_raw_keypress_ms"], errors="coerce"
    ) - pd.to_numeric(df["absolute_onset_time"], errors="coerce")
    df["rt_reconstruction_error_ms"] = df["rt_reconstructed_ms"] - pd.to_numeric(
        df["rt"], errors="coerce"
    )

    df["prestimulus_press_flag"] = df["prestimulus_press_ms"].notna()
    df["prestimulus_delta_to_onset_ms"] = [
        _prestimulus_delta_ms(value, onset)
        for value, onset in zip(
            df["prestimulus_press_ms"], df["absolute_onset_time"], strict=False
        )
    ]
    df["multiple_keypress_flag"] = df["n_raw_keypresses"].gt(1)

    rt = pd.to_numeric(df["rt"], errors="coerce")
    for threshold in (100, 150, 200):
        df[f"rt_candidate_lt_{threshold}_flag"] = rt.notna() & rt.lt(threshold)
    for threshold in (900, 1000, 1150):
        df[f"rt_candidate_gt_{threshold}_flag"] = rt.notna() & rt.gt(threshold)

    df["previous_second_raw_keypress_ms"] = df.groupby(["subject", "block_num"])[
        "second_raw_keypress_ms"
    ].shift(1)
    current_onset = pd.to_numeric(df["absolute_onset_time"], errors="coerce")
    previous_second = pd.to_numeric(df["previous_second_raw_keypress_ms"], errors="coerce")
    df["previous_second_press_to_current_onset_ms"] = current_onset - previous_second
    df["carryover_candidate_flag"] = (
        previous_second.notna()
        & df["previous_second_press_to_current_onset_ms"].ge(0)
        & df["previous_second_press_to_current_onset_ms"].le(float(carryover_ms))
    )

    is_go_omission = pd.to_numeric(df["is_no_go"], errors="coerce").eq(0) & pd.to_numeric(
        df["omission"], errors="coerce"
    ).eq(1)
    df["ambiguous_omission_flag"] = is_go_omission & (
        df["prestimulus_press_flag"] | df["carryover_candidate_flag"]
    )

    # Convenience descriptive flag only. It is intentionally not named "invalid".
    df["anticipatory_candidate_flag"] = (
        df["prestimulus_press_flag"]
        | df["rt_candidate_lt_100_flag"]
        | df["carryover_candidate_flag"]
    )
    return df
