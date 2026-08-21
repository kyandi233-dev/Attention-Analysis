from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from ..config import Config
from ..contracts import PROBE_LABELS
from ..io import subject_paths


RAW_REQUIRED_COLUMNS = {
    "block_num", "condition", "trial_num", "cycle_num", "position_in_cycle",
    "is_no_go", "response", "rt", "correct", "commission", "omission",
    "is_probe", "probe_response", "absolute_onset_time", "response_time",
}


def behavior_files(config: Config, subject: str) -> list[Path]:
    beh_dir = subject_paths(config.path_value("raw_root"), subject)["beh_dir"]
    result = []
    for block_num, condition in enumerate(config.section("protocol")["block_order"], start=1):
        candidates = sorted(beh_dir.glob(f"*Block{block_num}_{condition}_beh.csv"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"{subject} Block{block_num}_{condition} 行为文件应为 1 个，实际 {len(candidates)}")
        result.append(candidates[0])
    return result


def extract_trials(config: Config, subject: str) -> pd.DataFrame:
    frames = []
    for source in behavior_files(config, subject):
        frame = pd.read_csv(source)
        missing = RAW_REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{source} 缺少列: {sorted(missing)}")
        frame["source_file"] = str(source.resolve())
        frame["source_row"] = np.arange(2, len(frame) + 2)
        frames.append(frame)
    trials = pd.concat(frames, ignore_index=True)
    numeric = [
        "block_num", "trial_num", "cycle_num", "position_in_cycle", "is_no_go",
        "response", "rt", "correct", "commission", "omission", "is_probe",
        "probe_response", "absolute_onset_time", "response_time", "probe_onset_time",
    ]
    for column in numeric:
        if column in trials:
            trials[column] = pd.to_numeric(trials[column], errors="coerce")
    qc = config.section("behavior")["rt_qc_ms"]
    has_rt = trials["rt"].notna()
    trials["rt_qc_lt_100"] = has_rt & trials["rt"].lt(qc["very_early"])
    trials["rt_qc_lt_150"] = has_rt & trials["rt"].lt(qc["early"])
    trials["rt_qc_gt_1000"] = has_rt & trials["rt"].gt(qc["long"])
    trials["rt_qc_gt_1150"] = has_rt & trials["rt"].gt(qc["beyond_nominal"])
    trials["rt_timestamp_delta_ms"] = (
        trials["response_time"] - trials["absolute_onset_time"] - trials["rt"]
    )
    trials["rt_qc_timestamp_inconsistent"] = (
        has_rt
        & trials["response_time"].notna()
        & trials["absolute_onset_time"].notna()
        & trials["rt_timestamp_delta_ms"].abs().gt(qc["timestamp_tolerance"])
    )
    trials["probe_state_label"] = trials["probe_response"].map(PROBE_LABELS)
    trials["condition_x_position"] = (
        trials["condition"].astype(str) + "_p" + trials["position_in_cycle"].astype("Int64").astype(str)
    )
    # Invariant: QC only annotates. No RT row is removed here.
    return trials


def _corrected_rate(successes: int, opportunities: int) -> float:
    if opportunities <= 0:
        return float("nan")
    return (successes + 0.5) / (opportunities + 1.0)


def block_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    normal = NormalDist()
    for (subject_id, block_num, condition), block in trials.groupby(
        ["subject_id", "block_num", "condition"], dropna=False, sort=True
    ):
        go = block.loc[block["is_no_go"].eq(0)]
        nogo = block.loc[block["is_no_go"].eq(1)]
        hits = int(go["correct"].eq(1).sum())
        false_alarms = int(nogo["commission"].eq(1).sum())
        hit_rate = _corrected_rate(hits, len(go))
        false_alarm_rate = _corrected_rate(false_alarms, len(nogo))
        dprime = normal.inv_cdf(hit_rate) - normal.inv_cdf(false_alarm_rate)
        rows.append({
            "subject_id": subject_id,
            "block_num": int(block_num),
            "condition": condition,
            "go_opportunities": int(len(go)),
            "correct_go_hits": hits,
            "nogo_opportunities": int(len(nogo)),
            "nogo_commissions": false_alarms,
            "hit_rate_loglinear": hit_rate,
            "false_alarm_rate_loglinear": false_alarm_rate,
            "dprime_loglinear": dprime,
            "go_rt_count": int(go["rt"].notna().sum()),
            "go_rt_median_ms": float(go["rt"].median()) if go["rt"].notna().any() else np.nan,
            "omission_rate": float(go["omission"].mean()) if len(go) else np.nan,
            "commission_rate": float(nogo["commission"].mean()) if len(nogo) else np.nan,
        })
    return pd.DataFrame(rows)

