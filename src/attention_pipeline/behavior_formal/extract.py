"""正式 BBB SART 行为提取与校验。

读取 E:/正式实验/sub-XXX_/beh/sub-XXX_Block{N}_B_beh.csv（27 列，UTF-8 BOM），
拼接为统一长表并派生 QC/预警列；校验契约（行数、No-Go 数、探针位置、内部一致性、时间戳）。

口径：
- RT 永不静默删除；<100/<150/>1000/>1150 只作 QC 标注。
- go_rt_valid = 正确 Go 试次的 rt（不按区间裁剪），QC 由独立列保留。
- 探针 Q1=注意状态 4 分类（名义，不平均），Q2=警觉度 4 点（有序）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..io import subject_paths

FORMAL_COLUMNS = [
    "subject_id", "block_num", "condition", "trial_num", "cycle_num",
    "position_in_cycle", "stimulus_name", "stimulus_size", "is_no_go",
    "response", "rt", "response_time", "correct", "commission", "omission",
    "is_probe", "probe_response", "probe_rt", "probe_vigilance",
    "probe_vigilance_rt", "probe_onset_time", "probe_response_time",
    "absolute_onset_time", "block_onset_time", "raw_keypresses",
    "prestimulus_press_ms", "rest_duration",
]

REQUIRED_COLUMNS = set(FORMAL_COLUMNS)

# trial 级预警标记参数（移植 v1 extract_beh.add_trial_level_metrics）
RT_ZSCORE_WINDOW = 8
PRE_ERROR_K_MIN = 4
PRE_ERROR_K_MAX = 8
LOCAL_TREND_K = 6
POST_ERROR_BASELINE_K = 4

NUMERIC_COLUMNS = [
    "block_num", "trial_num", "cycle_num", "position_in_cycle", "is_no_go",
    "response", "rt", "correct", "commission", "omission", "is_probe",
    "probe_response", "probe_rt", "probe_vigilance", "probe_vigilance_rt",
    "probe_onset_time", "probe_response_time", "absolute_onset_time",
    "block_onset_time", "stimulus_size",
]


def formal_block_files(config: Config, subject: str) -> list[Path]:
    """返回 sub-XXX_/beh/ 下 3 个正式 block CSV（每 block 恰 1 个）。"""
    beh_dir = subject_paths(config.path_value("raw_root"), subject)["beh_dir"]
    result = []
    for block_num, condition in enumerate(config.section("protocol")["block_order"], start=1):
        candidates = sorted(beh_dir.glob(f"{subject}_Block{block_num}_{condition}_beh.csv"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"{subject} Block{block_num}_{condition} 行为文件应为 1 个，实际 {len(candidates)}")
        result.append(candidates[0])
    return result


def _add_derived_and_markers(trials: pd.DataFrame, config: Config) -> pd.DataFrame:
    df = trials.copy()
    qc = config.section("behavior")["rt_qc_ms"]
    has_rt = df["rt"].notna()
    df["rt_qc_lt_100"] = has_rt & df["rt"].lt(qc["very_early"])
    df["rt_qc_lt_150"] = has_rt & df["rt"].lt(qc["early"])
    df["rt_qc_gt_1000"] = has_rt & df["rt"].gt(qc["long"])
    df["rt_qc_gt_1150"] = has_rt & df["rt"].gt(qc["beyond_nominal"])
    df["rt_timestamp_delta_ms"] = (
        df["response_time"] - df["absolute_onset_time"] - df["rt"]
    )
    df["rt_qc_timestamp_inconsistent"] = (
        has_rt
        & df["response_time"].notna()
        & df["absolute_onset_time"].notna()
        & df["rt_timestamp_delta_ms"].abs().gt(qc["timestamp_tolerance"])
    )
    # 正确 Go RT（冻结口径：仅正确 Go，不按区间裁剪）
    df["go_rt_valid"] = df["rt"].where(df["is_no_go"].eq(0) & df["correct"].eq(1))
    # 探针语义
    probe_labels = config.section("protocol")["probe_labels"]
    df["probe_state_label"] = df["probe_response"].map(probe_labels)
    df["probe_vigilance_label"] = df["probe_vigilance"].map(
        {1: "极度困倦", 2: "困倦", 3: "清醒", 4: "极度清醒"})
    # block 内相对时间（秒）与周期 bin
    df["time_in_block_sec"] = df["block_onset_time"] / 1000.0
    n_bins = int(config.section("behavior")["cycle_bins"])
    df["cycle_bin"] = pd.cut(
        df["cycle_num"], bins=n_bins, labels=np.arange(1, n_bins + 1), right=True).astype("Int64")
    # trial 级预警标记（逐 block 独立）
    df = _add_trial_level_markers(df)
    return df


def _add_trial_level_markers(df: pd.DataFrame) -> pd.DataFrame:
    """移植 v1 add_trial_level_metrics：rt_zscore / pre_error_slope /
    post_error_flag / post_error_slowing / local_rt_trend（逐 block 独立）。"""
    out = df.copy()
    for col in ("rt_zscore", "pre_error_slope", "post_error_slowing", "local_rt_trend"):
        out[col] = np.nan
    out["post_error_flag"] = 0
    for block_num, mask in out.groupby("block_num").groups.items():
        idxs = out.index[mask]
        go_rt_window: list[float] = []
        go_rt_long: list[float] = []
        go_rt_local: list[float] = []
        prev_was_error = False
        pre_error_baseline: list[float] = []
        for idx in idxs:
            row = out.loc[idx]
            is_go = row["is_no_go"] == 0
            is_correct_go = is_go and row["correct"] == 1
            is_commission = row["commission"] == 1
            rt_val = row.get("go_rt_valid")
            rt_val = float(rt_val) if pd.notna(rt_val) else None
            out.at[idx, "post_error_flag"] = 1 if prev_was_error else 0
            if is_correct_go and rt_val is not None:
                rt = rt_val
                if len(go_rt_window) >= 2:
                    m = float(np.mean(go_rt_window))
                    s = float(np.std(go_rt_window, ddof=1))
                    if s > 1e-6:
                        out.at[idx, "rt_zscore"] = (rt - m) / s
                if len(go_rt_local) >= LOCAL_TREND_K:
                    y = np.asarray(go_rt_local[-LOCAL_TREND_K:], dtype=float)
                    out.at[idx, "local_rt_trend"] = float(np.polyfit(np.arange(len(y)), y, 1)[0])
                if prev_was_error and len(pre_error_baseline) >= POST_ERROR_BASELINE_K:
                    out.at[idx, "post_error_slowing"] = rt - float(np.mean(pre_error_baseline[-POST_ERROR_BASELINE_K:]))
                    prev_was_error = False
                elif prev_was_error:
                    prev_was_error = False
                go_rt_window.append(rt)
                go_rt_long.append(rt)
                go_rt_local.append(rt)
                if len(go_rt_window) > RT_ZSCORE_WINDOW:
                    go_rt_window.pop(0)
                if len(go_rt_long) > PRE_ERROR_K_MAX:
                    go_rt_long.pop(0)
                if len(go_rt_local) > LOCAL_TREND_K:
                    go_rt_local.pop(0)
            elif is_commission:
                if len(go_rt_long) >= PRE_ERROR_K_MIN:
                    y = np.asarray(go_rt_long, dtype=float)
                    out.at[idx, "pre_error_slope"] = float(np.polyfit(np.arange(len(y)), y, 1)[0])
                pre_error_baseline = list(go_rt_long)
                prev_was_error = True
                go_rt_window = []
                go_rt_long = []
                go_rt_local = []
    return out


def extract_formal_trials(config: Config, subject: str) -> pd.DataFrame:
    """读取 3 个 block CSV → 统一长表（含派生与预警列）。"""
    frames = []
    for source in formal_block_files(config, subject):
        frame = pd.read_csv(source, encoding="utf-8-sig")
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{source} 缺少列: {sorted(missing)}")
        frame["source_file"] = str(source.resolve())
        frame["source_row"] = np.arange(2, len(frame) + 2)
        frames.append(frame)
    trials = pd.concat(frames, ignore_index=True)
    for column in NUMERIC_COLUMNS:
        if column in trials.columns:
            trials[column] = pd.to_numeric(trials[column], errors="coerce")
    trials["condition"] = trials["condition"].astype(str).str.strip()
    trials = trials.sort_values(["block_num", "trial_num"]).reset_index(drop=True)
    return _add_derived_and_markers(trials, config)


def validate_formal(config: Config, trials: pd.DataFrame) -> pd.DataFrame:
    """契约校验。硬失败直接 raise；软问题记入校验报告行。

    返回逐 被试×block 校验报告 DataFrame。
    """
    protocol = config.section("protocol")
    expected_trials = protocol["trials_per_block"]
    expected_nogo = protocol["nogo_per_block"]
    expected_probes = protocol["probes_per_block"]
    rows = []
    for (subject, block_num), block in trials.groupby(["subject", "block_num"], sort=True):
        issues = []
        ok = True
        if len(block) != expected_trials:
            issues.append(f"行数={len(block)}≠{expected_trials}")
            ok = False
        if block["trial_num"].nunique() != len(block) or not block["trial_num"].between(1, expected_trials).all():
            issues.append("trial_num 不唯一或越界")
            ok = False
        nogo = int(block["is_no_go"].sum())
        if nogo != expected_nogo:
            issues.append(f"No-Go={nogo}≠{expected_nogo}")
            ok = False
        probes = int(block["is_probe"].sum())
        probe_positions = set(block.loc[block["is_probe"].eq(1), "trial_num"])
        if probes != expected_probes:
            issues.append(f"探针={probes}≠{expected_probes}")
            ok = False
        if block["cycle_num"].between(1, protocol["cycles_per_block"]).all() is False:
            issues.append("cycle_num 越界")
            ok = False
        if block["position_in_cycle"].between(1, protocol["trials_per_cycle"]).all() is False:
            issues.append("position_in_cycle 越界")
            ok = False
        if block["condition"].nunique() != 1:
            issues.append("condition 非单一")
            ok = False
        # 内部一致性
        nogo_only_comm = int(block.loc[block["commission"].eq(1) & block["is_no_go"].ne(1)].shape[0])
        go_only_omiss = int(block.loc[block["omission"].eq(1) & block["is_no_go"].ne(0)].shape[0])
        if nogo_only_comm or go_only_omiss:
            issues.append("commission/omission 与 is_no_go 不一致")
            ok = False
        rt_missing_on_response = int(block.loc[block["response"].eq(1) & block["rt"].isna()].shape[0])
        if rt_missing_on_response:
            issues.append(f"response=1 但 rt 缺失 {rt_missing_on_response}")
            ok = False
        ts_inconsistent = int(block["rt_qc_timestamp_inconsistent"].sum())
        rows.append({
            "subject": subject,
            "block_num": int(block_num),
            "condition": block["condition"].iloc[0] if len(block) else "",
            "trials": int(len(block)),
            "nogo": nogo,
            "probes": probes,
            "probe_positions": ";".join(str(p) for p in sorted(probe_positions)),
            "prestimulus_press_count": int(block["prestimulus_press_ms"].fillna("").astype(str).str.len().gt(0).sum()),
            "timestamp_inconsistent": ts_inconsistent,
            "hard_fail": not ok,
            "issues": ";".join(issues) if issues else "",
        })
    report = pd.DataFrame(rows)
    if report["hard_fail"].any():
        failed = report.loc[report["hard_fail"]]
        detail = "; ".join(f"{r.subject}B{r.block_num}:{r.issues}" for r in failed.itertuples())
        raise ValueError(f"校验硬失败：{detail}")
    # 跨被试探针位置一致性（设计=同一调度表）
    pos_by_block = {}
    for block_num, grp in report.groupby("block_num"):
        pos_by_block[block_num] = set(grp["probe_positions"])
    for block_num, pos_set in pos_by_block.items():
        if len(pos_set) != 1:
            raise ValueError(f"Block{block_num} 探针位置跨被试不一致")
    return report


def load_cohort(config: Config, subjects: list[str] | None = None) -> pd.DataFrame:
    """全部被试统一长表，带规范 subject 列。"""
    includes = subjects or config.section("subjects")["include"]
    frames = []
    for subject in includes:
        frame = extract_formal_trials(config, subject)
        frame.insert(0, "subject", subject)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    return result
