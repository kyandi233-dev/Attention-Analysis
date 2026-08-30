# -*- coding: utf-8 -*-
"""
section_55.py
版本: v1.0 (2026-08-31)
功能: 正式报告 5.5 节「RGB 可见行为结果」的聚合与解释层统计实现。

本模块只消费既有帧级产物 (10_analysis_ready 的 4 类 parquet) 与外部权威表
(行为 probe 表、毫米波 merge-ready 表)，不重跑任何视频/人脸/姿态模型。

实现范围:
  1. 严格 pre-probe 窗口聚合: 同 block、trial_num < 锚定 trial、unix_ms < probe
     onset 且 >= onset - 30 s。锚定 trial 与 post-probe 帧绝不进入窗口。
  2. block × cycle_bin 聚合 (cycle_bin 口径与行为 science-v3 一致:
     每场每 block 内 cycle_num 唯一值切 6 个等频 bin)。
  3. 场次级覆盖聚合。
  4. 解释层统计模型 (与行为/NIR 口径一致):
     - block×cycle: Gaussian GEE, exchangeable, `metric ~ block2 * cycle_bin`,
       参与者聚类;
     - Q1 四分类: MNLogit + 参与者聚类稳健协方差, 参照类别 1;
     - Q2 有序四级: OrdinalGEE (GlobalOddsRatio), 参与者聚类;
     - RGB 特征与近期行为指标 / 毫米波指标的窗口级对照: 单预测变量
       Gaussian GEE, 参与者聚类;
     - 所有模型只输出 B / SE / 95% CI, 不输出伪造 p 值; 失败模型写
       failures 表 (not_estimable + reason), 禁止空表冒充成功。
  5. within/between 分解 (复用 science.build_within_between 口径)。

用法: 由 scripts/rgb_55_analysis.py 编排调用; 本模块函数均为纯函数,
      数据路径一律由调用方注入, 不写死盘符。
依赖: pandas, numpy, statsmodels (>=0.14), scipy
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .science import build_within_between

# ---------------------------------------------------------------------------
# 硬编码参数 (集中声明)
# ---------------------------------------------------------------------------
# 主探针窗口长度 (毫秒), 与行为 probe_primary_30s 的 30 s 口径对齐
PROBE_WINDOW_MS = 30000.0

# block×cycle 时间进程的 bin 数; 与行为 science-v3 (configs/behavior_formal_v2.yaml)
# 的 cycle_bins=6 口径一致, 每场每 block 内按 cycle_num 唯一值等频切分
CYCLE_BINS = 6

# 模型可估计性门: 最少观测行数 / 最少参与者组数 (与行为 science-v3 口径一致)
MIN_MODEL_ROWS = 12
MIN_PARTICIPANT_GROUPS = 4

# 95% 置信区间系数 (正态近似, 与行为 science-v3 口径一致)
CI_Z = 1.96

# 前后方向候选的无量纲方向一致性分数语义; 全文禁止物理位移表述。
# 帧级 pose_direction_interpretation 已带 auxiliary_qc_candidate_not_physical_displacement。
RADIAL_INTERPRETATION = (
    "dimensionless_direction_agreement_score_candidate_not_physical_displacement"
)

# RGB 探针窗口特征 (进入 Q1/Q2/行为/毫米波模型的预定义特征集)
RGB_PROBE_METRICS: tuple[str, ...] = (
    "body_motion_energy_median",
    "exposure_change_abs_median",
    "pose_lateral_right_per_sec_median",
    "pose_vertical_up_per_sec_median",
    "pose_radial_proximity_direction_score_median",
    "blink_event_rate_per_min",
)

# 近期行为指标 (权威来源: Behavior/formal_v3/probe_primary_30s.csv)
BEHAVIOR_PROBE_METRICS: tuple[str, ...] = (
    "go_correct_rt_median_ms",
    "go_correct_rt_cv",
    "omission_rate",
    "commission_rate",
    "dprime_loglinear",
)

# 毫米波窗口级对照指标 (权威来源: mmWave/mmwave_probe_merge_ready*.csv)
MMWAVE_PROBE_METRICS: tuple[str, ...] = (
    "mmwave_motion_proxy_median",
    "mmwave_hr_fused_bpm_median",
    "mmwave_breath_rate_breaths_per_min_median",
)

# 失败表 schema (与行为 science-v3 MODEL_FAILURE_COLUMNS 对齐)
FAILURE_COLUMNS = [
    "model_name", "analysis_type", "outcome", "predictor",
    "participant_group_n", "session_n", "n_rows",
    "failure_type", "failure_reason",
]

# probe 窗口聚合表的标识列 (透传自行为权威 probe 表)
PROBE_ID_COLUMNS = [
    "session_id", "participant_group_id", "block_id", "probe_event_id",
    "probe_order_in_block", "anchor_trial_num", "probe_time_ms",
    "q1_nominal_4class", "q2_ordinal_4level",
]


# ---------------------------------------------------------------------------
# 严格 pre-probe 窗口
# ---------------------------------------------------------------------------
def probe_window_mask(
    frames: pd.DataFrame,
    *,
    block_num: int,
    anchor_trial_num: float,
    probe_time_ms: float,
    window_ms: float = PROBE_WINDOW_MS,
) -> pd.Series:
    """构造严格 pre-probe 窗口掩码 (布尔 Series, 与输入行对齐)。

    三个条件缺一不可 (宁缺勿滥):
      1. 同 block: frames['block'] 数值等于探针所在 block;
      2. trial number < 锚定 trial: 锚定 trial 的帧 (含 is_probe 帧) 全部排除,
         即使其时间早于 probe onset;
      3. unix_ms ∈ [probe_time_ms - window_ms, probe_time_ms): 严格早于探针,
         无 post-probe 信息泄漏。

    参数:
        frames: 帧级表, 须含 block / trial_num / unix_ms 列
        block_num: 探针所在 block 编号 (1 或 2)
        anchor_trial_num: 锚定 trial 号 (来自行为权威表)
        probe_time_ms: 探针 onset 时间 (unix 毫秒)
        window_ms: 窗口长度 (毫秒), 默认 30000
    返回:
        与 frames 行对齐的布尔掩码
    """
    required = {"block", "trial_num", "unix_ms"}
    missing = required - set(frames.columns)
    if missing:
        raise ValueError(f"严格 pre-probe 窗口缺少列: {sorted(missing)}")
    lo = float(probe_time_ms) - float(window_ms)
    hi = float(probe_time_ms)
    return (
        pd.to_numeric(frames["block"], errors="coerce").eq(float(block_num))
        & pd.to_numeric(frames["trial_num"], errors="coerce").lt(float(anchor_trial_num))
        & pd.to_numeric(frames["unix_ms"], errors="coerce").ge(lo)
        & pd.to_numeric(frames["unix_ms"], errors="coerce").lt(hi)
    )


def _window_coverage_sec(frames: pd.DataFrame, mask: pd.Series) -> float:
    """窗口实际覆盖秒数 = 窗口内帧 unix_ms 跨度; 帧数不足 2 时为 NaN。"""
    times = pd.to_numeric(frames.loc[mask, "unix_ms"], errors="coerce").dropna()
    if len(times) < 2:
        return math.nan
    return float((times.max() - times.min()) / 1000.0)


def _median_on_mask(frames: pd.DataFrame, mask: pd.Series, column: str) -> float:
    """窗口内某数值列的中位数; 无有效值时 NaN。"""
    values = pd.to_numeric(frames.loc[mask, column], errors="coerce").dropna()
    return float(values.median()) if len(values) else math.nan


def _mean_on_mask(frames: pd.DataFrame, mask: pd.Series, column: str) -> float:
    """窗口内某数值列的均值; 无有效值时 NaN。"""
    values = pd.to_numeric(frames.loc[mask, column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else math.nan


def _ratio_on_mask(frames: pd.DataFrame, mask: pd.Series, column: str) -> float:
    """窗口内布尔列的 True 比例; 窗口无帧时 NaN。"""
    if not mask.any():
        return math.nan
    return float(frames.loc[mask, column].fillna(False).astype(bool).mean())


def aggregate_probe_window(
    *,
    motion: pd.DataFrame | None,
    pose: pd.DataFrame | None,
    blink_frames: pd.DataFrame | None,
    blink_events: pd.DataFrame | None,
    probe_row: dict[str, Any],
    window_ms: float = PROBE_WINDOW_MS,
) -> dict[str, Any]:
    """聚合单个探针的严格 pre-probe 窗口特征 (四轨分离, 逐轨降级)。

    数据可用性分级:
      - rgb_source_status='missing_source': motion 与 pose 帧级源均缺失,
        全部特征 NaN;
      - 'partial_no_blink': 帧级源可用但眨眼轨 (帧或事件) 缺失,
        眨眼特征 NaN;
      - 'ok': 四轨均可用。
    缺失不补零、不删除行; 窗口内无帧时计数特征为 0 / 比率特征为 NaN,
    事件计数以窗口帧 blink_event_id 去重映射 (事件表 join 帧表时间范围),
    保证事件同样满足 block / trial / 时间三重严格条件。

    参数:
        motion/pose/blink_frames/blink_events: 单场四轨帧级/事件级表
            (DataFrame 或 None=该轨源缺失)
        probe_row: 行为权威探针行, 须含 PROBE_ID_COLUMNS 的标识列
        window_ms: 窗口长度 (毫秒)
    返回:
        单行特征 dict (含标识列 + rgb_source_status + 各轨特征)
    """
    row: dict[str, Any] = {c: probe_row.get(c) for c in PROBE_ID_COLUMNS}
    block_id = str(row["block_id"])
    block_num = int(block_id.lstrip("B"))
    anchor = float(row["anchor_trial_num"])
    onset = float(row["probe_time_ms"])

    motion_ok = motion is not None and not motion.empty and "unix_ms" in motion.columns
    pose_ok = pose is not None and not pose.empty and "unix_ms" in pose.columns
    blink_frames_ok = blink_frames is not None and not blink_frames.empty and "unix_ms" in blink_frames.columns
    if not motion_ok or not pose_ok:
        row["rgb_source_status"] = "missing_source"
        return row
    blink_partial = not blink_frames_ok or blink_events is None
    row["rgb_source_status"] = "partial_no_blink" if blink_partial else "ok"

    # ---- 运动 / 亮度轨 (motion_qc) ----
    m_mask = probe_window_mask(motion, block_num=block_num, anchor_trial_num=anchor,
                               probe_time_ms=onset, window_ms=window_ms)
    w_m = motion.loc[m_mask]
    row["win_n_frames"] = int(len(w_m))
    row["win_coverage_sec"] = _window_coverage_sec(motion, m_mask)
    row["win_valid_frame_ratio"] = _ratio_on_mask(motion, m_mask, "motion_valid") if "motion_valid" in motion else math.nan
    # 分轨可观测比例: body motion 与 exposure 各自的可观测帧占比
    row["body_motion_observable_ratio"] = _ratio_on_mask(motion, m_mask, "body_motion_observable") if "body_motion_observable" in motion else math.nan
    row["exposure_change_observable_ratio"] = _ratio_on_mask(motion, m_mask, "exposure_change_observable") if "exposure_change_observable" in motion else math.nan
    # 运动/曝光中位数只在该轨可观测帧上计算 (与 maintenance 脚本口径一致)
    obs_mask = m_mask & motion.get("body_motion_observable", pd.Series(True, index=motion.index))
    row["body_motion_energy_median"] = _median_on_mask(motion, obs_mask, "body_motion_energy") if "body_motion_energy" in motion else math.nan
    row["body_motion_energy_mean"] = _mean_on_mask(motion, obs_mask, "body_motion_energy") if "body_motion_energy" in motion else math.nan
    exp_mask = m_mask & motion.get("exposure_change_observable", pd.Series(True, index=motion.index))
    row["exposure_change_abs_median"] = _median_on_mask(motion, exp_mask, "exposure_change_abs") if "exposure_change_abs" in motion else math.nan
    row["exposure_change_signed_median"] = _median_on_mask(motion, exp_mask, "exposure_change_signed") if "exposure_change_signed" in motion else math.nan
    # 全窗口帧的整体量 (无 observable 过滤)
    row["global_motion_energy_median"] = _median_on_mask(motion, m_mask, "global_motion_energy") if "global_motion_energy" in motion else math.nan
    row["gray_mean_median"] = _median_on_mask(motion, m_mask, "gray_mean") if "gray_mean" in motion else math.nan

    # ---- 姿态方向轨 (pose_confirmation) ----
    p_mask = probe_window_mask(pose, block_num=block_num, anchor_trial_num=anchor,
                               probe_time_ms=onset, window_ms=window_ms)
    row["pose_n_frames"] = int(p_mask.sum())
    row["pose_shoulders_observable_ratio"] = _ratio_on_mask(pose, p_mask, "pose_shoulders_observable") if "pose_shoulders_observable" in pose else math.nan
    for column, out in (
        ("pose_lateral_right_per_sec", "pose_lateral_right_per_sec_median"),
        ("pose_vertical_up_per_sec", "pose_vertical_up_per_sec_median"),
        ("pose_radial_proximity_direction_score", "pose_radial_proximity_direction_score_median"),
        ("radial_world_z_proximity_rate", "radial_world_z_proximity_rate_median"),
    ):
        row[out] = _median_on_mask(pose, p_mask, column) if column in pose else math.nan
    row["radial_interpretation"] = RADIAL_INTERPRETATION if "pose_radial_proximity_direction_score" in pose else math.nan

    # ---- 眨眼候选轨 (blink_candidate_frames / events) ----
    if blink_frames_ok:
        b_mask = probe_window_mask(blink_frames, block_num=block_num, anchor_trial_num=anchor,
                                   probe_time_ms=onset, window_ms=window_ms)
        w_b = blink_frames.loc[b_mask]
        row["blink_win_n_frames"] = int(len(w_b))
        row["left_eye_observable_ratio"] = _ratio_on_mask(blink_frames, b_mask, "left_eye_observable") if "left_eye_observable" in blink_frames else math.nan
        row["right_eye_observable_ratio"] = _ratio_on_mask(blink_frames, b_mask, "right_eye_observable") if "right_eye_observable" in blink_frames else math.nan
        if "bilateral_eye_consistent" in blink_frames and "blink_bilateral_observable" in blink_frames:
            bilateral_mask = b_mask & blink_frames["blink_bilateral_observable"].fillna(False).astype(bool)
            row["bilateral_consistent_ratio"] = _ratio_on_mask(blink_frames, bilateral_mask, "bilateral_eye_consistent")
            row["bilateral_observable_ratio"] = _ratio_on_mask(blink_frames, b_mask, "blink_bilateral_observable")
        else:
            row["bilateral_consistent_ratio"] = math.nan
            row["bilateral_observable_ratio"] = math.nan
        row["blink_candidate_frame_n"] = int(w_b["blink_closed_bilateral_candidate"].fillna(False).astype(bool).sum()) if "blink_closed_bilateral_candidate" in w_b else 0
        row["blink_frame_ratio"] = (row["blink_candidate_frame_n"] / row["blink_win_n_frames"]) if row["blink_win_n_frames"] else math.nan
        # 事件计数: 窗口帧 blink_event_id 去重 (事件表 join 帧表时间范围),
        # 保证事件严格满足窗口条件, 而不是仅按事件 start 时间粗判。
        # 事件表文件存在但 0 行 = 该场真无候选事件 (计数 0); 事件表缺失/
        # 损坏 = 事件不可知 (NaN), 二者不得混淆。
        event_ids = (
            pd.to_numeric(w_b.get("blink_event_id"), errors="coerce").dropna()
            if "blink_event_id" in w_b else pd.Series(dtype=float)
        )
        if blink_events is not None:
            events_in = blink_events[blink_events["blink_event_id"].isin(event_ids.astype(int))] if len(event_ids) else blink_events.iloc[0:0]
            row["blink_event_n"] = int(len(events_in))
            if len(events_in):
                row["blink_event_duration_median_ms"] = float(pd.to_numeric(events_in["duration_ms"], errors="coerce").median())
                row["blink_event_duration_total_ms"] = float(pd.to_numeric(events_in["duration_ms"], errors="coerce").sum())
                starts = np.sort(pd.to_numeric(events_in["start_unix_ms"], errors="coerce").dropna().to_numpy(float))
                ibi = np.diff(starts) if len(starts) >= 2 else np.array([])
                row["blink_ibi_median_ms"] = float(np.median(ibi)) if len(ibi) else math.nan
            else:
                row["blink_event_duration_median_ms"] = math.nan
                row["blink_event_duration_total_ms"] = math.nan
                row["blink_ibi_median_ms"] = math.nan
        else:
            row["blink_event_n"] = math.nan
            row["blink_event_duration_median_ms"] = math.nan
            row["blink_event_duration_total_ms"] = math.nan
            row["blink_ibi_median_ms"] = math.nan
        # 事件率 = 事件数 / 窗口覆盖秒 * 60; 覆盖秒缺失时用眨眼帧窗口跨度兜底
        cov = row["win_coverage_sec"] if np.isfinite(row["win_coverage_sec"]) else _window_coverage_sec(blink_frames, b_mask)
        row["blink_event_rate_per_min"] = (row["blink_event_n"] / cov * 60.0) if (np.isfinite(cov) and cov > 0) else math.nan
    else:
        row["blink_win_n_frames"] = math.nan
        row["left_eye_observable_ratio"] = math.nan
        row["right_eye_observable_ratio"] = math.nan
        row["bilateral_consistent_ratio"] = math.nan
        row["bilateral_observable_ratio"] = math.nan
        row["blink_candidate_frame_n"] = math.nan
        row["blink_frame_ratio"] = math.nan
        row["blink_event_n"] = math.nan
        row["blink_event_duration_median_ms"] = math.nan
        row["blink_event_duration_total_ms"] = math.nan
        row["blink_ibi_median_ms"] = math.nan
        row["blink_event_rate_per_min"] = math.nan
    return row


# ---------------------------------------------------------------------------
# block × cycle_bin 聚合
# ---------------------------------------------------------------------------
def assign_cycle_bin(cycle_nums: pd.Series, n_bins: int = CYCLE_BINS) -> pd.Series:
    """把 cycle_num 序列切为等频 cycle_bin (与行为 science-v3 extract._add_derived 口径一致)。

    每场每 block 内按 cycle_num 唯一值切 min(n_bins, 唯一值数) 个 bin,
    labels 为 1..bins (Int64, 含 NA)。

    参数:
        cycle_nums: 数值序列 (允许 NaN)
        n_bins: 目标 bin 数, 默认 6
    返回:
        与输入对齐的 Int64 cycle_bin 序列
    """
    cycles = pd.to_numeric(cycle_nums, errors="coerce")
    valid = cycles.notna()
    out = pd.Series(pd.NA, index=cycle_nums.index, dtype="Int64")
    if valid.any():
        unique_n = int(cycles[valid].nunique())
        bins = max(1, min(int(n_bins), unique_n))
        binned = pd.cut(cycles[valid], bins=bins, labels=np.arange(1, bins + 1), include_lowest=True)
        out.loc[binned.index] = binned.astype("Int64")
    return out


def _assign_cells_by_time(frames: pd.DataFrame, cells: pd.DataFrame) -> pd.Series:
    """按 motion cell 时间范围把帧分配到 cycle_bin (帧无 cycle_num 列时的对齐)。

    pose 轨帧表没有 cycle_num 列 (真实 schema 事实), 因此 block×cycle 聚合
    统一以 motion 轨定义的 (block, cycle_bin) cell 时间范围 [t_min, t_max)
    为锚, 各轨帧按 unix_ms 与 block 落入 cell。帧落在 cell 间隙时保持 NA
    (宁缺勿滥), 间隙帧数由调用方审计。

    参数:
        frames: 帧级表 (须含 unix_ms / block)
        cells: motion cell 边界表 (block_num, cycle_bin, t_min, t_max)
    返回:
        与 frames 行对齐的 Int64 cycle_bin 序列
    """
    out = pd.Series(pd.NA, index=frames.index, dtype="Int64")
    for _, cell in cells.iterrows():
        # 闭区间 [t_min, t_max]: motion cell 的末帧时间戳属于该 cell,
        # 同时间线的 pose/blink 帧落在相同时间戳时归属一致。
        mask = (
            pd.to_numeric(frames["block"], errors="coerce").eq(float(cell["block_num"]))
            & pd.to_numeric(frames["unix_ms"], errors="coerce").ge(float(cell["t_min"]))
            & pd.to_numeric(frames["unix_ms"], errors="coerce").le(float(cell["t_max"]))
        )
        out.loc[mask] = int(cell["cycle_bin"])
    return out


def build_block_cycle_table(
    *,
    motion: pd.DataFrame | None,
    pose: pd.DataFrame | None,
    blink_frames: pd.DataFrame | None,
    blink_events: pd.DataFrame | None,
    session_id: str,
    participant_group_id: str,
) -> pd.DataFrame:
    """聚合单场的 block × cycle_bin 表 (每行 = 场 × block × cycle_bin)。

    cell 划分以 motion 轨的 (block, cycle_bin) 为锚: motion 帧按自身 cycle_num
    切 6 等频 bin (与行为 science-v3 口径一致); pose / blink 帧按 unix_ms 落入
    motion cell 时间范围 (pose 轨帧表无 cycle_num 列, 不静默补列)。
    事件归属: 事件 start_unix_ms 精确映射到眨眼帧行, 取该帧所在 cell。

    参数:
        motion/pose/blink_frames/blink_events: 单场四轨表 (可 None)
        session_id: 场次编号
        participant_group_id: 参与者聚类键
    返回:
        block×cycle 聚合表 (含身份列); 对齐审计列 _pose_unaligned_frames /
        _blink_unaligned_frames 记录落在 cell 间隙的帧数
    """
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    if motion is not None and not motion.empty:
        missing = {"block", "cycle_num", "unix_ms"} - set(motion.columns)
        if missing:
            raise ValueError(f"block×cycle 聚合的 motion 轨缺少列: {sorted(missing)}")
        m = motion.dropna(subset=["block", "cycle_num"]).copy()
        m["cycle_bin"] = assign_cycle_bin(m["cycle_num"])
        for (block, cycle_bin), g in m.groupby(["block", "cycle_bin"], sort=True, dropna=False):
            base: dict[str, Any] = {
                "session_id": session_id, "participant_group_id": participant_group_id,
                "block_id": f"B{int(block)}", "cycle_bin": int(cycle_bin),
                "win_n_frames": int(len(g)),
                "win_coverage_sec": float((g["unix_ms"].max() - g["unix_ms"].min()) / 1000.0) if len(g) >= 2 else math.nan,
                "body_motion_observable_ratio": float(g["body_motion_observable"].fillna(False).astype(bool).mean()) if "body_motion_observable" in g else math.nan,
                "exposure_change_observable_ratio": float(g["exposure_change_observable"].fillna(False).astype(bool).mean()) if "exposure_change_observable" in g else math.nan,
                "gray_mean_median": float(g["gray_mean"].median()) if "gray_mean" in g else math.nan,
                "global_motion_energy_median": float(g["global_motion_energy"].dropna().median()) if "global_motion_energy" in g else math.nan,
            }
            if "body_motion_energy" in g and "body_motion_observable" in g:
                vals = g.loc[g["body_motion_observable"].fillna(False).astype(bool), "body_motion_energy"]
                base["body_motion_energy_median"] = float(vals.dropna().median()) if len(vals.dropna()) else math.nan
            else:
                base["body_motion_energy_median"] = math.nan
            if "exposure_change_abs" in g and "exposure_change_observable" in g:
                vals = g.loc[g["exposure_change_observable"].fillna(False).astype(bool), "exposure_change_abs"]
                base["exposure_change_abs_median"] = float(vals.dropna().median()) if len(vals.dropna()) else math.nan
            else:
                base["exposure_change_abs_median"] = math.nan
            rows.append(base)
            cells.append({
                "block_num": float(block), "cycle_bin": int(cycle_bin),
                "t_min": float(g["unix_ms"].min()), "t_max": float(g["unix_ms"].max()),
            })
    cell_table = pd.DataFrame(cells)

    # ---- 姿态方向: 帧按 unix_ms 落入 motion cell (pose 轨无 cycle_num 列) ----
    pose_unaligned = 0
    if pose is not None and not pose.empty and not cell_table.empty:
        missing = {"block", "unix_ms"} - set(pose.columns)
        if missing:
            raise ValueError(f"block×cycle 聚合的 pose 轨缺少列: {sorted(missing)}")
        p = pose.dropna(subset=["block", "unix_ms"]).copy()
        p["cycle_bin"] = _assign_cells_by_time(p, cell_table)
        pose_unaligned = int(p["cycle_bin"].isna().sum())
        for (block, cycle_bin), g in p.dropna(subset=["cycle_bin"]).groupby(["block", "cycle_bin"], sort=True, dropna=False):
            rec = next(
                (r for r in rows if r["block_id"] == f"B{int(block)}" and r["cycle_bin"] == int(cycle_bin)),
                None,
            )
            if rec is None:
                continue  # 姿态轨没有对应运动轨 cell 时不新增行 (宁缺勿滥)
            rec["pose_n_frames"] = int(len(g))
            rec["pose_shoulders_observable_ratio"] = float(g["pose_shoulders_observable"].fillna(False).astype(bool).mean()) if "pose_shoulders_observable" in g else math.nan
            for column, out in (
                ("pose_lateral_right_per_sec", "pose_lateral_right_per_sec_median"),
                ("pose_vertical_up_per_sec", "pose_vertical_up_per_sec_median"),
                ("pose_radial_proximity_direction_score", "pose_radial_proximity_direction_score_median"),
            ):
                rec[out] = float(g[column].dropna().median()) if column in g and g[column].dropna().size else math.nan
            rec["radial_interpretation"] = RADIAL_INTERPRETATION if "pose_radial_proximity_direction_score" in g else math.nan
    if rows:
        for rec in rows:
            rec["_pose_unaligned_frames"] = int(pose_unaligned) if pose is not None and not pose.empty else 0

    # ---- 眨眼候选: 帧按 unix_ms 落入 motion cell, 事件经窗口帧 blink_event_id 映射 ----
    blink_unaligned = 0
    if blink_frames is not None and not blink_frames.empty and not cell_table.empty:
        missing = {"block", "unix_ms"} - set(blink_frames.columns)
        if missing:
            raise ValueError(f"block×cycle 聚合的 blink 帧轨缺少列: {sorted(missing)}")
        b = blink_frames.dropna(subset=["block", "unix_ms"]).copy()
        b["cycle_bin"] = _assign_cells_by_time(b, cell_table)
        blink_unaligned = int(b["cycle_bin"].isna().sum())
        for (block, cycle_bin), g in b.dropna(subset=["cycle_bin"]).groupby(["block", "cycle_bin"], sort=True, dropna=False):
            rec = next(
                (r for r in rows if r["block_id"] == f"B{int(block)}" and r["cycle_bin"] == int(cycle_bin)),
                None,
            )
            if rec is None:
                continue
            rec["blink_win_n_frames"] = int(len(g))
            rec["left_eye_observable_ratio"] = float(g["left_eye_observable"].fillna(False).astype(bool).mean()) if "left_eye_observable" in g else math.nan
            rec["right_eye_observable_ratio"] = float(g["right_eye_observable"].fillna(False).astype(bool).mean()) if "right_eye_observable" in g else math.nan
            if "blink_bilateral_observable" in g and "bilateral_eye_consistent" in g:
                bilat = g["blink_bilateral_observable"].fillna(False).astype(bool)
                rec["bilateral_observable_ratio"] = float(bilat.mean())
                rec["bilateral_consistent_ratio"] = float(g.loc[bilat, "bilateral_eye_consistent"].fillna(False).astype(bool).mean()) if bilat.any() else math.nan
            else:
                rec["bilateral_observable_ratio"] = math.nan
                rec["bilateral_consistent_ratio"] = math.nan
            rec["blink_candidate_frame_n"] = int(g["blink_closed_bilateral_candidate"].fillna(False).astype(bool).sum()) if "blink_closed_bilateral_candidate" in g else 0
            rec["blink_frame_ratio"] = (rec["blink_candidate_frame_n"] / rec["blink_win_n_frames"]) if rec["blink_win_n_frames"] else math.nan
            if "blink_event_id" in g:
                event_ids = pd.to_numeric(g["blink_event_id"], errors="coerce").dropna()
                if blink_events is not None:
                    # 事件表存在 (可为 0 行): 计数真实; 缺失/损坏时 NaN
                    events_in = blink_events[blink_events["blink_event_id"].isin(event_ids.astype(int))] if len(event_ids) else blink_events.iloc[0:0]
                    rec["blink_event_n"] = int(len(events_in))
                    if len(events_in):
                        rec["blink_event_duration_median_ms"] = float(pd.to_numeric(events_in["duration_ms"], errors="coerce").median())
                        starts = np.sort(pd.to_numeric(events_in["start_unix_ms"], errors="coerce").dropna().to_numpy(float))
                        ibi = np.diff(starts) if len(starts) >= 2 else np.array([])
                        rec["blink_ibi_median_ms"] = float(np.median(ibi)) if len(ibi) else math.nan
                    else:
                        rec["blink_event_duration_median_ms"] = math.nan
                        rec["blink_ibi_median_ms"] = math.nan
                    cov = rec["win_coverage_sec"] if np.isfinite(rec["win_coverage_sec"]) else float((g["unix_ms"].max() - g["unix_ms"].min()) / 1000.0)
                    rec["blink_event_rate_per_min"] = (rec["blink_event_n"] / cov * 60.0) if (np.isfinite(cov) and cov > 0) else math.nan
                else:
                    rec["blink_event_n"] = math.nan
                    rec["blink_event_rate_per_min"] = math.nan
                    rec["blink_event_duration_median_ms"] = math.nan
                    rec["blink_ibi_median_ms"] = math.nan
            else:
                rec["blink_event_n"] = math.nan
                rec["blink_event_rate_per_min"] = math.nan
                rec["blink_event_duration_median_ms"] = math.nan
                rec["blink_ibi_median_ms"] = math.nan
    if rows:
        for rec in rows:
            rec["_blink_unaligned_frames"] = int(blink_unaligned) if blink_frames is not None and not blink_frames.empty else 0
    return pd.DataFrame(rows)


def build_session_coverage(
    *,
    motion: pd.DataFrame | None,
    pose: pd.DataFrame | None,
    blink_frames: pd.DataFrame | None,
    blink_events: pd.DataFrame | None,
    session_id: str,
    participant_group_id: str,
) -> dict[str, Any]:
    """聚合单场的场次级覆盖 (各轨可观测帧比例 / 覆盖秒 / 事件统计)。

    参数:
        motion/pose/blink_frames/blink_events: 单场四轨表 (可 None)
        session_id: 场次编号
        participant_group_id: 参与者聚类键
    返回:
        单行覆盖 dict
    """
    rec: dict[str, Any] = {"session_id": session_id, "participant_group_id": participant_group_id}
    if motion is not None and not motion.empty:
        rec["motion_n_frames"] = int(len(motion))
        rec["motion_coverage_sec"] = float((motion["unix_ms"].max() - motion["unix_ms"].min()) / 1000.0) if len(motion) >= 2 else math.nan
        rec["motion_valid_ratio"] = float(motion["motion_valid"].fillna(False).astype(bool).mean()) if "motion_valid" in motion else math.nan
        rec["body_motion_observable_ratio"] = float(motion["body_motion_observable"].fillna(False).astype(bool).mean()) if "body_motion_observable" in motion else math.nan
        rec["exposure_change_observable_ratio"] = float(motion["exposure_change_observable"].fillna(False).astype(bool).mean()) if "exposure_change_observable" in motion else math.nan
        rec["block1_frames"] = int(motion["block"].eq(1.0).sum())
        rec["block2_frames"] = int(motion["block"].eq(2.0).sum())
    if pose is not None and not pose.empty:
        rec["pose_n_frames"] = int(len(pose))
        rec["pose_shoulders_observable_ratio"] = float(pose["pose_shoulders_observable"].fillna(False).astype(bool).mean()) if "pose_shoulders_observable" in pose else math.nan
        rec["pose_direction_valid_ratio"] = float(pose["pose_lateral_right_per_sec"].notna().mean()) if "pose_lateral_right_per_sec" in pose else math.nan
    if blink_frames is not None and not blink_frames.empty:
        rec["blink_n_frames"] = int(len(blink_frames))
        rec["left_eye_observable_ratio"] = float(blink_frames["left_eye_observable"].fillna(False).astype(bool).mean()) if "left_eye_observable" in blink_frames else math.nan
        rec["right_eye_observable_ratio"] = float(blink_frames["right_eye_observable"].fillna(False).astype(bool).mean()) if "right_eye_observable" in blink_frames else math.nan
        if "blink_bilateral_observable" in blink_frames and "bilateral_eye_consistent" in blink_frames:
            bilat = blink_frames["blink_bilateral_observable"].fillna(False).astype(bool)
            rec["bilateral_observable_ratio"] = float(bilat.mean())
            rec["bilateral_consistent_ratio"] = float(blink_frames.loc[bilat, "bilateral_eye_consistent"].fillna(False).astype(bool).mean()) if bilat.any() else math.nan
        else:
            rec["bilateral_observable_ratio"] = math.nan
            rec["bilateral_consistent_ratio"] = math.nan
        rec["blink_closed_frame_ratio"] = float(blink_frames["blink_closed_bilateral_candidate"].fillna(False).astype(bool).mean()) if "blink_closed_bilateral_candidate" in blink_frames else math.nan
        span = float((blink_frames["unix_ms"].max() - blink_frames["unix_ms"].min()) / 1000.0) if len(blink_frames) >= 2 else math.nan
        if blink_events is not None:
            # 事件表存在 (可为 0 行 = 真无候选事件): 计数与率均为真实值;
            # 事件表缺失/损坏时事件特征为 NaN (不可知)。
            rec["blink_event_n"] = int(len(blink_events))
            rec["blink_event_rate_per_min"] = (float(len(blink_events)) / span * 60.0) if (np.isfinite(span) and span > 0) else math.nan
            if not blink_events.empty:
                durations = pd.to_numeric(blink_events["duration_ms"], errors="coerce").dropna()
                rec["blink_event_duration_median_ms"] = float(durations.median()) if len(durations) else math.nan
                starts = np.sort(pd.to_numeric(blink_events["start_unix_ms"], errors="coerce").dropna().to_numpy(float))
                ibi = np.diff(starts) if len(starts) >= 2 else np.array([])
                rec["blink_ibi_median_ms"] = float(np.median(ibi)) if len(ibi) else math.nan
            else:
                rec["blink_event_duration_median_ms"] = math.nan
                rec["blink_ibi_median_ms"] = math.nan
        else:
            rec["blink_event_n"] = math.nan
            rec["blink_event_duration_median_ms"] = math.nan
            rec["blink_ibi_median_ms"] = math.nan
            rec["blink_event_rate_per_min"] = math.nan
    return rec


# ---------------------------------------------------------------------------
# 统计模型 (与行为 science-v3 口径一致; 只输出 B / SE / 95% CI)
# ---------------------------------------------------------------------------
def _failure(
    model_name: str,
    analysis_type: str,
    outcome: str,
    predictor: str,
    data: pd.DataFrame,
    reason: str,
) -> dict[str, Any]:
    """构造 not_estimable 失败记录 (含实际分母, 不伪造结果)。"""
    return {
        "model_name": model_name,
        "analysis_type": analysis_type,
        "outcome": outcome,
        "predictor": predictor,
        "participant_group_n": int(data["participant_group_id"].nunique()) if not data.empty else 0,
        "session_n": int(data["session_id"].nunique()) if not data.empty else 0,
        "n_rows": int(len(data)),
        "failure_type": "not_estimable",
        "failure_reason": reason,
    }


def _fit_gate(result: Any) -> str | None:
    """检查拟合结果是否可用; 不可用时返回原因字符串 (行为 science-v3 同款门)。"""
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


def _standardize(series: pd.Series) -> pd.Series:
    """标准化为 (x - mean) / SD (ddof=0, 与行为 science-v3 口径一致)。"""
    values = pd.to_numeric(series, errors="coerce")
    scale = float(values.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("predictor has zero/nonfinite variance")
    return (values - float(values.mean())) / scale


def _ci(estimate: float, se: float) -> tuple[float, float]:
    """正态近似 95% CI (与行为 science-v3 口径一致)。"""
    return float(estimate - CI_Z * se), float(estimate + CI_Z * se)


def fit_block_cycle_gee(
    cycle_table: pd.DataFrame,
    metrics: Sequence[str] = RGB_PROBE_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """block × cycle 时间进程: Gaussian GEE, exchangeable, 参与者聚类。

    模型公式与行为 science-v3 完全一致: `metric ~ block2 * cycle_bin`。
    输出全部参数项 (Intercept / block2 / cycle_bin / 交互项) 的 B、SE、95% CI;
    失败 (行数/参与者组/cycle 水平不足、未收敛、非有限参数) 写 failures。

    参数:
        cycle_table: block×cycle 聚合表, 须含 participant_group_id/session_id/
                     block_id/cycle_bin 与各 metric 列
        metrics: 进入模型的指标列
    返回:
        (results, failures) 两个 DataFrame
    """
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if cycle_table.empty:
        return pd.DataFrame(), pd.DataFrame([_failure(
            "block_cycle_gee", "block_cycle_gee", "all", "all", cycle_table,
            "cycle table is empty")])
    required = {"participant_group_id", "session_id", "block_id", "cycle_bin"}
    missing = required - set(cycle_table.columns)
    if missing:
        raise ValueError(f"block×cycle GEE 缺少列: {sorted(missing)}")
    for metric in metrics:
        if metric not in cycle_table:
            continue
        d = cycle_table[["participant_group_id", "session_id", "block_id", "cycle_bin", metric]].copy()
        d[metric] = pd.to_numeric(d[metric], errors="coerce")
        d["cycle_bin"] = pd.to_numeric(d["cycle_bin"], errors="coerce")
        d["block2"] = d["block_id"].astype(str).eq("B2").astype(int)
        d = d.dropna(subset=[metric, "cycle_bin", "participant_group_id"])
        if len(d) < MIN_MODEL_ROWS or d["participant_group_id"].nunique() < MIN_PARTICIPANT_GROUPS or d["cycle_bin"].nunique() < 2:
            failures.append(_failure(f"block_cycle_gee_{metric}", "block_cycle_gee", metric, "block2*cycle_bin", d,
                                     "insufficient rows/groups/cycle levels"))
            continue
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf

            model = smf.gee(
                formula=f"{metric} ~ block2 * cycle_bin",
                groups="participant_group_id",
                data=d,
                cov_struct=sm.cov_struct.Exchangeable(),
                family=sm.families.Gaussian(),
            )
            fit = model.fit(maxiter=100)
            reason = _fit_gate(fit)
            if reason:
                failures.append(_failure(f"block_cycle_gee_{metric}", "block_cycle_gee", metric, "block2*cycle_bin", d, reason))
                continue
            params = pd.Series(fit.params)
            bse = pd.Series(fit.bse)
            for term in params.index:
                estimate, se = float(params[term]), float(bse[term])
                low, high = _ci(estimate, se)
                results.append({
                    "analysis": "block_cycle_gee", "metric": metric, "term": str(term),
                    "estimate": estimate, "se": se, "ci_low": low, "ci_high": high,
                    "participant_group_n": int(d["participant_group_id"].nunique()),
                    "session_n": int(d["session_id"].nunique()), "n_rows": int(len(d)),
                    "correlation_structure": "Exchangeable within participant_group_id",
                    "status": "estimable",
                })
        except Exception as exc:
            failures.append(_failure(f"block_cycle_gee_{metric}", "block_cycle_gee", metric, "block2*cycle_bin", d,
                                     f"{type(exc).__name__}: {exc}"))
    return pd.DataFrame(results), pd.DataFrame(failures, columns=FAILURE_COLUMNS)


def fit_q1_mnlogit(
    probe_features: pd.DataFrame,
    predictors: Sequence[str] = RGB_PROBE_METRICS,
    reference_category: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Q1 四分类与 RGB 探针窗口特征: MNLogit + 参与者聚类稳健协方差。

    预测变量建模前标准化 (ddof=0), 参照类别 1 (完全专注)。
    只输出 predictor 项的 B (每预测变量 1 个 SD)、SE、95% CI, 不输出 p 值。

    参数:
        probe_features: 探针窗口特征表, 须含 q1_nominal_4class /
                        participant_group_id / session_id 与各 predictor 列
        predictors: 预测变量列
        reference_category: 参照类别 (默认 1)
    返回:
        (results, failures)
    """
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for predictor in predictors:
        if predictor not in probe_features:
            continue
        d = probe_features[["q1_nominal_4class", predictor, "participant_group_id", "session_id"]].copy()
        d["q1_nominal_4class"] = pd.to_numeric(d["q1_nominal_4class"], errors="coerce")
        d[predictor] = pd.to_numeric(d[predictor], errors="coerce")
        d = d.dropna()
        levels = sorted(set(d["q1_nominal_4class"].astype(int)))
        if len(d) < MIN_MODEL_ROWS or d["participant_group_id"].nunique() < MIN_PARTICIPANT_GROUPS:
            failures.append(_failure(f"Q1_{predictor}", "MNLogit_cluster_robust", "q1_nominal_4class", predictor, d,
                                     "insufficient rows or participant groups"))
            continue
        if set(levels) != {1, 2, 3, 4} or reference_category not in levels:
            failures.append(_failure(f"Q1_{predictor}", "MNLogit_cluster_robust", "q1_nominal_4class", predictor, d,
                                     f"all Q1 categories 1-4 and reference {reference_category} are required"))
            continue
        try:
            import statsmodels.api as sm

            ordered = [reference_category] + [x for x in levels if x != reference_category]
            mapping = {level: idx for idx, level in enumerate(ordered)}
            y = d["q1_nominal_4class"].astype(int).map(mapping).astype(int)
            x = d[[predictor]].astype(float)
            x[predictor] = _standardize(x[predictor])
            x = sm.add_constant(x, has_constant="add")
            model = sm.MNLogit(y, x)
            fit = model.fit(method="newton", maxiter=300, disp=False, cov_type="cluster",
                            cov_kwds={"groups": d["participant_group_id"].astype(str)})
            reason = _fit_gate(fit)
            if reason:
                failures.append(_failure(f"Q1_{predictor}", "MNLogit_cluster_robust", "q1_nominal_4class", predictor, d, reason))
                continue
            params = pd.DataFrame(fit.params)
            bse = pd.DataFrame(fit.bse)
            for equation_index, category in enumerate(ordered[1:]):
                if equation_index not in params.columns:
                    continue
                estimate = float(params.loc[predictor, equation_index])
                se = float(bse.loc[predictor, equation_index])
                low, high = _ci(estimate, se)
                results.append({
                    "model_name": f"Q1_{predictor}", "model_family": "MNLogit_cluster_robust",
                    "outcome": "q1_nominal_4class", "predictor": predictor,
                    "contrast_category": int(category), "reference_category": int(reference_category),
                    "estimate_per_predictor_sd": estimate, "se": se, "ci_low": low, "ci_high": high,
                    "status": "estimable", "observation_unit": "probe",
                    "participant_group_n": int(d["participant_group_id"].nunique()),
                    "session_n": int(d["session_id"].nunique()), "n_rows": int(len(d)),
                })
        except Exception as exc:
            failures.append(_failure(f"Q1_{predictor}", "MNLogit_cluster_robust", "q1_nominal_4class", predictor, d,
                                     f"{type(exc).__name__}: {exc}"))
    return pd.DataFrame(results), pd.DataFrame(failures, columns=FAILURE_COLUMNS)


def fit_q2_ordinal_gee(
    probe_features: pd.DataFrame,
    predictors: Sequence[str] = RGB_PROBE_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Q2 有序四级与 RGB 探针窗口特征: OrdinalGEE (GlobalOddsRatio), 参与者聚类。

    预测变量建模前标准化 (ddof=0); 只输出 B / SE / 95% CI, 不输出 p 值。
    正系数表示较高 RGB 特征值与较高 Q2 警觉等级相关 (与行为 science-v3 同向)。

    参数:
        probe_features: 探针窗口特征表
        predictors: 预测变量列
    返回:
        (results, failures)
    """
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for predictor in predictors:
        if predictor not in probe_features:
            continue
        d = probe_features[["q2_ordinal_4level", predictor, "participant_group_id", "session_id"]].copy()
        d["q2_ordinal_4level"] = pd.to_numeric(d["q2_ordinal_4level"], errors="coerce")
        d[predictor] = pd.to_numeric(d[predictor], errors="coerce")
        d = d.dropna()
        if len(d) < MIN_MODEL_ROWS or d["participant_group_id"].nunique() < MIN_PARTICIPANT_GROUPS:
            failures.append(_failure(f"Q2_{predictor}", "OrdinalGEE", "q2_ordinal_4level", predictor, d,
                                     "insufficient rows or participant groups"))
            continue
        levels = set(d["q2_ordinal_4level"].astype(int).unique())
        if not levels.issubset({1, 2, 3, 4}) or d["q2_ordinal_4level"].nunique() < 3:
            failures.append(_failure(f"Q2_{predictor}", "OrdinalGEE", "q2_ordinal_4level", predictor, d,
                                     "Q2 requires at least three observed ordered levels within 1-4"))
            continue
        try:
            import statsmodels.api as sm

            x = d[[predictor]].astype(float)
            x[predictor] = _standardize(x[predictor])
            gor = sm.cov_struct.GlobalOddsRatio("ordinal")
            model = sm.OrdinalGEE(d["q2_ordinal_4level"].astype(int), x,
                                  d["participant_group_id"].astype(str), cov_struct=gor)
            fit = model.fit(maxiter=100)
            reason = _fit_gate(fit)
            if reason:
                failures.append(_failure(f"Q2_{predictor}", "OrdinalGEE", "q2_ordinal_4level", predictor, d, reason))
                continue
            estimate = float(pd.Series(fit.params)[predictor])
            se = float(pd.Series(fit.bse)[predictor])
            low, high = _ci(estimate, se)
            results.append({
                "model_name": f"Q2_{predictor}", "model_family": "OrdinalGEE",
                "outcome": "q2_ordinal_4level", "predictor": predictor,
                "estimate_per_predictor_sd": estimate, "se": se, "ci_low": low, "ci_high": high,
                "status": "estimable", "observation_unit": "probe",
                "participant_group_n": int(d["participant_group_id"].nunique()),
                "session_n": int(d["session_id"].nunique()), "n_rows": int(len(d)),
            })
        except Exception as exc:
            failures.append(_failure(f"Q2_{predictor}", "OrdinalGEE", "q2_ordinal_4level", predictor, d,
                                     f"{type(exc).__name__}: {exc}"))
    return pd.DataFrame(results), pd.DataFrame(failures, columns=FAILURE_COLUMNS)


def fit_single_predictor_gee(
    data: pd.DataFrame,
    *,
    outcomes: Sequence[str],
    predictors: Sequence[str],
    analysis: str,
    model_family: str = "GaussianGEE_cluster",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """预定义网格的单预测变量 Gaussian GEE (参与者聚类, exchangeable)。

    用于两组窗口级对照 (均为解释层, 不输出 p 值):
      - RGB 探针窗口特征 ~ 近期行为指标 (behavior)
      - RGB 探针窗口特征 ~ 毫米波运动代理 / HR / BR (mmwave)

    每个 outcome × predictor 组合拟合 `outcome ~ standardized predictor`,
    只报告 predictor 项的 B (每 1 个 SD)、SE、95% CI; 失败写 failures。

    参数:
        data: 已合并的宽表, 须含 participant_group_id / session_id /
              outcomes / predictors 各列
        outcomes: 因变量列 (RGB 特征)
        predictors: 预测变量列 (行为或毫米波指标)
        analysis: 分析标识 (如 rgb_behavior_window_gee)
        model_family: 模型家族标识
    返回:
        (results, failures)
    """
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for outcome in outcomes:
        if outcome not in data:
            continue
        for predictor in predictors:
            if predictor not in data:
                continue
            d = data[["participant_group_id", "session_id", outcome, predictor]].copy()
            d[outcome] = pd.to_numeric(d[outcome], errors="coerce")
            d[predictor] = pd.to_numeric(d[predictor], errors="coerce")
            d = d.dropna()
            model_name = f"{analysis}_{outcome}__{predictor}"
            if len(d) < MIN_MODEL_ROWS or d["participant_group_id"].nunique() < MIN_PARTICIPANT_GROUPS:
                failures.append(_failure(model_name, analysis, outcome, predictor, d,
                                         "insufficient rows or participant groups"))
                continue
            try:
                import statsmodels.api as sm

                x = d[[predictor]].astype(float)
                x[predictor] = _standardize(x[predictor])
                x = sm.add_constant(x, has_constant="add")
                model = sm.GEE(
                    d[outcome].astype(float), x,
                    groups=d["participant_group_id"].astype(str),
                    cov_struct=sm.cov_struct.Exchangeable(),
                    family=sm.families.Gaussian(),
                )
                fit = model.fit(maxiter=100)
                reason = _fit_gate(fit)
                if reason:
                    failures.append(_failure(model_name, analysis, outcome, predictor, d, reason))
                    continue
                estimate = float(pd.Series(fit.params)[predictor])
                se = float(pd.Series(fit.bse)[predictor])
                low, high = _ci(estimate, se)
                results.append({
                    "model_name": model_name, "model_family": model_family,
                    "analysis": analysis, "outcome": outcome, "predictor": predictor,
                    "estimate_per_predictor_sd": estimate, "se": se, "ci_low": low, "ci_high": high,
                    "status": "estimable", "observation_unit": "probe",
                    "correlation_structure": "Exchangeable within participant_group_id",
                    "participant_group_n": int(d["participant_group_id"].nunique()),
                    "session_n": int(d["session_id"].nunique()), "n_rows": int(len(d)),
                })
            except Exception as exc:
                failures.append(_failure(model_name, analysis, outcome, predictor, d,
                                         f"{type(exc).__name__}: {exc}"))
    return pd.DataFrame(results), pd.DataFrame(failures, columns=FAILURE_COLUMNS)


# ---------------------------------------------------------------------------
# within / between 分解 (复用 science.build_within_between 口径)
# ---------------------------------------------------------------------------
def build_probe_within_between(
    probe_features: pd.DataFrame,
    metrics: Sequence[str] = RGB_PROBE_METRICS,
) -> pd.DataFrame:
    """探针窗口特征的参与者间均值与参与者内偏差分解。

    长格式化为 science.build_within_between 所需的
    (participant_group_id, session_id, scale, metric, median) 后复用其实现,
    participant mean = between, median - participant mean = within。

    参数:
        probe_features: 探针窗口特征表
        metrics: 进入分解的特征列
    返回:
        分解表 (含 participant_mean / within_deviation / decomposition_status)
    """
    if probe_features.empty:
        return pd.DataFrame()
    long_rows: list[dict[str, Any]] = []
    for metric in metrics:
        if metric not in probe_features:
            continue
        for _, r in probe_features.iterrows():
            long_rows.append({
                "participant_group_id": r.get("participant_group_id"),
                "session_id": r.get("session_id"),
                "scale": "probe_pre30s",
                "metric": metric,
                "median": r.get(metric),
            })
    return build_within_between(pd.DataFrame(long_rows))


# ---------------------------------------------------------------------------
# 毫米波合并键规范化
# ---------------------------------------------------------------------------
def normalize_mmwave_block(block_id: Any) -> str:
    """规范化毫米波 block 标识 ('block-1'/'block-2'/'B1'/'B2'/1/2 → 'B1'/'B2')。

    参数:
        block_id: 原始 block 标识
    返回:
        'B1' / 'B2' 或原值字符串 (无法识别时)
    """
    text = str(block_id).strip().lower()
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits in {"1", "2"}:
        return f"B{digits}"
    return str(block_id)


def merge_key_mmwave(block_id: Any, probe_index: Any) -> str:
    """构造毫米波行与行为 probe 行对齐的组合键。

    合并键 = 规范化 block + 块内探针序号 (行为侧 probe_order_in_block)。
    """
    return f"{normalize_mmwave_block(block_id)}|{int(float(probe_index))}"
