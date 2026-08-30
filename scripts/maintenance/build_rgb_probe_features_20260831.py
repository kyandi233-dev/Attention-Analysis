# -*- coding: utf-8 -*-
"""
build_rgb_probe_features_20260831.py
版本: v1.0 (2026-08-31)
功能: 生成 RGB 模态的探针级 pre_30s 特征表, 供多模态融合 (论文 5.6 节) 使用。
      每行 = 一个探针 x 探针 onset 前 30 秒窗口, 与行为/NIR/毫米波的 probe 表同粒度。
      特征覆盖: 整体运动能量、姿态方向 (左右/上下/前后)、眨眼候选事件、画面亮度。
用法: D:/CondaEnvs/attention-nir-formal/python.exe build_rgb_probe_features_20260831.py
依赖: pandas, numpy, pyarrow
输入:
  - 探针时间戳 (权威): D:/Project/厚粲杯/11_数据/_FormalAnalysis/Behavior/formal_v3/probe_primary_30s.csv
  - RGB 帧级派生 (116 场 cohort 内, 每场 4 个 parquet):
      D:/Project/厚粲杯/11_数据/_FormalAnalysis/RGB/10_analysis_ready/sub-XXX/sub-XXX_{motion_qc,pose_confirmation,blink_candidate_frames,blink_candidate_events}.parquet
输出:
  - D:/Project/厚粲杯/11_数据/_FormalAnalysis/RGB/11_analysis_tables_116cohort/rgb_probe_pre30s_features.csv
纪律:
  - 窗口 = [probe_time_ms - 30000, probe_time_ms), 严格早于探针, 无未来信息泄漏
  - 缺失不补零不删除行; RGB 缺失场 (sub-099) 行保留并标 missing
  - 身份用 participant_group_id (P 编码)
"""

import os
import glob
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 硬编码参数 (集中声明)
# ---------------------------------------------------------------------------
WINDOW_MS = 30000.0          # pre_30s 窗口长度 (毫秒), 默认 30000, 与行为表 window_seconds_nominal=30 对齐
PROBE_CSV = "D:/Project/厚粲杯/11_数据/_FormalAnalysis/Behavior/formal_v3/probe_primary_30s.csv"
RGB_DIR = "D:/Project/厚粲杯/11_数据/_FormalAnalysis/RGB/10_analysis_ready"
OUT_CSV = "D:/Project/厚粲杯/11_数据/_FormalAnalysis/RGB/11_analysis_tables_116cohort/rgb_probe_pre30s_features.csv"

# 从探针表透传的标识列 (用于与行为/NIR/毫米波 probe 表对齐)
PROBE_ID_COLS = [
    "session_id", "participant_group_id", "block_id", "probe_event_id",
    "probe_order_in_block", "probe_time_ms", "q1_nominal_4class", "q2_ordinal_4level",
]


def load_probe_table(csv_path: str) -> pd.DataFrame:
    """
    读取探针时间戳表 (权威来源), 只保留标识列, 输出每行一个探针。

    参数:
        csv_path: probe_primary_30s.csv 路径
    返回:
        含 PROBE_ID_COLS 的 DataFrame, probe_time_ms 转为 float64 (unix 毫秒)
    """
    df = pd.read_csv(csv_path, usecols=PROBE_ID_COLS)
    df["probe_time_ms"] = df["probe_time_ms"].astype("float64")
    return df


def aggregate_window(frames: pd.DataFrame, onset_ms: float) -> dict:
    """
    对单个探针的 pre_30s 窗口做聚合, 返回该探针的特征字典。
    窗口 = [onset_ms - WINDOW_MS, onset_ms), 严格早于探针 (无未来信息泄漏)。

    参数:
        frames: 帧级表 (必须含 unix_ms 列)
        onset_ms: 探针 onset 时间 (unix 毫秒)
    返回:
        特征 dict: n_frames / coverage_sec / valid_frame_ratio / 各数值列中位数均值
    """
    lo = onset_ms - WINDOW_MS
    # 严格早于 onset: 使用 unix_ms < onset_ms 而非 <=
    w = frames[(frames["unix_ms"] >= lo) & (frames["unix_ms"] < onset_ms)]
    out = {"n_frames": len(w), "coverage_sec": np.nan, "valid_frame_ratio": np.nan}
    if len(w) == 0:
        return out
    # 窗口实际覆盖时长 = 首末帧时间跨度 (秒), 用于事件率换算
    out["coverage_sec"] = (w["unix_ms"].max() - w["unix_ms"].min()) / 1000.0
    if "motion_valid" in w.columns:
        out["valid_frame_ratio"] = w["motion_valid"].mean()
    return out


def median_mean_on_observable(frames: pd.DataFrame, onset_ms: float,
                              value_col: str, observable_col: str) -> tuple:
    """
    在窗口内 observable=True 的帧上计算某数值列的中位数与均值。

    参数:
        frames: 帧级表
        onset_ms: 探针 onset
        value_col: 数值特征列名
        observable_col: 可观测标志列名 (仅该列为 True 的帧参与统计)
    返回:
        (median, mean) 元组, 窗口内无可观测帧时为 (np.nan, np.nan)
    """
    lo = onset_ms - WINDOW_MS
    w = frames[(frames["unix_ms"] >= lo) & (frames["unix_ms"] < onset_ms)]
    vals = w.loc[w[observable_col], value_col]
    if len(vals) == 0:
        return np.nan, np.nan
    return vals.median(), vals.mean()


def median_on_frames(frames: pd.DataFrame, onset_ms: float, value_col: str) -> float:
    """
    在窗口内全部帧上计算某数值列的中位数 (无 observable 过滤)。

    参数:
        frames: 帧级表
        onset_ms: 探针 onset
        value_col: 数值特征列名
    返回:
        中位数; 窗口内无帧或无有效值时返回 np.nan
    """
    lo = onset_ms - WINDOW_MS
    w = frames[(frames["unix_ms"] >= lo) & (frames["unix_ms"] < onset_ms)]
    vals = w[value_col].dropna()
    if len(vals) == 0:
        return np.nan
    return vals.median()


def count_blink_events(events: pd.DataFrame, onset_ms: float) -> int:
    """
    统计窗口内眨眼候选事件数量。
    计数标准: 事件 start_unix_ms 落在 [onset_ms - WINDOW_MS, onset_ms) 内
    (眨眼在探针前已开始即计入; end 允许越过 onset, 事件本身无泄漏)。

    参数:
        events: 眨眼候选事件表 (含 start_unix_ms 列)
        onset_ms: 探针 onset
    返回:
        窗口内候选事件数量
    """
    lo = onset_ms - WINDOW_MS
    return int(((events["start_unix_ms"] >= lo) & (events["start_unix_ms"] < onset_ms)).sum())


def read_parquet_safe(path: str, columns: list):
    """
    安全读取 parquet, 文件缺失或损坏时返回 None (不抛异常)。
    用于逐轨降级: 某一轨数据不可用时该轨特征标 NaN, 不影响其他轨。

    参数:
        path: parquet 路径
        columns: 需要的列
    返回:
        DataFrame 或 None (文件缺失/损坏)
    """
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:
        print(f"  WARN: 无法读取 {os.path.basename(path)}: {type(exc).__name__}")
        return None


def build_session_features(session_id: str, probe_sub: pd.DataFrame) -> pd.DataFrame:
    """
    对单场 session 的全部探针计算 RGB pre_30s 特征。

    数据可用性分级 (缺失不补零不删除行, 逐轨降级):
      - missing_source: motion_qc 或 pose_confirmation 帧级文件不可用, 全部特征 NaN
      - partial_no_blink: 帧级文件可用, 眨眼轨 (事件表或眨眼帧) 不可用, 眨眼特征 NaN
      - ok: 全部可用

    参数:
        session_id: 场次编号 (如 sub-031)
        probe_sub: 该场的探针行 (含 probe_time_ms)
    返回:
        与 probe_sub 同行的特征 DataFrame (行序一致)
    """
    base = os.path.join(RGB_DIR, session_id)
    prefix = os.path.join(base, session_id)
    rows = []

    # 只读聚合所需列, 控制内存; 逐轨安全读取
    mq = read_parquet_safe(f"{prefix}_motion_qc.parquet", [
        "unix_ms", "motion_valid", "body_motion_observable", "exposure_change_observable",
        "body_motion_energy", "global_motion_energy", "gray_mean",
        "exposure_change_abs", "exposure_change_signed"])
    pose = read_parquet_safe(f"{prefix}_pose_confirmation.parquet", [
        "unix_ms", "pose_lateral_right_per_sec", "pose_vertical_up_per_sec",
        "pose_radial_proximity_direction_score", "radial_world_z_proximity_rate"])
    blink_fr = read_parquet_safe(f"{prefix}_blink_candidate_frames.parquet",
                                 ["unix_ms", "blink_closed_bilateral_candidate"])
    events = read_parquet_safe(f"{prefix}_blink_candidate_events.parquet",
                               ["start_unix_ms", "end_unix_ms", "duration_ms"])

    if mq is None or pose is None:
        # RGB 帧级源缺失场 (如 sub-099): 行保留, 特征全 NaN, 不补零不删除
        for _, pr in probe_sub.iterrows():
            row = {c: pr[c] for c in PROBE_ID_COLS}
            row["rgb_source_status"] = "missing_source"
            rows.append(row)
        return pd.DataFrame(rows)

    # 眨眼轨可用性: 事件表与眨眼帧都可用才算完整; 帧级可用但事件表缺失则为部分可用
    blink_partial = (blink_fr is None) or (events is None)

    for _, pr in probe_sub.iterrows():
        onset = float(pr["probe_time_ms"])
        row = {c: pr[c] for c in PROBE_ID_COLS}
        row["rgb_source_status"] = "partial_no_blink" if blink_partial else "ok"

        # ---- 运动 / 亮度 (motion_qc 帧级) ----
        w_mq = aggregate_window(mq, onset)
        row["win_n_frames"] = w_mq["n_frames"]
        row["win_coverage_sec"] = w_mq["coverage_sec"]
        row["win_valid_frame_ratio"] = w_mq["valid_frame_ratio"]
        # 分轨有效帧比例 (motion 轨 / exposure 轨, 按 observable 标志)
        lo, hi = onset - WINDOW_MS, onset
        m = mq[(mq["unix_ms"] >= lo) & (mq["unix_ms"] < hi)]
        row["body_motion_observable_ratio"] = (m["body_motion_observable"].mean()
                                               if len(m) else np.nan)
        row["exposure_change_observable_ratio"] = (m["exposure_change_observable"].mean()
                                                   if len(m) else np.nan)
        # 运动轨: body motion 中位数/均值 (仅 body_motion_observable 帧)
        row["body_motion_energy_median"], row["body_motion_energy_mean"] = \
            median_mean_on_observable(mq, onset, "body_motion_energy", "body_motion_observable")
        # 整体运动能量中位数 (全窗口帧)
        row["global_motion_energy_median"] = median_on_frames(mq, onset, "global_motion_energy")
        # 亮度: 灰度均值中位数 + exposure 轨 (曝光变化绝对量/有符号量中位数)
        row["gray_mean_median"] = median_on_frames(mq, onset, "gray_mean")
        row["exposure_change_abs_median"] = \
            median_mean_on_observable(mq, onset, "exposure_change_abs", "exposure_change_observable")[0]
        row["exposure_change_signed_median"] = \
            median_mean_on_observable(mq, onset, "exposure_change_signed", "exposure_change_observable")[0]

        # ---- 姿态方向 (pose_confirmation 帧级) ----
        w_pose = aggregate_window(pose, onset)
        row["pose_n_frames"] = w_pose["n_frames"]
        row["pose_lateral_right_per_sec_median"] = \
            median_on_frames(pose, onset, "pose_lateral_right_per_sec")      # 左右, 正=右
        row["pose_vertical_up_per_sec_median"] = \
            median_on_frames(pose, onset, "pose_vertical_up_per_sec")        # 上下, 正=上
        row["pose_radial_proximity_direction_score_median"] = \
            median_on_frames(pose, onset, "pose_radial_proximity_direction_score")  # 前后方向分数
        row["radial_world_z_proximity_rate_median"] = \
            median_on_frames(pose, onset, "radial_world_z_proximity_rate")

        # ---- 眨眼候选 (事件表 + 帧级标记) ----
        # 眨眼轨不可用时特征为 NaN (缺失不补零), 不影响其他轨
        if events is not None:
            n_ev = count_blink_events(events, onset)
            row["blink_event_n"] = n_ev
            # 事件率 = 事件数 / 窗口实际覆盖秒数 * 60 (次/分钟); 覆盖为 0 时为 NaN
            cov = row["win_coverage_sec"]
            row["blink_event_rate_per_min"] = (n_ev / cov * 60.0
                                               if (cov and cov > 0) else np.nan)
        else:
            row["blink_event_n"] = np.nan
            row["blink_event_rate_per_min"] = np.nan
        if blink_fr is not None:
            w_bl = aggregate_window(blink_fr, onset)
            row["blink_win_n_frames"] = w_bl["n_frames"]
            cand = blink_fr[(blink_fr["unix_ms"] >= lo) & (blink_fr["unix_ms"] < hi)]
            row["blink_candidate_frame_n"] = (int(cand["blink_closed_bilateral_candidate"].sum())
                                              if len(cand) else 0)
            row["blink_frame_ratio"] = (row["blink_candidate_frame_n"] / row["blink_win_n_frames"]
                                        if row["blink_win_n_frames"] else np.nan)
        else:
            row["blink_win_n_frames"] = np.nan
            row["blink_candidate_frame_n"] = np.nan
            row["blink_frame_ratio"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """主流程: 读探针表 -> 按场逐场聚合 -> 合并输出 CSV 并打印统计摘要。"""
    probe = load_probe_table(PROBE_CSV)
    print(f"probe rows: {len(probe)}, sessions: {probe['session_id'].nunique()}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    all_rows = []
    for session_id, sub in probe.groupby("session_id", sort=True):
        feats = build_session_features(session_id, sub)
        all_rows.append(feats)
        n_missing = (feats["rgb_source_status"] == "missing_source").sum()
        print(f"  {session_id}: {len(feats)} probes, missing_source={n_missing}")
    out = pd.concat(all_rows, ignore_index=True)
    # 输出列顺序: 标识列 -> 状态 -> 运动/亮度 -> 姿态 -> 眨眼
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nOUTPUT: {OUT_CSV}")
    print(f"rows: {len(out)}, sessions: {out['session_id'].nunique()}, "
          f"missing_source rows: {(out['rgb_source_status'] == 'missing_source').sum()}")
    print("NaN counts (per feature):")
    print(out.isna().sum().sort_values(ascending=False).head(12).to_string())


if __name__ == "__main__":
    main()
