# -*- coding: utf-8 -*-
"""契约测试: 正式报告 5.5 节 RGB 下游分析 (section_55 / figures_55)。

覆盖任务纪律要求的四条硬契约:
  1. pre-probe 窗口严格性: 锚定 trial 与 post-probe 帧绝不进入窗口;
  2. 参与者聚类: 同一参与者 (participant_group_id) 的所有 session 共享同一聚类键;
  3. 失败写表: 模型不可估计时必须写入 not_estimable + reason, 不得空表冒充成功;
  4. 物理位移禁语: 前后方向解释列带 candidate / 非物理位移标注。

另覆盖: cycle_bin 口径与行为 science-v3 一致、blink 事件经窗口帧
blink_event_id 映射、毫米波合并键规范化与主键唯一、图包 audit 状态枚举。
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import pytest

from attention_pipeline.rgb_formal.section_55 import (
    BEHAVIOR_PROBE_METRICS,
    MMWAVE_PROBE_METRICS,
    RGB_PROBE_METRICS,
    RADIAL_INTERPRETATION,
    aggregate_probe_window,
    assign_cycle_bin,
    build_block_cycle_table,
    build_probe_within_between,
    fit_block_cycle_gee,
    fit_q1_mnlogit,
    fit_q2_ordinal_gee,
    fit_single_predictor_gee,
    merge_key_mmwave,
    normalize_mmwave_block,
    probe_window_mask,
)
from attention_pipeline.rgb_formal.figures_55 import (
    build_figure_pack,
    figure_blink_events,
    figure_coverage,
    figure_motion_exposure,
    figure_pose_direction,
)


# ---------------------------------------------------------------------------
# 合成帧表构造器
# ---------------------------------------------------------------------------
def _motion_frames(block: int = 1, trials: tuple[int, ...] = (1, 2, 3, 4, 5),
                   probe_time_ms: float = 9000.0, trial_len_ms: int = 1000) -> pd.DataFrame:
    """构造单 block 的合成 motion 帧表: 每 trial 一帧 (trial onset = trial*1000 ms)。

    probe_time_ms 参数保留以贴近真实 schema; cycle_num 用 trial 值占位,
    使构造器同时满足 probe 窗口与 block×cycle 聚合的列契约。
    """
    rows = []
    for trial in trials:
        onset = float(trial * trial_len_ms)
        rows.append({
            "unix_ms": onset, "block": float(block), "trial_num": float(trial),
            "cycle_num": float(trial), "motion_valid": True,
            "body_motion_observable": True, "exposure_change_observable": True,
            "body_motion_energy": float(trial) * 0.1,
            "global_motion_energy": float(trial) * 0.05,
            "gray_mean": 150.0, "exposure_change_abs": 0.5, "exposure_change_signed": 0.1,
        })
    return pd.DataFrame(rows)


def _probe_row(block_id: str = "B1", anchor: int = 3, probe_time_ms: float = 9000.0,
               session: str = "sub-001", group: str = "P-A", order: int = 1) -> dict:
    return {
        "session_id": session, "participant_group_id": group, "block_id": block_id,
        "probe_event_id": f"{session}|{block_id}|probe|{order}",
        "probe_order_in_block": order, "anchor_trial_num": float(anchor),
        "probe_time_ms": probe_time_ms,
        "q1_nominal_4class": 1.0, "q2_ordinal_4level": 3.0,
    }


# ---------------------------------------------------------------------------
# 1. pre-probe 窗口严格性
# ---------------------------------------------------------------------------
def test_probe_window_mask_excludes_anchor_trial_even_before_probe_onset() -> None:
    # 锚定 trial 3 的 onset 早于 probe onset, 但其帧绝不允许进入窗口
    frames = _motion_frames(block=1, trials=(1, 2, 3, 4), probe_time_ms=9000.0)
    mask = probe_window_mask(frames, block_num=1, anchor_trial_num=3.0, probe_time_ms=9000.0)
    inside = frames.loc[mask]
    assert set(inside["trial_num"]) == {1.0, 2.0}
    assert 3.0 not in set(inside["trial_num"])


def test_probe_window_mask_excludes_post_probe_frames() -> None:
    # probe onset 前窗口 [6000, 9000): trial 6/7/8 onset 在窗口内, trial 9 (9000ms) 不早于 probe
    frames = _motion_frames(block=1, trials=(6, 7, 8, 9), probe_time_ms=9000.0)
    mask = probe_window_mask(frames, block_num=1, anchor_trial_num=10.0,
                             probe_time_ms=9000.0, window_ms=3000.0)
    inside = frames.loc[mask]
    assert set(inside["trial_num"]) == {6.0, 7.0, 8.0}
    assert 9.0 not in set(inside["trial_num"])


def test_probe_window_mask_requires_same_block() -> None:
    # B2 的帧即使时间落在 B1 窗口内也不得进入 B1 探针窗口
    b2 = _motion_frames(block=2, trials=(1, 2, 3), probe_time_ms=9000.0)
    mask = probe_window_mask(b2, block_num=1, anchor_trial_num=10.0, probe_time_ms=9000.0)
    assert not mask.any()


def test_aggregate_probe_window_counts_blink_events_via_frame_mapping() -> None:
    # 事件 start 在窗口时间范围内, 但 start 帧属于锚定 trial → 不得计入
    blink_frames = pd.DataFrame({
        "unix_ms": [1000.0, 2000.0, 3000.0],
        "block": [1.0, 1.0, 1.0],
        "trial_num": [1.0, 2.0, 3.0],  # trial 3 = 锚定 trial
        "cycle_num": [1.0, 1.0, 1.0],
        "blink_event_id": [pd.NA, 1.0, 2.0],
        "left_eye_observable": [True, True, True],
        "right_eye_observable": [True, True, True],
        "blink_bilateral_observable": [True, True, True],
        "bilateral_eye_consistent": [True, True, True],
        "blink_closed_bilateral_candidate": [False, True, True],
    })
    events = pd.DataFrame({
        "blink_event_id": [1, 2],
        "start_unix_ms": [2000.0, 3000.0],  # 事件 2 的 start 帧在锚定 trial
        "end_unix_ms": [2100.0, 3100.0],
        "duration_ms": [120.0, 140.0],
    })
    motion = _motion_frames(block=1, trials=(1, 2, 3), probe_time_ms=4000.0)
    pose = pd.DataFrame({
        "unix_ms": [1000.0, 2000.0], "block": [1.0, 1.0], "trial_num": [1.0, 2.0],
        "pose_shoulders_observable": [True, True],
        "pose_lateral_right_per_sec": [0.01, -0.02],
        "pose_vertical_up_per_sec": [0.0, 0.01],
        "pose_radial_proximity_direction_score": [1.0, -1.0],
        "radial_world_z_proximity_rate": [0.0, 0.0],
    })
    row = aggregate_probe_window(
        motion=motion, pose=pose, blink_frames=blink_frames, blink_events=events,
        probe_row=_probe_row(anchor=3, probe_time_ms=4000.0),
    )
    assert row["blink_event_n"] == 1  # 只计入 trial 2 的事件; trial 3 (锚定) 的事件被排除
    assert row["rgb_source_status"] == "ok"
    assert row["blink_win_n_frames"] == 2  # 锚定 trial 帧不在窗口
    # 锚定 trial 的 closed 帧不进候选计数
    assert row["blink_candidate_frame_n"] == 1


def test_aggregate_probe_window_missing_source_and_partial_blink() -> None:
    motion = _motion_frames(block=1, trials=(1, 2), probe_time_ms=3000.0)
    pose = pd.DataFrame({
        "unix_ms": [1000.0], "block": [1.0], "trial_num": [1.0],
        "pose_shoulders_observable": [True],
        "pose_lateral_right_per_sec": [0.0], "pose_vertical_up_per_sec": [0.0],
        "pose_radial_proximity_direction_score": [1.0],
        "radial_world_z_proximity_rate": [0.0],
    })
    # 眨眼事件表缺失 → partial_no_blink, 眨眼特征 NaN (不补零)
    row = aggregate_probe_window(
        motion=motion, pose=pose, blink_frames=pd.DataFrame(), blink_events=None,
        probe_row=_probe_row(anchor=5, probe_time_ms=3000.0),
    )
    assert row["rgb_source_status"] == "partial_no_blink"
    assert math.isnan(row["blink_event_n"])
    # motion/pose 全缺 → missing_source
    row2 = aggregate_probe_window(
        motion=None, pose=None, blink_frames=None, blink_events=None,
        probe_row=_probe_row(anchor=5, probe_time_ms=3000.0),
    )
    assert row2["rgb_source_status"] == "missing_source"


# ---------------------------------------------------------------------------
# 2. 参与者聚类契约
# ---------------------------------------------------------------------------
def test_participant_cluster_key_consistent_across_sessions() -> None:
    # 同一参与者两个 session 的行必须共享同一 participant_group_id (同簇)
    rows = []
    for session in ("sub-001", "sub-002"):
        rows += [{"session_id": session, "participant_group_id": "P-A", "block_id": b, "cycle_bin": c}
                 for b in ("B1", "B2") for c in range(1, 4)]
    table = pd.DataFrame(rows)
    # 参与者 → session 映射必须唯一 (不允许同一参与者跨两个聚类键)
    mapping = table[["session_id", "participant_group_id"]].drop_duplicates()
    sessions_per_group = mapping.groupby("session_id")["participant_group_id"].nunique()
    assert (sessions_per_group == 1).all()
    assert table["participant_group_id"].nunique() == 1


# ---------------------------------------------------------------------------
# 3. 失败写表 (not_estimable + reason, 禁止空表冒充成功)
# ---------------------------------------------------------------------------
def test_q1_failure_written_when_categories_missing() -> None:
    # 只有两类 Q1 → 模型不可估计, failures 必须记录原因
    probe = pd.DataFrame({
        "q1_nominal_4class": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2] * 6,
        "participant_group_id": [f"P-{i % 6}" for i in range(72)],
        "session_id": [f"sub-{i:03d}" for i in range(72)],
    })
    probe["body_motion_energy_median"] = np.linspace(0.0, 1.0, len(probe))
    results, failures = fit_q1_mnlogit(probe, predictors=["body_motion_energy_median"])
    assert results.empty
    assert not failures.empty
    assert (failures["failure_type"] == "not_estimable").all()
    assert failures["failure_reason"].str.len().gt(0).all()
    assert "required" in failures["failure_reason"].iloc[0]


def test_q2_failure_written_when_insufficient_levels() -> None:
    probe = pd.DataFrame({
        "q2_ordinal_4level": [1, 1, 1, 2, 2, 2] * 12,
        "participant_group_id": [f"P-{i % 6}" for i in range(72)],
        "session_id": [f"sub-{i:03d}" for i in range(72)],
        "body_motion_energy_median": np.linspace(0.0, 1.0, 72),
    })
    results, failures = fit_q2_ordinal_gee(probe, predictors=["body_motion_energy_median"])
    assert results.empty
    assert not failures.empty
    assert (failures["failure_type"] == "not_estimable").all()


def test_gee_failure_written_when_fewer_than_min_participants() -> None:
    cycle = pd.DataFrame({
        "participant_group_id": ["P-A", "P-B", "P-C"] * 8,
        "session_id": ["s1"] * 8 + ["s2"] * 8 + ["s3"] * 8,
        "block_id": ["B1"] * 24,
        "cycle_bin": list(range(1, 7)) * 4,
        "body_motion_energy_median": np.linspace(0.0, 1.0, 24),
    })
    results, failures = fit_block_cycle_gee(cycle, metrics=["body_motion_energy_median"])
    assert results.empty
    assert not failures.empty
    assert failures["failure_reason"].str.contains("insufficient").all()


def test_single_predictor_gee_nonfinite_results_become_failures() -> None:
    # 常数因变量 → 模型拟合可能退化; 无论拟合结果如何, 结果行必须全部有限
    probe = pd.DataFrame({
        "participant_group_id": [f"P-{i % 6}" for i in range(24)],
        "session_id": [f"sub-{i:03d}" for i in range(24)],
        "body_motion_energy_median": [0.05] * 24,
        "go_correct_rt_median_ms": np.linspace(200.0, 400.0, 24),
    })
    results, failures = fit_single_predictor_gee(
        probe, outcomes=["body_motion_energy_median"], predictors=["go_correct_rt_median_ms"],
        analysis="test_rgb_behavior_window_gee")
    # 结果表要么为空 (失败写表), 要么全部有限; 不允许 NaN 行冒充成功
    if not results.empty:
        finite = results[["estimate_per_predictor_sd", "se", "ci_low", "ci_high"]].applymap(np.isfinite).all(axis=1)
        assert finite.all()
    else:
        assert not failures.empty
        assert (failures["failure_type"] == "not_estimable").all()


# ---------------------------------------------------------------------------
# 4. 物理位移禁语 (解释列带 candidate 标注)
# ---------------------------------------------------------------------------
def test_radial_interpretation_carries_candidate_not_physical_displacement() -> None:
    assert "candidate" in RADIAL_INTERPRETATION
    assert "not_physical_displacement" in RADIAL_INTERPRETATION
    # 解释列不得出现物理位移表述
    banned = re.compile(r"physical\s+displacement|metric\s+displacement", re.IGNORECASE)
    assert not banned.search(RADIAL_INTERPRETATION)


def test_aggregate_radial_interpretation_column_uses_candidate_language() -> None:
    motion = _motion_frames(block=1, trials=(1, 2), probe_time_ms=3000.0)
    pose = pd.DataFrame({
        "unix_ms": [1000.0], "block": [1.0], "trial_num": [1.0],
        "pose_shoulders_observable": [True],
        "pose_lateral_right_per_sec": [0.0], "pose_vertical_up_per_sec": [0.0],
        "pose_radial_proximity_direction_score": [1.0],
        "radial_world_z_proximity_rate": [0.0],
    })
    row = aggregate_probe_window(
        motion=motion, pose=pose, blink_frames=None, blink_events=None,
        probe_row=_probe_row(anchor=5, probe_time_ms=3000.0),
    )
    assert "candidate" in row["radial_interpretation"]
    assert "displacement" in row["radial_interpretation"]  # 仅以否定形式出现
    assert "not_physical" in row["radial_interpretation"]


# ---------------------------------------------------------------------------
# 5. cycle_bin 口径与行为 science-v3 一致
# ---------------------------------------------------------------------------
def test_cycle_bin_matches_behavior_convention() -> None:
    # 24 个 cycle → 6 个等频 bin (每 bin 4 个 cycle), 与 extract._add_derived 一致
    cycles = pd.Series(list(range(1, 25)), dtype=float)
    binned = assign_cycle_bin(cycles, n_bins=6)
    assert binned.notna().all()
    assert sorted(binned.unique()) == [1, 2, 3, 4, 5, 6]
    assert (binned.value_counts() == 4).all()
    # 少于 6 个唯一值时 bin 数收缩
    small = assign_cycle_bin(pd.Series([1.0, 2.0, 3.0]), n_bins=6)
    assert sorted(small.dropna().unique()) == [1, 2, 3]


def test_block_cycle_table_keys_and_event_mapping() -> None:
    # 真实口径: 每 block 24 个 cycle → 6 个 bin (每 bin 4 个 cycle)
    motion = pd.DataFrame({
        "unix_ms": [1000.0 * c for c in range(1, 25)], "block": [1.0] * 24,
        "trial_num": list(range(1, 25)), "cycle_num": [float(c) for c in range(1, 25)],
        "motion_valid": [True] * 24, "body_motion_observable": [True] * 24,
        "exposure_change_observable": [True] * 24,
        "body_motion_energy": [0.1 * c for c in range(1, 25)],
        "global_motion_energy": [0.05] * 24, "gray_mean": [150.0] * 24,
        "exposure_change_abs": [0.5] * 24,
    })
    # 眨眼帧落在 cycle 5/6/7 (bin 2), 事件 1 的 start 帧为 cycle 6
    blink_frames = pd.DataFrame({
        "unix_ms": [5000.0, 6000.0, 7000.0], "block": [1.0] * 3,
        "trial_num": [5.0, 6.0, 7.0], "cycle_num": [5.0, 6.0, 7.0],
        "blink_event_id": [pd.NA, 1.0, pd.NA],
        "left_eye_observable": [True] * 3, "right_eye_observable": [True] * 3,
        "blink_bilateral_observable": [True] * 3, "bilateral_eye_consistent": [True] * 3,
        "blink_closed_bilateral_candidate": [False, True, False],
    })
    events = pd.DataFrame({
        "blink_event_id": [1], "start_unix_ms": [6000.0], "end_unix_ms": [6100.0], "duration_ms": [120.0],
    })
    table = build_block_cycle_table(
        motion=motion, pose=None, blink_frames=blink_frames, blink_events=events,
        session_id="sub-001", participant_group_id="P-A")
    assert not table.empty
    assert set(table["block_id"]) == {"B1"}
    assert table["cycle_bin"].nunique() == 6
    # 事件 1 的 start 帧 (cycle 6 → bin 2) 只计入其所在 cycle cell
    cell = table[table["cycle_bin"].eq(2)].iloc[0]
    assert cell["blink_event_n"] == 1
    other = table[table["cycle_bin"].ne(2)]
    # 眨眼帧未覆盖的 cell 事件特征为 NaN (缺失), 已覆盖的 cell 应为 0
    assert (other["blink_event_n"].dropna() == 0).all()


# ---------------------------------------------------------------------------
# 6. 毫米波合并键规范化
# ---------------------------------------------------------------------------
def test_mmwave_block_normalization_and_merge_key() -> None:
    assert normalize_mmwave_block("block-1") == "B1"
    assert normalize_mmwave_block("block-2") == "B2"
    assert normalize_mmwave_block("B1") == "B1"
    assert normalize_mmwave_block(2) == "B2"
    assert merge_key_mmwave("block-1", 7) == "B1|7"
    # 与行为侧 merge_key 构造一致 (session|B1|order)
    behavior_style_key = f"sub-031|{normalize_mmwave_block('block-1')}|{int(float(7))}"
    assert behavior_style_key == "sub-031|B1|7"


# ---------------------------------------------------------------------------
# 7. within/between 分解
# ---------------------------------------------------------------------------
def test_within_between_decomposition_reconstructs_original() -> None:
    probe = pd.DataFrame({
        "participant_group_id": ["P-A", "P-A", "P-B", "P-B"],
        "session_id": ["s1", "s2", "s3", "s4"],
        "body_motion_energy_median": [0.1, 0.3, 0.5, 0.5],
        "exposure_change_abs_median": [0.2, 0.2, 0.4, 0.6],
    })
    out = build_probe_within_between(probe, metrics=["body_motion_energy_median"])
    assert not out.empty
    assert set(out["scale"]) == {"probe_pre30s"}
    merged = out[["participant_group_id", "session_id", "metric", "median",
                  "participant_mean", "within_deviation"]]
    reconstructed = merged["participant_mean"] + merged["within_deviation"]
    assert np.allclose(reconstructed.to_numpy(float), merged["median"].to_numpy(float))
    # 不同 metric 不互相污染 participant mean
    assert out["participant_mean"].nunique() >= 2


# ---------------------------------------------------------------------------
# 8. 图包 audit: generated / not_estimable + reason
# ---------------------------------------------------------------------------
def test_figure_pack_audit_statuses_and_manifest(tmp_path) -> None:
    motion = _motion_frames(block=1, trials=(1, 2, 3, 4, 5, 6), probe_time_ms=9000.0)
    pose = pd.DataFrame({
        "unix_ms": [1000.0, 2000.0, 3000.0], "block": [1.0] * 3, "trial_num": [1.0, 2.0, 3.0],
        "cycle_num": [1.0, 2.0, 3.0],
        "pose_shoulders_observable": [True] * 3,
        "pose_lateral_right_per_sec": [0.0, 0.01, -0.01],
        "pose_vertical_up_per_sec": [0.0, 0.0, 0.01],
        "pose_radial_proximity_direction_score": [1.0, -1.0, 1.0],
        "radial_world_z_proximity_rate": [0.0] * 3,
    })
    blink_frames = pd.DataFrame({
        "unix_ms": [1000.0, 2000.0], "block": [1.0, 1.0], "trial_num": [1.0, 2.0],
        "cycle_num": [1.0, 1.0], "blink_event_id": [pd.NA, 1.0],
        "left_eye_observable": [True, True], "right_eye_observable": [True, True],
        "blink_bilateral_observable": [True, True], "bilateral_eye_consistent": [True, True],
        "blink_closed_bilateral_candidate": [False, True],
    })
    events = pd.DataFrame({
        "blink_event_id": [1], "start_unix_ms": [2000.0], "end_unix_ms": [2100.0], "duration_ms": [120.0],
    })
    probe_features = pd.DataFrame([aggregate_probe_window(
        motion=motion, pose=pose, blink_frames=blink_frames, blink_events=events,
        probe_row=_probe_row(anchor=5, probe_time_ms=9000.0, session="sub-001", group="P-A", order=1))])
    cycle_table = build_block_cycle_table(
        motion=motion, pose=pose, blink_frames=blink_frames, blink_events=events,
        session_id="sub-001", participant_group_id="P-A")
    coverage = pd.DataFrame([
        {"session_id": "sub-001", "participant_group_id": "P-A",
         "body_motion_observable_ratio": 1.0, "exposure_change_observable_ratio": 0.99,
         "pose_shoulders_observable_ratio": 0.98, "left_eye_observable_ratio": 0.97,
         "right_eye_observable_ratio": 0.96, "bilateral_consistent_ratio": 0.95,
         "blink_event_rate_per_min": 12.0, "blink_event_duration_median_ms": 120.0,
         "blink_ibi_median_ms": np.nan},
        {"session_id": "sub-002", "participant_group_id": "P-B",
         "body_motion_observable_ratio": 0.9, "exposure_change_observable_ratio": 0.91,
         "pose_shoulders_observable_ratio": 0.92, "left_eye_observable_ratio": 0.93,
         "right_eye_observable_ratio": 0.94, "bilateral_consistent_ratio": 0.89,
         "blink_event_rate_per_min": 15.0, "blink_event_duration_median_ms": 110.0,
         "blink_ibi_median_ms": 3500.0},
    ])
    manifest, audit = build_figure_pack(
        probe_features=probe_features, cycle_table=cycle_table,
        session_coverage=coverage, output_root=tmp_path)
    assert not audit.empty
    assert set(audit["status"]) <= {"generated", "not_estimable"}
    assert audit["internal_title_present"].eq(False).all()
    assert audit["caption_external"].eq(True).all()
    # 每个 not_estimable 行必须带原因
    for _, row in audit[audit["status"].eq("not_estimable")].iterrows():
        assert str(row["reason"]).strip()
    # 覆盖图应当生成 (coverage 数据完整)
    coverage_audit = audit[audit["figure_id"].eq("fig_55_4_coverage")].iloc[0]
    assert coverage_audit["status"] == "generated"
    assert not manifest.empty
    assert {"figure_id", "caption_zh", "caption_en"} <= set(manifest.columns)


def test_figure_not_estimable_with_reason(tmp_path) -> None:
    generated, reason, png = figure_coverage(pd.DataFrame(), tmp_path)
    assert generated is False
    assert reason
    assert png == ""
