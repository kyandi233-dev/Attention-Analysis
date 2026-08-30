# RGB 探针级 pre_30s 特征表说明

- 生成日期: 2026-08-31
- 生成脚本: `scripts/maintenance/build_rgb_probe_features_20260831.py`
- 输出表: `D:/Project/厚粲杯/11_数据/_FormalAnalysis/RGB/11_analysis_tables_116cohort/rgb_probe_pre30s_features.csv`
- 行数: 2320 = 116 场 × 20 探针（与行为/NIR/毫米波 probe 表同粒度，供论文 5.6 节多模态融合）
- 窗口定义: `[probe_time_ms - 30000, probe_time_ms)`，严格早于探针 onset，无未来信息泄漏
- 身份: participant_group_id（P 编码），探针时间戳权威来源 `Behavior/formal_v3/probe_primary_30s.csv`

## 输入帧级 schema 要点

每场 4 个 parquet（`RGB/10_analysis_ready/sub-XXX/`），时间列均为 unix 毫秒：

| 文件 | 粒度 | 关键列 |
|------|------|--------|
| motion_qc | 帧级 (~20-30 fps) | unix_ms, motion_valid（总有效标志）, body_motion_observable / exposure_change_observable（motion/exposure 分轨可观测标志，合约值 `separate_tracks_no_combined_risk_score`）, body_motion_energy, global_motion_energy, gray_mean（亮度）, exposure_change_abs / exposure_change_signed |
| pose_confirmation | 帧级（约为 motion 的 1/3） | unix_ms, pose_lateral_right_per_sec（左右，正=右）, pose_vertical_up_per_sec（上下，正=上）, pose_radial_proximity_direction_score（前后方向分数）, radial_world_z_proximity_rate；注意 `pose_direction_interpretation` 全部为 `auxiliary_qc_candidate_not_physical_displacement`，即姿态方向值定位为辅助 QC 候选而非物理位移，下游解读需保留此限定 |
| blink_candidate_frames | 帧级 | unix_ms, blink_closed_bilateral_candidate（帧级眨眼候选标记） |
| blink_candidate_events | 事件级 | start_unix_ms, end_unix_ms, duration_ms, event_type（仅 `algorithm_defined_blink_candidate`） |

## 输出特征清单（30 列）

- 标识: session_id, participant_group_id, block_id, probe_event_id, probe_order_in_block, probe_time_ms, q1_nominal_4class, q2_ordinal_4level
- 状态: rgb_source_status（ok / partial_no_blink / missing_source）
- 窗口质量: win_n_frames（motion_qc 窗口帧数）, win_coverage_sec（首末帧时间跨度）, win_valid_frame_ratio（motion_valid 比例）, body_motion_observable_ratio, exposure_change_observable_ratio（分轨有效帧比例）
- 运动: body_motion_energy_median / _mean（仅 body_motion_observable 帧）, global_motion_energy_median
- 亮度: gray_mean_median, exposure_change_abs_median, exposure_change_signed_median（exposure 轨，仅 observable 帧）
- 姿态: pose_n_frames, pose_lateral_right_per_sec_median, pose_vertical_up_per_sec_median, pose_radial_proximity_direction_score_median, radial_world_z_proximity_rate_median
- 眨眼: blink_event_n（start 落在窗口内的事件数）, blink_event_rate_per_min（次/分钟，分母为 win_coverage_sec）, blink_win_n_frames, blink_candidate_frame_n, blink_frame_ratio

缺失一律 NaN，不补零不删除行。

## 缺失/降级场次清单

| session | 状态 | 说明 |
|---------|------|------|
| sub-099 | missing_source（20 行全 NaN） | 无 RGB 源目录 |
| sub-041 | partial_no_blink | blink_candidate_frames.parquet 损坏 + 无事件表 |
| sub-068, sub-128, sub-168, sub-173 | partial_no_blink（各 20 行） | 仅缺 blink_candidate_events.parquet，运动/姿态/亮度/眨眼帧特征正常 |

合计: ok 2200 行, partial_no_blink 100 行, missing_source 20 行。有效帧比例中位数 1.0（最低 0.70），窗口覆盖平均 29.96 秒。

## 对账

sub-031 probe 1 手工重算与输出表逐字段一致（win_n_frames=901、body_motion_energy_median=0.04636 等）；全表窗口边界检查 0 泄漏。
