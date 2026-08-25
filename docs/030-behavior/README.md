# Behavior

这里是 **Attention-Analysis 当前行为分析的唯一文档入口**。

## 当前正式版本

当前正式实验以 FocusWave **v3.1.3** 为准，正式 SART 结构为两个 B block：`B1 → B2`。当前实现已经建立，不再处于“待重写”状态：配置为 `configs/behavior_formal.yaml`，入口为 `scripts/sart_formal_analysis.py`，可复用实现位于 `src/attention_pipeline/behavior_formal/`。

当前说明按顺序阅读：`031-正式BB行为分析流程.md`、`032-行为指标定义.md`、`033-统计分析方法.md`、`034-行为QC与输出.md`。行为与 NIR 的跨模态时间对齐和 pupil/SART 分析见 `035-NIR与正式SART行为数据对齐分析方法.md`。旧 v3.0 BBB 的三 block 统计、B1↔B3、sub-011~030 队列和旧结果均不属于当前正式分析。

## 当前 NVIDIA 数据位置

当前 `nvidia-cuda` 分支对应的 RTX 5060 工作站正式数据根固定为：

```text
J:/Data
```

典型最终 BB 被试目录如下：

```text
J:/Data/sub-031_/
├── beh/
│   ├── esc_keypresses.csv
│   ├── master_timeline.csv
│   ├── SART_031_Practice_run1.csv
│   ├── sub-031_Block1_B_beh.csv
│   ├── sub-031_Block2_B_beh.csv
│   └── subject_summary.csv
└── nir/
    ├── sub-031_nir.avi
    └── sub-031_nir_timestamps.csv
```

当前 BB behavior reader 直接使用两个正式 block CSV；`master_timeline.csv`、practice、summary 与 NIR timestamps 是后续时序对齐、QC 和跨模态分析的重要辅助数据，但不是当前 block-level SART 指标提取的替代文件。

当前 NIR full-class CSV 已保留 `frame_idx / video_time_ms / unix_ms / phase_time_ms / phase`；正式 NIR ↔ SART trial alignment 作为独立下游步骤，通过 NIR `unix_ms` 与行为绝对时间戳匹配，不需要为行为对齐重新运行 RITnet。

> AMD/DirectML 工作站的数据根由 `amd-DirectML` 分支自己的配置维护；不要把 AMD 的路径复制进当前 NVIDIA 分支。

## 历史 BBB

历史材料统一放在 `history/`。`history/BBB-v3.0/` 保存 2026-08-16 的 BBB 分析计划、报告、图和冻结旧配置；`history/preformal/` 保存最终正式实验确定前的行为实验修改建议、图和统计审计。

用户要求 BBB 版本未来仍能直接重跑，因此当前主线同时保留一个**明确标记为历史**的可执行入口：`scripts/sart_bbb_v3_0_analysis.py`，配置为 `configs/sart_bbb_v3_0.yaml`，独立实现包为 `src/attention_pipeline/behavior_bbb_v3_0/`。它与当前 BB 的 `behavior_formal` 包分开，避免未来修改当前分析时污染历史 BBB 复现。

旧 BBB 的原工作记录位于 `docs/工作记录/08-16-08-SART-v3.0-BBB行为分析工作记录.md`；Git 历史继续承担完整版本追溯。
