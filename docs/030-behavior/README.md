# Behavior

这里是 **Attention-Analysis 当前行为分析的唯一文档入口**。

## 当前正式版本

当前正式实验以 FocusWave **v3.1.3** 为准，正式 SART 结构为两个 B block：`B1 → B2`。当前代码与配置为 `configs/behavior_formal.yaml`、`scripts/sart_formal_analysis.py` 和 `src/attention_pipeline/behavior_formal/`。

当前说明按顺序阅读：`031-正式BB行为分析流程.md`、`032-行为指标定义.md`、`033-统计分析方法.md`、`034-行为QC与输出.md`。旧 v3.0 BBB 的三 block 统计、B1↔B3、sub-011~030 队列和旧结果均不属于当前正式分析。

## 当前数据位置

正式数据在不同机器/硬盘上的已确认根目录为 `E:/正式实验`、`E:/Data`、`F:/Data`。典型最终 BB 被试目录如下：

```text
E:/正式实验/sub-031_/
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

## 历史 BBB

历史材料统一放在 `history/`。`history/BBB-v3.0/` 保存 2026-08-16 的 BBB 分析计划、报告、图和冻结旧配置；`history/preformal/` 保存最终正式实验确定前的行为实验修改建议、图和统计审计。

用户要求 BBB 版本仍可直接重跑，因此当前主线同时保留一个**明确标记为历史**的可执行入口：`scripts/sart_bbb_v3_0_analysis.py`，配置为 `configs/sart_bbb_v3_0.yaml`，独立实现包为 `src/attention_pipeline/behavior_bbb_v3_0/`。它不会与当前 BB 的 `behavior_formal` 包共享统计实现，从而避免未来重跑历史 BBB 时被当前修改污染。

旧 BBB 的原工作记录位于 `docs/工作记录/08-16-08-SART-v3.0-BBB行为分析工作记录.md`；`history/behavior-bbb-v3.0` 分支继续保存完整旧仓库快照。
