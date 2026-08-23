# 正式 SART 行为分析

> 2026-08-16｜基于 `E:/正式实验` 正式行为数据（v3.0 BBB 设计）；独立提取 SART 行为并完成 试次/block/时间窗/探针 四维分析与 主效应/交互/回归/相关 四类统计与图表。｜RT 永不删除、Q1 保持名义、推断单位=被试。

## 文件地图

| 文件 | 内容 |
|---|---|
| [000-正式SART行为分析计划.md](000-正式SART行为分析计划.md) | 分析计划（数据口径/指标/统计/图表/输出/校验/审批门） |
| [001-正式SART行为分析报告.md](001-正式SART行为分析报告.md) | **图文报告**（过程+结果+图表映射与说明，图在同目录 PNG 内嵌） |
| [002-工作记录.md](002-工作记录.md) | 工作记录（计划/执行决策/结果概览） |
| `051-01..20-*.png` | 20 张分析图（报告内嵌） |

## 数据与输出位置

- 原始数据（只读）：`E:/正式实验/sub-XXX_/beh/`
- 管线输出（规范主输出）：`D:/_AttentionData/output-v2/050-sart-formal/`
  - `040-behavior/`：051-trials.csv、051-block_metrics.csv、051-cycle_bin_metrics.csv、051-rolling_evidence.csv、051-probe_evidence.csv、051-probe_behaviour_link.csv、051-main_effects.csv、051-rt_drift_mixedlm.csv、051-pre_nogo_stats.csv、051-cross_block_consistency.csv、051-correlation_matrix.csv、051-validation.csv、051-subject_block_audit.csv
  - `090-manifests/`：051-extract/interaction/commission_gee/probe_association/correlation-*.json
- 配置：`configs/sart_formal.yaml`
- 代码：`src/attention_pipeline/behavior_formal/{extract,metrics,stats,figures,report}.py`
- 入口脚本：`scripts/sart_formal_analysis.py`

## 关键结论速查

- 有效样本 **19**（剔除 sub-015 完全无反应；sub-9504 试采排除；sub-025 慢反应/sub-029 高漏报保留标记）。
- **显著警戒衰退**：跨 block 漏按率上升（Holm p=0.007）、RT-CV 上升（p=0.021）。
- **错误前反应加速**：No-Go 前 lag−1/−2 RT 显著快（Holm p<0.002）。
- **block 内 RT 漂移**：B2/B3 显著上升，block×cycle 交互 p=0.001。
- **探针**：Q1 注意状态 1=完全专注占 64.5%；Q2 清醒(3-4) 占 72%。
- **相关**：d′×RT-CV ρ=−0.89；跨 block 一致性高（ρ 0.55–0.92）。
