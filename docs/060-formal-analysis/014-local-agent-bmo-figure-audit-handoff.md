# Behavior / Ocular / Movement 本地图件审计交接归档

> 原始日期：2026-09-13  
> 2026-09-15 cleanup 状态：`RETIRED_STAGE_A_HANDOFF / PROVENANCE_ONLY`

本文件原本是 `codex/reporting-figures-bmo-v1` 的 Stage-A（阶段 A）本地审计执行说明。该阶段已经被 2026-09-13 国赛报告的真实图件收口工作取代，因此**不再是待执行任务**，不得按旧说明重新启动一轮独立图件审计或重绘。

## 1. 历史用途

旧交接要求本地 agent（本地代理）枚举 `FormalScience/Behavior`、`FormalScience/Ocular`、`FormalScience/Movement` 的 tables / figures / manifests，回读 PNG/SVG，检查字体、裁切、图例、坐标轴、95% CI（95%置信区间）、程序字段暴露、科学问题混画以及 qualification/QC/sensitivity（测量资格/质量控制/敏感性）角色，并记录 checksum（校验和）与执行环境。

这些要求解释了历史 PR #77 中 `existing_figure_audit.csv` 和 `report_figure_plan.csv` 的来源，但旧 `pending_full_local_stage_a` 状态已经失效。

## 2. 当前状态入口

当前实际图件状态和最终上交版呈现应读取：

- `FocusWave-Formal-Analysis/国赛报告/assets/图件清单.md`；
- `FocusWave-Formal-Analysis/分析设计/1.16.19-Behavior_Movement_Ocular科研绘图与结果汇报第一阶段裁决_20260913.md`；
- `FocusWave-Formal-Analysis/分析设计/1.16.20-Behavior_RT水平中位数冻结与正式运行准入_20260913.md`；
- 对应第 5.2、5.3、5.4 章节及附录。

Ocular（眼部）正式解释性结果的执行来源仍为 `Attention-Analysis@4fdb1a22f99afaad604d6023a34796e9663fd4dd` 的 post-freeze（后冻结）分析资产；其可复现代码已由 cleanup PR #89 选择性再集成。

## 3. cleanup 边界

本归档只用于说明历史报告规划为什么产生以及哪些检查曾被要求，不授权：

- 重新拟合正式模型；
- 改变样本、缺失、冻结表示或正式效应；
- 恢复 RT mean/median（反应时均值/中位数）待决状态；
- 把 B/M/O 三模态专项汇报误写为 FocusWave 最终完整监督学习结构；
- 用历史 Stage-A 状态覆盖 2026-09-13 最终报告资产。

如未来需要重新设计图件，应从当前方法、当前正式结果和当前报告资产建立新的任务，而不是继续执行本归档中的旧阶段流程。
