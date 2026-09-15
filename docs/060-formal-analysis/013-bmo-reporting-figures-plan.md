# Behavior / Ocular / Movement 科研绘图与结果汇报语义合同

> 原始日期：2026-09-13  
> 2026-09-15 cleanup 状态：`historical_reporting_contract_reconciled_with_final_report`  
> 来源：历史 PR #77 `codex/reporting-figures-bmo-v1`，经当前 Formal 方法与最终报告资产重新对账后选择性再集成。

本文件保存 Behavior（行为）、Ocular（眼部）、Movement（动作）报告层的科学语义与可追溯约束。它不再是一个待执行的 Stage-A（阶段 A）任务说明，也不授权重新拟合模型或重新生成正式结果。实际 2026-09-13 国赛上交版的图件、图号和最终呈现以 `FocusWave-Formal-Analysis/国赛报告/assets/图件清单.md` 及相应章节为准。

## 1. 当前权威关系

报告语义首先受 Formal 当前方法文件约束：

- `分析设计/1.16.19-Behavior_Movement_Ocular科研绘图与结果汇报第一阶段裁决_20260913.md`：定义科学问题、正文/附录角色、within-person（人内）/between-person（人际）层级和允许的展示派生。
- `分析设计/1.16.20-Behavior_RT水平中位数冻结与正式运行准入_20260913.md`：已将 `go_correct_rt_median_ms` 冻结为 Behavior RT level（反应时水平）的唯一首轮科学表示；`go_correct_rt_mean_ms` 仅为 limited alternative（受限替代表示）。
- `国赛报告/assets/图件清单.md`：记录最终上交报告实际使用的图件、生成入口和已知上游限制。

本仓库的三份 `configs/reporting/*.csv` 只保存可复用的语义合同，不得覆盖上述 Formal 权威状态。

## 2. 三份合同文件

- `configs/reporting/report_figure_plan.csv`：保存每个科学问题、正式源表、结果角色和原设计目标；对未按原设计落地的图明确标为历史目标，并指向最终报告实际资产。
- `configs/reporting/scientific_label_dictionary.csv`：程序字段到正式中文科学语义的映射。RT 中位数已标为 `main_science_frozen`，RT 均值标为 `limited_alternative_qualification`。
- `configs/reporting/existing_figure_audit.csv`：保存 2026-09-13 图件重构裁决的来源追溯；旧 `pending_full_local_stage_a` 状态已移除，当前状态明确由最终报告资产接管。

## 3. 保持有效的科学报告原则

正文主图资格来自预先定义的科学角色与研究问题，不按 `p < .05` 筛选。无效应、区间跨 0 的结果仍属于正式结果。

Ocular 报告继续区分两个层级：`ocular_within_z` 的人内效应承担正文状态变化主叙事；`ocular_between_z` 的人际效应属于正式伴随结果，完整保留于结果表或附录，不因放入附录而降为 sensitivity（敏感性）。

Movement 第一轮主科学指标保持 `body_motion_energy_median`。姿态方向、径向代理以及曝光/来源指标继续承担 sensitivity（敏感性）或 QC（质量控制）角色。

Behavior RT level 已冻结为中位数。均值—中位数一致性与差异分布仍可保留为 qualification（测量资格）证据，但不得重新表述为当前待决的表示选择。

## 4. 原设计与最终上交版的差异

历史计划曾要求 Movement `M-M1` 与 Ocular `O-M1` 从完全相同冻结 GEE（广义估计方程）派生模型调整后轨迹。最终报告没有为了图形展示重新拟合冻结模型：

- Ocular 正式 `ocular_task_progression.csv` 不包含派生完整预测轨迹所需的截距和完整系数协方差，因此最终保留正式系数图；
- Movement 最终保留生产端正式的区块配对/任务进程图；
- Ocular Q1、Q2 与眼部—行为关系则按 1.16.19 拆出了正文的人内图，并把生产端同时含人内/人际结果的图保留为附录追溯资产。

cleanup 不补跑这些历史目标，也不据此改写任何正式效应数值。

## 5. 边界

本目录不修改冻结特征、模型公式、样本、缺失规则、正式效应数字或监督学习结果；不处理 Cardiopulmonary（心肺）/mmWave（毫米波）的预测资格；不恢复旧三模态主线作为当前完整项目结构。FocusWave 最终监督学习结构已包含 Cardiopulmonary，本文件仅保留 B/M/O 三个单模态结果汇报的专项语义合同。
