# 060 正式分析

本目录维护 `codex/formal-analysis-v2-portable` 的正式下游分析说明。这里记录分析入口、数据与身份合同、NIR pupil-only 适配、RGB 轻量范围、跨模态合并条件和正式报告所依据的代码来源；正式竞赛报告正文不保存在本仓库。

## 当前阅读顺序

1. [`001-正式多模态V2路径与分析契约.md`](001-正式多模态V2路径与分析契约.md)：V2 的路径、样本、行为、NIR、RGB、毫米波接入和合并总入口；2026-08-30 已按当前轻量 RGB 与外部毫米波来源更新。
2. [`002-NIR适配器provenance与merge-key契约.md`](002-NIR适配器provenance与merge-key契约.md)：NIR pupil-only 来源、字段与合并主键。
3. [`003-正式分析V2分支收口与历史分支处置_20260829.md`](003-正式分析V2分支收口与历史分支处置_20260829.md)：历史分支与当前正式分支的关系。
4. [`004-code-fix-ledger放行矩阵_20260829.md`](004-code-fix-ledger放行矩阵_20260829.md)：代码修复与证据仓库之间的内部验收映射。
5. [`005-当前分支角色与资产保全说明_20260829.md`](005-当前分支角色与资产保全说明_20260829.md)：当前分支职责及历史资产保全。
6. [`006-正式报告方法与结果权威来源_20260830.md`](006-正式报告方法与结果权威来源_20260830.md)：正式报告第 4.5 节和第 5 章写作时应采用的最新来源、术语转换和证据边界。
7. [`007-身份键与正式管线连续性联合审计_20260830.md`](007-身份键与正式管线连续性联合审计_20260830.md)：在最新代码 HEAD 上统一审计 `participant_key` / legacy `repeat_participant_id` / `participant_group_id` / NIR `analysis_group_token`，并记录 Behavior、NIR、RGB、merge 当前已确认的连续性缺陷与下一轮验收顺序。若旧文档在身份键或当前执行入口上与 007 冲突，以最新代码和 007 的审计结论为准，直至对应缺陷修复并再次更新合同。

## 当前分析职责

本仓库是 Behavior、NIR 和 RGB 正式下游分析的代码权威。RGB 当前范围已经收缩为整体运动与曝光变化分轨、必要的姿态确认/方向候选以及独立眨眼候选事件；AU、情绪、rPPG、PERCLOS、复杂 RGB 预测和正式多模态融合不属于当前 RGB 主线。

身份方面，`session_id` 始终是实验/采集场次键，不是参与者键；`participant_key` 是问卷/重复登记中的已核验匿名参与者来源；`participant_group_id` 是正式推断与 participant-disjoint prediction 的统一内部接口目标；旧 `repeat_participant_id` 与 NIR `analysis_group_token` 当前仍存在兼容用途，具体边界和未修复问题见 007。

毫米波信号处理与多模态增量/互补方法以 `greenboo26/focuswave-multimodal-attention-analysis@main` 的最新内容作为只读方法证据，但不在该外部仓库写入本项目正式报告或治理内容。报告、分析设计和证据整理统一维护在 `kyandi233-dev/FocusWave-Formal-Analysis@codex/code-fix-ledger`。

正式研究总体样本与阶段性已整理分析子集必须区分。具体结果只在真实分析产物形成后报告相应参与者数、实验记录数和有效试次/窗口数；代码通过、CI 通过和数据满足建模前提均不能替代科学结果。
