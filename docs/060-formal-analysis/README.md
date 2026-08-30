# 060 正式分析

本目录维护 `codex/formal-analysis-v2-portable` 的正式下游分析说明。这里记录分析入口、数据与身份合同、模态 availability、NIR pupil-only 适配、RGB 轻量范围、跨模态合并条件、新电脑迁移和正式报告所依据的代码来源；正式竞赛报告正文不保存在本仓库。

## 当前阅读顺序

1. [`001-正式多模态V2路径与分析契约.md`](001-正式多模态V2路径与分析契约.md)：V2 的路径、样本、行为、NIR、RGB、毫米波接入和合并总入口。
2. [`002-NIR适配器provenance与merge-key契约.md`](002-NIR适配器provenance与merge-key契约.md)：NIR pupil-only 来源、字段与合并主键。
3. [`003-正式分析V2分支收口与历史分支处置_20260829.md`](003-正式分析V2分支收口与历史分支处置_20260829.md)：历史分支与当前正式分支的关系。
4. [`004-code-fix-ledger放行矩阵_20260829.md`](004-code-fix-ledger放行矩阵_20260829.md)：代码修复与证据仓库之间的内部验收映射。
5. [`005-当前分支角色与资产保全说明_20260829.md`](005-当前分支角色与资产保全说明_20260829.md)：当前分支职责及历史资产保全。
6. [`006-正式报告方法与结果权威来源_20260830.md`](006-正式报告方法与结果权威来源_20260830.md)：正式报告第 4.5 节和第 5 章采用的来源、术语转换和证据边界。
7. [`007-身份键与正式管线连续性联合审计_20260830.md`](007-身份键与正式管线连续性联合审计_20260830.md)：前一轮参与者身份接口和 Behavior、NIR、RGB、merge 连续性审计；保留为修复前/修复中 provenance。
8. [`008-正式报告方法章叙事与颗粒度约束_20260830.md`](008-正式报告方法章叙事与颗粒度约束_20260830.md)：规定代码事实转写为报告时的整章逻辑、预实验结果压缩、Block 术语、设备参数表、算法名称与实现参数分层，以及伦理信息边界。
9. [`009-正式管线修复后完整复审_20260830.md`](009-正式管线修复后完整复审_20260830.md)：**当前修复后复审主文档**；覆盖 Behavior、NIR、RGB、mmWave 接口、模态缺失、legacy/deferred、路径与环境。若 007 与当前实现冲突，以 009 + 当前代码为准。
10. [`010-新电脑迁移与常见报错检查表_20260830.md`](010-新电脑迁移与常见报错检查表_20260830.md)：换电脑/硬盘/Conda 环境时的执行型 preflight、常见路径/接口/Parquet/manifest/Windows 报错清单。
11. [`011-当前116场输入输出与分析流程整理_20260830.md`](011-当前116场输入输出与分析流程整理_20260830.md)：当前机器的 149 场登记、116 场身份映射、61 个匿名参与者组、109 场 current-compatible NIR 与 115 场 RGB availability、配置使用边界、输入输出路径和全流程执行顺序。

仓库根目录的 `ANALYSIS_SETUP_FIRST.md` 是新电脑或新终端的第一入口；010 是其更细的检查表。

## 当前分析职责

本仓库是 Behavior、NIR 和 RGB 正式下游分析的代码权威。RGB 当前范围已经收缩为整体运动与曝光变化分轨、必要的姿态确认/方向候选以及独立眨眼候选事件；AU、情绪、rPPG、PERCLOS、复杂 RGB 预测和正式多模态融合不属于当前 RGB 主线。

身份方面，`session_id` 是实验/采集场次键，`participant_key` 是当前已核验的重复参与者来源，`participant_group_id` 是正式推断与参与者独立预测的内部统一接口；旧 `repeat_participant_id` 与 NIR `analysis_group_token` 只保留兼容边界。

**模态 availability 与 participant identity 是正交的。** 某个已知 session/participant 可能没有录到 RGB、mmWave 或 NIR；这种情况只在对应模态记录 `source_missing` / `structurally_invalid` / `not_estimable`，不能因此从 Behavior 或其他模态删除该 session，也不能改变 participant 分组。未来显式 paired 多模态分析可以建立 common-available subset，但必须同时报告原 governed cohort 和各模态覆盖率。

NIR 当前主线只把 pupil geometry 作为正式生理信号；历史 PIR/iris geometry 不进入正式 endpoint。producer 已保存的 `fullclass_ocular_aperture_ratio_median/p90` 可作为 eye-opening QC candidate 保留，但不是 EAR、blink 或 PERCLOS，也不自动成为正式 endpoint。

毫米波信号处理与正式生产逻辑以 `greenboo26/focuswave-multimodal-attention-analysis` 的当前正式路线作为外部权威，不在本仓库复制 producer engine。报告、分析设计和证据整理统一维护在 `kyandi233-dev/FocusWave-Formal-Analysis@codex/code-fix-ledger`。

正式研究总体样本、当前 governed queue、某单一模态可用子集和显式 common-available paired subset 必须分别报告。具体科学结果只在真实分析产物形成后报告相应参与者数、实验记录数和有效试次/窗口数；代码通过、CI 通过和数据满足建模前提均不能替代科学结果。
