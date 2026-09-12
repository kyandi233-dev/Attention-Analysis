# 060 正式分析

更新日期：2026-09-12。

本目录维护 `codex/formal-analysis-v2-portable` 的正式下游分析说明。代码事实以当前分支源码、配置和对应开发 PR 为准；方法决策以 `kyandi233-dev/FocusWave-Formal-Analysis@codex/code-fix-ledger` 为唯一权威。历史文档继续用于 provenance（来源追踪），但与 1.16 系列后出决策冲突时不得覆盖当前方法。

## 当前 cohort、身份与 availability

当前正式配置已经统一到 **116 sessions（场次）、61 participant groups（参与者组）**。`configs/behavior_formal_v2.yaml` 与 `configs/nir_analysis_ready.yaml` 均声明该拓扑；问卷或单一模态缺失不得反向删除 governed cohort（治理队列）。`participant_group_id` 是正式推断、bootstrap（自助法）与 participant-disjoint prediction（参与者互斥预测）的统一内部统计键。

当前单模态 availability（可用性）与 governed cohort 分开管理：`011-当前116场输入输出与分析流程整理_20260830.md` 记录 149 场登记、116 场治理队列、61 个匿名参与者组、109 场 current-compatible NIR（当前兼容近红外）与 115 场 RGB（可见光视频）availability。模态缺失只能记录 `source_missing` / `structurally_invalid` / `not_estimable` 等状态，不能改写 participant identity（参与者身份）或 Behavior（行为）队列。

不要继续使用历史 `44/38/6` 或先前 `115/61/11` 作为当前代码事实。参与次数分布与重复组细节必须从当前 cohort manifest / repeat registry 重新审计，不写成永久常量。

## 当前方法入口：1.16 系列优先

Formal 当前优先读取：

1. `分析设计/1.16.1-监督学习心理意义、训练权重与多层评价修订_20260911.md`：Q1 预测的心理学解释、参与者等权训练/预处理/评价、特征级解释、D4-D7 时间报告及 M0–M7 设备/信息包比较。
2. `分析设计/1.16.2-瞳孔相关_眨眼联合清洗与探针前动态分析当前决策_20260912.md`：NIR（近红外）瞳孔、RGB（可见光视频）眨眼、联合清洗、线性/二次动态的当前科学决策。
3. `分析设计/1.16.3-瞳孔与眨眼测量审计及代码修改实施计划_20260912.md`：1.16.2 的代码实现与真实数据 measurement audit（测量审计）合同。

`1.15.x` 系列保留为方案演变和 A/B/C/D 初始实现来源。若其“唯一 NIR 基础信号”“旧 slope（斜率）”“统一覆盖率门槛”等描述与 1.16 冲突，以 1.16 为准。

## 当前监督学习代码状态

- **Task A / Draft PR #33**：D1 参与者等权训练、D8 参与者等权预处理、D9 participant-macro inner log loss（参与者宏平均内层对数损失）、D2/D3 参与者等权折外评价、D10 fixed-OOF participant-cluster bootstrap（固定折外参与者簇自助法）、D11 LOSO→5-fold GroupKFold、feature registry（特征登记）/具体特征比较框架和 D4-D7 probe trajectory（探针轨迹）/session discrimination（场内区分）报告层均已有合成合同实现。当前 Task A HEAD `20439a467b495e8a9a8df5cbafa60099cc630667`，专用 Task A CI #86 PASS；真实结果仍未运行。
- **Task B / Draft PR #35**：comparison-specific analysis set（按比较分析集合）、质量/缺失状态和 prediction archive（预测归档）基础实现已存在；当前监督学习集成修复由 #41、#42、#48 接续。
- **#41 / Draft PR #47**：Behavior 权威 probe-key mapper（探针键映射器）已实现并通过专用 Task B CI #12；最终接入 `alignment.py` 等待 #44 冻结当前 NIR 输出 schema，避免和正在推进的 NIR 测量链重复造键。
- **#42 / Draft PR #46**：prediction archive 已补 Behavior 权威 Q1 回联、期望 `analysis_set × model × outcome` 全集覆盖、空档案/整模型缺失拒绝与 membership 检查；当前正式 Task-A 归档强制要求 `run_id`、`feature_set_id`、`membership_type`、`model_failed`、`failure_reason`，成功/失败行分别执行相应完整性检查。当前 HEAD `e1bc7cf7cfa38a5b57ce8861d180c7608f674d83`，专用 Task B CI #18 PASS。
- **#48 / Draft PR #49**：已增加 comparison-specific supervised input materialization（按比较监督学习输入物化），将 Task-B `analysis_set_id + membership_type + probe_feature_status.value` 生成 Task A 可直接消费的一行一 probe 宽表；complete 与 missing-aware 规则分开，并已补 CSV 字符串布尔严格解析、Behavior 权威 metadata 冲突检查、Task-A 必需 `probe_event_id`/Q1 检查和 `comparison_models` JSON 校验。当前 HEAD `c21054d6d3f30a92e3c1c5efdff69b04466f2b20`，专用 Task B CI #17 PASS。
- **Task C**：行为监督学习接口与 30 s probe（探针）候选输出已完成；原 Task C issue 已关闭。具体哪些 Behavior 特征最终构成监督学习参照集合 `B` 仍需按上游科学合同冻结。
- **Task D / PR #39**：保留为 1.15.7 NIR 接口 draft（草稿）基线。其实现仍以 `pupil_geom_mean_diameter` 与旧 `robust_binned_slope_per_sec` 为中心，不能再视为最终 1.16 NIR 科学合同。
- **Issue #44 / Draft PR #45**：堆叠在 PR #39 上，实现 pupil×blink（瞳孔×眨眼）measurement audit，包括 `R_seg,hard`、RGB blink mask（眨眼掩码）、candidate buffer（候选缓冲）、probe-locked fixed bins（探针锁定固定时间箱）、linear slope（线性斜率）与 quadratic curvature（二次曲率）。其真实测量审计与最终参数冻结由当前 NIR 工作线继续负责。

## 当前 active issues

当前监督学习/测量集成相关开放问题包括：

| Issue | 当前职责 | 执行关系 |
|---|---|---|
| #41 | A/B/D probe 键统一 | mapper 已实现；最终 NIR 接线等待 #44 schema 冻结 |
| #42 | prediction archive 回联权威 Q1 标签与期望全集 | 核心合同已实现，Task B CI #18 PASS；最终随 A/B 集成验收 |
| #48 | Task-B analysis set / audited values → Task-A 输入宽表物化 | 核心合同已实现，Task B CI #17 PASS；不负责科学特征资格 |
| #44 | 瞳孔×眨眼真实测量审计及 NIR 动态接口 | 与监督学习核心并行；参数冻结后回 Formal |
| #40 | B 消费 NIR 显式可估计状态 | 核心缺陷成立；最终字段映射等待 #44 冻结后一次性接线 |
| #43 | 参与者等权正式评价、特征级解释、M0–M7 与时间报告 | 当前不依赖上游的通用核心已基本实现；正式结果仍受 #41/#42/#48、#40、feature freeze 与最终集成约束 |

旧 #19/#20/#21/#22/#30/#32/#34/#36/#38 已按“历史调研 / 已取代 / 已完成”退出 active 队列；关闭不删除其代码证据、讨论和历史意义。

## 推荐执行顺序

当前监督学习代码线已进入等待上游科学合同的阶段：

- #42：保持已通过 CI 的 prediction archive 权威 Q1、membership、formal audit fields 与完整覆盖合同；
- #48：保持已通过 CI 的 comparison-specific input materialization；
- #43：维持已经通过专用 CI 的 participant-equal 训练/评价、特征比较和 D4-D7 报告核心，不继续无依据扩模型。

与另一条测量工作线的汇合顺序为：

- #44 完成真实 pupil×blink measurement audit，并回 Formal 冻结最终 NIR feature/status contract（特征/状态合同）；
- #41 按最终 NIR schema 接入 Behavior 权威 probe-key mapper；
- #40 按最终 NIR status 字段接 B，不为旧 slope/旧字段另做临时路线；
- 冻结实际 Behavior reference `B`、NIR、RGB、mmWave 正式 feature registry；
- 将 Task A、Task B、#41、#42、#48 与最终单模态接口做统一 schema integration test（数据结构集成测试）和小规模真实 smoke test（冒烟测试）；
- 只有上述接口和科学特征均冻结后，才运行正式 FocusWave 监督学习并形成性能结论。

问卷当前只注册了 `questionnaire_derived_data` 路径，尚无冻结的机器可读字段合同；因此 Task A 只保留 Q1/Q2/预测时间轨迹，后续按 session 或预先规定的粗阶段做探索性问卷对照，不在当前监督学习核心中硬编码问卷题目。

## 历史与基础文档

1. [`001-正式多模态V2路径与分析契约.md`](001-正式多模态V2路径与分析契约.md)：V2 路径与基础数据合同。
2. [`002-NIR适配器provenance与merge-key契约.md`](002-NIR适配器provenance与merge-key契约.md)：NIR pupil-only（仅瞳孔）来源、字段与合并主键。
3. [`003-正式分析V2分支收口与历史分支处置_20260829.md`](003-正式分析V2分支收口与历史分支处置_20260829.md)：历史分支与正式分支关系。
4. [`006-正式报告方法与结果权威来源_20260830.md`](006-正式报告方法与结果权威来源_20260830.md)：报告代码来源和证据边界。
5. [`009-正式管线修复后完整复审_20260830.md`](009-正式管线修复后完整复审_20260830.md)：前一阶段完整管线复审，保留为历史基线；当前状态需再结合本 README、active issues 与 1.16 系列。
6. [`010-新电脑迁移与常见报错检查表_20260830.md`](010-新电脑迁移与常见报错检查表_20260830.md)：迁移与执行 preflight（预检）。
7. [`011-当前116场输入输出与分析流程整理_20260830.md`](011-当前116场输入输出与分析流程整理_20260830.md)：当前 116 场治理队列及模态 availability 的资产入口。

任何新 AI 或 Codex 开始正式分析代码工作前，应先读本 README，再读对应 active issue 与 Formal 1.16 文档；不得仅按已关闭的 1.15.7 issue 继续开发。