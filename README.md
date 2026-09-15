# FocusWave Attention-Analysis｜正式科学分析与特征生产

> **状态：CURRENT（当前入口）**  
> 最后核验：2026-09-15  
> 当前正式科学分析权威分支：`codex/formal-analysis-v2-portable`  
> 国赛提交快照：`national-competition-submission-20260913` → `470529b7e048d0caf4f55d2ffec680e3ce2e4c26`

本仓库负责 FocusWave 的可执行分析代码：把正式实验中的任务记录与多设备测量整理为 Behavior（行为）、Ocular（眼部）、Movement（动作）和 Cardiopulmonary（心肺）四类科学信息，完成单模态科学分析、特征交接、参与者互斥监督学习、多模态增量比较和设备组合评价。NIR（近红外）、RGB（可见光视频）与 mmWave（毫米波）是设备/数据来源，不与科学模态混用。

方法裁决、正式结果总账和国赛报告以 `kyandi233-dev/FocusWave-Formal-Analysis@main` 为权威；本仓库回答“代码实际怎样实现”。

## 1. 当前研究结构

```text
SART / 思维探针 ───────────────→ Behavior（行为）
NIR 瞳孔 ───────┐
RGB 眨眼辅助 ───┴─────────────→ Ocular（眼部）
RGB 身体运动 / 姿态 ──────────→ Movement（动作）
mmWave 心率 / 呼吸率估计 ─────→ Cardiopulmonary（心肺）
                                      ↓
单模态解释性分析与质量控制
                                      ↓
冻结特征登记与共同分析集合
                                      ↓
participant-disjoint supervised learning
（参与者互斥监督学习）
                                      ↓
模态增量、条件价值与设备组合评价
```

Q1（注意内容自我报告）是当前监督学习要预测的变量；Q2（困倦/清醒报告）用于解释性分析，不作为首轮 Q1 预测输入。Q1 二分类为主要监督学习任务，四分类为后续扩展分析。

## 2. 当前正式样本口径

当前 governed cohort（治理队列）为 **116 sessions（场次）、61 participant groups（参与者组）、2,320 个 Behavior 权威 probes（思维探针）**。`participant_group_id` 是重复测量推断、bootstrap（自助法）和 participant-disjoint validation（参与者互斥验证）的统一参与者键。

模态 availability（可用性）与 cohort membership（队列成员资格）必须分开。NIR、RGB 或 mmWave 缺失只影响对应分析集合，不得反向删除 Behavior 场次或改变参与者身份。历史 `44/38/6`、`115/61/11` 以及两台机器各自挂载的数据量只保留为当时工程运行记录，不再作为当前总体样本口径。

## 3. 当前正式分析状态

截至本次核验，仓库已经越过“只做摄像头/瞳孔提取”和“等待正式监督学习”的阶段：

- Behavior、Ocular、Movement 已形成冻结后的科学输出与监督学习特征交接；
- Cardiopulmonary 已在 mmWave producer（生产端）来源追踪和 pre-probe time-legality（探针前时间合法性）闭环后正式进入比较，但心率/呼吸率仍属于支持性指标，不能解释为已经完成独立生理效度验证，HRV（心率变异性）继续阻塞；
- Q1 二分类 participant-disjoint LOSO（参与者互斥留一参与者验证）、概率诊断、多模态增量和设备组合比较已经实际运行；
- Q1 四分类分析、汇总和多分类概率诊断代码已经进入当前正式分支；其结果需要按 Formal 结果总账和第 5 章的解释边界报告，不能仅凭模型已运行就称为可靠四分类识别。

正式结果数字不要从本 README 复制。当前结果以 `FocusWave-Formal-Analysis/main/国赛报告/完整结果/` 和 `国赛报告/章节草稿/5.*` 为准。

## 4. 当前仅保留的长期分支

| 分支 | 角色 | 是否用于当前科学结论 |
|---|---|---|
| `codex/formal-analysis-v2-portable` | **正式科学分析权威线**：单模态、特征登记、监督学习、结果生成 | **是** |
| `amd-DirectML` | NIR AMD / DirectML producer（生产端） | 只负责生产来源 |
| `nvidia-cuda-v8` | NIR NVIDIA / CUDA producer（生产端） | 只负责生产来源 |
| `rgb-amd` | RGB AMD producer（生产端） | 只负责生产来源 |
| `rgb-nvidia` | RGB NVIDIA producer（生产端） | 只负责生产来源 |

producer 分支可以保留硬件环境、批处理和恢复运行说明，但不能覆盖正式分析分支已经冻结的科学定义。旧开发分支已在清理后退出长期入口；需要追溯时使用 Git 历史、PR（拉取请求）和 archive tag（归档标签）。

## 5. 第一次进入仓库应读什么

按任务选择入口，不再从某一台机器的摄像头运行手册开始：

| 目的 | 当前入口 |
|---|---|
| 理解正式分析全貌 | [`docs/060-formal-analysis/README.md`](docs/060-formal-analysis/README.md) |
| Behavior（行为） | [`docs/030-behavior/README.md`](docs/030-behavior/README.md) |
| NIR / Ocular（近红外 / 眼部） | [`docs/020-nir/README.md`](docs/020-nir/README.md) |
| RGB / Movement（可见光 / 动作） | [`docs/040-rgb/README.md`](docs/040-rgb/README.md) |
| 当前可执行脚本 | [`scripts/README.md`](scripts/README.md) |
| NIR 生产端 runtime（运行时） | `runtime/nir-formal/`；按实际硬件切对应 producer 分支 |
| 方法、结果与报告权威 | `kyandi233-dev/FocusWave-Formal-Analysis@main` |

## 6. 当前正式脚本层级

正式科学分析主要包括：

```text
单模态科学输出
├─ build_behavior_science_output.py
├─ nir_pupil_blink_measurement_audit.py
├─ build_ocular_science_output_frozen.py
├─ run_ocular_postfreeze_analysis.py
└─ build_movement_science_output.py

心肺接入
├─ build_m1_cardiopulmonary_taskb_source.py
└─ promote_cardiopulmonary_registry.py

监督学习
├─ validate_supervised_feature_registry.py
├─ materialize_supervised_input.py
├─ build_supervised_comparison_sets.py
├─ supervised_learning_analysis.py
├─ build_probability_diagnostics.py
├─ supervised_learning_analysis_4class.py
├─ summarise_four_class_runs.py
└─ build_probability_diagnostics_multiclass.py
```

完整分类和执行边界见 [`scripts/README.md`](scripts/README.md)。

## 7. 结果、代码与历史材料的边界

- **CURRENT（当前）** README：只描述现在有效的入口、科学角色和执行状态。
- **PRODUCER（生产端）** README：可以详细记录 GPU（图形处理器）、环境、批量运行、恢复和硬件差异，但其职责止于形成可追溯测量产物。
- **HISTORICAL（历史）** 文档：保留当时原文用于 provenance（来源追踪）；即使其中写有旧样本数、旧分支或当时未完成事项，也不得反向覆盖当前状态。

如果当前 README 与更早的日期型工作记录冲突，先核对当前代码、配置和 `FocusWave-Formal-Analysis@main` 的后出方法裁决，不按旧工程阶段继续执行。