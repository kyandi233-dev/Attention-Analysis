# 060 正式分析｜当前科学分析入口

> **状态：CURRENT（当前）**  
> 最后核验：2026-09-15  
> 代码权威：`Attention-Analysis@codex/formal-analysis-v2-portable`  
> 方法、结果与报告权威：`FocusWave-Formal-Analysis@main`

本目录是 Attention-Analysis 当前正式下游分析的文档入口。旧 1.15、早期 1.16、#40–#44 等 Issue（问题单）记录继续用于 provenance（来源追踪），但它们描述的“待冻结”“待运行”“active issue（活动问题）”阶段已经结束，不能作为当前执行队列。

## 1. 当前研究对象与身份合同

当前 governed cohort（治理队列）为 **116 sessions（场次）、61 participant groups（参与者组）、2,320 个 Behavior 权威 probes（思维探针）**。`participant_group_id` 是重复测量推断、participant-cluster bootstrap（参与者簇自助法）和 participant-disjoint validation（参与者互斥验证）的统一参与者键。

模态 availability（可用性）独立于 cohort membership（队列成员资格）。NIR（近红外）、RGB（可见光视频）或 mmWave（毫米波）缺失只影响对应分析集合，不得删除治理队列中的 Behavior 场次，也不得改变参与者身份。

## 2. 科学模态与设备来源

```text
Behavior（行为）
  ← SART / 思维探针

Ocular（眼部）
  ← NIR 瞳孔
  ← RGB 眨眼作为眼部特征与瞳孔伪迹辅助信息

Movement（动作）
  ← RGB 身体运动 / 姿态

Cardiopulmonary（心肺）
  ← mmWave 心率 / 呼吸率估计
```

RGB、NIR、mmWave 是设备/来源命名空间；Behavior、Ocular、Movement、Cardiopulmonary 才是正式科学信息类别。设备依赖关系由 feature registry（特征登记表）和 provenance（来源追踪）记录。

## 3. 当前正式流程已经完成到哪里

当前正式代码链已经覆盖：

```text
治理队列 / 身份键
        ↓
单模态测量与科学输出
        ↓
Behavior / Ocular / Movement handoff（特征交接）
        ↓
mmWave Cardiopulmonary 来源与时间合法性闭环
        ↓
冻结 feature registry（特征登记表）
        ↓
比较特异 analysis sets（分析集合）
        ↓
Q1 二分类 participant-disjoint LOSO
（参与者互斥留一参与者验证）
        ↓
概率诊断 / 增量 / 条件价值 / 设备组合
        ↓
Q1 四分类扩展与多分类诊断
        ↓
Formal 结果总账与国赛报告
```

因此，不再使用“只有接口 smoke（冒烟测试），尚未进入正式监督学习”“等待 #40/#43/#44 完成后才能训练”等旧状态描述。

## 4. 当前主要代码入口

### 单模态科学输出

| 科学信息 | 主要入口 | 当前角色 |
|---|---|---|
| Behavior | `scripts/build_behavior_science_output.py` | 构建正式行为科学输出与交接 |
| Ocular | `scripts/nir_pupil_blink_measurement_audit.py` | 瞳孔×眨眼测量审计 |
| Ocular | `scripts/nir_ocular_g1_freeze_support.py` | G1 冻结支持证据 |
| Ocular | `scripts/build_ocular_science_output_frozen.py` | 冻结后眼部科学输出 |
| Ocular | `scripts/run_ocular_postfreeze_analysis.py` | 冻结后解释性分析 |
| Movement | `scripts/build_movement_science_output.py` | 动作科学输出与交接 |

### Cardiopulmonary（心肺）正式接入

| 入口 | 作用 |
|---|---|
| `scripts/build_m1_cardiopulmonary_taskb_source.py` | 从满足 M1 合同的 mmWave 来源构建正式下游输入 |
| `scripts/promote_cardiopulmonary_registry.py` | 将符合合同的心肺特征登记到正式 feature registry |

Cardiopulmonary 已获得正式比较资格，但只关闭 producer provenance（生产端来源追踪）与 pre-probe time-legality（探针前时间合法性）门。毫米波估计心率、呼吸率仍为支持性生理信息，HRV（心率变异性）继续阻塞。

### 监督学习与诊断

| 入口 | 作用 |
|---|---|
| `scripts/validate_supervised_feature_registry.py` | 校验冻结特征登记与依赖 |
| `scripts/materialize_supervised_input.py` | 物化监督学习输入 |
| `scripts/build_supervised_comparison_sets.py` | 构建比较特异分析集合 |
| `scripts/supervised_learning_analysis.py` | Q1 二分类正式监督学习 |
| `scripts/build_probability_diagnostics.py` | 二分类折外概率诊断 |
| `scripts/supervised_learning_analysis_4class.py` | Q1 四分类扩展分析 |
| `scripts/summarise_four_class_runs.py` | 四分类运行汇总 |
| `scripts/build_probability_diagnostics_multiclass.py` | 多分类折外概率与类别诊断 |
| `scripts/verify_report_number_consistency.py` | 结果与报告数字一致性核验 |

完整脚本索引见 [`../../scripts/README.md`](../../scripts/README.md)。

## 5. 当前监督学习解释合同

Q1 是 attention-content self-report（注意内容自我报告），不是潜在注意状态的“真值”。当前主要预测任务将 Q1 的任务聚焦与其余三类报告进行二分类；四分类作为扩展分析。Q2（困倦/清醒报告）不作为首轮 Q1 预测输入。

正式外层验证按 `participant_group_id` 做 participant-disjoint LOSO（参与者互斥留一参与者验证）；训练内部的模型/参数选择按参与者分组完成。预处理、插补、标准化与训练不得读取外层测试参与者的信息。主要总体评价采用参与者等权汇总，并以固定折外预测进行 participant-cluster bootstrap（参与者簇自助法）不确定性评估。

不同模态往往使用不同可用样本，因此单独模型的原始损失或 AUROC（受试者工作特征曲线下面积）不能跨不同比较集合直接排名。真正的“增加了多少信息”使用同一分析集合内的成对增量或 `Full-x → Full` 条件价值比较。

## 6. 正式结果去哪里看

本目录说明代码和数据合同，不作为最终数字抄录来源。正式结果优先读取：

```text
FocusWave-Formal-Analysis@main
└─ 国赛报告/
   ├─ 完整结果/
   └─ 章节草稿/5.*
```

当前第 5 章已经包含 Behavior、Ocular、Movement、Cardiopulmonary，以及 Q1 跨参与者监督学习、多模态增量和设备组合结果。四分类扩展的解释必须同时查看其概率/类别诊断，不能把“代码已运行”写成“已实现可靠四分类识别”。

## 7. 历史文档如何使用

以下内容继续保留，但默认视为历史层：

- `001`–`011` 等早期正式管线与迁移说明；
- 1.15.x / 早期 1.16 形成过程；
- 已关闭 Issue 与 PR（拉取请求）讨论；
- 旧 NIR `10_analysis_ready → 11_analysis_tables → 12_pipeline_validation` 验证阶段；
- 已删除开发分支的分支名和当时状态。

历史材料可以回答“为什么后来采用当前方案”，不能覆盖当前代码、配置、Formal 后出裁决和真实正式结果。

新 AI、Codex 或本地执行者接手正式分析时，应先读本 README，再按任务读 `FocusWave-Formal-Analysis@main` 对应方法与结果文件；不要从已关闭的旧 Issue 直接继续开发。