# Scripts｜当前正式入口索引

> **状态：CURRENT（当前）**  
> 最后核验：2026-09-15  
> 当前科学分析分支：`codex/formal-analysis-v2-portable`

`scripts/` 同时包含当前正式科学分析、producer（生产端）/工程工具、敏感性与历史复现入口。**文件存在不等于它仍是当前正式主入口。** 本页只负责导航；具体方法以 `FocusWave-Formal-Analysis@main` 为准，具体参数以当前 config（配置）、代码和运行证据为准。

## 1. 当前正式科学输出

| 脚本 | 科学信息 | 当前作用 |
|---|---|---|
| `build_behavior_science_output.py` | Behavior（行为） | 物化正式行为科学输出与 feature handoff（特征交接） |
| `nir_pupil_blink_measurement_audit.py` | Ocular（眼部） | NIR（近红外）瞳孔 × RGB（可见光）眨眼 G1 测量审计 |
| `nir_ocular_g1_freeze_support.py` | Ocular | G1 表示、时间支持、同步等冻结支持证据 |
| `build_ocular_science_output_frozen.py` | Ocular | 物化冻结后的眼部 probe（探针）特征、coverage（覆盖）和 handoff |
| `run_ocular_postfreeze_analysis.py` | Ocular | 冻结后 Q1/Q2、任务进程、近期行为等解释性分析 |
| `build_movement_science_output.py` | Movement（动作） | 从 RGB 5.5 资产物化正式动作科学输出与 handoff |
| `build_m1_cardiopulmonary_taskb_source.py` | Cardiopulmonary（心肺） | 从满足 M1 合同的 mmWave（毫米波）来源构建正式下游输入 |
| `promote_cardiopulmonary_registry.py` | Cardiopulmonary | 将合格心肺特征登记到正式 feature registry（特征登记表） |

单模态正式结果不是通过重新跑全部原始视频“现场生成”的。已有 producer 资产时，优先消费冻结后的正式表和 manifest（清单）；是否需要重跑 producer 由来源追踪和 QC（质量控制）决定。

## 2. 当前监督学习主线

| 脚本 | 当前作用 |
|---|---|
| `validate_supervised_feature_registry.py` | 校验科学模态、设备依赖、时间合法性、特征角色和注册表合同 |
| `materialize_supervised_input.py` | 按治理身份和冻结特征物化监督学习输入 |
| `build_supervised_comparison_sets.py` | 构建比较特异 analysis sets（分析集合），避免用无关缺失缩小比较样本 |
| `supervised_learning_analysis.py` | Q1 二分类 participant-disjoint supervised learning（参与者互斥监督学习） |
| `build_probability_diagnostics.py` | 二分类 OOF（折外）概率、Brier（布里尔分数）、校准等诊断 |
| `build_window_sensitivity_sets.py` | 构建行为窗口长度敏感性分析集合 |
| `summarise_window_sensitivity.py` | 汇总窗口敏感性结果 |
| `build_supplementary_cardiopulmonary_sets.py` | 心肺补充/比较集合构建；历史补充角色需结合当前正式 promoter 状态解读 |

Q1 二分类是当前主要预测任务；外层评价按 `participant_group_id` 做 LOSO（留一参与者），不能让同一参与者同时进入训练与测试。

## 3. Q1 四分类扩展

| 脚本 | 当前作用 |
|---|---|
| `supervised_learning_analysis_4class.py` | Q1 四分类扩展分析 |
| `verify_four_class_contract.py` | 校验四分类标签、分析集合、折和输出合同 |
| `verify_four_class_reuses_binary_sets.py` | 核验四分类与既定比较集合的复用关系 |
| `summarise_four_class_runs.py` | 汇总四分类各路线运行结果 |
| `build_probability_diagnostics_multiclass.py` | 多分类 OOF 概率、Brier、类别/校准诊断 |
| `verify_multiclass_probability_diagnostics.py` | 核验多分类概率诊断产物 |

四分类脚本进入正式分支只表示分析已经具备正式实现与结果链。科学结论必须同时读取类别分布、balanced accuracy（平衡准确率）、macro F1（宏平均 F1）、macro AUROC（宏平均曲线下面积）、概率损失和预测偏置，不能只报告单一损失改善。

## 4. 正式一致性与交付核验

| 脚本 | 作用 |
|---|---|
| `validate_supervised_prediction_archive.py` | 校验预测归档与权威标签/身份全集 |
| `verify_sensitivity_runs_contract.py` | 校验敏感性运行合同 |
| `verify_report_number_consistency.py` | 核对结果输出与报告使用数字的一致性 |
| `run_p5_interface_smoke.py` | 历史 P5 三模态接口 smoke（冒烟测试）；当前主要作为接口追溯证据 |
| `audit_formal_modality_availability.py` | 审计治理队列与各模态 availability（可用性） |
| `build_formal_local_manifests.py` | 构建本地正式资产 manifest（清单） |

## 5. Behavior 工程/复现入口

| 脚本 | 当前角色 |
|---|---|
| `sart_formal_analysis.py` | 正式 Behavior 底层分析入口；已有正式结果时不因 README 更新而无理由重跑 |
| `sart_formal_analysis_v2.py` | 后续正式流程相关入口；实际使用前核对当前 config 和调用方 |
| `sart_formal_redraw.py` | 从正式表重绘 Behavior 图件 |
| `behavior_supervised_interface.py` | Behavior 监督学习接口历史/基础实现 |
| `sart_bbb_v3_0_analysis.py` | **HISTORICAL（历史）** v3.0 BBB 可复现入口，不属于当前 BB 正式结果 |

当前 Behavior 科学结果入口优先看 `build_behavior_science_output.py` 和 `docs/030-behavior/README.md`，不要从历史 BBB 入口反推当前方法。

## 6. NIR producer / 旧下游验证入口

以下脚本仍可用于 NIR 工程、诊断和历史复现：

```text
nir_materialize_analysis_ready.py
nir_build_analysis_tables.py
nir_formal_pipeline.py
nir_pipeline_validation.py
nir_behavior_alignment.py
nir_behavior_cohort_qc.py
nir_pir_*.py
nir_validate_pupil_formal.py
```

其中旧 `10_analysis_ready → 11_analysis_tables → 12_pipeline_validation` 是历史 NIR/PIR 阶段形成的管线，不再是当前 Ocular 正式科学分析的状态机。当前 Ocular 以 G1 measurement audit（测量审计）、冻结后的 science output（科学输出）和 post-freeze analysis（冻结后分析）为准，见 `docs/020-nir/README.md`。

真正需要重新生产 NIR 测量时，入口在 `runtime/nir-formal/`，并按硬件切 `amd-DirectML` 或 `nvidia-cuda-v8`；不要在正式科学分支中把 producer 运行和结果分析混成一步。

## 7. RGB producer / Movement 入口

当前主要相关脚本：

```text
rgb_55_analysis.py                 # RGB 5.5 正式分析表/模型资产
build_movement_science_output.py   # Movement 科学输出
rgb_formal_downstream.py
rgb_formal_motion_pose.py
rgb_formal_report.py
rgb_analysis.py                    # producer/工程入口
run_rgb_formal_subject.ps1         # 单场正式 producer 辅助
face_*                             # Face/Pose/眼睑等工程与验证资产
```

`face_*` 大量脚本保留完整工程历史，但它们不是当前报告的“科学模态”。RGB 眨眼进入 Ocular，身体运动/姿态进入 Movement，曝光和全局画面量进入 QC。需要重新生产视频级资产时切 `rgb-amd` 或 `rgb-nvidia`。

## 8. 多模态旧入口与当前监督学习的关系

`formal_multimodal_analysis.py`、`multimodal_fusion_analysis.py`、`multimodal_pupil_audit.py`、`multimodal_pupil_correction_pilot.py` 等保存早期多模态整合、瞳孔校正与验证代码。它们仍可作为 provenance（来源追踪）、敏感性或工程诊断使用，但当前正式 Q1 预测和增量比较应走冻结 feature registry、comparison sets（比较集合）和 `supervised_learning_analysis*.py` 主线。

不要因为旧脚本名包含 `formal` 或 `multimodal` 就默认它是当前最高权威入口。

## 9. 当前执行前的最低检查

在任何正式重跑之前至少确认：

```powershell
git status --short --branch
git branch --show-current
git log -1 --oneline
```

科学分析应处于：

```text
codex/formal-analysis-v2-portable
```

然后核对本轮任务对应 config、paths config（路径配置）、输入 manifest 和 Formal 当前方法裁决。README 中不再维护固定本机盘符作为跨机器“真值”；本地绝对路径只属于对应运行记录或本机配置。

## 10. 结果从哪里读取

正式科学数字以：

```text
FocusWave-Formal-Analysis@main
└─ 国赛报告/
   ├─ 完整结果/
   └─ 章节草稿/5.*
```

为当前报告入口。脚本 README 只说明“怎样找到当前代码”，不替代结果总账、真实输出和运行 provenance。