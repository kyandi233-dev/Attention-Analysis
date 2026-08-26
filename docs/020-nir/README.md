# NIR

## 2026-08-27 当前 NIR 状态

NIR 的**工程管线结构**已经推进到下游分析层，但用户已确认：**当前得到的 NIR/PIR 数值本身是错误的，不能用于科学解释。** 因此现在停止继续 44 人 `11_analysis_tables` 批量构表，也禁止进入 `20_formal_statistics`。

当前实际状态：

```text
runtime/nir-formal / current extracted NIR
        ↓
10_analysis_ready                 # 工程结构已物化，但当前 PIR 数值不可作科学解释
        ↓
11_analysis_tables                # 仅已有部分 completed subject，用于验证构表接口
        ↓
12_pipeline_validation            # 当前允许：假定 PIR 正确，调通分析/模型/绘图代码
        ↓
20_formal_statistics              # 当前禁止；待 NIR 数值修正后重跑上游再进入
```

当前 AMD 44 人仍是 exploratory/development cohort；当前错误 PIR 的任何方向、差异、相关、p 值都不得作为 exploratory scientific result。

## 当前最重要的方法入口

| 文档 | 作用 | 当前地位 |
|---|---|---|
| [022-2026-08-25-NIR正式分析设计与待验证项.md](022-2026-08-25-NIR正式分析设计与待验证项.md) | 总体科学问题、Behavior/Probe/time-on-task/统计路线 | 总体科学设计 |
| [212-2026-08-27-NIR数据清洗逻辑与正式分析纳入规则.md](212-2026-08-27-NIR数据清洗逻辑与正式分析纳入规则.md) | primary/strict validity、左右眼、baseline、coverage 的现行规则 | 清洗设计基线；待正确 PIR 后复用/复核 |
| [213-2026-08-27-NIR-analysis-ready数据契约与物化规范.md](213-2026-08-27-NIR-analysis-ready数据契约与物化规范.md) | production → `10_analysis_ready` | 数据层工程契约 |
| [214-2026-08-27-NIR正式下游分析表数据契约.md](214-2026-08-27-NIR正式下游分析表数据契约.md) | `10_analysis_ready` → trial / probe / time-on-task | 下游构表工程契约 |
| [215-2026-08-27-NIR正式下游分析管线运行手册.md](215-2026-08-27-NIR正式下游分析管线运行手册.md) | 新终端、测试、构表与续跑规则 | 下游运行手册 |
| [217-2026-08-27-NIR错误值条件下下游分析管线验证方案.md](217-2026-08-27-NIR错误值条件下下游分析管线验证方案.md) | 利用当前少量 completed subject 假定 PIR 正确，验证分析/模型/专业代码绘图 | **当前实际工作入口** |
| [027-44人全量分析数据边界与资料清单.md](027-44人全量分析数据边界与资料清单.md) | 当前 44 人数据边界 | exploratory 数据边界 |
| [028-2026-08-26-NIR-cohort44分析实施计划与进度.md](028-2026-08-26-NIR-cohort44分析实施计划与进度.md) | cohort 分析阶段与进度 | 进度记录 |

## 1. Production / NIR 提取层

```text
runtime/nir-formal/
```

当前问题不是下游统计脚本，而是 NIR/PIR 数值本身已被确认错误。后续必须先纠正真正的 NIR 数值来源/计算，再重新生成可信的 downstream snapshot。

在错误来源定位清楚前，不因为下游需要而继续批量跑更多 subject。

## 2. `10_analysis_ready`

入口：

```text
scripts/nir_materialize_analysis_ready.py
configs/nir_analysis_ready.yaml
src/attention_pipeline/nir_analysis_ready/
```

Issue #18 已完成的是**数据层工程验收**：逐帧 validity、subject×eye baseline、左右眼保留、binocular source mode、provenance、时间键等逻辑能够运行并生成完整 analysis-ready 结构。

但由于当前 PIR 数值本身错误，该 snapshot 不能作为正式科学分析输入。之前得到的 production/primary/strict 有效率等数字仅保留为这一次错误值 snapshot 的工程记录，不作为科学质量结论。

## 3. `11_analysis_tables`

正式构表入口：

```text
scripts/nir_formal_pipeline.py
scripts/nir_build_analysis_tables.py
configs/nir_formal_analysis.yaml
src/attention_pipeline/nir_formal_analysis/
```

该层负责：

```text
subject-level continuous PIR
→ trial_level
→ trial_pir_windows
→ probe_pir_windows
→ time_on_task_1s
→ trial/probe coverage
```

当前 `sub-031` representative 和部分 subject 已证明构表逻辑可运行；批量在用户要求下主动停止，不是程序失败。已生成部分结果只用于管线验证，不解释为科学结果。

再次运行时 completion identity 一致的 subject 会自动 skip；中断/续跑机制会记录 cohort manifest。但在 PIR 修正前，**不要继续剩余 subject**。

## 4. 当前允许：`12_pipeline_validation`

入口：

```text
scripts/nir_pipeline_validation.py
configs/nir_pipeline_validation.yaml
src/attention_pipeline/nir_pipeline_validation/
```

它只发现已有有效 `11_analysis_tables` completion 的 subject，并假定 PIR 数值正确来验证未来正式分析代码。

当前可以做：

- Behavior-only subject×Block 描述；
- Block1→Block2 配对数据结构；
- time-on-task 1 s → 30 s 轨迹；
- trial outcome × pre-trial PIR；
- Go RT / commission / omission 模型接口；
- Probe 10/20/30/60 s 数据结构；
- within-person / between-person PIR decomposition；
- LMM / GEE smoke test；
- coverage / missingness QC；
- 专业 PNG/PDF 代码绘图。

输出：

```text
D:/_AttentionData/Beijing-NIR/analysis/nir-behavior-v2/cohort-44-exploratory/12_pipeline_validation/
├── tables/
├── figures/
└── validation_summary.json
```

图形由 `matplotlib` 代码直接生成，不使用图片生成模型，包括：

```text
fig00_pipeline_validation_schematic
fig01_time_on_task_trajectory
fig02_block_paired_pir
fig03_trial_outcome_pir_pre_5s
fig04_probe_vigilance_windows
fig05_coverage_heatmap_pre_5s
fig06_model_smoke_forest
```

所有图自动标记：

```text
PIPELINE VALIDATION ONLY — CURRENT NIR VALUES KNOWN INVALID
```

## 5. 当前运行最短路径

只运行 pipeline validation：

```powershell
git status --short --branch
git pull --ff-only

conda activate "D:\CondaEnvs\nir-amd"
python -m pip install -e .

python -m pytest `
  tests/test_nir_pipeline_validation.py `
  tests/test_nir_formal_analysis.py -q

python scripts/nir_pipeline_validation.py
```

不要执行：

```text
python scripts/nir_formal_pipeline.py --stage tables
```

除非之后明确要求继续构表。

## 6. PIR 方法设计仍如何保留

PIR 的计划定义仍为直径比：

```text
D_pupil = sqrt(pupil_axis_a × pupil_axis_b)
D_iris  = sqrt(iris_axis_a × iris_axis_b)
PIR     = D_pupil / D_iris
```

primary validity、subject×eye centering、binocular fallback 等方法决策继续保留，后续正确数值出来时应重新审计并复用相同数据契约，而不是从错误 snapshot 直接继续统计。

## 7. Behavior / Probe / coverage

FocusWave v3.1.3 Behavior 的 trial/probe 时间轴和已验证对齐逻辑可以继续保留。即使当前 PIR 数值错误，以下工程检查仍有价值：

- trial/probe key 是否正确；
- Block 边界截断是否正确；
- internal coverage 是否正确；
- max temporal gap 是否合理；
- join 是否重复或遗漏；
- Behavior-only 变量是否正常。

旧 [024-2026-08-26-NIR行为对齐原型与数据契约.md](024-2026-08-26-NIR行为对齐原型与数据契约.md) / `scripts/nir_behavior_alignment.py` 继续只作历史 prototype / provenance，不恢复成正式入口。

## 8. 当前严格禁止的解释

当前不要解释：

- PIR 随 time-on-task 上升或下降；
- Block1 与 Block2 PIR 谁更高；
- PIR 是否预测 RT / commission / omission；
- PIR 是否预测 probe vigilance；
- 哪个 pre-trial / pre-probe window “效果最好”；
- mixed model / GEE 的 PIR p 值或置信区间。

这些输出现在只回答一个问题：**未来正确 NIR 数据进入后，这套分析管线能否无缝运行并生成可用于论文/报告的结构化结果和专业图形。**

## 9. OAR / blink 与其他资料

[021-眨眼检测边界与RITnet派生开合度.md](021-眨眼检测边界与RITnet派生开合度.md) 继续记录 OAR、blink/PERCLOS 的解释边界。

其他仍有效资料：

- [025-2026-08-26-SART刺激视觉协变量重建.md](025-2026-08-26-SART刺激视觉协变量重建.md)：正式 SART 视觉协变量；
- [029-2026-08-26-PIR有效性与usable筛选定义.md](029-2026-08-26-PIR有效性与usable筛选定义.md)：历史 production gate 定义；
- [210-2026-08-26-PIR六条gate失败原因QC结果.md](210-2026-08-26-PIR六条gate失败原因QC结果.md)：历史 strict gate QC；
- [211-2026-08-26-左右眼PIR处理与标准化冻结决策.md](211-2026-08-26-左右眼PIR处理与标准化冻结决策.md)：左右眼与标准化设计依据。

更早日期文档和 `docs/工作记录/` 继续保留历史 provenance，不追溯改写。