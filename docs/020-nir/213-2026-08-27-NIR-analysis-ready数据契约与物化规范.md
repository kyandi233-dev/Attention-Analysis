# 213｜NIR analysis-ready 数据契约与物化规范

> 2026-08-27｜工作分支：`analysis/multimodal-integration`。本文定义从 frozen `ritnet-fullclass-v1.2-fast-qc` production CSV 到正式统计分析基础数据层的只读物化契约。当前对象为 44 人 exploratory/development cohort；后续约 116 人正式北京 cohort 应复用同一逻辑并写入独立 snapshot，不覆盖本阶段结果。

## 1. 本文解决什么问题

现有 `00_inventory`、`01_qc`、`02_standardization`、`03_time_on_task`、`04_frame_quality_audit` 均属于资料清单、质量控制或方法校准结果。它们不是后续 Behavior、Probe、time-on-task 和 mixed-effects model 共同读取的最终 cleaned base table。

因此新增独立的 analysis-ready derived layer：

```text
frozen production full-class CSV
        ↓ 只读
逐眼派生 primary / strict validity
        ↓
subject×eye 跨两个正式 Block 的 baseline / robust scale
        ↓
逐眼 centered / robust-z
        ↓
同一时间点 left/right wide materialization
        ↓
base-layer binocular PIR + source mode
        ↓
10_analysis_ready/
        ↓
trial / probe / time-bin 等 analysis-specific windowing
        ↓
formal statistical results
```

本层只完成“可直接进入后续分析构造”的连续时间序列物化，不运行显著性检验，不冻结 trial/probe/time-bin coverage cutoff，也不根据 outcome 选择窗口。

## 2. 文档优先级与历史字段边界

当前正式主分析的逐帧纳入规则以：

```text
docs/020-nir/212-2026-08-27-NIR数据清洗逻辑与正式分析纳入规则.md
```

为准。

`029-2026-08-26-PIR有效性与usable筛选定义.md` 与 `050-decisions/058-NIR-cohort数据质量与筛选规则.md` 中关于 `fullclass_normalization_valid` 的描述继续作为**生产字段历史真实定义和旧 strict 轨道说明**保存，不删除、不改写历史事实，但不再等价于 primary main-analysis inclusion。

因此 production 字段：

```text
fullclass_normalization_valid
```

永久只读保留；下游新增：

```text
pir_valid_primary
pir_valid_strict
```

二者不能反向覆盖 production CSV。

## 3. 输入边界

唯一权威 NIR 输入仍为每名被试 completed `ritnet-fullclass-v1.2-fast-qc` production CSV，由现有 `nir_behavior.discovery.find_nir_source()` 按 completion marker 发现。validation-mode、未完成或无 canonical CSV 的运行不得混入。

只物化正式：

```text
phase ∈ {block1, block2}
```

读取必须按列名，不允许依赖列位置，因为当前 44 人 source header 已知存在 106/107 列两种版本差异。

analysis-ready 最低必需 source 字段：

```text
subject
phase
phase_segment
frame_idx
video_time_ms
unix_ms
phase_time_ms
eye

fullclass_pupil_to_iris_diameter_ratio
fullclass_normalization_valid
fullclass_pupil_fit_valid
fullclass_iris_outer_fit_valid
fullclass_pupil_center_in_iris_outer
fullclass_pupil_geom_mean_diameter
fullclass_iris_outer_geom_mean_diameter
```

若 source 中存在下列字段，同时保留到左右眼 QC/provenance 列，不把它们作为 primary hard gate：

```text
fullclass_pupil_touches_roi_edge
fullclass_iris_outer_touches_roi_edge
roi_clipped
ritnet_found
fullclass_pupil_confidence
fullclass_ocular_component_count
fullclass_ocular_largest_component_fraction
```

## 4. PIR 数值与两条 validity 轨道

PIR（pupil-to-iris diameter ratio，瞳孔/虹膜直径比）沿用 frozen production 数值：

```text
D_pupil = sqrt(pupil_axis_a × pupil_axis_b)
D_iris  = sqrt(iris_axis_a × iris_axis_b)
PIR     = D_pupil / D_iris
```

其中 `iris_outer = iris OR pupil`，即 RITnet class 2 与 class 3 的联合区域；PIR 不是像素面积比。

### 4.1 Primary frame validity

```text
pir_valid_primary =
    pupil ellipse fit valid
AND iris outer ellipse fit valid
AND pupil center in iris outer
AND D_iris > D_pupil
AND PIR finite
```

whole-mask edge、`roi_clipped`、fragmentation、pupil confidence、blur 等只保留为 QC / sensitivity 信息，不进入 primary hard exclusion。

### 4.2 Strict sensitivity validity

```text
pir_valid_strict =
    fullclass_normalization_valid
AND PIR finite
```

该轨道用于与历史生产 gate 保持可比。

由于 strict production gate 比 primary 多包含 whole-mask edge 限制，因此在当前定义下必须满足：

```text
pir_valid_strict ⊆ pir_valid_primary
```

若实际数据出现 `strict=True` 但 `primary=False`，物化程序必须停止该被试并报 contract error；不能把这种冲突作为普通 QC 继续吞掉。

## 5. Raw 值、缺失与标准化

原始 PIR 数值即使没有通过 validity，也不从 analysis-ready 表中物理删除。wide 表永久保留：

```text
left_raw_PIR
right_raw_PIR
left_valid_primary / right_valid_primary
left_valid_strict / right_valid_strict
```

“无效”表示对应轨道不能把该值用于分析计算，不表示把原值改写为 0。禁止用邻帧自动插补 PIR。

Primary 个体内动态使用 subject×eye median centering（被试×眼中位数中心化）：

```text
PIR_centered = PIR - median(PIR_subject,eye,primary-valid across Block1+Block2)
```

baseline 必须跨两个正式 Block 一起计算，不能按 Block 单独中心化。这样能够去除稳定 subject×eye offset，同时保留 Block1→Block2 的平均水平变化。

同时生成 robust-z（稳健 z 标准化）：

```text
robust_sigma = 1.4826 × MAD_subject,eye
PIR_robust_z = (PIR - median_subject,eye) / robust_sigma
```

仅当 robust sigma 为有限正值时生成；否则保留 denominator-valid flag，并令 z 值缺失，不强行除法。

strict sensitivity 轨道使用 strict-valid 帧单独计算其 baseline / robust scale，避免 primary 恢复帧反向进入 strict 标准化分母。

## 6. 时间点与左右眼宽表

最底层 source 仍是一帧×一眼。analysis-ready 为方便后续多模态对齐，再增加一份一时间点一行的 wide table，但不会删除 eye identity。

同一时间点的配对键：

```text
subject + phase + phase_segment + frame_idx
```

左右眼对应行的 `unix_ms / video_time_ms / phase_time_ms` 应一致。当前配置允许最多 1 ms 数值误差；超过容差视为时间契约异常并停止该被试，不能静默按最近邻配对。

wide 表保留左右眼各自 `source_row`，从而可以回溯到 frozen production CSV。

## 7. Base-layer binocular PIR

`binocular_PIR` 在本层定义为**完成 subject×eye baseline correction 后的共同尺度 PIR**，不是 raw left/right PIR 的无条件平均。

对每个时间点按 `pir_valid_primary`：

| left | right | `binocular_PIR` | `binocular_source_mode` |
|---|---|---|---|
| valid | valid | `(left_centered + right_centered) / 2` | `binocular` |
| valid | invalid | `left_centered` | `left_only` |
| invalid | valid | `right_centered` | `right_only` |
| invalid | invalid | missing | `missing` |

同时保留 `binocular_centered_PIR` 作为显式同义字段，并生成 `binocular_robust_z_PIR`。strict sensitivity 生成平行的 strict binocular 字段和 strict source mode。

这里**不应用** 30 s / 60 s coverage gate，也不应用 Issue #17/211 中曾讨论的 window-level binocular concordance gate。原因是这些 gate 回答的是“某个具体 trial/probe/time-bin 是否具有足够代表性”，不属于基础逐时间点清洗。后续具体分析窗口仍可在不看 outcome 的前提下冻结 coverage / temporal-representativeness / concordance 规则。

## 8. 输出目录与职责隔离

当前 44 人 snapshot：

```text
D:/_AttentionData/Beijing-NIR/analysis/nir-behavior-v2/cohort-44-exploratory/
├── 00_inventory/                 # 已有：输入资料清单
├── 01_qc/                        # 已有：QC
├── 02_standardization/           # 已有：方法比较/QC
├── 03_time_on_task/              # 已有：coverage 方法校准
├── 04_frame_quality_audit/       # 已有：frame-quality audit
├── 10_analysis_ready/            # 本次新增：正式分析基础派生数据
│   ├── frame_level/
│   │   └── sub-XXX/
│   │       └── sub-XXX_nir_analysis_ready.csv
│   ├── baselines/
│   │   ├── sub-XXX_eye_baselines.csv
│   │   └── subject_eye_baselines.csv
│   ├── qc/
│   │   ├── subject_eye_block_inclusion.csv
│   │   ├── known_low_usable_subject_changes.csv
│   │   ├── subject_load_failures.csv
│   │   └── cohort_inclusion_summary.json
│   └── provenance/
│       ├── source_files.csv
│       └── analysis_ready_manifest.json
└── 20_formal_statistics/          # 后续正式模型结果；本脚本不生成
```

`10_analysis_ready` 与 production source、既有 QC、后续 formal statistical results 明确分层。当前脚本不得向 production run directory 写任何文件。

## 9. Wide frame table 最低字段契约

每行至少包括：

```text
subject
block
phase
phase_segment
frame_idx
unix_ms
video_time_ms
phase_time_ms

left_source_row
right_source_row
left_raw_PIR
right_raw_PIR
left_valid_primary
right_valid_primary
left_valid_strict
right_valid_strict

left_centered_PIR
right_centered_PIR
left_robust_z_PIR
right_robust_z_PIR
left_strict_centered_PIR
right_strict_centered_PIR
left_strict_robust_z_PIR
right_strict_robust_z_PIR

binocular_PIR
binocular_centered_PIR
binocular_robust_z_PIR
binocular_source_mode

binocular_strict_PIR
binocular_strict_robust_z_PIR
binocular_strict_source_mode
```

可用 source QC 字段以 `left_... / right_...` 前缀保留。

## 10. 每次物化必须重新报告的内容

cohort summary 至少报告：

1. production formal eye-row 总数；
2. primary-valid eye-row 数与逐帧纳入率；
3. strict-valid eye-row 数与逐帧纳入率；
4. `primary AND NOT strict` 恢复的帧数、占全部帧比例、占旧 strict-invalid 帧比例；
5. `strict AND NOT primary` 数，必须为 0；
6. subject×eye×Block 的 total / primary / strict / recovered 数和比例；
7. primary `binocular / left_only / right_only / missing` 时间点数量与比例；
8. strict source-mode 的平行统计；
9. 已知低 usable 被试在 primary 与 strict 下的变化；
10. subject load failure 与所有 provenance。

这些统计只描述新旧纳入规则如何改变数据可用性，不是科学效应检验。

## 11. Provenance 与覆盖保护

每名被试记录：source CSV 绝对路径、文件大小、mtime、completion marker 路径与 SHA256；是否计算 source CSV SHA256 由配置控制。

程序必须拒绝把 analysis-ready output root 设置到任一 frozen NIR source root 内。默认也拒绝覆盖已经存在的 derived subject 文件；只有显式指定 `--overwrite-derived` 时才允许重建 `10_analysis_ready` 中的派生文件，该参数绝不授权修改 production 数据。

## 12. 当前 44 人与后续 116 人的解释边界

44 人仍是 exploratory/development cohort。本次允许冻结的是：字段语义、清洗逻辑、标准化定义、物化方式和 provenance 规则。44 人得到的 primary/strict 保留率、恢复帧数、被试分布和任何后续效应都不能直接写成最终 116 人正式样本结果。

后续补齐约 116 人时，应使用相同代码和冻结规则，输出到独立 `cohort-116-final/10_analysis_ready`，并重新计算所有 cohort-level 数据质量统计。
