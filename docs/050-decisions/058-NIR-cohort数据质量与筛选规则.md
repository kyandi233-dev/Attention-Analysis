# 058｜NIR cohort 数据质量与筛选规则

> 适用范围：当前北京 44 人 exploratory/development cohort；后续 116 人正式样本原则上复用相同逻辑。当前分析主线：`analysis/multimodal-integration`。

## 1. 决策目的

本文件冻结的是“数据质量控制与筛选的原则”，不是某个科学效应，也不是为了让当前 44 人结果更显著而挑选阈值。

当前 44 人已经完成：

- 44/44 被试结构读取；
- 176/176 个 subject×eye×Block NIR QC；
- 88/88 个 subject×Block Behavior QC；
- 双眼 PIR 可用性与 PIR 无效时段连续性复核；
- PIR 六条 normalization gate 的失败原因审计；
- 左右眼测量结构与质量分层 agreement 复核；
- 30 s / 60 s time-bin coverage calibration；
- 行为极端记录的跨 Block 完整性核查。

## 2. 全局被试筛选结论

当前 44 人 **不因行为极端表现、单眼 PIR 低可用率、ROI clipped 高比例或单个 Block 的局部时间缺口而整被试排除**。

理由：

1. 44 人的 Behavior B1/B2 文件均完整，每个 Block 均为 432 trials；
2. 行为极端被试 B1/B2 结构完整，现有证据支持将其视为真实行为差异，而非文件损坏；
3. PIR 低可用问题多数具有明显的单眼或局部时段特征，另一眼在很多 Block 能覆盖大部分时段；
4. `roi_clipped` 只表示放大的 RITnet ROI 碰到原始画面边界，不等于眼睛未完整检测，因此不能作为自动排除条件；
5. 直接按整个被试或整个 Block 的平均 PIR 可用率删除数据，会同时丢弃仍然完整的 trial、probe 和时间窗信息。

因此，当前全局 subject-level inclusion 状态为：44/44 保留作为后续分析候选样本。

## 3. Behavior 筛选原则

### 3.1 不按表现好坏删被试

以下现象本身不构成数据排除理由：

- commission rate 很高或很低；
- omission rate 很高或很低；
- RT 很快或很慢；
- RT variability 很高；
- B1/B2 表现差异很大。

只有在出现文件缺失、trial 数错误、时间字段损坏、无法恢复的记录结构异常时，才属于 Behavior 数据质量排除问题。

### 3.2 原始评分永久保留

程序原始 `commission`、`omission`、`correct`、RT 和按键记录不被覆盖。prestimulus、carry-over、multiple press、ambiguous omission 等只作为附加 QC / subtype 标记，用于主分析与敏感性分析。

## 4. NIR / PIR 帧级有效性：正式数据筛选条件

这一节定义的是**正式的数据筛选条件**，不是仅用于描述的 QC 指标。

PIR 数值进入任何正式计算的基本条件为：

```text
fullclass_normalization_valid == True
AND
fullclass_pupil_to_iris_diameter_ratio 为 finite 数值
```

其中 `fullclass_normalization_valid=True` 由以下 6 条结构 gate 共同决定，必须全部通过：

1. pupil ellipse 有效；
2. iris outer ellipse 有效；
3. pupil mask 不触碰 320×160 analysis ROI 边缘；
4. iris outer mask 不触碰 320×160 analysis ROI 边缘；
5. pupil center 位于 iris outer contour 内部或边界上；
6. iris geometric-mean diameter > pupil geometric-mean diameter。

下游再要求 PIR 本身为 finite 数值。因此：

```text
PIR usable frame
=
六项结构 gate 全部通过
AND
PIR finite
```

不满足时，该帧的 PIR 视为 unavailable；不对 PIR 进行强行插补，也不把 invalid 帧改写为 0。

`roi_clipped`、单纯的 OAR 高低或行为表现不参与这一帧级 PIR 有效性定义。

完整代码条件、gate 失败逻辑和正式报告方法口径统一见：

```text
docs/020-nir/029-2026-08-26-PIR有效性与usable筛选定义.md
```

PIR 六条 gate 在 cohort 中的实际失败原因结果见：

```text
docs/020-nir/210-2026-08-26-PIR六条gate失败原因QC结果.md
```

## 5. 不设置全局 eye / Block 硬删除阈值

当前不采用例如“整只眼 PIR usable fraction < 20% 就删除”“整个 Block either-eye usable < 50% 就删除”的固定全局规则。

原因是筛选应与实际分析单位一致：

- trial-level 分析关心该 trial 周围的窗口是否有足够有效 PIR；
- probe 分析关心 probe 前指定时间窗是否有足够有效 PIR；
- 60 s time-on-task 分析关心该 60 s bin 自身的覆盖与时间代表性；
- subject/session baseline 与标准化关心 subject×eye 的有效样本量、跨 Block 表征和稳定性。

因此，同一个低质量 Block 可能不能用于某些长窗口分析，但仍可提供多个完整短窗口；反之亦然。

## 6. 正式分析采用 analysis-specific inclusion

后续每一种分析在建模前先冻结自己的 window-level coverage / temporal-representativeness rule，并在结果中报告实际有效 N。规则必须在查看该分析的 Behavior / Probe / time-on-task 效应显著性之前确定。

至少分别记录：

- 输入 subject 数；
- 输入 Block 数；
- 输入 eye / eye×Block 数；
- 候选 trial / probe / time-bin 数；
- 因 NIR coverage 不足被排除的分析单元数；
- binocular / single-eye fallback / binocular-discordant / unavailable 的分析单元数；
- 最终进入模型的分析单元数；
- 每名被试实际贡献的分析单元范围。

不同分析允许拥有不同有效 N，不要求人为制造一份“所有分析完全相同”的删人名单。

### 6.1 30 s / 60 s time-bin coverage calibration 的当前证据

Issue #17 已完成 60 s primary candidate 与 30 s sensitivity candidate 的 coverage calibration，但**尚未冻结正式 cutoff**。

60 s full-duration eye-bin 的 `usable_expected_fraction` 分布为：P25≈0.571、median≈0.831、P75≈0.910；30 s 与 60 s 分布整体相近。

对 60 s paired bins（n=788）的候选 threshold sensitivity：

| 候选 threshold | both eyes pass | either eye pass | single-eye fallback | neither eye pass |
|---:|---:|---:|---:|---:|
| 0.25 | 0.779 | 0.968 | 0.189 | 0.032 |
| 0.50 | 0.643 | 0.928 | 0.284 | 0.072 |
| 0.75 | 0.434 | 0.826 | 0.392 | 0.174 |
| 0.80 | 0.362 | 0.773 | 0.411 | 0.227 |
| 0.90 | 0.151 | 0.449 | 0.298 | 0.551 |

这说明提高 coverage 门槛会连续降低双眼可融合候选比例，并显著改变 single-eye fallback / unavailable 的构成；当前不存在仅凭保留率即可认定为唯一正确的 coverage cutoff。

因此下一步冻结 window gate 时，不能只看 overall usable fraction，还应同时考虑 5 s 子窗的时间代表性，防止一个 60 s 窗口的有效帧只集中在很短片段。

## 7. 左右眼与 single-eye fallback：正式筛选流程

左右眼处理已经不再是“整只眼删不删”的问题，而是**逐分析窗口做两道门控**。

### 7.1 第一道门：每只眼独立通过 window gate

对每个 baseline / time-bin / trial window / probe window：

1. left/right 分别仅使用 frame-level usable PIR；
2. 分别计算 coverage、usable N 和时间代表性；
3. 按该分析预先冻结的 window gate 判断每只眼是否合格。

此时结果分为：

```text
左不合格 + 右不合格
→ 该窗口 PIR unavailable

左合格 + 右不合格
→ left-only single-eye fallback

左不合格 + 右合格
→ right-only single-eye fallback

左合格 + 右合格
→ 进入第二道 binocular concordance gate
```

**single-eye fallback 的正式边界是：恰好只有一只眼通过预先冻结的 window gate。**

如果两眼都通过 window gate 但彼此明显冲突，不能把这种情况重新解释成 single-eye fallback，也不能事后挑选“更符合假设”的那只眼。

### 7.2 第二道门：双眼都合格时检查 binocular concordance

Issue #16 表明：低质量眼会显著拉低左右眼总体 agreement；当双眼质量较高时，subject×Block median PIR 的 cohort-level ICC 可达到约 0.89–0.91。但即便高质量记录中，within-block exact-timestamp Pearson 的中位数仍约 0.49–0.50，说明两眼短时动态并非完全等价。

因此：

```text
两眼都通过 window gate
↓
比较 baseline-adjusted 的左右眼同定义特征
↓
concordant
→ binocular equal-weight mean

discordant
→ 不强行平均
→ fusion_mode = binocular_discordant
→ 主融合分析中该窗口不生成 binocular single value
→ 左右眼原特征保留到 eye-preserved sensitivity
```

这里的 concordance / discordance 必须基于 subject×eye baseline-adjusted 特征，而不是直接比较 raw PIR 的绝对左右眼差。原因是两只眼可能存在稳定 baseline offset，但相对于各自 baseline 的变化方向和幅度仍然一致。

**binocular discordance 的具体数值判据当前尚未冻结**；在进入正式 time-on-task / Behavior / Probe 效应模型前，必须先用不涉及 outcome 的测量分布完成校准。

### 7.3 为什么双眼 concordant 后才允许平均

如果两眼都高质量且都表示“相对于各自 baseline 同方向、相近幅度的变化”，等权平均可降低单眼随机测量误差。

反之，如果左眼显示明显增加、右眼显示明显减少，直接平均可能人为制造接近 0 的假象，因此不允许用“平均后看起来更稳定”作为融合理由。

左右眼处理与标准化的完整冻结依据见：

```text
docs/020-nir/211-2026-08-26-左右眼PIR处理与标准化冻结决策.md
```

## 8. 当前已知重点 QC 记录

以下记录继续保留为 review / provenance 信息，不等于正式排除名单：

- sub-171：left PIR 在 B1/B2 均极低，但 right 在两 Block 高可用；
- sub-153：left 在 B1/B2 均低可用，双眼并集仍有中等覆盖，但存在较长双眼同时无效段；
- sub-165：主要为 B2 left 局部异常，其他 eye×Block 明显更好；
- sub-047：left 尤其低，但 right 可部分补偿；
- sub-150 B2：双眼都存在低可用时段，time-bin 层面尤其需要 window gate；
- sub-050 B1 right：存在较大时间缺口，但另一眼和 B2 信息明显更完整；
- sub-163 B1：左右眼同时出现较长 gap；
- sub-176：ROI clipped 长期偏高，但 PIR 本身并非因此自动无效；
- sub-035 right、sub-164 B2 right：在 60 s coverage calibration 中还出现额外局部低 coverage bin。

这些记录后续通过 analysis-specific window gate 自然影响可进入的 trial / probe / bin 数，而不是预先整被试删除。

## 9. 正式报告写法边界

正式报告的数据质量与筛选部分必须区分：

1. **结构完整性检查**：文件、trial、eye×Block 是否存在；
2. **帧级数据筛选**：PIR 是否通过 6 条 normalization gate，并且为 finite 数值；
3. **单眼窗口筛选**：left/right 各自是否满足该分析预先冻结的 coverage + temporal-representativeness rule；
4. **双眼融合筛选**：双眼都合格时，baseline-adjusted 特征是否满足预先冻结的 binocular concordance rule；
5. **single-eye fallback**：仅一眼通过 window gate 时使用该眼；
6. **binocular discordance**：双眼都合格但明显冲突时不强行平均；
7. **行为异常值**：真实极端表现与记录损坏分开；
8. **敏感性分析**：bilateral-only、eye-preserved、coverage threshold sensitivity 等用于验证稳健性，而不是事后挑最好看的结果。

正式报告不能只写“删除异常值”或“删除无效帧”，而必须说明每一层筛选条件与最终各类 analysis unit 的数量。

建议报告用语：

> NIR 数据先在帧级应用预设的 pupil–iris 几何有效性门控，仅保留 normalization-valid 且 PIR 为有限数值的帧。随后在各分析时间窗内分别评估左右眼的有效覆盖率与时间代表性。若双眼均不满足窗口质量标准，则该时间窗 PIR 记为不可用；若仅单眼满足标准，则使用该眼作为 single-eye fallback；若双眼均满足质量标准，则进一步比较经 subject×eye baseline 校正后的双眼同定义特征，只有双眼结果一致时才进行等权融合。双眼均高质量但结果明显冲突的窗口不强行平均，并在主融合分析中标记为 binocular-discordant，同时保留双眼独立特征用于敏感性分析。

## 10. 当前落盘要求

本地 44 人分析目录应保留：

```text
01_qc/
├── subject_eye_block_qc.csv
├── subject_qc.csv
├── behavior_cohort_qc.csv
├── cohort_anomaly_flags.csv
├── screening_review/
│   ├── binocular_pir_usability.csv
│   ├── pir_temporal_usability_60s.csv
│   ├── pir_invalid_run_summary.csv
│   ├── either_eye_invalid_run_summary.csv
│   ├── behavior_extreme_crossblock_review.csv
│   ├── pir_gate_failure_qc/
│   │   ├── pir_gate_failure_by_eye_block.csv
│   │   ├── pir_gate_failure_distribution.csv
│   │   ├── pir_gate_failure_top_records.csv
│   │   └── README.md
│   └── README.md
├── analysis_inclusion.csv
└── qc_decisions.md

02_standardization/
└── eye_structure/
    ├── subject_eye_baselines.csv
    ├── subject_block_eye_structure.csv
    ├── standardization_comparison.csv
    └── quality_sensitivity/

03_time_on_task/
└── coverage_calibration/
    ├── time_bin_eye_features.csv
    ├── time_bin_coverage_distribution.csv
    ├── time_bin_threshold_sensitivity.csv
    └── README.md
```

`analysis_inclusion.csv` 只记录当前的全局候选状态与 QC 注释，不提前伪造某个尚未定义分析的最终 N；真正的 trial / probe / bin 纳入结果由后续各正式分析模块单独输出。

## 11. 当前状态

截至当前：

- 44/44 被试保留为正式下游分析候选；
- Behavior 无结构性排除；
- NIR 不进行 subject-level 或 Block-level 一刀切排除；
- PIR 帧级筛选已冻结：6 条结构 gate + finite 数值；
- PIR gate failure 只作为筛选原因诊断，不作为新的全局排除阈值；
- 左右眼不提前在 frame level 融合；
- subject×eye median-centered PIR 为个体内动态主候选，raw PIR 保留 between-subject / absolute-level 信息；
- single-eye fallback 仅发生在“恰好一眼通过 window gate”时；
- 双眼都通过 window gate 后仍需经过 binocular concordance gate；
- 30 s / 60 s coverage calibration 已完成，但正式 window coverage cutoff 尚未冻结；
- binocular discordance 的数值判据尚未冻结；
- 在这两项 window-level gate 冻结前，不进入正式 time-on-task / Behavior / Probe 效应模型。
