# 058｜NIR cohort 数据质量与筛选规则

> 适用范围：当前北京 44 人 exploratory/development cohort；后续 116 人正式样本原则上复用相同逻辑。当前分析主线：`analysis/multimodal-integration`。

## 1. 决策目的

本文件冻结的是“数据质量控制与筛选的原则”，不是某个科学效应，也不是为了让当前 44 人结果更显著而挑选阈值。

当前 44 人已经完成：

- 44/44 被试结构读取；
- 176/176 个 subject×eye×Block NIR QC；
- 88/88 个 subject×Block Behavior QC；
- 双眼 PIR 可用性与 PIR 无效时段连续性复核；
- 行为极端记录的跨 Block 完整性核查。

## 2. 全局被试筛选结论

当前 44 人 **不因行为极端表现、单眼 PIR 低可用率、ROI clipped 高比例或单个 Block 的局部时间缺口而整被试排除**。

理由：

1. 44 人的 Behavior B1/B2 文件均完整，每个 Block 均为 432 trials；
2. 行为极端被试（如 sub-051、sub-144、sub-174）B1/B2 结构完整，现有证据支持将其视为真实行为差异，而非文件损坏；
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

特别注意：

- `roi_clipped` 表示扩大后的眼睛裁剪 ROI 是否触及原视频画面边界；
- `fullclass_pupil_touches_roi_edge` 与 `fullclass_iris_outer_touches_roi_edge` 表示 RITnet 分割结构是否触及 320×160 analysis ROI 边界；
- 只有后两者直接参与 PIR 的帧级筛选 gate。

完整代码条件与每个 gate 的实现依据见：

```text
docs/020-nir/029-2026-08-26-PIR有效性与usable筛选定义.md
```

面向正式报告的方法写法与筛选流程见：

```text
docs/020-nir/030-2026-08-26-PIR数据筛选与正式报告口径.md
```

## 5. 不设置全局 eye / Block 硬删除阈值

当前不采用例如“整只眼 PIR usable fraction < 20% 就删除”“整个 Block either-eye usable < 50% 就删除”的固定全局规则。

原因是筛选应与实际分析单位一致：

- trial-level 分析关心该 trial 周围的窗口是否有足够有效 PIR；
- probe 分析关心 probe 前指定时间窗是否有足够有效 PIR；
- 60 s time-on-task 分析关心该 60 s bin 自身的覆盖；
- subject/session baseline 与标准化关心 subject×eye 的有效样本量、跨 Block 表征和稳定性。

因此，同一个低质量 Block 可能不能用于某些长窗口分析，但仍可提供多个完整短窗口；反之亦然。

## 6. 正式分析采用 analysis-specific inclusion

后续每一种分析在建模前先冻结自己的 coverage rule，并在结果中报告实际有效 N。规则必须在查看该分析的 Behavior / Probe 效应显著性之前确定。

至少分别记录：

- 输入 subject 数；
- 输入 Block 数；
- 输入 eye / eye×Block 数；
- 候选 trial / probe / time-bin 数；
- 因 NIR coverage 不足被排除的分析单元数；
- 最终进入模型的分析单元数；
- 每名被试实际贡献的分析单元范围。

不同分析允许拥有不同有效 N，不要求人为制造一份“所有分析完全相同”的删人名单。

## 7. 左右眼规则

当前仍不提前平均或融合左右眼。

已知 88 个 subject×Block 的 either-eye PIR usable fraction 中位数约为 0.949，说明双眼互补通常明显；同时也存在 sub-150 B2 等双眼都低可用的 Block。因此：

1. 左右眼在 eye-structure / standardization 阶段继续独立评估；
2. 单眼坏不自动升级为整 Block 或整被试坏；
3. 双眼如何进入最终模型（平均、重复测量、单眼 fallback 或其他方案）必须由左右眼一致性、偏移和稳定性分析后再冻结；
4. 不能因为某一种眼处理方式得到更显著结果而选择它。

## 8. 当前已知重点 QC 记录

以下记录继续保留为 review / provenance 信息，不等于正式排除名单：

- sub-171：left PIR 在 B1/B2 均极低，但 right 在两 Block 高可用；
- sub-153：left 在 B1/B2 均低可用，双眼并集仍有中等覆盖，但存在较长双眼同时无效段；
- sub-165：主要为 B2 left 局部异常，其他 eye×Block 明显更好；
- sub-047：left 尤其低，但 right 可部分补偿；
- sub-150 B2：当前最值得后续谨慎处理的双眼低可用 Block，either-eye usable fraction 约 0.199；
- sub-050 B1 right：存在较大时间缺口，但另一眼和 B2 信息明显更完整；
- sub-163 B1：左右眼同时出现较长 gap；
- sub-176：ROI clipped 长期偏高，但 PIR 本身并非因此自动无效。

这些记录后续通过 analysis-specific coverage 自然影响可进入的 trial / probe / bin 数，而不是预先整被试删除。

另外，当前增加 `scripts/nir_pir_gate_failure_qc.py`，用于统计每个 subject×eye×Block 在上述 6 条结构 gate 上各自的失败比例。该结果只用于解释“为什么 PIR 被筛除”，不生成新的自动删除阈值；同一帧可同时违反多条 gate，因此各失败比例不可直接相加。

## 9. 正式报告写法边界

正式报告的数据质量与筛选部分应区分：

1. **结构完整性检查**：文件、trial、eye×Block 是否存在；
2. **帧级数据筛选**：PIR 是否通过 6 条 normalization gate，并且为 finite 数值；
3. **分析单元覆盖率筛选**：trial / probe / time-bin 是否满足对应分析预先冻结的 coverage rule；
4. **行为异常值**：真实极端表现与记录损坏分开；
5. **敏感性分析**：对 clipped、单眼低可用、不同 coverage 门槛等做稳健性验证，而不是事后挑最好看的结果。

正式报告不能只写“删除异常值”或“删除无效帧”，而必须说明无效帧的判定条件。推荐使用 `030-2026-08-26-PIR数据筛选与正式报告口径.md` 中已经准备好的方法段落，并在最终分析冻结 coverage rule 后补入具体阈值与最终 N。

不得把 `roi_clipped` 描述成“眼睛没检测完整”，也不得把行为极端值未经记录错误证据直接称为坏数据。

## 10. 当前落盘要求

本地 44 人分析目录的 `01_qc/` 应保留：

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
```

其中 `analysis_inclusion.csv` 只记录当前的全局候选状态与 QC 注释，不提前伪造某个尚未定义分析的最终 N；真正的 trial / probe / bin 纳入结果由后续各正式分析模块单独输出。

## 11. 当前状态

截至本规则冻结：

- 44/44 被试保留为正式下游分析候选；
- Behavior 无结构性排除；
- NIR 不进行 subject-level 或 Block-level 一刀切排除；
- PIR 采用明确的帧级数据筛选条件：6 条结构 gate + finite 数值；
- PIR gate failure 只作为筛选原因诊断，不作为新的全局排除阈值；
- 后续 trial / probe / time-bin 按 analysis-specific coverage 进入模型；
- 左右眼融合方案尚未冻结，下一步进入 eye structure / standardization 分析。
