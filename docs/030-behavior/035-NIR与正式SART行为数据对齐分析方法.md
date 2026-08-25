# 035 NIR 与正式 SART 行为数据对齐分析方法

## 1. 数据职责分离

正式 v3.1.3 行为分析继续由 `031-正式BB行为分析流程.md`、`032-行为指标定义.md`、`033-统计分析方法.md` 与 `034-行为QC与输出.md` 定义。NIR full-class 文件负责逐帧/逐眼结构信号，不把行为 trial 内容复制进 segmentation CSV。

建议保持三层数据：

```text
NIR full-class per-eye time series
        +
SART behavior trial table
        ↓
独立 alignment / analysis table
```

这样以后更新 trial 定义或统计窗口时，不需要重新运行 YOLO/RITnet。

## 2. 时间对齐键

NIR full-class CSV 保留 source `eyes.csv` 中：

```text
subject
phase
phase_segment
frame_idx
video_time_ms
unix_ms
phase_time_ms
```

正式行为文件的绝对时间字段用于与 NIR `unix_ms` 对齐；`master_timeline.csv` 同时提供 block/phase 时间边界。Practice 已使用 `absolute_onset_time` 做真实 trial 定位，因此正式 trial-level alignment 也应坚持绝对时间戳，而不是按“第几帧≈第几个 trial”推断。

对齐精度受 NIR 帧间隔和采集时间戳抖动约束，不要求行为 onset 恰好等于某一帧时间。

## 3. 行为指标保持现有定义

行为侧主要保留：

- 正确 Go RT：median / mean / SD / RT-CV / ex-Gaussian μ/σ/τ；
- commission rate；
- omission rate；
- d′、c、β；
- cycle-bin / trial-level RT drift；
- No-Go 前兆；
- probe_response / probe_vigilance 及 probe 前 RT。

RT 阈值仍按行为 QC 文档用于标记，不在跨模态对齐阶段静默删除 trial。

## 4. NIR 主指标

正式 pupil signal 默认使用：

```text
fullclass_pupil_to_iris_diameter_ratio
```

并优先保留：

```text
fullclass_normalization_valid == True
```

的样本。`pupil_equiv_diameter` 等原 pupil 列继续用于 provenance、QC 与敏感性检查，但不替代 pupil/iris 主尺度指标。

NIR `fullclass_ocular_aperture_ratio_median`、ocular fraction、iris/sclera visibility 等目前定位为眼睛开合/QC候选信号，不直接当作 blink/PERCLOS 或注意状态标签。

## 5. 两个主要跨模态分析层级

### 5.1 持续注意 / 时间过程

行为侧已有 block 与 cycle-bin 时间结构，因此可在相同时间窗口汇总 NIR：

```text
median pupil/iris ratio
pupil/iris variability
valid coverage
必要时的时间趋势/slope
```

再与同一 block / cycle-bin 中的 RT、RT variability、commission、omission 等比较。这样对应 `033-统计分析方法.md` 中的 B1/B2 与 block 内时间趋势框架。

### 5.2 Trial / event-level

以行为 trial 的绝对 onset 为零点，从 NIR `unix_ms` 选择相应时间窗，生成独立的 trial-aligned 表。建议至少保留：

```text
subject
block
trial
behavior_absolute_onset
nir_unix_ms
relative_time_ms
eye
pupil_to_iris_diameter_ratio
normalization_valid
behavior response / RT / correctness
```

具体 pre/post window 应在正式分析阶段根据 SART trial 时序和瞳孔响应建模方案冻结，不写死在 RITnet extraction 中。

## 6. 左右眼与有效率

full-class 保留左右眼身份。跨模态聚合前先计算每个时间窗的有效覆盖率，并保留左右眼是否同时有效的信息；不要在 extraction 阶段直接把双眼压成单列。

被试/phase/block 若出现大量 `roi_clipped`、`ritnet_missing` 或 `normalization_invalid`，应将 valid coverage 与主要统计结果一起检查，避免把数据质量变化误解释成注意变化。

## 7. “瞳孔变化”与“专注状态”的解释边界

项目当前不把 pupil/iris ratio 单独定义成一个直接的“专注分数”。正式注意状态推断由：

```text
NIR pupil dynamics
+ SART behavioral performance
+ NIR structural QC
+ 后续 RGB blink / head-pose 信息
```

共同支持。行为结果仍是任务表现层证据，NIR 是连续生理/眼部动态证据，两者通过时间戳对齐后分析，不互相替代。

## 8. 推荐输出

跨模态结果建议另写 subject-numbered 文件，例如：

```text
sub-031_nir_beh_aligned.csv
```

必要时再生成跨被试汇总表。该步骤不能覆盖：

```text
sub-XXX_ritnet_fullclass_v1-2-fast-qc.csv
```

也不需要重新运行 RITnet。
