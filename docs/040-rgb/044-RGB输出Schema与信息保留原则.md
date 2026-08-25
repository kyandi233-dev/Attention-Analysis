# 044｜RGB 输出 Schema 与信息保留原则

> **开发前优先阅读。** 本文件定义 RGB 分支的长期数据保存原则。后续 Face、Pose、Motion、DirectML 适配、QC 和全量运行在设计输出字段时都必须先检查本文件，避免因为过早过滤而重新运行昂贵模型。

## 1. 核心工作目标

RGB 分析遵循一条硬规则：

> **昂贵推理尽量只运行一次；第一次推理时完整保存模型已经能够获得、以后可能需要且无法从现有结果直接恢复的信息。筛选、派生指标选择和统计聚合尽量后移。**

这条规则来自 NIR 分析阶段的实际经验：如果模型已经计算出的信息没有在第一次运行时写入结果，后续补字段可能必须重新运行模型。RGB 不重复这一问题。

这里的目标不是无差别保存所有中间数组，而是最大化可复用性：

- 原始 RGB 视频本来长期保留、可低成本重新读取的普通图像中间量，不必全部复制；
- 需要重新运行 Py-Feat / LibreFace / MediaPipe / 其他昂贵模型才能恢复的输出，优先完整落盘；
- QC 先标记，不在 raw 层静默删除；
- derived feature 和 summary 可以从 raw 输出重新计算，因此可以后移。

## 2. 三层输出职责

### 2.1 Raw model / measurement layer

这是最重要的保险层。保存模型或测量方法第一次运行时能够稳定导出的原始结果，不根据当前论文假设提前裁字段。

#### Face

`sub-XXX_face_raw.parquet` 原则上保存所选 backend 能够获得的完整可用输出，例如：

- `video_frame_position`、`capture_frame_idx`、`unix_ms`、`dt_ms`；
- face bbox / detection confidence / face id；
- primary-face 标记，但不在 raw 层静默删除其他检测到的人脸；
- 全部可用 Action Units；
- 全部 expression scores；
- gaze；
- head pose；
- valence / arousal（backend 支持时）；
- FaceMesh / landmarks（backend 支持时）；
- blendshapes（backend 支持时）；
- valid / missing / multi-face / model-specific QC flags。

在 Py-Feat vs LibreFace benchmark 完成前，不提前把 schema 缩减到某几个 AU。

#### Pose

`sub-XXX_pose_landmarks.parquet` 保存 MediaPipe Pose 能够稳定返回的完整 landmark 层，包括：

- 33 个 body landmarks 的 `x/y/z`；
- visibility / presence（可用时）；
- world coordinates（可用时）；
- detection / tracking validity；
- timestamp、frame identity 和 QC 字段。

`wrist_motion`、`upper_body_motion`、`trunk_lean` 等属于 derived feature，不替代完整 landmarks。

#### Motion

`sub-XXX_motion_raw.parquet` 不只保存一个最终 Motion Energy。第一版至少保留：

- frame identity 与真实 Unix timestamp；
- `dt_ms`；
- `mean_abs_difference`；
- `sum_abs_difference`；
- `changed_pixel_ratio`；
- frame gray mean / std 等低成本亮度 QC；
- `global_motion_energy`；
- `gap_before` / `gap_duration_ms`；
- 后续获得人体 ROI 后的 `body_motion_energy` 与 ROI 有效性信息。

跨 timestamp gap 的相邻帧不计算伪运动；该位置保留 raw identity 并把依赖相邻帧的指标记为 missing。

### 2.2 Derived feature layer

`sub-XXX_rgb_features.parquet` 保存从 raw 输出可重复计算的研究变量，例如：

- AU activity / variability；
- head angular velocity / variability；
- gaze variability；
- wrist / elbow / shoulder / upper-body motion；
- trunk lean / posture variability；
- global / body motion energy 等。

这一层允许以后重新定义算法而无需重新跑原始模型。

### 2.3 Summary layer

`sub-XXX_rgb_summary.csv` 保存 phase、block、trial-centered window 等统计汇总。summary 是最容易从前两层重建的一层，因此不能替代 raw 输出。

## 3. 每条结果都应保留的身份与时间字段

只要粒度允许，原始/逐帧结果至少保留：

- `subject`；
- `video_frame_position`：AVI 中实际第几帧；
- `capture_frame_idx`：FocusWave 采集线程原始 frame index；
- `unix_ms`；
- `dt_ms`；
- `phase`；
- `block`；
- 后续可映射时的 `trial` / `condition`；
- `gap_before` / missing / valid / QC flags。

必须区分 `video_frame_position` 与 `capture_frame_idx`。像 `sub-033` 一样发生采集掉帧时，两者不能互相替代。

## 4. QC 原则：flag first, filter later

Raw 层原则上采用：

```text
模型/测量输出
→ 完整落盘
→ 添加 QC flag
→ 派生特征
→ 统计阶段按冻结后的 QC 规则筛选
```

避免采用：

```text
模型输出
→ 当前觉得无用/低质量
→ continue / 丢弃
```

典型 QC 包括 timestamp gap、face confidence、multi-face、pose visibility、ROI validity、极端头部姿态等。除非数据无法建立正确身份或模型根本没有输出，否则先保留并标记。

## 5. Manifest 与可复现信息

每个正式被试最终保存 `sub-XXX_manifest.json`。至少应记录：

- 原始 RGB video / timestamp / behavior source paths；
- Attention-Analysis Git commit；
- config digest 与 schema version；
- Face/Pose backend、模型版本和模型 hash；
- ONNX 导出模型 hash（适用时）；
- ONNX Runtime / PyTorch / MediaPipe 等版本；
- Execution Provider（CPU / DirectML 等）与设备；
- 分析开始/结束 Unix 时间；
- 实际采样策略；
- run timestamp；
- 输出文件及关键行数/coverage。

模型或 backend 改变时，不覆盖成看起来像同一次运行；manifest 必须能区分不同结果 provenance。

## 6. 输出目录与命名

正式输出统一位于 `D:/_AttentionData/Beijing-RGB`。测试/benchmark 进入 `_test/`；被试结果仅在实际产生时创建 `sub-XXX/`，内部文件重复带 `sub-XXX_` 前缀。

不为 raw / processed / face / pose / motion 再建立空套目录。信息层级由文件名和 schema 表达。

## 7. 后续开发检查清单

每增加一个模型或 stage，在全量运行前必须回答：

1. 这个库/模型一次推理到底能返回哪些字段？
2. 哪些字段以后若需要会迫使模型重跑？这些必须优先保留。
3. 哪些字段只是从已保存 raw 输出可重新计算？放 derived 层。
4. QC 是否只做了 flag，而没有在 raw 层过早删除？
5. frame identity、timestamp、phase/block 和 provenance 是否足够完整？
6. schema 是否经过 representative pilot 检查后才进入全量？

只有上述问题明确后，才允许进行大规模 RGB 推理。
