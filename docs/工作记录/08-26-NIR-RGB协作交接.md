# NIR ↔ RGB 协作交接｜2026-08-26

## 当前目标

厚粲杯多模态分析当前按两条独立工作线并行推进：

- NIR / 多模态统计线：`nvidia-cuda`
- RGB NVIDIA 正式部署线：`rgb-nvidia`

两条工作线不要互相重复开发。当前优先级是比赛完成度：先完成正式可用的数据与结果，再做非关键优化。

## NIR 当前正式状态

NIR 已完成：

- `NIR_V1_FORMAL_STATS_V1`
- `NIR_INCREMENTAL_VALUE_V1`

当前 NIR 最新正式提交：

```text
2f52396613921a4057a1e110e9065802c0578707
```

核心结论：

- NIR 单独对 fully task-focused 的区分能力较弱，不能包装成已验证的独立注意分类器；
- NIR 对 Behavior baseline 有小幅正向增量：ROC-AUC 约从 0.5659 提升到 0.5989，ΔAUC≈+0.033；
- participant-bootstrap 95% CI 跨 0，因此增量尚不能称为稳定独立提升；
- NIR 提高 specificity、降低 sensitivity/F1；
- NIR 仍保留为最终多模态系统候选输入，不再继续单独优化；
- 后续继续保持 participant-level split，并保留 vigilance 作为协变量或敏感性分析。

NIR 当前工作线不再接管 RGB 部署。

## RGB NVIDIA 当前正式状态

RGB 真正的 NVIDIA 正式工作线是：

```text
rgb-nvidia
```

当前远端 HEAD：

```text
e82f48137468c79d75e09a99ab8f04eef09f22b1
```

该分支已经具备正式 raw pipeline：

- Face：Py-Feat 2.1.1 Detectorv2 + native PyTorch CUDA；
- Pose：MediaPipe 10 Hz；
- Motion：原视频 full FPS；
- 单被试正式 runner；
- cohort runner；
- resume / skip；
- final validator；
- schema regression test；
- CPU reference，用于 NVIDIA representative CPU↔CUDA parity。

当前 RGB 的重点不是重新设计 feature，也不是做统计，而是完成 NVIDIA 工作站实机部署：

```text
sub-130 环境/schema 检查
→ CPU reference ↔ CUDA representative parity
→ sub-130 full-span raw extraction
→ final validator
→ FaceBatch 16/32/64 快速 benchmark并冻结最快稳定配置
→ 启动 cohort raw 全量
```

正式 raw cohort 的最低完成标准是每个 eligible subject 生成并通过：

```text
motion_raw.parquet
pose_landmarks.parquet
face_raw.parquet
manifest completion_status=complete
```

以下全部后移，不阻塞 raw cohort：

- tracking；
- primary-face selection；
- EAR / eyelid；
- blink / PERCLOS；
- Pose features；
- QC 聚合；
- probe-level aggregation；
- 注意状态统计；
- 最终多模态融合。

单个 subject 失败时应记录并继续 cohort，不应停止全量。

## 工作线边界

RGB 窗口当前只负责：

```text
RGB NVIDIA Gate + cohort raw 正式部署
```

NIR / 多模态窗口当前只负责：

```text
保存 NIR 正式结论，等待 RGB raw cohort 完成
```

RGB raw cohort 完成后，再由多模态分析线接手：

```text
RGB probe-level feature aggregation
→ Behavior + mmWave + NIR + RGB 共同 probe intersection
→ participant-level 统一 cross-validation
→ 最终四模态增量价值与比赛展示
```

在 RGB raw cohort 完成前，不要求 RGB 窗口提前生成 probe matrix，也不要求 NIR 窗口介入 CUDA 部署。

## 下一次交接需要返回

RGB 工作线完成 cohort raw 后，至少返回：

- `RGB_NVIDIA_COHORT_RAW_V1`；
- 最新 commit SHA；
- eligible / complete / failed / skipped subject 数；
- 冻结 FaceBatch；
- Face / Pose / Motion 基本 coverage；
- 仍需后续处理的失败 subject。

随后再启动最终多模态阶段。
