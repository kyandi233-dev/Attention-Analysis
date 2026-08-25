# 042｜面部分析工具与 Benchmark

## 1. 当前问题

RGB Face 分支当前不直接冻结单一模型，而是比较 **Py-Feat** 与 **LibreFace 2.0**。二者都能提供面部行为信息，但覆盖范围和工程重点不同。

## 2. Py-Feat 能得到什么

当前计划关注 Py-Feat Detectorv2 的多任务输出，包括：

- Facial Action Units（AU）；
- categorical facial expression；
- valence / arousal；
- gaze；
- head pose；
- 3D FaceMesh landmarks；
- face blendshapes。

因此 Py-Feat 更像一套“一体化面部行为分析器”。如果正式录像上的稳定性和运行效率可以接受，它可以同时覆盖 AU、头姿、视线和部分表情相关特征。

## 3. LibreFace 2.0 能得到什么

当前主要考虑：

- AU detection / intensity；
- facial expression；
- gaze 等面部行为输出。

LibreFace 的重点更偏向 AU 与面部行为本身，输出覆盖范围没有 Py-Feat 那么宽，但它更适合作为 AU 专项候选。

## 4. 两者的核心差异

| 维度 | Py-Feat | LibreFace 2.0 |
|---|---|---|
| AU | 支持 | 支持，核心能力 |
| 表情类别 | 支持 | 支持 |
| Valence / Arousal | 支持 | 不是当前主要输出 |
| Gaze | 支持 | 支持 |
| Head pose | 直接覆盖 | 不是当前主要选择理由 |
| FaceMesh / blendshapes | 覆盖较完整 | 不是当前主要选择理由 |
| 一体化程度 | 高 | 中 |
| 当前项目角色 | 信息最完整候选 | AU 专项候选 |

因此 benchmark 的目标不是比较“谁输出数字更大”，而是决定：**在本项目正式 RGB 视频和 AMD 运行环境下，哪一个能以更好的稳定性、速度和信息覆盖完成全量分析。**

## 5. Benchmark 比较什么

正式选择至少比较：

1. **检测覆盖率**：正常正脸、低头、转头、戴眼镜等情况下是否持续得到有效面部输出；
2. **时间连续性**：AU、head/gaze 等曲线是否出现大量非生理性的跳变或断裂；
3. **明显动作 spot-check**：对明显微笑、张嘴、皱眉、转头等片段人工检查输出是否方向合理；
4. **缺失率**：每分钟/每阶段有效输出比例；
5. **计算效率**：FPS、CPU/GPU 占用、内存与长视频稳定性；
6. **信息覆盖**：哪些输出是后续研究真正需要且可稳定获得的。

模型间一致性可以作为参考，但在没有人工 FACS ground truth 时不能把二者一致性直接称为“准确率”。

## 6. AMD / DirectML 路线

本项目已经在 NIR 上验证过“模型导出 ONNX → ONNX Runtime → DirectML”可以在 Windows AMD GPU 上运行，因此 Face 模型也允许测试相同工程路线。

但这里需要区分：

```text
官方 PyTorch pipeline
= preprocessing + face alignment/detection + neural network + postprocessing
```

即使核心网络可以导出 ONNX，也必须验证输入预处理、输出解释和数值结果与官方 pipeline 保持一致。因此当前只把 DirectML 视为**待验证的部署路线**，不能因为 NIR 已成功就直接宣布 Py-Feat 或 LibreFace 的 ONNX 版本已经可替代官方实现。

## 7. 环境策略

Face benchmark 与当前稳定 NIR 环境隔离。优先建立独立 RGB/benchmark 环境，避免 PyTorch、OpenCV、ONNX Runtime、NumPy 等依赖污染已经可复现的 NIR runtime。

最终可能出现三种结果：

1. Py-Feat 官方运行已足够快且稳定 → 直接采用；
2. LibreFace 更稳定/更适合 AU → 采用 LibreFace；
3. 某候选结果最好但官方 backend 太慢 → 再验证该模型的 ONNX + DirectML 适配。

最终 backend、采样率和 DirectML 是否正式采用，都必须在 benchmark 后写入 decision record。
