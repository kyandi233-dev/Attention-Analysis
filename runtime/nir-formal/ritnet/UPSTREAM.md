# RITnet upstream provenance

本目录保存 **Attention-Analysis 正式 NIR runtime 实际使用的冻结 RITnet 最小源码**。它不是一份新的 RITnet fork，也不是 `tools/` 辅助工具。

## 上游来源

- 上游项目：`AayushKrChaudhary/RITnet`
- 上游仓库：`https://github.com/AayushKrChaudhary/RITnet`
- 上游默认分支：`master`
- 本项目核对的上游 commit：`6431c57ce7bf0eda935fb6178b926ae9440b50bf`
- 论文：Chaudhary et al. (2019), *RITnet: Real-time Semantic Segmentation of the Eye for Gaze Tracking*, ICCVW 2019.

## 正式 runtime 中保留的上游文件

### `densenet.py`

正式 runtime：

```text
runtime/nir-formal/ritnet/densenet.py
```

Git blob SHA：

```text
9bc49f0e285a9dc26d4885ab7f74cf3c5fdbe59a
```

该 SHA 与上游 commit `6431c57...` 的 `densenet.py` 完全一致。

### `License.md`

正式 runtime：

```text
runtime/nir-formal/ritnet/License.md
```

Git blob SHA：

```text
001a801151794f0a2c72d3ab3f79738e91ece0e0
```

该文件直接保留上游 license。

### 正式权重

正式 runtime：

```text
runtime/nir-formal/models/ritnet-best_model.pkl
```

Git blob SHA：

```text
f0864e6651f578525a9101c7ca787e23d2d201d7
```

该 SHA 与上游 commit `6431c57...` 的 `best_model.pkl` 完全一致，也与当前 `models/external/ritnet/best_model.pkl` 一致。

## 本项目自己的适配层

```text
runtime/nir-formal/ritnet_runtime.py
```

该文件不是上游 RITnet 原文件，而是 Attention-Analysis 为正式 NIR pipeline 编写的运行适配层。它负责将项目的 eye ROI、batch inference、预处理、输出坐标与 QC 语义接入冻结的 RITnet 网络和权重。

因此正式实现应理解为：

```text
上游冻结 densenet.py + 上游冻结 best_model.pkl
                    ↓
Attention-Analysis ritnet_runtime.py
                    ↓
正式 NIR pipeline
```

## 与 `models/external/ritnet/` 的关系

`models/external/ritnet/` 当前保存的是更完整的上游仓库副本，其中还包括训练、测试、dataset、environment、augmentation 示例等内容。

正式 NIR 运行并不依赖该完整仓库；正式 runtime 已直接保存其运行所需的：

- 网络定义；
- license；
- 正式权重；
- 本项目适配层。

因此在确认没有历史脚本仍要求直接从 `models/external/ritnet/` 调用文件后，完整上游副本可以作为主线精简候选。任何实际删除仍需用户明确批准。

## 复现原则

正式分析复现时，以本 `runtime/nir-formal/` 中的冻结文件为准，不自动跟随上游仓库未来变化。若未来升级 RITnet、替换权重或修改 `densenet.py`，必须记录新的来源 commit、文件 SHA 和验证结果。
