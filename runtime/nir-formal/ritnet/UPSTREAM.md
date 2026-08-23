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
runtime/nir-formal/models/ritnet-b16-fp32.onnx
runtime/nir-formal/models/ritnet-b16-fp32.onnx.data
```

Git blob SHA：

```text
f0864e6651f578525a9101c7ca787e23d2d201d7
```

该 SHA 是导出 ONNX 所用的上游 `best_model.pkl` 权重 blob，与上游 commit `6431c57...` 完全一致。历史 `models/external/ritnet/` 副本已从当前主线删除，可通过 Git 历史追溯。

ONNX 的网络主干来自上述冻结权重。AMD 版本为减少 DirectML→CPU 搬运，在网络 logits 后追加确定性的 ArgMax/Softmax/Gather 后处理节点，正式输出为：

```text
labels_u8  [16,400,640]  # 0 background / 1 sclera / 2 iris / 3 pupil
pupil_prob [16,400,640]  # class-3 softmax probability
```

这不改变网络参数或四分类 argmax，只改变 runtime 图的输出接口；模型文件与 external data 的发布 SHA 以 `runtime/nir-formal/SHA256SUMS.txt` 为准。

## 本项目自己的适配层

```text
runtime/nir-formal/ritnet_runtime.py
```

该文件不是上游 RITnet 原文件，而是 Attention-Analysis 为正式 NIR pipeline 编写的运行适配层。它负责将项目的 eye ROI、batch inference、预处理、输出坐标与 QC 语义接入冻结的 RITnet 网络和权重。

因此正式实现应理解为：

```text
上游冻结 densenet.py + 从冻结 best_model.pkl 导出的 batch-16 FP32 ONNX
                    ↓
Attention-Analysis ritnet_runtime.py
                    ↓
正式 NIR pipeline
```

## 与历史 `models/external/ritnet/` 的关系

`models/external/ritnet/` 曾保存更完整的上游仓库副本，其中还包括训练、测试、dataset、environment、augmentation 示例等内容；该历史副本当前已不在主线工作树。

正式 NIR 运行并不依赖该完整仓库；正式 runtime 已直接保存其运行所需的：

- 网络定义；
- license；
- 正式权重；
- 本项目适配层。

完整副本的来源与删除过程由 Git 历史和工作记录追溯；当前正式运行不依赖该路径。

## 复现原则

正式分析复现时，以本 `runtime/nir-formal/` 中的冻结文件为准，不自动跟随上游仓库未来变化。若未来升级 RITnet、替换权重或修改 `densenet.py`，必须记录新的来源 commit、文件 SHA 和验证结果。
