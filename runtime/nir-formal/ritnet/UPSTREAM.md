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

上游 `best_model.pkl` 的 Git blob SHA-1：

```text
f0864e6651f578525a9101c7ca787e23d2d201d7
```

该 blob 与上游 commit `6431c57...` 完全一致。注意它是 Git blob SHA-1，不是运行模型文件的 SHA256；正式 `.onnx` 与 `.onnx.data` 的内容 SHA256 以 `runtime/nir-formal/SHA256SUMS.txt` 和每次 v2 manifest 为准。

## 官方网络输出与本项目 ONNX 接口的边界

上游 `DenseNet2D` 网络的直接输出是四分类 logits。上游 `utils.get_predictions()`/`test.py` 会在推理后对 channel 取最大值得到 hard class prediction，并把预测 label 保存为 `.npy`。因此四分类 hard segmentation 的任务语义来自官方 RITnet。

AMD 版本为了减少 DirectML→CPU 搬运，在网络 logits 后追加确定性的 ArgMax/Softmax/Gather 后处理节点，正式 production ONNX 暴露：

```text
labels_u8  [16,400,640]  # 0 background / 1 sclera / 2 iris / 3 pupil
pupil_prob [16,400,640]  # class-3 softmax probability
```

这些节点不改变网络参数，也不改变四分类 argmax，但 **`labels_u8` / `pupil_prob` 这两个 ONNX 输出接口是 Attention-Analysis 的适配层，不是上游仓库原本提供的官方 ONNX 接口**。同理，ellipse、PIR、component、edge、OAR、置信度摘要、gate、QC、label store、manifest/completion 全部属于本项目后处理/审计层。

## 本项目自己的适配层

```text
runtime/nir-formal/ritnet_runtime.py
runtime/nir-formal/ritnet_fullclass_runtime.py
```

这些文件不是上游 RITnet 原文件，而是 Attention-Analysis 为正式 NIR pipeline 编写的 DirectML 运行适配层。它们负责把项目 eye ROI、fixed batch、预处理、ONNX 输出与正式 pipeline 接起来。

预处理也要区分来源：上游 RITnet 明确要求 grayscale 输入、gamma=0.8、CLAHE clipLimit=1.5 tileGridSize=(8,8) 和 Normalize([0.5],[0.5])；而“复用本项目 YOLO ROI 并把任意 ROI resize 到固定 640×400”属于 Attention-Analysis 的 pipeline 适配。

因此正式实现应理解为：

```text
上游冻结 densenet.py + best_model.pkl 权重 + 上游预处理原则
                    ↓
Attention-Analysis ONNX/DirectML 适配与固定 ROI 输入接口
                    ↓
Attention-Analysis 原生 label store 与派生指标
```

## 与历史 `models/external/ritnet/` 的关系

`models/external/ritnet/` 曾保存更完整的上游仓库副本，其中还包括训练、测试、dataset、environment、augmentation 示例等内容；该历史副本当前已不在主线工作树。

正式 NIR 运行并不依赖该完整仓库；正式 runtime 已直接保存其运行所需的网络定义、license、正式权重与本项目适配层。完整副本的来源与删除过程由 Git 历史和工作记录追溯。

## 复现原则

正式分析复现时，以本 `runtime/nir-formal/` 中的冻结文件为准，不自动跟随上游仓库未来变化。若未来升级 RITnet、替换权重、修改 `densenet.py` 或更换 ONNX 后处理接口，必须记录新的来源 commit、对应哈希、DirectML parity 和验证结果。
