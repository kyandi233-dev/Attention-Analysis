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

该 blob 与上游 commit `6431c57...` 完全一致。注意它是 Git blob SHA-1，不是运行模型文件的 SHA256。当前正式 `.onnx` 与 `.onnx.data` 的内容 SHA256 均在每次正式 full-class run 的 manifest / resume identity 中现场计算并分别记录；代码身份由 clean working tree（干净工作区）下的 exact Git commit 确定。`SHA256SUMS.txt` 不再作为会随源码修改而失真的静态运行证明。

## 官方网络输出与本项目 ONNX 接口的边界

上游 `DenseNet2D` 网络的直接输出是四分类 logits（未归一化分类得分）。上游 `utils.get_predictions()` / `test.py` 会在推理后对 channel 取最大值得到 hard class prediction（硬分类预测），并把预测 label 保存为 `.npy`。因此四分类 hard segmentation（硬分割）的任务语义来自官方 RITnet。

AMD 版本为了减少 DirectML→CPU 搬运，在网络 logits 后追加确定性的 ArgMax / Softmax / Gather 后处理节点，正式 production ONNX 暴露：

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
Attention-Analysis 400×640 hard-label evidence store
                    ↓
Attention-Analysis 派生几何 / QC / provenance / completion
```

## 当前唯一正式 full-class 口径

当前 full-class 只认 `ritnet-fullclass-v2-native640` 这一套生产 Schema。这里的版本号用于标记当前完整证据结构，并不表示旧 fast/320×160 路径仍是并行正式版本。历史产物可以保留用于 provenance，但当前用户生产入口只有：

```text
run_ritnet_fullclass_extension.py
run_ritnet_fullclass_batch.py
```

`native_*` 仅表示“在 640×400 RITnet hard-label 坐标系中保存或测量”，不表示“RITnet 官方原生变量”。

## 与历史 `models/external/ritnet/` 的关系

`models/external/ritnet/` 曾保存更完整的上游仓库副本，其中还包括训练、测试、dataset、environment、augmentation 示例等内容；该历史副本当前已不在主线工作树。

正式 NIR 运行并不依赖该完整仓库；正式 runtime 已直接保存其运行所需的网络定义、license、正式权重与本项目适配层。完整副本的来源与删除过程由 Git 历史和工作记录追溯。

## 复现原则

正式分析复现时，以本 `runtime/nir-formal/` 中的冻结文件、exact Git commit、每次 run 的 config/model/input SHA256 与 completion 验证链为准，不自动跟随上游仓库未来变化。若未来升级 RITnet、替换权重、修改 `densenet.py` 或更换 ONNX 后处理接口，必须升级当前正式 Schema，并记录新的来源 commit、对应哈希、DirectML parity（等价性）和验证结果。
