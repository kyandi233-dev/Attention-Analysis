# RGB NVIDIA 性能优化实测｜2026-08-27

## 目的与边界

本记录只总结 `rgb-nvidia` 正式 raw pipeline 在 NVIDIA 工作站上的工程性能优化与负结果。科学核心不变：Face 仍为 Py-Feat 2.1.1 Detectorv2、`identity_model=None`、15 Hz；Pose 10 Hz；Motion full FPS；不改变正式采样频率、检测阈值、时间跨度或保留字段。

## 实机环境

- GPU：NVIDIA GeForce RTX 5070，12 GB
- PyTorch：2.13.0+cu130
- CUDA：13.0
- cuDNN：92000
- Py-Feat：2.1.1
- Python：3.11.15
- Windows 10/11 build 26100

## 1. Face streaming Parquet 写入优化

早期正式 Face runner 中，Parquet 写入成为明显 wall-clock 瓶颈。通过把小批次频繁写入改为较大聚合 flush（约 512 planned frames 一次），在不改变 raw 科学字段和帧覆盖语义的前提下，写入耗时由数百秒级降低到约 20–30 秒级。

实测代表：

- sub-130 早期 B16：`parquet_write_sec ≈ 655.5 s`，`total_wall ≈ 1068.6 s`
- 后续 streaming 聚合写版本：同量级正式输出的 Parquet 写入约 20–30 s

因此正式 Face pipeline 保留聚合 streaming Parquet 写法。

## 2. FaceBatch / prefetch

RTX 5070 12 GB 上，FaceBatch=32 具有稳定显存余量，且明显优于过小 batch。

sub-062 全跨度、B32 对比：

- prefetch=2：`58.51 fps`，`total_wall ≈ 380.23 s`
- prefetch=4：`61.70 fps`，`total_wall ≈ 360.57 s`
- 两者峰值 CUDA allocated memory 均约 5.13 GB

因此当前正式工程配置优先冻结：

```text
FaceBatch = 32
prefetch_batches = 4
```

该调整只改变执行调度，不改变科学输出定义。

## 3. Motion CPU reduction 优化

Motion 原实现对每帧 720p 灰度数组进行多次独立 NumPy reduction：mean/std/min/max，以及相邻帧 absdiff 后再次 mean/std/sum/max/threshold count。纯计算 A/B 显示，把等价 reduction 改为 OpenCV C++ reduction 可获得约 `12.55×` 的微基准加速，数值差异仅浮点舍入量级（约 `2.2e-16`）。

该优化不改变 Motion 的科学变量定义，只减少同一大数组被 Python/NumPy 多次完整扫描的执行成本。

## 4. torch.compile / TorchInductor / Triton 实测：REJECT

为确认是否存在隐藏的 GPU 大幅加速空间，系统测试了 `torch.compile`。

### 4.1 直接 compile Detectorv2.forward

直接编译 `Detectorv2.forward` 失败，核心错误：

```text
RuntimeError: Cannot set version_counter for inference tensor
```

定位发现 Py-Feat 2.1.1 中：

- `Detectorv2.forward` 使用 `@torch.inference_mode()`；
- `MultitaskModel.__call__` 也使用 `@torch.inference_mode()`。

因此外围 compile 会与 nested inference tensor / Inductor graph capture 冲突。

### 4.2 no_grad wrapper

绕开 `Detectorv2.forward` 的 inference decorator、改用 `no_grad` 后，仍在内部 `MultitaskModel.__call__` / ConvNeXt GRN 路径触发同类 inference tensor version-counter 错误，因此该方案 REJECT。

### 4.3 只 compile MEGraphAUv2

进一步只编译真正的 GPU 神经网络主体 `detector.multitask.model`（`MEGraphAUv2`）。最小测试证明：

```text
NO_GRAD_OK
INFERENCE_MODE_OK
```

说明这种 compile 粒度可以正常执行，不再触发上述兼容性错误。

但同一批 3600 帧、B32 实测：

```text
普通 eager CUDA B32：约 67.8 fps
MEGraphAUv2 torch.compile：20.76 fps
```

compile 版本：

- `detector_detect ≈ 172.87 s`
- `detect_plus_write ≈ 173.39 s`
- `input_frames_per_sec_detect_plus_write ≈ 20.76 fps`
- CUDA peak allocated ≈ 5.14 GB

即吞吐仅约 eager 的 30.6%，整体约慢 3.3 倍。即使非常宽松地扣除首次 compile 的约 49 s，稳态估算仍显著慢于 eager。

因此当前环境下以下路线全部冻结为：

```text
torch.compile whole Detectorv2.forward     REJECT
torch.compile + no_grad wrapper            REJECT
torch.compile MEGraphAUv2 only             REJECT
```

不再继续投入 fullgraph、其他 compile mode、GRN 特判或 Triton 调参。

## 5. 当前正式推荐配置

```text
Py-Feat 2.1.1 Detectorv2
FP32 eager CUDA
FaceBatch = 32
prefetch_batches = 4
streaming Parquet aggregate flush ≈ 512 frames
identity_model = None
Face inference = 15 Hz
Pose = 10 Hz MediaPipe
Motion = full FPS OpenCV
```

这些均属于工程执行优化，不改变正式科学核心。

## 6. 当前结论

本轮优化贡献包括：

1. 找到并消除 Face Parquet 高频 flush 的主要写入瓶颈；
2. 实测冻结 RTX 5070 上更优的 B32 + prefetch4 调度；
3. 找到 Motion 多次 reduction 扫描的 CPU 热点，并验证等价 OpenCV reduction 可显著加速；
4. 完整定位 Py-Feat 2.1.1 nested `inference_mode` 与 TorchInductor 的兼容问题；
5. 找到能够成功 compile 的正确模型粒度后，仍通过同批数据证明 `torch.compile` 在当前 Windows + RTX 5070/Blackwell + PyTorch 2.13 + CUDA 13 环境中实际显著慢于 eager，因此正式排除该路线。

这组结果作为可复现的部署工程记录保留，避免后续重复探索无效路线。

## 7. 下一步

停止 compiler 实验，回到正式 RGB raw cohort：

```text
冻结 Face B32 + prefetch4
→ 完成/确认 representative parity Gate
→ 继续 eligible subject 正式 raw extraction
→ 单被试失败记录后继续
→ cohort 完成后返回 RGB_NVIDIA_COHORT_RAW_V1
```
