# 046｜NVIDIA CUDA RGB 运行路线

**Branch:** `nvidia-cuda`

## 1. 目的

在不改变已经由 RGB 开发阶段冻结的 scientific contract 的前提下，为 NVIDIA GPU 建立 CUDA 执行版本。当前长期主线已经收口到 `nvidia-cuda`，不再把 `rgb-nvidia-cuda` 当作正式运行入口。

当前硬盘分工：NVIDIA 工作站连接的是剩余约 72 名被试的数据盘，AMD 工作站连接另一块约 44 名被试的数据盘。因此 NVIDIA representative Face Gate 使用 `sub-130`；AMD 历史 `sub-031` / `sub-033` 继续作为 AMD/开发阶段 provenance，不要求在 NVIDIA 本机存在。

## 2. 与 NIR 双后端策略一致

现有 NIR 正式链采用“不同硬件后端、相同科研口径”的维护方式：

- NVIDIA：CUDA；
- AMD：ONNX Runtime DirectML；
- 跨硬件一致性由 shared scientific contract + representative parity 证明，而不是强制使用相同执行框架。

RGB 同样采用：

```text
AMD:    Py-Feat scientific core → ONNX Runtime DirectML
NVIDIA: Py-Feat 2.1.1 native     → PyTorch CUDA
```

## 3. 必须保持与 AMD RGB 一致的科学定义

- Face backend：Py-Feat 2.1.1 Detectorv2 scientific core；
- timestamp-driven Face 15 Hz；
- 正式阶段最终使用 original AVI direct decode；
- Detectorv2 / RetinaFace 检测语义、bbox expansion/crop contract、canonical pose/gaze 定义一致；
- 所有 detected faces 先保留；
- 20 AU、7 emotion、V/A、gaze、6DoF pose、478×3 mesh、68 compatibility view、52 blendshapes 全保留；
- identity 不属于当前 accepted scientific core；
- `eyeBlinkLeft/Right` 保留 native raw；
- EAR / aperture-iris / normalized openness / closure proxy 为可重算 derived；
- primary-face 规则、QC 可视化和 output provenance 与 AMD 版本一致。

CUDA 可以改变执行框架、device scheduling 和最优 batch，但不得删减科学字段、改变采样时间点或改变变量定义。

## 4. NVIDIA 默认实现：native PyTorch CUDA

当前 CUDA dry-run runner 已实现：

```text
scripts/face_formal_dryrun_cuda.py
```

第一轮 Gate 故意消费已经抽取好的 timestamp-driven JPEG sample，而不是直接把 I/O 优化和 backend 变化混在一起：

```text
sub-130 original RGB/timestamps
→ timestamp-driven 15 Hz dry-run sample (~3600 frames)
→ Py-Feat 2.1.1 Detectorv2 native CPU reference
→ 同一 sample Py-Feat 2.1.1 Detectorv2 native CUDA
→ field-level parity
→ tracking / eyelid / QC
```

通过 scientific parity 后，正式 NVIDIA runner 才升级为：

```text
original AVI
→ timestamp-driven 15 Hz
→ decode/prefetch
→ Py-Feat Detectorv2 on CUDA
→ native Py-Feat crop / multitask semantics
→ same raw schema
→ same tracking / eyelid derived / QC
```

具体 CUDA batch 由 RTX 5070 benchmark 决定。AMD DirectML 的 RetinaFace B8 / multitask B16 是 AMD 工程参数，不强制作为 CUDA 最优参数。

## 5. 为什么不再写“sub-031 AMD↔NVIDIA 逐帧 parity”

当前两台机器连接不同正式数据盘：

```text
AMD: 约 44 名被试的数据盘
NVIDIA: 剩余约 72 名被试的数据盘
```

因此把 `sub-130` CUDA 输出和 `sub-031` DirectML 输出逐帧比较在科学上无意义。当前 validation chain 改为：

```text
AMD 已完成：同输入 Py-Feat CPU reference ↔ DirectML parity
NVIDIA 待完成：同一 sub-130 sample Py-Feat CPU reference ↔ PyTorch CUDA parity
```

两边共同锚定同一个 Py-Feat 2.1.1 scientific reference。如果以后需要真正的 cross-device row-wise parity，再复制一小份 representative sample 到另一台机器即可，无需迁移完整数据盘。

## 6. NVIDIA sub-130 parity Gate

进入 NVIDIA full-video 之前检查：

1. sample timestamp grid 与 15 Hz 目标一致；
2. CPU/CUDA input frame identity 完全一致；
3. Face coverage / face count agreement；
4. bbox / FaceScore；
5. AU20 / emotion7 / V-A；
6. pose / gaze；
7. 478 mesh / blendshapes；
8. primary tracking / eyelid derived；
9. output schema / subject provenance；
10. CUDA device evidence、peak GPU memory、throughput。

允许 CPU/CUDA 正常浮点微小误差；不允许采样点、schema、primary 语义或科学变量定义分叉。

## 7. 当前仍为两硬件共同待冻结的项目

- NVIDIA representative timestamp/capture-gap stress subject（从 NVIDIA 当前 72 人数据盘选择，不硬编码 AMD `sub-033`）；
- blink event threshold；
- `perclos80_proxy` rolling/event threshold；
- full-video formal runner 的 completion/resume/QC orchestration；
- Face/Pose/Motion/body_motion_energy 正式统一。

这些项目最终形成同一 scientific definition，由 AMD / NVIDIA 两个 runtime 分别实现。
