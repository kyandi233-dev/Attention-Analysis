# 046｜NVIDIA CUDA RGB 运行路线

**当前开发 Branch:** `rgb-nvidia`

> `nvidia-cuda` 是 NVIDIA 综合线，目前有人在实际使用；本轮 RGB 文档同步不直接修改它。`rgb-nvidia` 与 `nvidia-cuda` 属于同一个 Attention-Analysis 项目，拆分只是为了让 CUDA RGB 能独立开发、减少冲突。分支关系见 `docs/010-overview/015-并行分支与同步约定.md`。

## 1. 目的

在不改变已冻结 scientific contract 的前提下，为 NVIDIA GPU 建立 CUDA 执行版本。当前 RGB CUDA 开发在 `rgb-nvidia` 继续；成熟改动之后再同步/回并 NVIDIA 综合线。

当前硬盘分工：NVIDIA 工作站连接剩余约 72 名被试的数据盘，AMD 工作站连接另一块约 44 名被试的数据盘。因此 NVIDIA representative Face Gate 使用 `sub-130`；AMD 历史 `sub-031` / `sub-033` 继续作为 AMD provenance，不要求在 NVIDIA 本机存在。

## 2. 与 NIR 双后端策略一致

现有 NIR 采用“不同硬件后端、相同科研口径”的方式：

```text
NVIDIA: CUDA
AMD:    ONNX Runtime DirectML
```

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
- 20 AU、7 emotion、V/A、gaze、6DoF pose、478×3 mesh、68 compatibility landmarks、blendshapes 全保留；
- identity 不属于当前 accepted scientific core；
- `eyeBlinkLeft/Right` 保留 native raw；
- EAR / aperture-iris / normalized openness / closure proxy 为可重算 derived；
- primary-face、QC 与 output provenance 与 AMD 版本保持同一 scientific semantics。

CUDA 可以改变执行框架、device scheduling 和最优 batch，但不得删减科学字段、改变采样时间点或改变变量定义。

## 4. 当前 NVIDIA 实现状态

CUDA representative dry-run runner 已实现：

```text
scripts/face_formal_dryrun_cuda.py
```

第一轮 Gate 故意使用已抽取好的 timestamp-driven sample，把 scientific parity 与 I/O 优化分开：

```text
sub-130 original RGB/timestamps
→ timestamp-driven 15 Hz representative sample (~3600)
→ 同一 sample Py-Feat 2.1.1 native CPU reference
→ 同一 sample Py-Feat 2.1.1 native CUDA
→ field-level parity
→ tracking / eyelid / QC
```

通过后，正式 NVIDIA full-video runner 应升级为：

```text
original AVI
→ timestamp-driven 15 Hz
→ decode / prefetch
→ Py-Feat Detectorv2 on CUDA
→ same scientific raw schema
→ same tracking / eyelid derived / QC
```

RTX 5070 的最优 batch 由 NVIDIA 实机 benchmark 决定。AMD DirectML 的 RetinaFace B8 / multitask B16 只是 AMD 工程参数，不强制照搬。

## 5. AMD formal runner 进展如何同步到 NVIDIA

AMD `rgb-amd` 已经实现并开始验收：

- full-span timestamp-driven Face frame preparation；
- Motion/Pose 正式 subject wrapper；
- original-AVI DirectML full-span Face runner；
- continuous tracking/eyelid formal derive；
- single-subject orchestration；
- subject output / manifest / completion 结构。

这些设计中，共享逻辑可以同步到 NVIDIA；DirectML execution 不能直接照搬。NVIDIA 后续 full-video runner 应尽量复用相同：

- frame manifest schema；
- output naming；
- manifest/completion semantics；
- raw-first retention；
- Motion/Pose orchestration；
- tracking/eyelid derived。

只把 Face executor 换成 native CUDA，并记录 CUDA-specific runtime/provider/batch/GPU memory。

## 6. 为什么不写 sub-031 AMD↔sub-130 NVIDIA 逐帧 parity

不同被试之间不能做 row-wise parity。当前验证链：

```text
AMD 已完成：同输入 Py-Feat CPU reference ↔ DirectML parity
NVIDIA 待完成：同一 sub-130 sample Py-Feat CPU reference ↔ PyTorch CUDA parity
```

两边共同锚定 Py-Feat 2.1.1 scientific reference。需要真正 cross-device parity 时，再复制同一 representative sample 到另一台机器即可。

## 7. NVIDIA sub-130 parity Gate

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

允许正常浮点误差；不允许 sampling、schema、primary semantics 或科学变量定义分叉。

## 8. 当前 NVIDIA 仍未完成

- sub-130 native CPU ↔ CUDA parity 实机结果；
- NVIDIA representative timestamp/capture-gap stress；
- native CUDA original-AVI full-video Face runner；
- single-subject formal CUDA orchestration；
- cohort completion/resume；
- `body_motion_energy`；
- blink event threshold；
- `perclos80_proxy` 最终窗口、分母与 QC。

## 9. 当前执行边界

当前开发 branch：

```text
rgb-nvidia
```

NVIDIA 综合线：

```text
nvidia-cuda
```

由于 `nvidia-cuda` 当前有人使用，本轮不直接修改。`rgb-nvidia` 中通过实机验收的 CUDA RGB 改动再选择性同步回综合线。
