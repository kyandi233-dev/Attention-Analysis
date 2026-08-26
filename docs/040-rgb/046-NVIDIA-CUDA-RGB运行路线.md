# 046｜NVIDIA CUDA RGB 运行路线

**Branch:** `rgb-nvidia-cuda`

## 1. 目的

在不改变已经由 `rgb-dev` 冻结的 RGB scientific contract 的前提下，为 NVIDIA GPU 建立 CUDA 执行版本。该分支从稳定 `nvidia-cuda` 创建，避免直接 merge `rgb-dev` 时把 AMD-specific NIR runtime / ONNX 资产变化一并带入 NVIDIA 主线。

## 2. 与 NIR 双后端策略一致

现有 NIR 正式链已经证明可以采用“不同硬件后端、相同科研口径”的维护方式：

- NVIDIA 正式历史基线默认 `pytorch-cuda`，直接使用 `.pt/.pkl`；
- AMD 正式主链使用 ONNX Runtime DirectML；
- NVIDIA 后来新增的 `ort-cuda` 只是可选高速 profile，不改变既有 PyTorch CUDA 正式基线。

因此 RGB 不要求 AMD 与 NVIDIA 使用完全相同的执行框架。跨硬件一致性由 scientific contract + representative parity 保证，而不是强制两边都使用 ONNX。

## 3. 必须保持与 AMD RGB 一致的科学定义

- Face backend：Py-Feat 2.1.1 Detectorv2 scientific core；
- timestamp-driven Face 15 Hz；
- original AVI direct decode；
- RetinaFace / multitask 的检测阈值、NMS、1.2 square-reflect crop、ImageNet normalize 与 canonical postprocess 一致；
- 所有检测 faces 先保留；
- 20 AU、7 emotion、V/A、gaze、6DoF pose、478×3 mesh、68 compatibility view、52 blendshapes 全保留；
- `eyeBlinkLeft/Right` 保留 native raw；
- EAR / aperture-iris / normalized openness / closure proxy 为可重算 derived；
- primary-face 规则、QC 可视化和输出 schema 与 AMD 版本一致。

CUDA 版本可以改变执行框架、设备调度和最优 batch 组织，但不得因硬件变化删减科学字段、改变采样时间点或改变变量定义。

## 4. NVIDIA 默认实现：PyTorch CUDA

NVIDIA RGB 默认沿用 NIR 的平台策略，优先实现原生 PyTorch CUDA / Py-Feat 2.1.1 路线，而不是强制使用 AMD 的 ONNX graph：

```text
original AVI
→ timestamp-driven 15 Hz
→ decode/prefetch
→ Py-Feat Detectorv2 / RetinaFace on CUDA
→ Py-Feat 2.1.1 square-pad crop contract
→ multitask scientific core on CUDA
→ canonical postprocess
→ same raw schema
→ same tracking / eyelid derived / QC
```

优先复用 Py-Feat 2.1.1 原生模型与已核对 source contract。具体 CUDA batch size 由目标 NVIDIA 机器 benchmark 决定；AMD 上冻结的 RetinaFace B8 / multitask B16 是 DirectML 工程参数，不强制作为 CUDA 最优参数，但任何 batch 变化不得改变逐样本输出定义。

ONNX Runtime CUDA 可以保留为可选 profile / parity 辅助路线，不作为首要要求。

## 5. AMD ↔ NVIDIA parity Gate

在进入 NVIDIA 全量前，对同一 `sub-031` representative 片段做跨后端 parity：

1. 时间采样点完全一致；
2. Face coverage / face count agreement；
3. bbox IoU / FaceScore；
4. AU20 / emotion7 / V-A；
5. pose / gaze；
6. 478 mesh / blendshapes；
7. primary tracking / eyelid derived；
8. output schema / subject provenance；
9. throughput 与实际 CUDA device evidence。

允许 PyTorch CUDA 与 ONNX Runtime DirectML 的正常浮点/插值微小差异；不允许采样点、schema、primary 语义或科学变量定义分叉。

特别注意：AMD 当前正式候选使用 OpenCV remap/resize 来近似 Py-Feat `grid_sample(align_corners=False)`，其与 PyTorch reference 的 small interpolation drift 已在 real-300 parity 中验证可接受。NVIDIA 原生 PyTorch 路线应优先保留 Py-Feat 2.1.1 原生 crop/sampling 行为，因此跨后端 parity 的目标是科学等价，而不是 bitwise identity。

## 6. 当前仍为两硬件共同待冻结的项目

- `sub-033` timestamp/capture-gap stress test；
- blink event threshold；
- `perclos80_proxy` rolling/event threshold；
- full-video formal runner 的 completion/resume/QC orchestration。

这些项目最终应形成同一 scientific definition，由 AMD / NVIDIA 两个 runtime 共同实现。