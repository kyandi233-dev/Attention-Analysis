# 046｜NVIDIA CUDA RGB 运行路线

**Branch:** `rgb-nvidia-cuda`

## 1. 目的

在不改变已经由 `rgb-dev` 冻结的 RGB scientific contract 的前提下，为 NVIDIA GPU 建立 CUDA 执行版本。该分支从稳定 `nvidia-cuda` 创建，避免直接 merge `rgb-dev` 时把 AMD-specific NIR runtime / ONNX 资产变化一并带入 NVIDIA 主线。

## 2. 必须保持与 AMD RGB 一致的科学定义

- Face backend：Py-Feat 2.1.1 Detectorv2 scientific core；
- timestamp-driven Face 15 Hz；
- original AVI direct decode；
- RetinaFace batch 8；
- face chips 跨 detector batch pending；
- multitask batch 16；
- 所有检测 faces 先保留；
- 20 AU、7 emotion、V/A、gaze、6DoF pose、478×3 mesh、68 compatibility view、52 blendshapes 全保留；
- `eyeBlinkLeft/Right` 保留 native raw；
- EAR / aperture-iris / normalized openness / closure proxy 为可重算 derived；
- primary-face 规则、QC 可视化和输出 schema 与 AMD 版本一致。

CUDA 版本只能改变执行后端、设备调度和必要的环境依赖，不得因硬件变化删减科学字段或改变时间采样定义。

## 3. CUDA backend 推荐实现

优先使用与 AMD 相同的 ONNX 模型和预处理/后处理定义，执行 provider 改为 CUDA，以最大化跨硬件 parity：

```text
original AVI
→ timestamp-driven 15 Hz
→ reader/preprocess prefetch
→ RetinaFace ONNX / CUDA B8
→ same decode + NMS + 1.2 square-reflect crop
→ pending face chips
→ multitask ONNX / CUDA B16
→ same postprocess/schema
→ parquet
```

环境建议使用 `onnxruntime-gpu`，并显式要求 `CUDAExecutionProvider` 为 primary provider；不得静默退化为 CPU-only。

## 4. 验证 Gate

在进入 NVIDIA 全量前，至少对同一 representative 片段做 AMD-vs-NVIDIA parity：

1. Face coverage / face count agreement；
2. bbox IoU / FaceScore；
3. AU20 / emotion7 / V-A；
4. pose / gaze；
5. 478 mesh / blendshapes；
6. primary tracking / eyelid derived；
7. throughput / device-provider evidence。

只允许浮点执行带来的微小数值差异；不允许 schema、采样点或 primary 语义分叉。

## 5. 当前仍为两硬件共同待冻结的项目

- `sub-033` timestamp/capture-gap stress test；
- blink event threshold；
- `perclos80_proxy` rolling/event threshold；
- full-video formal runner 的 completion/resume/QC orchestration。

这些项目最终应形成同一 scientific definition，由 AMD / NVIDIA 两个 runtime 共同实现。
