# 08-26-16｜RGB Face 全脸 QC 与 NVIDIA 分支启动

## sub-031 当前状态

`sub-031` window-aware primary/eyelid dry-run 已通过：3600/3600 primary-frame coverage、3600/3600 eye geometry valid；baseline open reference 正确使用 900 个 baseline 样本。30 个 multi-face frames 均位于 `baseline_start`，主轨迹仍完整覆盖 450/450 frames，secondary tracks 仅短暂出现。

## QC 可视化 v0.2

原 `face_qc_visualize.py` 首版是针对 blink/EAR/iris gate 的局部可视化，因此只主动绘制 EAR 所用眼睑点与 iris ring。完整 478-point mesh 一直保留在 raw parquet 中，并非模型只输出眼睛点。

新增：

- `scripts/face_qc_visualize_v02.py`

v0.2 在保留原 bbox / primary-secondary track / eye-iris highlight / EAR / eyeBlink / openness / pose / gaze / emotion 文本的同时，给 primary face 叠加全部 478 个 mesh 点，并额外画 face oval、outer lips、nose bridge 轮廓，使整脸 landmark drift 可直接做视觉 QC。

正式 scientific raw 不因可视化需求改变；QC 仍只从原 AVI + saved raw/derived 后生成，不重跑 Py-Feat。

## NVIDIA RGB 分支策略

现有 `rgb-dev` 与 `nvidia-cuda` 已明显分叉，不能直接整分支 merge，否则会把 AMD-specific NIR runtime / ONNX 资产变化等一并带入 NVIDIA 稳定线。

因此从当前 `nvidia-cuda` 新建：

```text
rgb-nvidia-cuda
```

作为 NVIDIA RGB 开发分支。后续仅选择性同步 RGB scientific contract / schema / sampling / tracking / derived / QC 逻辑，并实现 CUDA execution backend；通过 AMD-vs-NVIDIA parity 后再考虑合并回 `nvidia-cuda`。

CUDA 版本应保持：

- Py-Feat 2.1.1 Detectorv2 scientific core；
- timestamp-driven 15 Hz；
- original AVI direct decode；
- RetinaFace B8；
- pending face chips → multitask B16；
- full native outputs retained；
- identical primary/eyelid derived semantics；
- identical QC visualization contract。

只允许执行后端与必要的设备调度不同，不因 NVIDIA 版本改变科学字段定义。

## 尚未冻结的共同项

NVIDIA 同步可以现在开始，但以下仍应由 AMD/NVIDIA 共用同一最终定义并在 representative QC 后冻结：

1. `sub-033` timestamp/capture-gap stress test；
2. blink event threshold；
3. `perclos80_proxy` rolling/event threshold；
4. full-video formal runner completion/resume/QC orchestration。
