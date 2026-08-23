# Models

`models/` 现在只作为**模型资产注册与来源说明入口**，不再保存已经退出正式路线的第三方完整源码或历史候选模型二进制。

## 当前正式模型资产

YOLO26n 眼框训练产物：

```text
../training/nir-eye-yolo/runs/yolo26n_eye_100epoch/weights/best.pt
```

正式 NIR runtime 冻结副本：

```text
../runtime/nir-formal/models/nir-eye-yolo26n-best.pt
```

RITnet 正式运行实现与权重：

```text
../runtime/nir-formal/ritnet/
../runtime/nir-formal/ritnet_runtime.py
../runtime/nir-formal/models/ritnet-best_model.pkl
```

RITnet 上游来源、冻结 commit、license 与文件一致性记录见：

```text
../runtime/nir-formal/ritnet/UPSTREAM.md
```

## 2026-08-23 主线精简

经审计并获得用户明确授权，当前 `main` 删除了以下**不参与正式分析**的模型资产：

```text
models/external/ritnet/
models/external/deepvog/
models/external/deepvog-3d/
models/external/pye3d-detector/
models/external/yolo-face-parts-detector/
models/historical/face_landmarker.task
models/historical/yolov8n-face.onnx
models/historical/yunet_2023mar.onnx
```

删除原因：

1. 当前正式 NIR 主链不从这些路径读取任何模型或源码；
2. YOLO26n 的训练 lineage 已由 `datasets/ → training/ → runtime/` 保存；
3. RITnet 正式所需源码、权重、license 与上游 provenance 已冻结在 `runtime/nir-formal/`；
4. DeepVOG、pye3d、MediaPipe、YuNet、YOLO-face、face-parts 等已经退出正式路线；
5. 旧方案仍可通过 Git 历史、`history/tracking-era-2026-08` 与 `docs/工作记录/` 追溯，不需要把完整第三方仓库和历史权重长期放在当前主线；
6. 删除可避免新设备配置或后续维护时误把历史候选当成当前正式依赖。

## 历史脚本与配置

`configs/formal.yaml` 以及部分 `scripts/` 文件属于历史候选阶段，可能仍保留对上述已删除路径的文字或代码引用。它们**不再保证可在当前 `main` 直接运行**。

需要复现旧 ROI / pupil 候选链时，应使用对应 Git 历史版本或：

```text
history/tracking-era-2026-08
```

当前正式运行始终以：

```text
runtime/nir-formal/
```

为准。

## 后续规则

- 第三方模型若只是短期候选，默认不再把完整上游仓库长期 vendor 到 `main`；优先记录上游 URL、commit/version、license、使用目的与验证结果。
- 只有正式 runtime 直接依赖且无法通过常规依赖安装获得的必要源码/权重，才冻结进入 runtime。
- 本项目自行训练的模型继续保留训练工作区与正式 runtime 冻结副本，两者职责不同，不视为无意义重复。
