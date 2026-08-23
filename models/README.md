# Models

`models/` 保存历史比较、候选算法和第三方模型资源；当前正式全量分析使用的冻结权重仍保存在自包含的 `runtime/nir-formal/models/` 中。

## 目录结构

```text
models/
├── external/      # 第三方源码/算法仓库
├── pretrained/    # 独立预训练模型文件
└── README.md
```

### `external/`

- `ritnet/`：RITnet 第三方源码与相关资源。
- `deepvog/`：DeepVOG 第三方源码与模型资源。
- `deepvog-3d/`：DeepVOG 3D 相关第三方资源。
- `pye3d-detector/`：pye3d 第三方检测资源。
- `yolo-face-parts-detector/`：历史 face-parts ROI 检测资源。

### `pretrained/`

- `face_landmarker.task`：MediaPipe Face Landmarker 模型。
- `yolov8n-face.onnx`：历史 YOLO-face ROI 候选模型。
- `yunet_2023mar.onnx`：YuNet 人脸检测模型。

## 当前正式分析权重

YOLO26n 眼框训练产物：

```text
../training/nir-eye-yolo/runs/yolo26n_eye_100epoch/weights/best.pt
```

正式 runtime 使用的冻结副本：

```text
../runtime/nir-formal/models/nir-eye-yolo26n-best.pt
```

RITnet 正式 runtime 权重：

```text
../runtime/nir-formal/models/ritnet-best_model.pkl
```

训练目录中的 YOLO `best.pt` 与 runtime 中的 `nir-eye-yolo26n-best.pt` 为同一模型资产；runtime 保留副本是为了形成可直接迁移和运行的正式分析包。

## 边界

- 根 `models/` 主要服务历史比较、诊断和方法复现，不作为正式 NIR runtime 的运行时依赖。
- 当前正式权重不从 runtime 中抽走。
- `configs/formal.yaml` 已同步到新的 `external/` 与 `pretrained/` 路径，以保持历史诊断脚本可运行。
- 后续新增本项目训练模型时，应明确区分训练工作区产物与正式 runtime 冻结副本。
