# Models

`models/` 保存历史比较、候选算法和第三方预训练资源。当前正式全量分析实际使用的冻结权重保存在 `runtime/nir-formal/models/`，以保证正式运行包自包含。

## 根目录现有模型资产

| 路径 | 角色 |
|---|---|
| `RITnet-master/` | RITnet 第三方源码与相关资源 |
| `DeepVOG-master/` | DeepVOG 第三方源码与模型资源 |
| `DeepVOG-deepvog3d/` | DeepVOG 3D 相关第三方资源 |
| `pye3d-detector-master/` | pye3d 第三方检测资源 |
| `yolo-face-parts-detector-main/` | 历史/候选 face-parts ROI 检测资源 |
| `face_landmarker.task` | MediaPipe Face Landmarker 预训练模型 |
| `yolov8n-face.onnx` | 历史 YOLO-face ROI 候选模型 |
| `yunet_2023mar.onnx` | YuNet 人脸检测模型 |

## 当前正式分析权重

YOLO26n 训练产物：

```text
../yolotrain/runs/yolo26n_eye_100epoch/weights/best.pt
```

正式 runtime 中使用的冻结副本：

```text
../runtime/nir-formal/models/nir-eye-yolo26n-best.pt
```

RITnet 正式 runtime 权重：

```text
../runtime/nir-formal/models/ritnet-best_model.pkl
```

训练目录中的 YOLO `best.pt` 与 runtime 中的 `nir-eye-yolo26n-best.pt` 对应同一正式眼框模型资产；runtime 保留副本是为了形成可直接迁移的运行包。

## 整理原则

- 根 `models/` 主要保留第三方、历史和候选资源；
- 正式运行所需权重继续跟随 `runtime/nir-formal/`；
- 后续整理第三方资源时，只做移动/重命名并同步当前有效引用；
- 日期型历史工作记录中的旧路径不追溯改写。
