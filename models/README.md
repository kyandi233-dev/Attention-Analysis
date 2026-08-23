# models｜模型与外部算法资产说明

## 当前整理原则

`models/` 保存了项目不同阶段使用、比较或镜像到本仓库中的模型权重与外部算法源码。这里不是单纯的“当前生产模型目录”，而是同时承载了**正式路线依赖、历史 ROI 候选、瞳孔算法比较资产和外部源码镜像**。

当前**不物理拆分**成 `trained/`、`pretrained/`、`external/` 等新目录。原因不是这些分类没有意义，而是仓库中的历史配置和脚本仍直接引用现有路径；为了目录整齐而移动会降低历史复现性，并引入大量路径修改。

例如历史兼容配置 `configs/formal.yaml` 仍显式引用：

```text
models/face_landmarker.task
models/yunet_2023mar.onnx
models/yolov8n-face.onnx
models/yolo-face-parts-detector-main/weights/yolov8n.pt
models/yolo-face-parts-detector-main/weights/yolov8s.pt
models/RITnet-master/best_model.pkl
models/DeepVOG-master/deepvog/model/DeepVOG_weights.h5
models/DeepVOG-master/deepvog/model/DeepVOG3D_weights.h5
```

因此目前采用“**保留路径 + README 分类说明**”而不是“移动后再修全仓引用”。

## 顶层资产分类

| 资产 | 当前角色 | 整理判断 |
| --- | --- | --- |
| `RITnet-master/` | RITnet 源码/权重资产；与后续 NIR 瞳孔、虹膜分割路线相关 | **保留原路径**；正式全量入口尚未完成独立核验前不移动 |
| `DeepVOG-master/` | DeepVOG 历史瞳孔算法比较资产，历史配置直接引用其权重 | **保留原路径** |
| `DeepVOG-deepvog3d/` | DeepVOG/3D 相关外部源码或历史比较资产 | **保留原路径**，后续如需进一步清理先核验与 `DeepVOG-master/` 的真实关系 |
| `pye3d-detector-master/` | pye3d 外部算法源码镜像 / 历史研究资产 | **保留原路径** |
| `yolo-face-parts-detector-main/` | 历史 face-parts ROI 候选；旧配置直接引用内部权重 | **保留原路径** |
| `face_landmarker.task` | MediaPipe FaceLandmarker 模型；历史完整人脸 ROI 候选 | **保留原路径** |
| `yunet_2023mar.onnx` | YuNet 人脸检测模型；历史 ROI 候选 | **保留原路径** |
| `yolov8n-face.onnx` | 旧 YOLO-face 人脸检测模型；历史 ROI 候选 | **保留原路径** |

## 与当前 YOLO26n 眼框模型的区别

当前正式使用过的 YOLO26n 是项目自行训练的**双眼 eye bounding-box detector**，它与本目录中的 `yolov8n-face.onnx` 不是同一个模型，也不是同一用途。

当前 GitHub 分支中，YOLO26n 正式训练权重实际位于：

```text
yolotrain/runs/yolo26n_eye_100epoch/weights/best.pt
```

所以不要因为目录名叫 `models/` 就把 `best.pt` 的真实位置描述成这里，也不要把历史 `yolov8n-face.onnx` 误认为正式眼框模型。

## 历史 ROI 候选为什么继续保留

正式 NIR 路线确定前，项目比较过多种完整人脸 / face-parts ROI 后端。当前 `scripts/00-目录与映射.md` 已将 MediaPipe、YuNet、旧 YOLO-face、face-parts 等明确标记为历史 ROI 候选，而不是当前正式双眼检测路线。

这些模型和源码仍有研究 provenance 价值：

- 可以复现早期 ROI selection / benchmark；
- 可以解释为什么完整人脸检测方案后来被放弃；
- 可以追溯历史 artifacts 的生成条件；
- 可以避免把失败或淘汰路线误写成“从未尝试过”。

因此，**历史路线被淘汰不等于资产应被删除**。

## 后续如需进一步整理

只有在以下条件满足后，才值得考虑物理移动：

1. 已确认某资产的真实角色和来源；
2. 已搜索所有代码、配置、文档和工作记录中的路径引用；
3. 已确认移动不会破坏正式流程或历史复现；
4. 已设计兼容路径或同步修改全部引用；
5. 如移动方案涉及删除旧路径，必须先取得用户明确同意。

在当前阶段，`models/` 的最佳整理方式是**说明职责，而不是重排目录**。
