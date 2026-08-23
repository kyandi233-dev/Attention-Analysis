# Models

`models/` 当前主要保存**历史候选模型和第三方上游源码副本**。它不是当前正式 NIR runtime 的模型入口。

当前正式 NIR 使用的模型资产位于：

```text
training/nir-eye-yolo/runs/yolo26n_eye_100epoch/weights/best.pt
runtime/nir-formal/models/nir-eye-yolo26n-best.pt
runtime/nir-formal/models/ritnet-best_model.pkl
```

因此整理 `models/` 时，判断标准不是“以前是否用过”，而是：

1. 当前正式 pipeline 是否直接依赖；
2. 新设备正式运行是否必须包含；
3. 是否只是历史方法复现所需；
4. 是否已经可以通过工作记录、decision、Git 历史/冻结历史版本和上游来源重新获得。

## 当前目录结构

```text
models/
├── external/      # 第三方完整源码/算法仓库副本
├── historical/    # 历史候选的独立预训练模型文件
└── README.md
```

## 当前正式模型资产不在这里

### YOLO26n 眼框模型

训练产物：

```text
../training/nir-eye-yolo/runs/yolo26n_eye_100epoch/weights/best.pt
```

正式 runtime 冻结副本：

```text
../runtime/nir-formal/models/nir-eye-yolo26n-best.pt
```

### RITnet

正式 runtime：

```text
../runtime/nir-formal/ritnet/
../runtime/nir-formal/ritnet_runtime.py
../runtime/nir-formal/models/ritnet-best_model.pkl
```

正式 runtime 内的 RITnet 来源和冻结版本见：

```text
../runtime/nir-formal/ritnet/UPSTREAM.md
```

## `external/` 保留状态审计

| 目录 | 历史用途 | 当前正式依赖 | 当前建议 |
|---|---|---|---|
| `ritnet/` | RITnet 完整上游仓库副本 | **否**；正式运行使用 `runtime/nir-formal/` 中的冻结最小实现 | **可删候选**；先确认历史脚本不再需要直接调用上游 `infer_ritnet.py` 等文件 |
| `deepvog/` | DeepVOG 瞳孔候选 | 否 | **可删候选**；保留来源/版本/淘汰依据即可 |
| `deepvog-3d/` | DeepVOG 3D 历史资源 | 否 | **可删候选** |
| `pye3d-detector/` | pye3d 历史候选/比较资源 | 否 | **可删候选**；删除前确认无仍需主线运行的历史入口 |
| `yolo-face-parts-detector/` | 特写 eye ROI 历史候选 | 否 | **可删候选**；相关 `roi_faceparts.py` 已退出正式路线 |

上述“可删候选”表示从当前 `main` 精简是合理方向，**不表示已经授权删除**。实际删除仍需用户明确批准。

## `historical/` 保留状态审计

| 文件 | 历史用途 | 当前正式依赖 | 当前建议 |
|---|---|---|---|
| `face_landmarker.task` | MediaPipe 完整人脸 ROI / Iris 历史候选 | 否 | **可删候选** |
| `yolov8n-face.onnx` | YOLO-face ROI 历史候选 | 否 | **可删候选** |
| `yunet_2023mar.onnx` | YuNet ROI 历史候选 | 否 | **可删候选** |

这些模型当前仍由历史 `configs/formal.yaml` 和部分历史脚本引用。该配置文件已经明确标记为 `HISTORICAL CONFIG ONLY`，不属于当前正式 NIR pipeline。

如果后续从 `main` 删除这些模型，应同时处理相关历史脚本的状态，避免留下看似可直接运行、实际上已缺少模型依赖的入口。

## 为什么 `training/` 和 `runtime/` 不能一起删

虽然 `datasets/`、`training/` 也记录了研究过程，但它们仍然是当前正式结果的直接 provenance：

```text
datasets/
→ training/
→ YOLO best.pt
→ runtime 冻结副本
→ 正式 NIR 全量分析
```

因此它们属于**当前正式结果的可追溯资产**，与已经退出路线的 DeepVOG / YuNet / face-parts 等第三方候选不同。

## 目标状态

如果后续历史依赖审计确认无阻碍，`models/` 可以大幅瘦身，甚至最终只保留一个 `README.md` 作为历史模型注册表，而正式运行资产继续分别由：

```text
datasets/
training/
runtime/
```

承担。

这样能够避免“看到 `models/` 就误以为里面都是当前要用的模型”。

## 删除原则

- 不因为“重复”就删除正式 runtime 的冻结权重；runtime 副本用于形成自包含正式分析包。
- 已淘汰且可重新获取的第三方完整源码，不需要仅为了目录完整而永久留在 `main`。
- 删除第三方源码前，应保存上游 URL、版本/commit、license、项目中使用目的和淘汰依据。
- 删除历史模型文件前，应确保其研究结论已由工作记录 / decisions / Git 历史稳定记录。
- 任何实际删除仍需用户明确批准。
