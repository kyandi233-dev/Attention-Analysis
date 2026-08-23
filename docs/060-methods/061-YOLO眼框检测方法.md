# 061｜YOLO 眼框检测方法

## 方法定位

YOLO 在当前 NIR 路线中的职责是**定位眼睛 bbox**，不是直接测量瞳孔，也不是只“记住坐标”。模型以整帧 NIR 图像像素为输入，学习能够区分眼睛区域的视觉特征，并同时输出目标类别、置信度与 bounding box 坐标。后续程序再根据 bbox 裁剪单眼 ROI，交给 RITnet 做像素级分割。

因此当前链路中的职责关系是：

```text
NIR frame
    ↓
YOLO26n
    ↓
eye bbox
    ↓
ROI crop
    ↓
RITnet segmentation
```

YOLO 解决“眼睛在哪里”，RITnet 解决“ROI 内哪些像素属于瞳孔/虹膜等结构”。

## 训练数据

冻结训练数据来源于 `datasets/nir-eye-dataset-v1/`，在 `yolotrain/` 中合并 batch1 / batch2 后按被试级别固定划分 train / val / test。这样做的原因是连续视频相邻帧高度相似；如果按图片随机拆分，同一被试的近邻帧可能同时出现在训练集和测试集，造成数据泄漏和虚高指标。

最终 test split 为 7 名被试、85 张图片、169 个 eye boxes。test 不参与训练、epoch 选择和运行阈值调节。

## 正式模型

当前正式训练资产为：

```text
yolotrain/runs/yolo26n_eye_100epoch/weights/best.pt
```

正式 run 使用 YOLO26n，训练 100 epochs，`imgsz=640`、`batch=16`、`workers=0`、`seed=0`，训练机器记录为 CPU。CPU/GPU 会显著影响训练速度，但在相同软件、算法、随机性控制和数值条件下，不因为“用 CPU”就自动得到质量更差的模型。

runtime package 中：

```text
runtime/nir-yolo-tracking-ritnet-v1/models/nir-eye-yolo26n-best.pt
```

与上述 `best.pt` 是同一个 Git blob，因此是逐字节相同的权重副本，而不是重新训练的另一版本。

## Validation 与 held-out test

训练第 100 epoch 的 validation 记录为：

| 指标 | validation |
|---|---:|
| Precision | 0.99922 |
| Recall | 1.00000 |
| mAP50 | 0.99500 |
| mAP50-95 | 0.71810 |

2026-08-22 的冻结 test 评价记录为：

| 指标 | test |
|---|---:|
| Precision | 0.9754 |
| Recall | 0.9645 |
| mAP50 | 0.9913 |
| mAP50-95 | 0.6589 |

val 与 test 不能混为同一组结果。val 用于训练过程监控和阈值选择；test 用于冻结后的最终泛化评价。

运行置信度阈值 `0.40` 是先在 val 上按 IoU=0.50 下 F1 选择，再冻结到 test，因此没有使用 test 调参。

## 推理时如何选眼框

portable runtime 中 YOLO 只保留类别 `eye`，按 confidence 从高到低排序；正式 tracking 链需要两只眼时取最高的两个检测框，再按画面横坐标排序为：

```text
frame_left
frame_right
```

这里的 `frame_left / frame_right` 只描述图像中的左右位置，**不能直接解释为被试解剖学左眼/右眼**。如果后续分析需要解剖学眼别，必须经过相机镜像方向和实验记录的额外映射。

## 为什么不必每帧都跑 YOLO

视频相邻帧中的眼睛位置通常连续变化。若每帧都执行神经网络检测，会重复计算大量相似信息，因此实际管线可以采用：

```text
YOLO 重新检测
    ↓
若干帧 tracking
    ↓
周期性 YOLO 校正
```

这样 YOLO 充当较昂贵但可靠的“定位锚点”，tracking 负责中间帧的廉价位置更新。tracking 发生失败、越界或异常跳变时再提前回退到 YOLO。

具体 tracking 机制见 `062-Tracking策略.md`。

## 失败状态与 QC

检测阶段应至少区分：

- `yolo_missing`：没有检测到眼睛；
- `single_eye`：只得到一个 eye bbox；
- `extra_boxes`：得到两个以上候选框；
- 正常双眼检测：得到两个候选并进入 tracking / ROI。

这些状态不能直接补成“成功”。正式分析中需要保留失败信息，以便后续计算可用率、检查特定被试或视频片段是否存在系统性问题。

## 当前 provenance 边界

当前可以确定模型、test 结果、val 选择阈值和 runtime 中的实现逻辑。但当前 Git 分支没有保存正式全量运行的最终 `run_manifest.json`，因此不能仅根据 2026-08-22 portable package 默认配置声明全量最终运行一定使用相同置信度、tracker 或重检测周期。能核验的事实与尚缺的 final-run provenance 必须分开记录。
