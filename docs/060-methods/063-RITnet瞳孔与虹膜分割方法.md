# 063｜RITnet 瞳孔与虹膜分割方法

## 方法定位

RITnet 位于当前 NIR 管线的 ROI 之后。YOLO / tracking 先确定眼睛位置并裁剪单眼区域，RITnet 再对眼睛 ROI 做像素级语义分割。因此两者的输出层级不同：

```text
YOLO / tracking
    ↓
眼睛 bounding box
    ↓
单眼 ROI
    ↓
RITnet
    ↓
像素级类别图
    ↓
瞳孔 mask → 椭圆 / 中心 / 直径等指标
```

YOLO 的 bbox 只定义“在哪里分析”；真正的瞳孔形状、中心和尺寸来自 RITnet 分割结果，而不是从 YOLO 框大小直接推导。

## 当前权重与实现

仓库中的 RITnet 权重为：

```text
models/RITnet-master/best_model.pkl
```

portable runtime 中保存一份运行副本：

```text
runtime/nir-yolo-tracking-ritnet-v1/models/ritnet-best_model.pkl
```

两者 Git blob SHA 完全相同，因此 runtime 使用的是同一份 RITnet 权重。

当前 portable 推理封装位于：

```text
runtime/nir-yolo-tracking-ritnet-v1/ritnet_runtime.py
```

网络定义来自 package 内的 `ritnet/densenet.py`，运行时加载 `DenseNet2D` 并进入 `eval()` 模式。

## ROI 与输入预处理

`run_pipeline.py` 首先依据 YOLO/tracking bbox 扩展并裁剪眼睛区域，将 ROI 转为灰度图，再缩放为 320×160 的标准单眼 ROI。之后 `RitnetRuntime.infer()` 会把该 ROI 再缩放到 RITnet 的网络输入尺寸；portable package 当前配置为 640×400。

进入网络前还执行：

1. gamma 变换，指数为 0.8；
2. CLAHE 局部直方图均衡，`clipLimit=1.5`、`tileGridSize=(8,8)`；
3. 像素归一化到近似 `[-1, 1]`；
4. 增加 batch/channel 维度后送入 PyTorch 模型。

这意味着 RITnet 并不是直接读取原始整帧 NIR 图像，而是在已经定位和标准化后的单眼灰度 ROI 上工作。

## 从网络输出到瞳孔椭圆

网络输出 logits 后，代码在类别维度执行 softmax，同时取 argmax 得到每个像素的预测类别。portable runtime 将类别 `3` 作为 pupil：

```text
pred == 3 → pupil mask
```

随后把 pupil mask 缩回 320×160 ROI 坐标系，提取外轮廓，并选择面积最大的轮廓作为当前瞳孔候选。

只有满足以下最低几何条件才继续：

- 存在轮廓；
- 最大轮廓至少有 5 个点；
- 轮廓面积至少为 5 px²。

通过后使用 OpenCV `fitEllipse()` 拟合椭圆，输出：

- `center_x`, `center_y`：瞳孔中心；
- `axis_a`, `axis_b`：拟合椭圆两轴；
- `angle_deg`：椭圆角度；
- `mask_area`：瞳孔 mask 面积；
- `equiv_diameter`：按等面积圆计算的等效直径；
- `pupil_confidence`：网络在预测为 pupil 的像素上的 pupil softmax 概率均值。

等效直径计算为：

```text
equiv_diameter = 2 × sqrt(mask_area / π)
```

因此它描述的是与当前 pupil mask 面积相同的圆的直径，而不是简单取椭圆长轴或短轴。

## `pupil_confidence` 的含义

当前代码中的 `pupil_confidence` 不是 YOLO detection confidence。它来自 RITnet softmax：只在最终被 argmax 判为 pupil 的像素上读取类别 3 的概率，再求平均。

因此：

- YOLO confidence 表示某个 bbox 是 eye 的检测置信度；
- RITnet pupil confidence 表示已预测 pupil 区域内部的分类置信程度。

两者处于完全不同的模型和处理阶段，不能直接比较数值高低。

portable package 的 `pupil_confidence_min` 当前为 `null`，即 package 创建时只输出该值，没有把它作为生产拒绝阈值。最终全量是否后来设置了额外门控，需要 final-run provenance 才能确认。

## 失败与状态语义

如果不存在有效 pupil 轮廓，`infer()` 返回：

```text
found = False
```

上游 `run_pipeline.py` 再将其记录为 `ritnet_missing`。如果成功形成椭圆，则记录 `observed`；如果 ROI 扩展碰到图像边界，状态还会保留 `roi_clipped` 信息。

当前 portable 路线不把 `ritnet_missing` 自动插值成 `observed`。这符合测量层与插值层分离的原则：原始观测失败应先被保留，后续若有时间序列插值，也必须明确标记为插值值而不是伪装成直接观测。

## 为什么 ROI 质量会影响 RITnet

RITnet 的输入已经不是完整画面，所以它高度依赖前一级 ROI。若 YOLO/tracking 框偏移、眼睛被裁掉、扩展范围过小，或者 ROI 包含过多眉毛/鼻梁/背景，RITnet 的输入分布都会发生改变。

因此“RITnet 没找到瞳孔”不一定意味着 RITnet 本身失效，可能来自：

```text
YOLO / tracking 定位错误
        ↓
ROI 裁剪错误
        ↓
RITnet segmentation 失败
```

正式 QC 需要分开记录 `yolo_missing`、tracking 回退、`roi_clipped` 和 `ritnet_missing`，才能定位失败发生在哪一级，而不能只统计最终“有没有瞳孔”。

## GPU 与 CPU

`RitnetRuntime` 会根据传入 device 选择 CUDA 或 CPU。如果明确指定 `cpu`，或者 PyTorch 检测不到 CUDA，则使用 CPU；数字设备号如 `0` 会映射到 `cuda:0`。

设备主要决定推理速度。对于大量 25 分钟视频，RITnet 每帧要处理两只眼睛，因此它可能和 YOLO 一样成为重要计算成本；tracking 只能减少 YOLO 调用次数，不能自动减少每帧两次 RITnet 分割。

## 当前 provenance 边界

当前已经可以核验 RITnet 权重身份、预处理、类别 3 pupil mask、椭圆拟合、confidence 计算和 portable package 的状态输出。但当前 Git 分支没有正式全量运行的最终 manifest，因此以下内容仍不能从 08-22 package 默认值推断为最终冻结方法：

- 最终 ROI expansion；
- 是否设置 `pupil_confidence` 拒绝阈值；
- 是否增加其他几何/光度门；
- 全量运行时的实际 device / 软件环境；
- 后续时序插值与正式指标汇总规则。

找到 final-run manifest 或等价工作记录后，再把这些内容补成最终方法版本。
