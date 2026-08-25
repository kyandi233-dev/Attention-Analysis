# 08-25-01 NIR RITnet 全分类补全与瞳孔分析方法

## 1. 当前问题与修复目标

已完成的正式 NIR pipeline 保存了逐眼 `eyes.csv`，其中包含 subject、phase、frame/time、YOLO ROI 坐标以及 pupil 几何，但历史 downstream 只把 RITnet class 3（pupil）转成了正式变量，没有保留 class 0/1/2 的完整结构指标。

本次修复不重新运行 YOLO。利用已有：

```text
video
frame_idx
roi_x1 / roi_y1 / roi_x2 / roi_y2
```

从原始 AVI 重建同一 ROI，只重新运行 RITnet full-class segmentation。

## 2. 冻结的 RITnet 方法

两台 GPU 机器统一使用：

```text
RITnet input: 640×400
precision: FP32
batch: 16
analysis coordinates: 320×160
class 0: background
class 1: sclera
class 2: iris
class 3: pupil
```

不使用 FP16，不使用 512×320 或其他降分辨率方案。

AMD 与 NVIDIA full-class 扩展统一使用 `ritnet-b16-fp32.onnx`，SHA256：

```text
1933f44f483b350e17249a37b4a2ebe8b5e32f83fc8c1eb1a21c27e96477e621
```

AMD 的 execution provider 为 DirectML；NVIDIA 为 CUDAExecutionProvider。

## 3. 四分类与几何变量

逐眼保留四类 pixel count / fraction，并计算：

- pupil ellipse：center、axis、angle、contour area、ellipse area、equivalent diameter、geometric-mean diameter；
- `iris_outer = iris OR pupil` 后拟合虹膜外轮廓 ellipse；
- `ocular = sclera OR iris OR pupil` 的可见眼球区域；
- pupil 与 iris center offset；
- iris fill ratio；
- connected components / largest-component fraction；
- ROI edge touch；
- ocular bbox / aperture geometry。

class 2 的 `iris_pixels` 只代表可见虹膜组织，本身不包含中央 pupil hole，因此不把 `pupil_pixels / iris_pixels` 作为主 pupil normalization。

## 4. 主瞳孔指标

正式主指标固定为：

```text
fullclass_pupil_to_iris_diameter_ratio
```

\[
PIR_D=\frac{D_{pupil}}{D_{iris}}
=\frac{\sqrt{a_p b_p}}{\sqrt{a_i b_i}}
\]

其中 iris 不是单独 class 2，而是对 `class 2 OR class 3` 的外边界拟合椭圆。这样 pupil hole 不会缩小虹膜尺度参照。

该比例是无量纲尺度指标，比直接比较原始 pupil pixel diameter 更适合抵消相机距离、ROI尺度等影响。

辅助变量保留：

```text
fullclass_pupil_to_iris_ellipse_area_ratio
fullclass_pupil_to_iris_contour_area_ratio
```

但不作为默认主结果替代 diameter ratio。

## 5. normalization_valid

正式 normalized pupil 分析优先要求：

```text
fullclass_normalization_valid == True
```

门控要求包括：

- pupil ellipse valid；
- iris outer ellipse valid；
- pupil / iris outer 不触碰 ROI edge；
- pupil center 位于 iris outer contour 内；
- iris geometric-mean diameter > pupil geometric-mean diameter。

因此不能因为某帧存在 pupil 数值就默认其 pupil/iris ratio 可用于正式分析。

## 6. Ocular aperture

`fullclass_ocular_aperture_ratio_median` 和 `fullclass_ocular_aperture_ratio_p90` 是基于 RITnet 可见眼球分割得到的眼睛开合几何候选量，不是 EAR。

计算先定义：

\[
ocular=sclera\cup iris\cup pupil
\]

在 ocular bbox 中间 80% 横向范围内，对每列计算可见眼球纵向高度：

\[
h_x=y_{max,x}-y_{min,x}+1
\]

然后：

\[
aperture\ ratio_{median}=\frac{median(h_x)}{ocular\ bbox\ width}
\]

`p90` 将 median 换为 90th percentile。median 更稳健；p90 更接近眼睛最开处，但更容易受局部分割异常影响。

解释边界：该量适合作为 NIR 眼睛开合/QC/闭眼候选信号，不直接定义 blink 或 PERCLOS。正式 blink/PERCLOS 后续仍以 RGB MediaPipe 眼睑 landmark / EAR 分析为主，并可与 NIR aperture、ocular fraction、iris/sclera visibility 交叉验证。

## 7. Sparse QC 图

每个被试不保存全部数万张 segmentation 图，而使用固定抽样：

```text
每3000帧约一次
+ 每个phase/segment first/middle/last
+ 每phase每种异常最多2例
```

异常包括：

```text
roi_clipped
ritnet_missing
normalization_invalid
ocular_fragmented
```

抽样保存：

```text
*_labels.png
*_overlay.png
```

并保存 subject-numbered QC index，供异常追踪与论文方法示意图使用。

## 8. 后续瞳孔分析原则

逐帧主信号使用 `fullclass_pupil_to_iris_diameter_ratio`，先执行 QC；左右眼不要在最早阶段丢失身份。后续可在被试内进一步基于 baseline 做相对变化或标准化，但 baseline normalization 是统计分析层，不改写 full-class 原始输出。

瞳孔变化本身不直接等价为“专注分数”。正式注意状态解释需要与 SART 行为指标、数据质量以及后续 RGB blink/head-pose 信息结合。

## 9. 时间字段

full-class CSV 继续保留：

```text
phase
phase_segment
frame_idx
video_time_ms
unix_ms
phase_time_ms
```

行为—NIR trial 对齐通过绝对时间戳在下游完成，不把行为 trial 字段硬写回 RITnet segmentation 文件。
