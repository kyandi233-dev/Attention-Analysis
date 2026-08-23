# 062｜Tracking 策略

## 方法定位

Tracking 的作用不是替代 YOLO，而是在连续视频中利用“相邻帧位置变化通常较小”的时间连续性，减少神经网络检测的调用次数。当前仓库已经核验到的 portable 实现支持 `none`、CSRT 和 KCF 三种模式，其中 `none` 表示每帧都重新运行 YOLO。

基本逻辑是：

```text
YOLO 检测双眼
    ↓
初始化两个 tracker
    ↓
中间帧 tracker 更新 bbox
    ↓
到达重检测周期或 tracker 异常
    ↓
重新 YOLO
```

因此它是一个“检测器 + 跟踪器”的混合系统，而不是纯 tracking 系统。

## 当前代码实现

完整实现位于：

```text
runtime/nir-yolo-tracking-ritnet-v1/run_pipeline.py
```

YOLO 首次得到两个眼框后，程序按画面横坐标排序，并分别初始化两个 tracker。之后如果当前帧不需要强制重新检测，就调用 tracker 的 `update()` 更新两个 bbox。

portable package 支持：

| 模式 | 含义 |
|---|---|
| `none` | 不使用 tracker，每帧 YOLO |
| `csrt` | 使用 OpenCV CSRT tracker |
| `kcf` | 使用 OpenCV KCF tracker |

CSRT/KCF 依赖 `opencv-contrib-python` 中的 tracker 实现；代码也兼容部分 OpenCV 版本将 tracker 放在 `cv2.legacy` 下的情况。

## 为什么需要周期性重检测

Tracker 只根据上一段时间的目标外观和位置继续追踪。如果连续运行过久，误差可能逐渐积累，即所谓 drift。周期性 YOLO 的作用是重新使用完整图像特征“校正”位置，因此系统不是：

```text
YOLO 一次 → 永远 tracking
```

而是：

```text
YOLO → tracking → YOLO校正 → tracking → ...
```

portable package 的 2026-08-22 默认配置为 `redetect_interval=10`，但这只是当时准入 package 的默认值；当前仓库没有最终 full-run manifest，因此不能把“每 10 帧重检测”直接写成正式全量最终冻结参数。

## 提前回退到 YOLO 的条件

代码并不会机械等待计划周期。如果 tracker 出现异常，会立即在当前帧触发 YOLO。当前实现至少包含以下检查：

1. 两个 tracker 必须都成功返回 bbox；任一失败即判定 tracking 失败。
2. bbox 必须仍位于合法图像范围，且宽高不能退化到极小值。
3. 当前 bbox 中心相对上一 bbox 的跳变不能超过设定比例。

portable package 中 `center_jump_width_fraction=0.50`，即当前框中心与上一框中心的距离若超过上一框宽度的 50%，该 tracker 更新被视为不可信，并回退 YOLO。这个阈值同样属于 package 默认准入配置，不自动等于正式 full-run 最终参数。

`redetect_reason` 会区分：

- `scheduled`：按周期计划重新检测；
- `tracker_failure`：tracker 更新失败或异常后回退；
- `initial_or_previous_detection_missing`：首次运行或上一帧没有有效双眼；
- `tracker_disabled`：未启用 tracker，因此每帧 YOLO。

这种记录方式很重要，因为它允许后续区分“模型正常周期校正”和“系统因为异常被迫重新检测”。

## 双眼身份问题

portable runtime 将两个框按横坐标命名为：

```text
frame_left
frame_right
```

tracking 时两个 tracker 沿用初始化顺序，因此目标是保持“画面左/画面右”的时序连续性。但这并不能解决解剖学左右眼问题，也不能在双眼交叉、严重遮挡或检测重新初始化后完全保证身份永不交换。

对于正常正面 NIR 双眼特写，左右眼通常不会在图像平面真正交叉，因此横坐标排序是一种合理的工程约束；若研究指标必须区分解剖学左/右眼，仍应单独建立并验证映射规则。

## Tracking 带来的计算收益

假设视频有 30 fps、重检测间隔为 N 帧，那么理论上 YOLO 调用比例约为 `1/N`，其余帧主要承担 tracker 更新与 RITnet 推理。例如 N=10 时，正常情况下约 10% 帧进行 YOLO、90% 帧进行 tracking；实际比例会因为 tracker 失败回退而略高。

这只能说明计算结构，不能直接推导总速度。整条管线耗时还取决于：

- YOLO 单帧推理速度；
- CSRT/KCF 更新速度；
- 两眼 ROI 裁剪；
- 每帧两次 RITnet 推理；
- 视频解码与磁盘写入；
- GPU/CPU 型号和软件环境。

因此正式性能应以运行 `summary.json` 中的 `processing_fps` 和状态计数为准，而不是只按 N 做理论估计。

## 与 QC 的关系

Tracking 的目标是降低计算量，但不能以“连续输出框”为成功标准。如果 tracker 错跟到眉毛、鼻梁、眼镜边缘或背景，表面上仍可能每帧产生 bbox，所以必须依靠 YOLO 周期校正、中心跳变检查、ROI 可视化和下游 RITnet 状态共同判断。

在方法论上，tracking 是一个**计算优化层**，不是新的眼睛识别真值来源。最终 ROI 是否可用于瞳孔分析仍应接受下游 QC。

## 当前 provenance 边界

当前可以核验 portable runtime 的 CSRT/KCF 实现、异常回退逻辑和 package 默认参数，但当前 Git 分支没有正式 full-run 的最终 `run_manifest.json`。因此最终全量使用的是 CSRT、KCF、每帧 YOLO，还是某个具体重检测间隔，目前不能只靠 package 默认值作结论；找到实际运行 manifest 后再冻结到最终方法文档。
