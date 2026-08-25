# 眨眼检测边界与 RITnet 派生开合度

> 2026-08-25（Asia/Shanghai）｜本文件说明当前正式 NIR 输出能否解释为眨眼，以及 `fullclass_ocular_aperture_ratio_*` 的真实计算语义和后续验证边界。

## 当前结论

当前 `runtime/nir-formal/` **不输出正式眨眼事件，也不输出 PERCLOS**。AMD/DirectML `0.1.1` 保持 NVIDIA 正式 CSV schema，负责逐帧眼框、ROI、RITnet 分割、瞳孔椭圆与成功/失败状态；它没有新增或暗含一个已经验证的 blink 分类器。

RITnet 把每个 ROI 像素分为四类：

```text
0 background
1 sclera
2 iris
3 pupil
```

因此除了瞳孔椭圆外，完整四分类图还可以派生“当前可见眼球开口有多高”的几何量。当前 CSV 已经存在：

```text
fullclass_ocular_aperture_ratio_median
fullclass_ocular_aperture_ratio_p90
```

这两个字段是连续的 NIR 眼睑开合辅助指标，但**不是 EAR，也不能直接等同于 blink 或 PERCLOS**。

## 不能使用的替代定义

- `ritnet_missing` 不能解释为闭眼或眨眼；它也可能来自反光、模糊、遮挡、ROI 偏移、分割失败或瞳孔轮廓不满足椭圆条件。
- `yolo_missing` 不能解释为闭眼；它只表示当前帧没有达到有效眼框条件。
- 单眼框、双眼框和 extra boxes 只描述眼框检测状态，不描述眼睑开合。
- 瞳孔面积、短轴或置信度骤降可用于 QC 定位，但未经标注校准不能直接作为 blink 阈值。
- 缺失不是闭眼；不得把 unknown/missing 填成 closed，也不得跨缺失段插值制造连续眨眼事件。
- `roi_clipped` 也不能自动解释为“眼睛检测不完整”：它主要表示 YOLO 眼框进一步放大供 RITnet 使用后，扩大的 ROI 碰到了原始图像边界；很多 clipped 帧实际仍包含完整眼睛。

## `fullclass_ocular_aperture_ratio` 的实际定义

先把 RITnet 中属于眼球的三类合起来：

```text
ocular = sclera ∪ iris ∪ pupil
```

也就是把巩膜、虹膜和瞳孔视为当前可见的眼球区域。

然后取得该 ocular 区域的水平 bbox 宽度：

```text
width = ocular_bbox_width
```

为减少左右边缘异常，只在 ocular bbox 中间 80% 的横向范围内逐列计算可见眼球高度：

```text
h_x = y_max,x - y_min,x + 1
```

最终：

```text
fullclass_ocular_aperture_ratio_median
    = median(h_x) / ocular_bbox_width

fullclass_ocular_aperture_ratio_p90
    = p90(h_x) / ocular_bbox_width
```

直观上：眼睛睁得较开时，可见眼球纵向高度较大，ratio 较高；半闭或眨眼过程中，可见眼球纵向高度下降，ratio 较低。

由于分母是同帧可见眼球水平宽度，该指标已经通过“高度 / 宽度”降低 ROI 尺寸和摄像距离整体缩放带来的影响，是无量纲比例。它不需要再为了单纯消除画面缩放而机械地除一次虹膜直径。

### median 与 p90

`median` 表示中间大部分眼球区域的典型开口高度，对少数异常列更稳健，因此后续作为 NIR 连续开合度特征时优先考虑：

```text
fullclass_ocular_aperture_ratio_median
```

`p90` 更接近“眼睛局部最开处”的高度，对局部遮挡有时较不敏感，但更容易受到异常分割列影响，适合作为辅助特征和 QC 对照。

## 它与 MediaPipe EAR 的区别

EAR 依赖眼睑 landmark：

```text
EAR = 上下眼睑距离 / 眼角水平距离
```

而当前 RITnet aperture ratio 依赖分割：

```text
ocular_aperture_ratio
    = RITnet 可见眼球纵向高度 / RITnet 可见眼球水平宽度
```

两者都与“眼睛有多开”相关，但测量来源不同。EAR 更直接描述眼睑 landmark 几何；RITnet aperture ratio 描述 NIR 分割中当前可见眼球形状。

因此当前 NIR 阶段不把 `fullclass_ocular_aperture_ratio_median` 直接命名为 blink/PERCLOS。它适合用于：

- 连续眼睑开合状态；
- blink/closure candidate 定位；
- 与 ocular fraction、iris/sclera 可见面积、时间连续性联合做 QC；
- 后续与 RGB EAR/blink 结果交叉验证。

## 个体差异与 closure 阈值

虽然 aperture ratio 已经对整体缩放较稳健，不同被试天生眼裂形状、左右眼形态仍可能不同。因此若要进一步把它转换成“相对闭合程度”“closed state”或 PERCLOS-like 指标，需要额外验证是否采用 `subject × eye` 的个体内睁眼基线，例如可靠睁眼帧的稳健高分位数。

这一步用于解决个体眼形差异，而不是重复解决摄像距离缩放。baseline 不能默认取整个视频单个最大值，也不能在实时预测任务中使用未来数据计算。

## 从开合度到 blink/PERCLOS 的验证门槛

1. **三态逐眼观测**：每只眼每帧至少区分 `open / closed / unknown`。眼框缺失、ROI 严重异常、严重反光、结构不合理和低可信分割进入 unknown。
2. **时间戳驱动**：事件起止与持续时间使用实际视频/NIR 时间戳，不假设所有帧严格等间隔。
3. **时序事件规则**：用经人工标注验证的进入/退出阈值、最短闭合时长、允许的短缺口和双眼一致性规则，把逐帧 closure 合并为 blink，避免阈值附近抖动。
4. **不跨 unknown 补齐**：缺失段不继承前一 open/closed 状态。若研究需要插值，只能写入独立副轨并保留原始观测轨。
5. **PERCLOS 分母**：按“有效闭眼观察时长 ÷ 有效眼睑观察时长”计算，同时报告有效覆盖率与 unknown 时长；不能直接用 phase 或视频总时长作分母。
6. **人工时序真值**：从多名被试中分层抽取完整睁眼、半闭、自然眨眼、长闭眼、反光、头动、无眼和模糊片段，人工检查原始 ROI 与 RITnet mask；如冻结阈值，再报告事件级 precision/recall、持续时间误差和 PERCLOS 误差。
7. **无需重新训练网络**：上述人工验证是测量效度和阈值验证，不等于训练一个新的 blink 模型，也不要求 116 名被试全部逐帧人工标注。
8. **显式版本迁移**：若未来新增正式 openness/blink/PERCLOS 列，需要升级 package/schema 版本，保留 AMD `0.1.x` 既有正式结果的可复现性，不静默改写已有全量输出。

## 当前 NIR 分析边界

当前可以把 `fullclass_ocular_aperture_ratio_median/p90` 当作 RITnet-derived continuous openness features 进入 NIR 分析和 QC，但不能仅凭理论关系把某个固定 ratio 阈值命名为“闭眼”，也不能直接从连续 ratio 计算正式 PERCLOS。

与 pupil 不同，aperture ratio 本身已经通过宽高比做了尺度归一化；pupil 主指标仍需联合 iris 构造同帧尺度参照。完整分析设计与尚未冻结的问题见 [022-2026-08-25-NIR正式分析设计与待验证项.md](022-2026-08-25-NIR正式分析设计与待验证项.md)。
