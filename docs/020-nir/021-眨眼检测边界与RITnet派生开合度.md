# 眨眼检测边界与 RITnet 派生开合度

> 2026-08-24（Asia/Shanghai）｜本文件说明当前正式 NIR 输出能否解释为眨眼，以及如何验证 RITnet 四分类结果中尚未使用的眼睑开合信息。

## 当前结论

当前 `runtime/nir-formal/` **不输出正式眨眼事件，也不输出 PERCLOS**。AMD/DirectML `0.1.0` 保持 NVIDIA 正式 CSV schema，负责逐帧眼框、ROI、RITnet 分割、瞳孔椭圆与成功/失败状态；它没有新增或暗含一个已经验证的 blink 分类器。

RITnet 同时也不只是“找瞳孔”。它把每个 ROI 像素分为四类：

```text
0 background
1 sclera
2 iris
3 pupil
```

当前正式后处理对完整四分类图执行 argmax 后，只用 `pred == 3` 拟合瞳孔椭圆，其他三类没有进入 CSV。由于 sclera/iris/pupil 与 background 的交界会随眼睑遮挡而变化，完整分割图可以派生候选眼裂开合度；这是一条值得验证的后续路线，但不是当前已经成立的正式眨眼指标。

## 不能使用的替代定义

- `ritnet_missing` 不能解释为闭眼或眨眼；它也可能来自反光、模糊、遮挡、ROI 偏移、分割失败或瞳孔轮廓不满足椭圆条件。
- `yolo_missing` 不能解释为闭眼；它只表示当前帧没有达到 `conf=0.40` 的有效眼框。
- 单眼框、双眼框和 extra boxes 只描述眼框检测状态，不描述眼睑开合。
- 瞳孔面积、短轴或置信度骤降可用于 QC 定位，但未经标注校准不能直接作为 blink 阈值。
- 缺失不是闭眼；不得把 unknown/missing 填成 closed，也不得跨缺失段插值制造连续眨眼事件。

因此，当前 `eyes.csv` 的 `status`、`ritnet_found` 和 `pupil_*` 字段仍按瞳孔观测语义解释。AMD 迁移没有改变这些列的名称、分母或科研含义。

## 候选 RITnet-derived openness

候选可见眼球 mask 定义为：

```text
ocular = (pred == sclera) OR (pred == iris) OR (pred == pupil)
       = pred > 0
```

对通过结构 QC 的 ocular mask，可以计算每个 x 位置上最上与最下可见眼球像素的距离，再用中部有效列的稳健百分位数汇总垂直 aperture；横向宽度也应使用稳健百分位数而非单个极值点。候选尺度无关指标可写为：

```text
aperture_ratio = robust_ocular_height / robust_ocular_width
```

为了区分个体眼形、相机距离和左右眼差异，还需要按“被试 × 眼别”从可靠睁眼帧估计基线：

```text
normalized_openness = aperture_ratio / open_eye_baseline
```

这里的 baseline 不能默认取整个视频最大值。应从时间戳对应的可用阶段中筛选 YOLO/ROI 正常、ocular mask 连通且无遮挡的高质量帧，再使用稳健上分位数；具体阶段与分位数必须经真实数据验证后冻结。完全闭眼时 ocular mask 可能接近空，但空 mask 同样可能是 RITnet/ROI 失败，所以单帧仍只能先进入 `closed_candidate` 或 `unknown`，不能直接定为 blink。

## 从开合度到 blink/PERCLOS 的验证门槛

1. **三态逐眼观测**：每只眼每帧至少区分 `open / closed / unknown`。眼框缺失、ROI 异常、严重反光、结构不合理和低置信度进入 unknown。
2. **时间戳驱动**：事件起止与持续时间使用实际视频/NIR 时间戳，并限定在 FocusWave v3.1.3 phase window 内，不假设所有帧严格等间隔。
3. **时序事件规则**：用经人工标注验证的进入/退出阈值、最短闭合时长、允许的短缺口和双眼一致性规则，把逐帧 closure 合并为 blink；避免阈值附近抖动。
4. **不跨 unknown 补齐**：缺失段不继承前一 open/closed 状态。若研究需要插值，只能写入独立副轨并保留原始观测轨。
5. **PERCLOS 分母**：按“有效闭眼观察时长 ÷ 有效眼睑观察时长”计算，同时报告有效覆盖率与 unknown 时长；不能直接用 phase 或视频总时长作分母。
6. **人工时序真值**：标注完整睁眼、半闭、自然眨眼、长闭眼、反光、头动、无眼和模糊片段；至少报告事件级 precision/recall、起止误差、持续时间误差和 PERCLOS 误差。
7. **显式版本迁移**：新增 openness/blink/PERCLOS 正式列时升级 package/schema 版本，保留 AMD `0.1.0` 输出的可复现性，不静默改写已有全量结果。

## 本次 AMD 版本的边界

AMD `0.1.0` 继续输出原 CSV schema，因此不会仅凭理论可行性增加未经验证的 blink 列。当前四分类 argmax 已在内存中产生，未来计算 ocular mask 不需要第二次 RITnet forward，新增几何计算本身预计不是 GPU 瓶颈。真正需要解决的是基线、unknown 门控、时序规则和人工效度，而不是把 `pred > 0` 直接命名成“眨眼”。
