# 042｜面部分析工具与 Benchmark

## 1. 当前问题

RGB Face 分支比较 **Py-Feat Detectorv2** 与 **LibreFace 2.0**。目标不是比较“哪个模型给出的数字更大”，而是决定：在本项目正式 RGB 视频、Windows/AMD 环境和有限时间预算下，哪一个候选能够以更好的**输出覆盖、时间稳定性、科学可解释性、速度和工程可部署性**完成全量分析。

没有人工 FACS / gaze / head-pose ground truth，因此两个模型之间的一致性只能作为描述性参考，不能直接称为准确率。

## 2. 第一轮稀疏 benchmark

共同输入为 `sub-031` 正式分析时间窗中按 phase 分层确定性抽取的 350 张 JPEG。Py-Feat 与 LibreFace 使用完全相同的 frame manifest。

### Py-Feat Detectorv2

CPU / batch=8，Py-Feat 2.1.1 / Python 3.11.15 / Torch 2.13.0：

- 350/350 张均检测到 face，image-level coverage = **100%**；
- 348 张恰好 1 个 face，2 张 baseline 图存在 multi-face；
- 350 张输入产生 352 个 face rows；
- FaceScore 最低约 0.9945，中位数约 0.9996；
- 20 AU、7 emotion、valence/arousal、gaze、6DoF head pose、68-point landmark、478×3 mesh、51 blendshape、identity 均完整保留；
- raw 共 2,182 列；
- inference 约 662.1 s，约 **0.529 image/s**。

第一轮优势是检测覆盖完整、multi-face 被显式保留、输出信息量显著更完整；主要劣势是 CPU reference backend 很慢。

### LibreFace 2.0

Python 3.9 / LibreFace 0.2.0 / Torch 2.0.0：

- 349/350 张 alignment 成功，coverage = **99.71%**；
- 唯一失败位于 baseline benchmark index 0；
- 该图也是 Py-Feat 明确检测到 2 张脸的 multi-face 图之一；
- failure 为 `AttributeError: 'numpy.ndarray' object has no attribute 'append'`，不是普通 no-face；
- 349 张成功图中 head pose、landmarks、12 AU detection、12 AU intensity、expression、gaze 均完整覆盖。

LibreFace 当前 Python `get_aligned_image()` 允许 `max_num_faces=2`，但 multi-face 路径在第二张脸处存在 list → ndarray 后继续 `.append()` 的实现缺陷。正式 pipeline 若采用 LibreFace，必须在外层实现可靠 primary-face / multi-face 处理，不能把该 failure 记作“无人脸”。

第一轮中 12 个 binary AU detection 有 8 列为常数，而 12 个 AU intensity 均有变化，因此后续连续分析优先考察 AU intensity，不把 binary detection 作为连续主变量。

第一轮成功运行 manifest 的 LibreFace `alignment_reused=true`，所以 5.02 image/s 只覆盖下游 AU/expression/gaze，不能与 Py-Feat 端到端速度直接比较。

## 3. 第二轮：连续 30 秒时序 benchmark

第二轮使用 `sub-031` Block1 中段连续 30 秒，按真实 timestamp 采样到 10 fps，共 **300 个时点**。中位实际采样间隔为 **99 ms**，`temporal_gap_rows=0`，因此该窗口可以用于相邻时点连续性比较。

### 3.1 Coverage 与端到端速度

连续窗口中：

- Py-Feat：300/300 检测成功，0 multi-face；
- LibreFace：300/300 alignment 成功；
- 两者连续 coverage 都为 **100%**。

端到端 CPU reference：

- Py-Feat：568.05 s / 300 张，约 **0.528 image/s**；
- LibreFace：67.38 s / 300 张，约 **4.452 image/s**；
- LibreFace 此次 `alignment_reused=false`，其中 alignment 16.77 s、AU 22.24 s、expression 16.68 s、gaze 11.68 s；
- 在该连续窗口与当前 CPU reference implementation 下，LibreFace 端到端吞吐约为 Py-Feat 的 **8.43×**。

该速度差只描述当前 CPU reference runtime，不代表最终 AMD/DirectML 速度。

### 3.2 Py-Feat 时间连续性

20 个 AU 中 18 个在该 30 秒窗口内有连续变化，AU11 与 AU20 为常数。非恒定 AU 的 lag-1 correlation 中位数约 **0.723**，范围约 0.496–0.814；整体表现为中等到较高的相邻时点连续性。

AU43 的 lag-1 较低且存在较大的瞬时 step，但 AU43 本身与眼闭合/眨眼相关，因此不能仅凭较大 frame-to-frame step 判定为 jitter，需要结合对应视频帧做可视化 review。

Py-Feat gaze 三个输出在全部 300 帧有效，lag-1 约 0.60–0.67。其 gaze 为弧度制；换算后 100 ms 相邻样本的 p95 变化约为：pitch 3.69°、yaw 2.62°，最大约 7.78° / 6.38°。

Py-Feat head pose 三轴全部有效，lag-1：Pitch≈0.828、Roll≈0.956、Yaw≈0.722。按弧度转角度后，相邻样本 p95 step 约 Pitch 0.54°、Roll 0.31°、Yaw 0.69°，最大约 1.22° / 0.65° / 1.46°，连续性较好。

### 3.3 LibreFace 时间连续性

12 个 AU intensity 全部为连续变量，lag-1 correlation 中位数约 **0.648**，范围约 0.502–0.749。AU intensity 的时间连续性总体可用，但在该窗口中多数 AU 的绝对变化幅度较小。

12 个 binary AU detection 中多数仍为常数，仅 AU4 / AU14 / AU17 / AU24 出现 0/1 切换；因此 binary AU detection 不适合作为本项目主要连续 Face 指标，保留 raw/QC 即可。

LibreFace gaze 全部 300 帧有效，但连续稳定性较弱：

- gaze pitch lag-1≈0.626，median step≈1.50°，p95≈8.10°，max≈17.22°；
- gaze yaw lag-1≈0.425，median step≈5.68°，p95≈17.30°，max≈41.81°。

在本任务连续 10 fps 窗口中，尤其 gaze yaw 出现较大的高频变化。没有 gaze ground truth，不能直接判为错误，但在可视化 review 前**不把 LibreFace gaze 作为正式主变量**。

LibreFace head pose 连续性较好：pitch/roll/yaw lag-1 分别约 0.787 / 0.749 / 0.874。与 Py-Feat 的 cross-model rank correlation 为 Pitch≈0.718、Roll≈0.693、Yaw≈-0.407；Yaw 的负号可能受到两套工具坐标/符号约定影响，不能直接按原符号解释为相反运动。

LibreFace categorical expression 在该 30 秒窗口 `expression_change_fraction=0`，即始终保持同一类别。对持续 SART 任务而言这不一定是模型故障，但 categorical expression 对本项目连续注意分析的信息量有限。

## 4. Cross-model agreement

共同 AU intensity 的 Spearman correlation 整体偏弱：11 个可计算 common AU 中相关范围约 -0.419 到 0.386，相关绝对值中位数约 0.298。部分 AU 甚至方向相反。

该结果说明：**不能把“两个模型都输出 AU”视为可互换测量。** 在没有 FACS ground truth 的情况下，弱一致性既可能来自模型训练域、尺度/定义、face alignment 和预处理差异，也可能来自其中一个或两个模型的误差；因此 cross-model correlation 只作为方法不确定性证据，不作为准确率排序。

Gaze 的模型间一致性也较弱：pitch Spearman≈-0.580、yaw≈0.054。Pitch 的负相关可能包含符号约定差异；yaw 近零则说明两套 gaze 在该窗口对时间变化基本没有一致趋势。

Head pose 的一致性明显好于 AU/gaze，Pitch 与 Roll 为中高正相关，Yaw 需先统一符号/坐标约定后再比较。

## 5. AMD DirectML：下一阶段主 benchmark

当前暂不以 RTX/CUDA 作为开发前提，先在 Windows AMD 上公平比较两套模型的 ONNX/DirectML 方案。**两边都必须尝试**，不能因为 LibreFace 已有官方 ONNX derivative 就提前淘汰 Py-Feat。

### 5.1 LibreFace 路线

LibreFace 官方 derivative 已公开 ONNX 权重，并明确推荐通过 ONNX Runtime Execution Provider 使用硬件加速。当前计划：

```text
raw frame
→ MediaPipe alignment / primary-face handling
→ LibreFace ONNX heads
→ ONNX Runtime DmlExecutionProvider
→ AMD GPU
```

需要分别记录 model-core 和 end-to-end wall time，因为 MediaPipe alignment 仍可能主要在 CPU 上执行。

### 5.2 Py-Feat 路线

Detectorv2 当前结构为：

```text
RetinaFace
→ isotropic square-pad 256 crop
→ single multitask model
   (AU / emotion / VA / gaze / head pose / 478 mesh / blendshape)
→ optional identity branch
```

因此 Py-Feat DirectML 不能只导出一个随意的子模型后称为“Py-Feat ONNX”。第一阶段至少要验证 RetinaFace + multitask 主模型，严格复刻 256 crop、224 模型输入 normalization 和 postprocessing。identity 不属于当前科学核心变量，可单独记录为 optional overhead，不让它影响 Py-Feat vs LibreFace 的核心速度比较。

### 5.3 公平速度比较

AMD benchmark 使用现有完全相同的连续 300 帧，并同时报告：

1. **model-core throughput**：在固定 crop/aligned input 上比较 ONNX 头部速度；
2. **end-to-end throughput**：从原始 RGB frame 到最终 AU/head/gaze 等结果的总 wall time；
3. provider 必须确认实际为 `DmlExecutionProvider`，不能静默 fallback 到 CPU；
4. 每条路线先 warm-up，再测试至少 batch 1 / 8 / 16 / 32 中可运行的配置；
5. 输出与 CPU/PyTorch reference 做逐字段数值 parity；
6. 速度测试和科学输出 parity 分开报告。

最终选择看的是“AMD GPU 上完整 scientific-core pipeline 的速度 + 输出可信度”，不是某个单独 ONNX 文件的理论 FPS。

## 6. 可视化 review：替代繁琐人工逐帧标注

不要求人工逐帧进行 FACS 编码。Py-Feat 官方本身支持 478 FaceMesh、gaze arrow、AU 驱动 mesh 和动画等 visualization；本项目另外增加统一的 side-by-side review，使两套候选在同一原图上直接可见。

新增 stage：

```powershell
python scripts/rgb_analysis.py --stage face-visual-review --subject sub-031
```

它读取已有连续 30 秒结果，不重新运行模型，生成：

```text
D:\_AttentionData\Beijing-RGB\_test\face-continuous\sub-031\
└── sub-031_face-visual-review.mp4
```

视频左侧为 Py-Feat：face bbox、68 landmark、gaze direction cue、head pose 与当前 top AU；右侧为 LibreFace：MediaPipe landmarks（可解析时）、gaze direction cue、head pose 与 AU intensity。该视频用于肉眼检查模型预测是否与真实动作方向一致。

**注意：overlay 是模型预测的可视化，不是人工 ground truth。** 它可以快速发现明显 jitter、方向错误、漏脸和 AU/gaze 与画面不一致，但不能据此计算严格准确率。

## 7. 当前候选定位

| 维度 | Py-Feat Detectorv2 | LibreFace 2.0 |
|---|---|---|
| 连续 coverage | 300/300 | 300/300 |
| CPU end-to-end | 0.528 image/s | **4.452 image/s** |
| AU 连续性 | 18/20 有变化，lag-1 中位≈0.723 | 12 intensity 均变化，lag-1 中位≈0.648 |
| binary AU | 不适用 | 多数常数，不作为主连续指标 |
| gaze | 连续性相对较好 | yaw 高频变化较大，待 visual review |
| head pose | 连续性高 | 连续性高；需统一坐标符号 |
| expression | 7 类概率 + VA | categorical label 在本窗口恒定 |
| 信息完整性 | **明显更高** | 较精简 |
| multi-face | 原生保留多 face rows | 当前 Python alignment 有缺陷 |
| AMD/ONNX | **进入自定义 DirectML 验证** | **进入官方 ONNX DirectML 验证** |

当前**不冻结 backend**。下一阶段必须完成两套 AMD DirectML benchmark、reference parity 和可视化 review，之后再冻结 Face backend / fps / primary-face 策略。

## 8. 信息保留与环境规则

Face 是昂贵推理，继续遵循 `044-RGB输出Schema与信息保留原则.md`：候选推理保存原生可用字段；multi-face 不在 raw 层静默删除；QC 先 flag；正式结果记录 candidate/version/model/device/batch/source-frame manifest。

主 `attention-rgb` 环境负责 sample/QC/比较/visual review；Py-Feat 使用独立 Python 3.11 环境；LibreFace 使用独立 Python 3.9 环境。DirectML benchmark 将使用独立环境，避免污染当前稳定的候选 reference runtime。