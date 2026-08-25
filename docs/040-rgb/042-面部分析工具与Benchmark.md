# 042｜面部分析工具与 Benchmark

## 1. 当前问题

RGB Face 分支比较 **Py-Feat Detectorv2** 与 **LibreFace 2.0**。目标不是比较“哪个模型给出的数字更大”，而是决定：在本项目正式 RGB 视频、Windows/AMD 环境和有限时间预算下，哪一个候选能够以更好的**输出覆盖、检测稳定性、速度和工程可部署性**完成全量分析。

没有人工 FACS ground truth，因此两个模型之间的一致性只能作为描述性参考，不能直接称为准确率。

## 2. Py-Feat Detectorv2

截至 2026-08 的官方 Py-Feat 主线，推荐新工作使用 `Detectorv2`。当前本项目实测 raw 输出包括：

- 20 个 AU；
- 7 类 emotion：`Neutral / Happy / Sad / Surprise / Fear / Disgust / Anger`；
- valence / arousal；
- gaze pitch / yaw / angle；
- 6-DoF head pose；
- 68-point 兼容 landmark；
- 完整 478×3 FaceMesh；
- 51 个 blendshape；
- 512 维 identity embedding + identity 字段。

### sub-031 第一轮 benchmark 实测

共同输入集 350 张正式实验 JPEG，CPU / batch=8：

- Py-Feat 2.1.1 / Python 3.11.15 / Torch 2.13.0；
- 模型初始化约 51.2 s；
- inference 约 662.1 s（约 11.0 min）；
- 约 0.529 input images/s；
- 350 张输入产生 352 个 face rows；
- 输出 2,182 列；
- raw 原生字段完整保存在 `pyfeat_raw.parquet`。

修正版 `rgb-face-benchmark-pyfeat-qc-v0.2` 确认：

- 350/350 张均检测到 face，image-level coverage = **100%**；
- 348 张恰好 1 个 face；
- 2 张存在 multi-face，均位于 baseline；
- multi-face 分别为 benchmark index 0 与 1；
- FaceScore 最低约 0.9945，中位数约 0.9996；
- 20 AU、7 emotion、VA、gaze、6DoF head pose、68-point landmark、478×3 mesh、51 blendshape、identity 均为 100% row/cell 有效；
- 除 `FrameHeight/FrameWidth` 外未出现整体常数型数值输出。

因此 Py-Feat 第一轮的主要优势是：**检测覆盖完整、multi-face 被显式保留、输出信息量显著更完整**。主要劣势是当前 CPU reference backend 较慢。

## 3. LibreFace 2.0

LibreFace 2.0 当前 Python inference 在本项目实测可返回：

- 12 个 AU detection；
- 12 个 AU intensity（0–5；与 detection AU 集合并不完全相同）；
- categorical facial expression；
- gaze yaw / pitch；
- alignment 阶段的 head pose 与 facial landmarks。

LibreFace Python 官方安装说明仍推荐 Python 3.9，并要求 CMake；Windows 下直接 pip 安装可能触发 `dlib` 源码编译。当前 benchmark 环境使用 conda-forge 的 `dlib==19.24.6` / `dlib-cpu==19.24.6`，LibreFace 0.2.0 / Torch 2.0.0。

### sub-031 第一轮 benchmark 实测

共同 350 张输入中：

- 349 张 alignment 成功，coverage = **99.71%**；
- 唯一失败位于 baseline、benchmark index 0；
- 该图恰好也是 Py-Feat 明确检测到 2 张脸的 multi-face 图之一；
- failure 为 `AttributeError: 'numpy.ndarray' object has no attribute 'append'`，不是普通 no-face；
- alignment 成功的 349 张中，headpose 与 landmarks 字段均 100% 可用；
- AU detection 12 列、AU intensity 12 列、expression 1 列、gaze 2 列均完整覆盖 349 个 aligned indices。

LibreFace 官方 `get_aligned_image()` 当前以 `max_num_faces=2` 运行 MediaPipe FaceMesh，但其实现把 `face_2d/face_3d` 在第一张 face 后转换成 `numpy.ndarray`，第二张 face 再进入循环时仍调用 `.append()`。因此 benchmark index 0 的报错与**多人脸路径实现缺陷**高度一致。正式 pipeline 若采用 LibreFace，不能把该 failure 误记为“无人脸”，必须另外实现可靠的 multi-face / primary-face 处理。

### AU detection 的额外观察

349 张已对齐图片中，12 个 binary AU detection 里有 8 列在整个稀疏 benchmark 上为常数：

`au_1 / au_2 / au_6 / au_7 / au_10 / au_12 / au_15 / au_23`

这不等同于模型故障：可能是当前实验中这些 binary threshold 很少被触发，也可能说明 binary detection 对本数据的区分度有限。值得注意的是 **12 个 AU intensity 均不是常数**，因此连续 intensity 比 binary detection 更值得进入下一轮时序验证。

### 速度口径

当前成功 manifest 显示 `alignment_reused=true`，所以约 69.7 s / 5.02 images/s 只覆盖：

- AU joint detection/intensity：约 26.4 s；
- expression：约 29.7 s；
- gaze：约 13.6 s。

**该数值不包含首次 face alignment，不能与 Py-Feat 的 662.1 s 端到端 wall time 直接相除并称为整体 9.5× 加速。** 下一轮连续窗口必须记录 LibreFace 首次 alignment + downstream heads 的完整时间。

LibreFace 2.0 另有 ONNX derivative 路线，因此在 Windows AMD / DirectML 上具有较低的潜在移植门槛，但正式采用前仍必须验证 preprocessing、face alignment、输出命名和数值一致性。

## 4. 第一轮 benchmark 结论

| 维度 | Py-Feat Detectorv2 | LibreFace 2.0 |
|---|---|---|
| image-level face coverage | **350/350 = 100%** | **349/350 = 99.71%** |
| multi-face | 2 张明确保留为多 face rows | 当前 Python alignment 在至少 1 张 multi-face 图上崩溃 |
| AU | **20 个连续 AU 输出** | 12 binary detection + 12 intensity，集合不完全相同 |
| emotion | **7 类概率** | 1 个 categorical label |
| valence/arousal | **有** | 当前 Python默认 pipeline 无 |
| gaze | pitch/yaw/angle | yaw/pitch |
| head pose | **6DoF** | alignment head pose |
| landmarks | 68-point + **478×3 mesh** | alignment landmarks |
| blendshape | **51** | 当前 benchmark 无 |
| identity | **有** | 当前 benchmark 无 |
| CPU timing | 662.1 s / 350，完整端到端 | downstream heads 69.7 s；alignment 被复用，不能直接公平比较 |
| AMD/ONNX | 需要额外移植验证 | 官方已有 ONNX derivative 路线 |

**第一轮不冻结 backend。** 当前证据表明 Py-Feat 在科学信息完整性和 multi-face raw handling 上领先；LibreFace 在下游模型头速度与 ONNX/DirectML 工程潜力上更有优势，但 multi-face alignment 缺陷和 binary AU 区分度必须纳入决策。

## 5. 下一轮：连续短窗口

不再扩充稀疏 benchmark。下一轮只需要同一个 `sub-031` 正式 Block 内的一段连续短窗口，两个候选使用完全相同的 frame/timestamp 集。目标是一次性回答：

1. AU / gaze / head pose 的时间连续性与非生理跳变；
2. LibreFace AU intensity 与 binary detection 在连续任务中的实际区分度；
3. Py-Feat 多任务输出是否存在明显 frame-to-frame jitter；
4. LibreFace **包含首次 alignment** 的真实端到端 wall time；
5. 结合科学输出和总速度，决定 backend；
6. 若最终候选 CPU 仍不足，再进入 ONNX / DirectML benchmark。

连续窗口仍不能替代人工 FACS ground truth，因此只能验证稳定性、coverage 与相对合理性，不把模型间一致性称为准确率。

## 6. 第一轮共同输入

第一轮从 `sub-031` 的正式时间窗提取固定、按 phase 分层的共同 JPEG 输入集：

```powershell
python scripts/rgb_analysis.py --stage face-sample --subject sub-031
```

抽样为 baseline 50、instructions 25、practice 35、transition 15、block1 90、interblock transition 45、block2 90，共 **350** 张。两候选必须使用完全相同输入。

输出：

```text
D:\_AttentionData\Beijing-RGB\_test\face-benchmark\sub-031\
├── frames\
├── sub-031_face-benchmark_frames.csv
└── sub-031_face-benchmark_manifest.json
```

## 7. 信息保留规则

Face 是昂贵推理，必须遵循 `044-RGB输出Schema与信息保留原则.md`：第一次候选推理尽量保存原生全部可用字段；multi-face 不在 raw 层直接删掉；QC 先 flag；candidate/version/model/device/batch/source-frame manifest 必须记录。

## 8. 环境策略

不要为了 Face benchmark 破坏已经能工作的 RGB/Pose 环境：Py-Feat 使用独立 Python 3.11 环境；LibreFace 使用独立 Python 3.9 环境；两个候选读取同一批 benchmark 图。主 `attention-rgb` 环境负责 sample/QC/比较。完整环境与运行命令见 `045-RGB开发环境与运行指令.md`。