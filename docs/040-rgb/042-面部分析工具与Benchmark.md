# 042｜面部分析工具与 Benchmark

## 1. 当前问题

RGB Face 分支当前比较 **Py-Feat Detectorv2** 与 **LibreFace 2.0**。目标不是比较“哪个模型给出的数字更大”，而是决定：在本项目正式 RGB 视频、Windows/AMD 环境和有限时间预算下，哪一个候选能够以更好的**输出覆盖、检测稳定性、速度和工程可部署性**完成全量分析。

没有人工 FACS ground truth，因此两个模型之间的一致性只能作为描述性参考，不能直接称为准确率。

## 2. Py-Feat Detectorv2

截至 2026-08 的官方 Py-Feat 主线，推荐的新工作使用 `Detectorv2`。它是单一 multi-task 模型，一次 forward 可覆盖：

- 20 个 Action Units；
- 7 类 emotion；
- valence / arousal；
- gaze；
- 6-DoF head pose；
- 478-point 3D FaceMesh；
- 52 个 MediaPipe/ARKit blendshapes；
- face identity（可选）。

Py-Feat 当前官方要求 Python 3.11+，因此与本项目 `attention-rgb` 的 Python 3.11 方向兼容。官方 Python backend 支持 CPU、NVIDIA CUDA 和 Apple MPS；Windows AMD 没有直接可选的 DirectML device，因此当前 AMD 主机首先按 CPU benchmark，只有结果明显优于其他候选但 CPU 太慢时才评估 ONNX/DirectML 移植。

Py-Feat 支持 image/video 输入、batching、`skip_frames` 和逐批保存。Raw benchmark 必须尽量保留 Detectorv2 原生全部字段，而不是只挑 AU。

## 3. LibreFace 2.0

LibreFace 主仓库已在 2026 年更新到 LibreFace 2.0。当前 Python inference 默认可返回：

- AU detection；
- AU intensity（0–5）；
- facial expression；
- gaze yaw / pitch；
- face alignment 阶段产生的 head pose；
- facial landmarks。

当前 joint AU 模型覆盖的 AU detection / intensity 集合并不完全相同，因此 benchmark 必须保存模型原生字段和 AU 名称，不能先强行对齐成共同子集后再丢掉其他 AU。

LibreFace Python 官方安装说明仍推荐独立 Python 3.9 环境，并支持 CPU / CUDA；视频推理支持 batch。其 2.0 项目还提供基于 ONNX 权重的 .NET / OpenSense derivative tools，并明确建议使用带硬件加速的 ONNX Runtime Execution Provider。这使 LibreFace 在本项目的 Windows AMD / DirectML 路线上具有较低的潜在移植门槛，但正式采用前仍必须验证 preprocessing、face alignment、输出命名和数值一致性。

LibreFace 官方主仓库使用研究许可证，而 Py-Feat 本体为 MIT；正式报告与代码发布时需要保留各自许可/provenance 信息。

## 4. 第一轮 benchmark：先固定共同输入

为了赶时间，不让两个候选各自先跑完整 26 分钟视频。第一轮先从同一个正式被试 (`sub-031`) 的真实分析时间窗中提取**固定、可复现、按 phase 分层的共同 JPEG 输入集**：

```powershell
python scripts/rgb_analysis.py --stage face-sample --subject sub-031
```

该 stage **不运行任何 Face 模型**，只读取原 AVI 和真实 timestamp，按 phase 均匀抽取：

| phase | frames |
|---|---:|
| baseline | 50 |
| instructions | 25 |
| practice | 35 |
| transition | 15 |
| block1 | 90 |
| interblock_transition | 45 |
| block2 | 90 |
| **合计** | **350** |

输出归档到：

```text
D:\_AttentionData\Beijing-RGB\_test\face-benchmark\sub-031\
├── frames\
├── sub-031_face-benchmark_frames.csv
└── sub-031_face-benchmark_manifest.json
```

每张 benchmark 图片都保留 `video_frame_position`、`capture_frame_idx`、`unix_ms`、phase/block 和可映射的 trial/probe context。Py-Feat 和 LibreFace 后续必须使用**完全相同的这 350 张图**，以避免候选之间因为采样差异造成假比较。

350 张图片只用于第一轮工具筛选，不代表正式 Face 的最终采样率。最终 fps 只有在 backend 冻结并验证时序稳定性后决定。

## 5. Benchmark 比较什么

第一轮工具筛选至少比较：

1. **能否稳定安装/初始化**：当前 Windows 环境能否复现，权重能否正常下载和缓存；
2. **检测覆盖率**：正常正脸、低头、转头、戴眼镜、不同实验 phase 中是否持续产生有效 face 输出；
3. **输出完整性**：AU、head pose、gaze、expression、VA、landmarks/blendshapes 中实际可获得哪些字段；
4. **缺失与多人处理**：无脸/多人时是否保留明确 missing / face identity，而不是静默错位；
5. **计算效率**：模型初始化耗时、350-frame inference wall time、frames/s、CPU/RAM；
6. **明显动作 spot-check**：转头、张嘴、微笑等明显事件的曲线方向是否合理；
7. **时间连续性**：候选胜出后才在连续短窗口/完整视频上检查 AU/head/gaze 的非生理跳变；
8. **AMD 部署成本**：官方 CPU 是否已够快；若不够，核心网络是否存在官方/可靠 ONNX 路径及 DirectML 一致性验证成本。

## 6. 信息保留规则

Face 是昂贵推理，必须遵循 `044-RGB输出Schema与信息保留原则.md`：

- 第一次候选推理尽量保存该工具原生返回的全部可用字段；
- 不因为当前只关心几个 AU 就提前删除 gaze/head pose/landmarks/blendshapes；
- multi-face 不在 raw 层直接删掉，先保存并标记 primary face；
- QC 先 flag，正式筛选后移；
- candidate/version/model weights/hash/device/batch/config/source-frame manifest 必须写入 benchmark manifest。

## 7. 环境策略

不要为了 Face benchmark 破坏已经能工作的 RGB/Pose 环境。

- Py-Feat：优先建立独立 Python 3.11 benchmark 环境；
- LibreFace 2.0：按官方建议优先独立 Python 3.9 benchmark 环境；
- 两个候选都读取同一个 `_test/face-benchmark/sub-031/frames`；
- 当前先比较官方 CPU reference backend；只有胜出候选的 CPU 性能不足时，再投入 DirectML 移植。

最终可能出现：

1. Py-Feat 输出更完整且 CPU 已足够快 → 直接采用；
2. LibreFace 的 AU/face pipeline 更稳且速度更合适 → 采用 LibreFace；
3. 某候选科学表现更好但 CPU 太慢 → 再进行该候选 ONNX + DirectML 验证。

正式 backend、Face fps、batch/QC、DirectML 与否必须在 benchmark 后写入 decision record。
