# RGB

> 2026-08-26（Asia/Shanghai）｜`rgb-dev`：RGB 模态处于正式视频方法验证阶段；分支基于 `amd-DirectML`。

> **后续 RGB 开发前优先阅读：** 本页 → [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md) → [`045-RGB开发环境与运行指令.md`](045-RGB开发环境与运行指令.md) → [`042-面部分析工具与Benchmark.md`](042-面部分析工具与Benchmark.md) → [`../050-decisions/053-RGB分析路线与开发边界.md`](../050-decisions/053-RGB分析路线与开发边界.md)。新增模型字段、QC 或全量运行前必须先检查 044；运行命令前若不确定环境则先检查 045。

> **当前阶段交接记录：** [`../工作记录/08-26-01-RGB-Face阶段交接与DirectML计划.md`](../工作记录/08-26-01-RGB-Face阶段交接与DirectML计划.md)。新对话续接时优先读取该文件。

RGB 主线为 **Face、Pose、Motion**。

| 分支 | 当前工具路线 | 当前状态 |
|---|---|---|
| Face | Py-Feat Detectorv2 vs LibreFace 2.0 | **350 张稀疏 + 30 s 连续 benchmark/QC 已完成；下一步 AMD DirectML 双路线** |
| Pose | MediaPipe Tasks Pose Landmarker | **sub-031 10 fps representative pilot/QC/features 已完成** |
| Motion | OpenCV Motion Energy | **sub-031 global Motion pilot/QC/review 已完成** |

## 当前最重要：Face backend 仍未冻结

Py-Feat Detectorv2 与 LibreFace 2.0 已完成两轮共同输入验证。

第一轮：`sub-031` 350 张 phase-stratified 稀疏共同样本。

- Py-Feat：350/350 image-level coverage；348 单脸 + 2 multi-face；20 AU、7 emotion、VA、gaze、6DoF head pose、68-point compatibility landmarks、478×3 FaceMesh、51 blendshape、identity 均完整保存；CPU 约 0.529 image/s。
- LibreFace：349/350 alignment；唯一失败为 multi-face 路径实现 bug，不是 no-face；成功帧保存 12 AU detection、12 AU intensity、expression、gaze、head pose、landmarks。

第二轮：`sub-031` Block1 中段连续 30 秒、真实 timestamp 10 fps，共 300 个时点，中位间隔 99 ms、0 temporal gap。

- 两边 coverage 均为 300/300；
- Py-Feat CPU：568.05 s / 300，约 0.528 image/s；
- LibreFace fresh end-to-end CPU：67.38 s / 300，约 4.452 image/s；
- 当前 CPU reference 下 LibreFace 约快 8.43×，但**不能据此直接决定正式 backend**；
- Py-Feat 信息明显更完整；LibreFace AU intensity 和 head pose 连续性可用，但 binary AU 多数为常数；LibreFace gaze 尤其 yaw 高频变化较大；
- cross-model AU/gaze 一致性整体较弱，不能把两个模型的输出当作可互换测量，也不能把相关当准确率。

当前用户明确要求：**在 AMD GPU 上同时尝试 LibreFace ONNX/DirectML 与 Py-Feat ONNX/DirectML，再比较 GPU 条件下的 parity、coverage 和 end-to-end 速度。** 暂时没有 RTX 5070 可用；未来可补 CUDA benchmark，但不是当前前置条件。

## Face visual review

不做繁琐逐帧人工 FACS 标注，优先用模型输出 overlay 做肉眼 sanity check。

`face-visual-review` 已升级到 v0.2：

```powershell
conda activate "D:\CondaEnvs\attention-rgb"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-dev"
python scripts/rgb_analysis.py --stage face-visual-review --subject sub-031
```

输出：

```text
D:\_AttentionData\Beijing-RGB\_test\face-continuous\sub-031\
├── sub-031_face-visual-review-v2.mp4
└── sub-031_face-visual-review-v2_manifest.json
```

新版尽量统一两边 FaceMesh 关键轮廓，并把眼睛/虹膜、gaze 与 head pose 分开显示：黄色箭头只表示 gaze；红 X / 绿 Y / 蓝 Z 三轴表示 head pose；AU 附中文含义并标明两边不同量纲；Py-Feat 显示 top emotion probability，LibreFace 显示 expression label。该视频是模型预测可视化，不是人工 ground truth。

## Pose representative validation 已完成

`sub-031` 从真实 baseline 起点到 Block2 结束：10 fps、15,494 个采样时点、`pose_valid_fraction=1.0`、0 个 multi-pose frame，约 429.1 s / 7.15 min，实际约 36.1 inference/s；Raw 输出 511,302 landmark rows / 约 22.7 MB。因此正式开发参数保留 **10 fps**。

逐 landmark QC 确认：nose 和双肩质量极高且始终在画面内；肘、腕、髋大多属于画外模型外推。Raw 层仍保存全部 33 个 landmark，但 derived 层使用 visibility + presence + in-frame 质量门控。

`rgb-pose-features-v0.2` 在 `sub-031` 上得到：15,494 个 Pose 时点、2 个 >300 ms gap reset、shoulder motion 有效 15,491 行、elbow 0 行、wrist 6 行、trunk angle 0 行。因此当前 Pose 主测量收敛到 **shoulder motion / shoulder center / shoulder-line posture**；腕、肘、髋为 opportunistic 指标。

`body_motion_energy` 与 Pose-derived motion 不同：前者是在人体 ROI 内计算像素帧差，需要重新消费视频。它继续等待 Face backend 冻结后，与 Motion/Pose/Face 的正式统一视频读取一起实现。

## Motion / 数据审计已完成

- RGB 数据审计：45 个唯一记录基础完整，`sub-9504` 排除，44 个可分析；
- timestamp gap QC：完成；
- global Motion：46,479 行，约 78.3 fps；关键 timestamp gap 正确置 missing；
- Motion 分布 QC / representative review：完成；baseline 最大 global Motion 已确认主要来自主试进入/离开画面。

## 下一步：AMD DirectML 双路线

使用独立环境：

```text
D:\CondaEnvs\attention-face-directml
```

下一步顺序：

1. 建立/核验 `onnxruntime-directml` 环境并确认 `DmlExecutionProvider`；
2. LibreFace official ONNX → DirectML，测 model-core + raw-frame end-to-end，和现有 PyTorch reference 做逐字段 parity；
3. Py-Feat RetinaFace + Detectorv2 multitask → ONNX → DirectML，严格复刻 square-pad crop、normalization 和 postprocessing，再做同样 parity / speed；
4. 两边都使用现有完全相同的连续 300 帧，比较 provider、CPU fallback、batch、coverage、raw schema 与速度；
5. 通过后再冻结 Face backend / fps / primary-face 规则；
6. Face backend 冻结后再实现 `body_motion_energy` 与统一正式视频读取。

## 输出目录

正式结果统一位于 `D:\_AttentionData\Beijing-RGB`。数据集级 QC 直接放根目录；pilot/benchmark/review 放 `_test/`；正式被试只创建一个 `sub-XXX/`，内部文件重复带被试编号，不建立 face/pose/raw/processed 等空套子目录。

## 文档入口

- [`041-RGB分析目标与数据流.md`](041-RGB分析目标与数据流.md)
- [`042-面部分析工具与Benchmark.md`](042-面部分析工具与Benchmark.md)
- [`043-姿态与运动量分析方法.md`](043-姿态与运动量分析方法.md)
- [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md)：**开发前必读。**
- [`045-RGB开发环境与运行指令.md`](045-RGB开发环境与运行指令.md)：**环境/命令速查。**
- [`../050-decisions/053-RGB分析路线与开发边界.md`](../050-decisions/053-RGB分析路线与开发边界.md)
- [`../工作记录/08-26-01-RGB-Face阶段交接与DirectML计划.md`](../工作记录/08-26-01-RGB-Face阶段交接与DirectML计划.md)：**当前阶段交接。**
