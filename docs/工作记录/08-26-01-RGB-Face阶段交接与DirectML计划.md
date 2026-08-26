# 08-26-01｜RGB Face阶段交接与 DirectML 计划

> 2026-08-26 01:56（Asia/Shanghai）｜分支：`rgb-dev`｜用途：记录当前 RGB Face 方法验证的完成状态、现阶段结论、未冻结事项和下一步 AMD DirectML 双路线开发入口。该记录作为新对话续接时的优先上下文，不改写此前历史工作记录。

## 1. 当前总状态

RGB 主线仍为 **Face + Pose + Motion**。Pose 与 global Motion 的 representative pilot/QC 已完成；当前阻塞项集中在 Face backend 选择与 AMD GPU 部署验证。

Face 候选仍为：

- Py-Feat Detectorv2；
- LibreFace 2.0。

当前**不冻结任一 Face backend**。CPU benchmark 只能说明 reference implementation 的速度差，不能直接决定最终正式方案。后续必须在同一 AMD GPU / ONNX Runtime DirectML 条件下比较两条路线。

## 2. 已完成的 Face benchmark

### 2.1 第一轮：350 张稀疏共同样本

共同输入来自 `sub-031` 正式分析时间窗，按 phase 分层确定性抽取 350 张 JPEG。

Py-Feat：

- 350/350 image-level coverage；
- 348 张单脸，2 张 baseline 图为 multi-face；
- 20 AU、7 emotion、valence/arousal、gaze、6DoF head pose、68-point compatibility landmarks、478×3 FaceMesh、51 blendshape、identity 均完整保存；
- CPU 约 0.529 image/s；
- 信息完整性明显更高，multi-face raw handling 更自然。

LibreFace：

- 349/350 alignment 成功；
- 唯一失败样本为 multi-face 图，失败原因是当前 Python alignment 实现的 list→ndarray 后继续 `.append()` bug，不是 no-face；
- 成功帧保存 12 AU binary detection、12 AU intensity、expression、gaze、head pose 与 landmarks；
- binary AU detection 多数为常数，AU intensity 更适合连续分析。

### 2.2 第二轮：连续 30 秒 / 10 fps

使用 `sub-031` Block1 中段连续 30 秒，真实 timestamp 采样到 10 fps，共 300 个时点；中位间隔 99 ms，0 temporal gap。

两候选 coverage 均为 300/300。

CPU reference 端到端速度：

- Py-Feat：568.05 s / 300，约 0.528 image/s；
- LibreFace：67.38 s / 300，约 4.452 image/s；
- 当前 CPU reference 下 LibreFace 约快 8.43×；
- 该结果**不代表 AMD/DirectML 下最终速度关系**。

时间连续性：

- Py-Feat：20 AU 中 18 个有变化，非恒定 AU lag-1 中位约 0.723；head pose 连续性较好；gaze 连续性中等；
- LibreFace：12 AU intensity 全部有变化，lag-1 中位约 0.648；head pose 连续性较好；binary AU 多数为常数；categorical expression 在该窗口不变；
- LibreFace gaze 尤其 yaw 高频变化较大，当前不能直接作为正式主变量；
- cross-model AU/gaze 一致性整体较弱；head pose 的 Pitch/Roll 一致性明显更高；
- 无人工 FACS/gaze ground truth，因此模型间相关只作为 measurement uncertainty 描述，不能称为准确率。

## 3. AU / expression 当前解释边界

两套工具都使用 FACS AU 编号，但输出集合和量纲不同。

Py-Feat Detectorv2 当前输出 20 个 AU，值为 0–1 probability-like activation output；LibreFace 同时有：

- 12 个 binary AU detection：0/1，表示是否检测到；
- 12 个 AU intensity：0–5 连续强度。

“LibreFace 12 个 AU intensity 可作为连续候选”不意味着另外 12 个 binary AU 无效，而是 binary 输出在当前持续任务窗口里多数为常数，连续信息量低。正式 raw 仍保留 binary AU，主连续分析优先 intensity。

Emotion/expression：

- Py-Feat：7 类概率（Neutral/Happy/Sad/Surprise/Fear/Disgust/Anger）+ valence/arousal；
- LibreFace：8 类 categorical expression（多 Contempt）。

这些 emotion/expression 是独立神经网络分类输出，不是按 AU 规则硬编码推导；在持续注意研究中不直接解释为被试真实心理情绪状态。

## 4. Visual review 状态

已实现 `face-visual-review`，并于本阶段升级到 v0.2。它只读取现有 300 帧模型结果，不重跑 Py-Feat/LibreFace。

命令：

```powershell
conda activate "D:\CondaEnvs\attention-rgb"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-dev"
git pull --ff-only
python scripts/rgb_analysis.py --stage face-visual-review --subject sub-031
```

新版输出：

```text
D:\_AttentionData\Beijing-RGB\_test\face-continuous\sub-031\
├── sub-031_face-visual-review-v2.mp4
└── sub-031_face-visual-review-v2_manifest.json
```

v0.2 目标：

- 两边尽量统一到同语义的 FaceMesh 关键轮廓；
- 眼睛/虹膜单独区分；
- 黄色箭头只表示 gaze；
- 红 X / 绿 Y / 蓝 Z 三轴表示 head pose；
- AU 显示中文含义；
- Py-Feat 标明 AU 为 0–1 probability-like；
- LibreFace 标明 AU intensity 为 0–5；
- Py-Feat 显示 top emotion probability；
- LibreFace 显示 expression label；
- `Alignment=True/False` 改为“人脸对齐：成功/失败”。

该视频用于快速 sanity check，不是人工 ground truth，不计算人工准确率。

## 5. 用户最新明确决策

1. 暂时没有 RTX 5070 可用，因此**当前先在 AMD 上验证 ONNX Runtime DirectML**；
2. LibreFace 与 Py-Feat 两条 DirectML 路线都必须考虑，不能因为 LibreFace CPU 更快或已有官方 ONNX 就提前淘汰 Py-Feat；
3. Py-Feat 信息更完整，因此只要 AMD GPU 速度可接受，它仍可能成为正式 Face backend；
4. 不做繁琐逐帧人工 FACS 标注，优先用可视化 overlay 做肉眼 sanity check；
5. 未来 RTX 5070 可用时，可再补 Py-Feat / LibreFace CUDA reference benchmark，但这不是当前前置条件。

## 6. 下一步主计划：AMD DirectML 双路线

### 6.1 独立环境

使用：

```text
D:\CondaEnvs\attention-face-directml
```

建议环境：

```powershell
conda create -p "D:\CondaEnvs\attention-face-directml" python=3.11 -y
conda activate "D:\CondaEnvs\attention-face-directml"
python -m pip install onnx onnxruntime-directml numpy pandas pyarrow opencv-python pillow
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

必须看到 `DmlExecutionProvider`。不要在该环境同时安装普通 `onnxruntime` / `onnxruntime-gpu`。

DirectML runner 必须设置：

- `enable_mem_pattern=False`；
- `execution_mode=ORT_SEQUENTIAL`；
- 显式记录 provider；
- 检查/记录 CPU fallback，不能把名义 DirectML 当成实际 GPU 全执行。

### 6.2 LibreFace ONNX → DirectML

路线：

```text
raw RGB frame
→ MediaPipe alignment / primary-face handling
→ LibreFace official ONNX heads
→ ONNX Runtime DmlExecutionProvider
→ AU intensity / AU detection / expression / gaze
```

注意：MediaPipe alignment 可能仍主要在 CPU，因此必须同时测：

- model-core throughput；
- raw-frame end-to-end throughput。

并与已有 LibreFace PyTorch CPU reference 做逐字段 parity。

### 6.3 Py-Feat Detectorv2 → ONNX → DirectML

路线至少包含：

```text
RetinaFace
→ isotropic square-pad 256 crop
→ multitask model
   AU / emotion / VA / gaze / head pose / 478 mesh / blendshape
→ ONNX Runtime DmlExecutionProvider
```

identity branch 暂不纳入第一轮 scientific-core 速度比较，可后续单独补 overhead。

不能只导出 multitask 子模型后把结果称为“完整 Py-Feat DirectML”；正式 end-to-end 需要复刻 RetinaFace、256 square-pad crop、224 模型输入 normalization 与 postprocessing。

### 6.4 公平 benchmark 规则

两套 DirectML 都使用现有完全相同的 `sub-031` 连续 300 帧输入。至少报告：

1. provider 与模型/权重 hash；
2. warm-up 后 batch 1 / 8 / 16 / 32 中可运行配置；
3. model-core images/s；
4. end-to-end images/s；
5. GPU/CPU fallback 情况；
6. 与 PyTorch reference 的逐字段 parity；
7. coverage / missing / multi-face 行为；
8. 输出 schema 是否完整保留昂贵推理可获得的信息。

速度与科学 parity 分开判断：模型“能跑得快”不等于输出等价。

## 7. 暂不做的事情

- 不跑 44 个被试全量 Face；
- 不冻结 Face fps/backend；
- 不因为 CPU 速度直接选 LibreFace；
- 不把 cross-model correlation 当准确率；
- 不做大规模人工 FACS 标注；
- 不重复跑已经保存的 350 稀疏 benchmark 和 300 连续 CPU reference；
- 不提前删除/压缩 raw candidate outputs；
- `body_motion_energy` 继续等待 Face backend 冻结后，与正式统一视频读取一起实现。

## 8. 新对话续接入口

新对话开始后优先读取：

1. `docs/040-rgb/README.md`；
2. `docs/040-rgb/044-RGB输出Schema与信息保留原则.md`；
3. `docs/040-rgb/045-RGB开发环境与运行指令.md`；
4. `docs/040-rgb/042-面部分析工具与Benchmark.md`；
5. `docs/050-decisions/053-RGB分析路线与开发边界.md`；
6. 本工作记录 `docs/工作记录/08-26-01-RGB-Face阶段交接与DirectML计划.md`。

然后直接进入：**建立/核验 `attention-face-directml` 环境 → LibreFace ONNX/DirectML → Py-Feat ONNX/DirectML → 同 300 帧 parity + speed benchmark**。
