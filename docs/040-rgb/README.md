# RGB

> 2026-08-26（Asia/Shanghai）｜`rgb-dev`：RGB 模态处于正式视频方法验证阶段；分支基于 `amd-DirectML`。

> **后续 RGB 开发前优先阅读：** 本页 → [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md) → [`045-RGB开发环境与运行指令.md`](045-RGB开发环境与运行指令.md) → [`042-面部分析工具与Benchmark.md`](042-面部分析工具与Benchmark.md) → [`../050-decisions/053-RGB分析路线与开发边界.md`](../050-decisions/053-RGB分析路线与开发边界.md)。新增模型字段、QC 或全量运行前必须先检查 044；运行命令前若不确定环境则先检查 045。

> **当前阶段续接记录：** [`../工作记录/08-26-06-RGB-Face-PyFeat-DirectML-v03诊断收口.md`](../工作记录/08-26-06-RGB-Face-PyFeat-DirectML-v03诊断收口.md)。LibreFace Gate 1 见 `08-26-04`；Py-Feat 前序异常诊断保留在 `08-26-05`。

RGB 主线为 **Face、Pose、Motion**。

| 分支 | 当前工具路线 | 当前状态 |
|---|---|---|
| Face | Py-Feat Detectorv2 vs LibreFace 2.0 | **两轮 CPU benchmark/QC 已完成；LibreFace Gate 0/1 PASS；Py-Feat Gate 0 PASS，RetinaFace DML PASS，multitask v0.3 strict-DML batch=1 已证明无 CPU kernel；下一步修复后 probe 小范围收口** |
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

当前目标仍是：在 AMD GPU 上比较 LibreFace ONNX/DirectML 与 Py-Feat ONNX/DirectML 的 parity、coverage、raw schema 与 end-to-end 速度，再冻结 backend。未来 CUDA benchmark 不是当前前置条件。

### LibreFace DirectML 当前进度

LibreFace 当前 Python reference 已成功导出 3 个 ONNX：AU joint、expression、gaze MLP，并记录源权重/ONNX SHA256 与 preprocessing/postprocessing contract。随后在 Windows AMD 开发机使用 ONNX Runtime 1.24.4 / `DmlExecutionProvider` 完成 Gate 1：3 个模型 × batch 1/8/16/32 共 12 个组合全部 `status=ok`，所有输出 finite，profile 中 `cpu_kernel_events=0`、未观察到 CPU fallback。

AU joint model-core：batch 1/8/16/32 分别约 883 / 2078 / 2739 / 2827 images/s；Expression 约 684 / 1966 / 2241 / 2291 images/s。两者 batch 16→32 仅再增加约 3.2% / 2.2%，因此 batch 16 暂作为真实 pipeline 的默认候选拐点，batch 32 保留 secondary candidate。Gaze MLP 本身极小，其高 model-core throughput 不代表包含 MediaPipe landmark feature extraction 的 raw-frame end-to-end。

Gate 1 只证明 ONNX graph / provider / fallback / batch 可运行性，**尚未证明与 CPU reference 的逐字段 parity，也不是最终 AMD end-to-end speed**。因此 LibreFace 仍不能单独冻结为正式 backend。

### Py-Feat DirectML 当前进度

Py-Feat 2.1.1 Gate 0 已成功导出 RetinaFace R34 与 multitask scientific core。RetinaFace 原 Gate 1 中 batch 1/8/16 均由 DML profile 证明执行、0 CPU kernel，吞吐约 79.83 / 85.05 / 81.98 frame/s，当前 batch 8 最优；batch 32 异常，不作为候选。

第一版 Gate 1 随后曾出现 multitask 整体落到 CPU，但后续诊断发现原 probe 允许 Python `InferenceSession` provider-level fallback，因此 session 创建失败时可能被静默重建为 CPU-only session。`rgb-face-directml-diagnostic-v0.3` 已在 batch=1 对同一个 multitask ONNX 做实际 profiling：

- normal DML+CPU 模式：DML fused kernel=1，CPU kernel=0；
- strict-DML 模式：关闭 ORT graph-node CPU fallback + Python wrapper fallback，仅请求 DML；结果仍为 DML fused kernel=1，CPU kernel=0；
- 7 类 scientific outputs 全部 finite。

因此 **multitask ONNX 本身可以完整运行在当前 AMD / DirectML runtime 上**。前一轮 CPU-only 结果不能再解释成模型架构不支持 DML。`face_directml_probe.py` 已升级为 v0.2：保留 CPU EP 用于 graph-level fallback 检测，但禁止 Python wrapper 整体静默退回 CPU。

当前只差使用修复后 probe 对 Py-Feat 做一次干净的 batch 1/8/16 小范围复测后收口 Gate 1；不重新导出模型，不重跑 CPU benchmark。

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

## 下一步：Py-Feat Gate 1 小范围收口，再进入真实 300 帧

使用既有独立环境：

```text
D:\CondaEnvs\attention-face-pyfeat
D:\CondaEnvs\attention-face-directml
```

下一步顺序：

1. 使用修复后的 `face_directml_probe.py`，分别对 RetinaFace 与 multitask 做 batch 1/8/16 干净复测；
2. 不把 RetinaFace 已异常的 batch 32 放进这次收口；
3. 若两者均 DML kernel >0、CPU kernel=0，则 Py-Feat Gate 1 正式 PASS；
4. 然后用现有完全相同的连续 300 帧实现两条真实输入 DirectML runner；
5. LibreFace：MediaPipe alignment/landmark feature extraction → ONNX heads；Py-Feat：RetinaFace → isotropic square-pad crop → multitask；
6. 分别与现有 CPU parquet 做逐字段 parity，并单独报告 raw-frame end-to-end speed、coverage、missing/multi-face 与 raw schema；
7. 若真实工程复杂度/速度明显倾向 LibreFace，可基于证据冻结 LibreFace；不要求为了保留 Py-Feat 而增加正式 pipeline 复杂度；
8. Face backend 冻结后再实现 `body_motion_energy` 与统一正式视频读取。

## 输出目录

正式结果统一位于 `D:\_AttentionData\Beijing-RGB`。数据集级 QC 直接放根目录；pilot/benchmark/review 放 `_test/`；正式被试只创建一个 `sub-XXX/`，内部文件重复带被试编号，不建立 face/pose/raw/processed 等空套子目录。

## 文档入口

- [`041-RGB分析目标与数据流.md`](041-RGB分析目标与数据流.md)
- [`042-面部分析工具与Benchmark.md`](042-面部分析工具与Benchmark.md)
- [`043-姿态与运动量分析方法.md`](043-姿态与运动量分析方法.md)
- [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md)：**开发前必读。**
- [`045-RGB开发环境与运行指令.md`](045-RGB开发环境与运行指令.md)：**环境/命令速查。**
- [`../050-decisions/053-RGB分析路线与开发边界.md`](../050-decisions/053-RGB分析路线与开发边界.md)
- [`../工作记录/08-26-04-RGB-Face-LibreFace-DirectML-Gate1通过.md`](../工作记录/08-26-04-RGB-Face-LibreFace-DirectML-Gate1通过.md)
- [`../工作记录/08-26-05-RGB-Face-PyFeat-DirectML-Gate1阻断诊断.md`](../工作记录/08-26-05-RGB-Face-PyFeat-DirectML-Gate1阻断诊断.md)
- [`../工作记录/08-26-06-RGB-Face-PyFeat-DirectML-v03诊断收口.md`](../工作记录/08-26-06-RGB-Face-PyFeat-DirectML-v03诊断收口.md)：**当前阶段续接。**
