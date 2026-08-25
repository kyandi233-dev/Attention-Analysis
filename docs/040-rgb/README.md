# RGB

> 2026-08-25（Asia/Shanghai）｜`rgb-dev`：RGB 模态处于正式视频方法验证阶段；分支基于 `amd-DirectML`。

> **后续 RGB 开发前优先阅读：** 本页 → [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md) → 当前具体方法文档。新增模型字段、QC 或全量运行前必须先检查 044，避免因过早过滤而重新运行昂贵模型。

RGB 当前目标不是直接生成“注意力分数”，而是从正式实验 RGB 视频中提取可审计、可与 NIR/Behavior 对齐的连续行为特征。主线为 **Face、Pose、Motion**。

| 分支 | 当前工具路线 | 当前状态 |
|---|---|---|
| Face | Py-Feat vs LibreFace 2.0 | 待 benchmark 后二选一 |
| Pose | MediaPipe Tasks Pose Landmarker | **正式视频长窗口 pilot 已实现，待 sub-031 实跑** |
| Motion | OpenCV Motion Energy | **sub-031 global Motion pilot/QC/review 已完成** |

## 时间轴与信息保留

RGB 不物理切段。模型从真实 3 分钟 baseline 起点连续分析到 Block2 结束；阶段和 trial/probe 通过 FocusWave 的真实 Unix 时间映射。Raw 层遵循“昂贵推理尽量只跑一次、完整保留可得原始信息、QC 先 flag 后 filter”的硬规则，详见 044。

## 已完成：数据审计 / gap QC / global Motion

- 45 个唯一 RGB 被试基础完整性通过；`sub-9504` 按研究口径排除，44 个进入 RGB 可分析队列；
- timestamp gap 已全体扫描；`sub-031` 的 13.853 s / 0.758 s Block2 gap 在 Motion 中正确置为相邻帧指标 missing；
- `sub-031` global Motion：46,479 行，处理约 78.3 fps，zstd Parquet 约 3 MB；
- Motion 分布 QC 与人工 review 已确认最高 baseline Motion 主要来自主试进入/离开画面，不是简单曝光跳变；
- `global_motion_energy` 保留为“整画面变化量/QC”，后续用 Pose ROI 构建 `body_motion_energy` 作为更接近被试自身运动的候选指标。

## 当前下一步：MediaPipe Pose 正式视频 pilot

使用 MediaPipe Tasks Pose Landmarker 的 `VIDEO` 模式，直接跑真实正式分析时间窗。当前开发参数：

- Pose Lite 官方 `.task` 模型；
- `inference_fps=10`，先验证速度、覆盖率和 landmark 稳定性；
- `num_poses=2`，baseline 主试入镜时不静默丢掉第二个人；
- 完整保存每个返回人体的 33 个 normalized landmarks；
- 同时保存 33 个 world landmarks；
- 保存 `visibility` / `presence`；
- 保存 pose bbox、frame/timestamp identity、phase/block、trial/probe 上下文；
- 检测失败的采样帧仍保留一行 `pose_valid=False`，不从 raw 层消失；
- 第一版不请求高体积 segmentation mask，body ROI 先由完整 landmarks 可重复重建。

运行：

```powershell
python scripts/rgb_analysis.py --stage pose --subject sub-031
```

输出：

```text
D:\_AttentionData\Beijing-RGB\_test\
├── pose_landmarker_lite.task
├── sub-031_pose-test.parquet
└── sub-031_pose-test_manifest.json
```

Manifest 会记录模型 hash、MediaPipe/OpenCV/Python 版本、采样 fps、正式时间窗、frames-with-pose、multi-pose frame 数、mean visibility/presence、处理速度和输出体积。只有这一轮通过后，才冻结正式 Pose 参数并创建 `sub-XXX_pose_landmarks.parquet`。

## 输出目录

正式结果统一位于 `D:\_AttentionData\Beijing-RGB`。数据集级 QC 直接放根目录；pilot/benchmark/review 放 `_test/`；正式被试只创建一个 `sub-XXX/`，内部文件重复带被试编号，不建立 face/pose/raw/processed 等空套子目录。

## 文档入口

- [`041-RGB分析目标与数据流.md`](041-RGB分析目标与数据流.md)
- [`042-面部分析工具与Benchmark.md`](042-面部分析工具与Benchmark.md)
- [`043-姿态与运动量分析方法.md`](043-姿态与运动量分析方法.md)
- [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md)：**开发前必读。**
- [`../050-decisions/053-RGB分析路线与开发边界.md`](../050-decisions/053-RGB分析路线与开发边界.md)

## 当前工程边界

当前已经可以开始真实正式 RGB 视频的完整链路验证：Motion 已通过首个正式视频验证；Pose stage 已实现并等待本地正式视频实跑；Pose 通过后立即进入 Face benchmark。Face backend、正式 Pose fps、body Motion ROI/QC 和最终全量参数仍需在 representative pilot 后冻结。
