# RGB

> 2026-08-25（Asia/Shanghai）｜`rgb-dev`：RGB 模态处于正式视频方法验证阶段；分支基于 `amd-DirectML`。

> **后续 RGB 开发前优先阅读：** 本页 → [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md) → 当前具体方法文档。新增模型字段、QC 或全量运行前必须先检查 044，避免因过早过滤而重新运行昂贵模型。

RGB 主线为 **Face、Pose、Motion**。

| 分支 | 当前工具路线 | 当前状态 |
|---|---|---|
| Face | Py-Feat vs LibreFace 2.0 | **现在进入 benchmark** |
| Pose | MediaPipe Tasks Pose Landmarker | **sub-031 10 fps representative pilot/QC/features 已完成** |
| Motion | OpenCV Motion Energy | **sub-031 global Motion pilot/QC/review 已完成** |

## Pose representative validation 已完成

`sub-031` 从真实 baseline 起点到 Block2 结束：10 fps、15,494 个采样时点、`pose_valid_fraction=1.0`、0 个 multi-pose frame，约 429.1 s / 7.15 min，实际约 36.1 inference/s；Raw 输出 511,302 landmark rows / 约 22.7 MB。因此正式开发参数保留 **10 fps**。

逐 landmark QC 进一步确认当前 RGB 机位的真实边界：nose 和双肩质量极高且始终在画面内；肘、腕、髋大多属于画外模型外推。Raw 层仍保存全部 33 个 landmark，但 derived 层使用 visibility + presence + in-frame 质量门控。

修正版 `rgb-pose-features-v0.2` 在 `sub-031` 上得到：

- 15,494 个 Pose 时点；
- 2 个 >300 ms gap reset；
- shoulder motion 有效 15,491 行；
- elbow motion 有效 0 行；
- wrist motion 有效 6 行；
- trunk angle 有效 0 行。

因此当前 Pose 主测量明确收敛到 **shoulder motion / shoulder center / shoulder-line posture**。腕、肘、髋保留为其他被试可能可用的 opportunistic 指标，但不作为当前机位的必需测量；`upper_body_motion` 也不宽泛解释为完整上半身运动，因为在本机位下它可能实际由 shoulder-only 组成。

`body_motion_energy` 与 Pose-derived motion 不同：前者是在人体 ROI 内计算像素帧差，需要重新消费视频。为避免现在再次独立扫描大体积 AVI，它留到 Face backend 冻结后，与 Motion/Pose/Face 的正式统一视频读取一起实现。

## 下一步：Face benchmark

Pose 已完成首个 representative 正式视频所需的速度、coverage、landmark QC 和派生特征 sanity check，不再作为 Face 的前置阻塞项。下一开发步骤直接进入 **Py-Feat vs LibreFace 2.0 benchmark**，重点比较 AU、head pose、gaze、expression/VA 的输出覆盖、稳定性、缺失率、正式视频鲁棒性、速度和 AMD/DirectML 可行性。

## 已完成的其他环节

- RGB 数据审计：45 个唯一记录基础完整，`sub-9504` 排除，44 个可分析；
- timestamp gap QC：完成；
- global Motion：46,479 行，约 78.3 fps；关键 timestamp gap 正确置 missing；
- Motion 分布 QC / representative review：完成；baseline 最大 global Motion 已确认主要来自主试进入/离开画面。

## 输出目录

正式结果统一位于 `D:\_AttentionData\Beijing-RGB`。数据集级 QC 直接放根目录；pilot/benchmark/review 放 `_test/`；正式被试只创建一个 `sub-XXX/`，内部文件重复带被试编号，不建立 face/pose/raw/processed 等空套子目录。

## 文档入口

- [`041-RGB分析目标与数据流.md`](041-RGB分析目标与数据流.md)
- [`042-面部分析工具与Benchmark.md`](042-面部分析工具与Benchmark.md)
- [`043-姿态与运动量分析方法.md`](043-姿态与运动量分析方法.md)
- [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md)：**开发前必读。**
- [`../050-decisions/053-RGB分析路线与开发边界.md`](../050-decisions/053-RGB分析路线与开发边界.md)
