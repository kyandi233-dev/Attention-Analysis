# RGB

> 2026-08-25（Asia/Shanghai）｜`rgb-dev`：RGB 模态处于正式视频方法验证阶段；分支基于 `amd-DirectML`。

> **后续 RGB 开发前优先阅读：** 本页 → [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md) → 当前具体方法文档。新增模型字段、QC 或全量运行前必须先检查 044，避免因过早过滤而重新运行昂贵模型。

RGB 主线为 **Face、Pose、Motion**。

| 分支 | 当前工具路线 | 当前状态 |
|---|---|---|
| Face | Py-Feat vs LibreFace 2.0 | **下一开发步骤：benchmark** |
| Pose | MediaPipe Tasks Pose Landmarker | **sub-031 10 fps 正式长窗口 pilot 已完成** |
| Motion | OpenCV Motion Energy | **sub-031 global Motion pilot/QC/review 已完成** |

## 已验证：Pose 10 fps

`sub-031` 从真实 baseline 起点到 Block2 结束：

- 10 fps；
- 15,494 个采样时点；
- `pose_valid_fraction=1.0`；
- 0 个 multi-pose frame；
- 约 429.1 s / 7.15 min；
- 实际约 36.1 inference/s；
- 511,302 landmark rows；
- zstd Parquet 约 22.7 MB。

因此正式开发参数恢复并保留 **10 fps**，不采用此前基于“可能过慢”预判的 5 fps。Raw 层继续完整保留 33 normalized + 33 world landmarks、visibility/presence 和 frame/timestamp/behavior identity。

## 当前本地下一步：不重跑 Pose

现有：

```text
D:\_AttentionData\Beijing-RGB\_test\sub-031_pose-test.parquet
```

直接运行：

```powershell
python scripts/rgb_analysis.py --stage pose-qc --subject sub-031
python scripts/rgb_analysis.py --stage pose-features --subject sub-031
```

`pose-qc` 逐项检查 shoulder/elbow/wrist/nose，并把 hip 单独作为 optional trunk landmark；不会再被长期出画的膝/踝/脚拉低整体判断。

`pose-features` 从现有 landmark raw 派生 shoulder-width-normalized wrist/elbow/shoulder/upper-body motion 和可用时的 trunk angle。它不重新读取 RGB 视频。

这里的 Pose-derived `upper_body_motion` 与 `body_motion_energy` 必须区分：后者是人体 ROI 内的像素帧差，留到正式统一视频读取时和 Face/Motion 一起计算，避免现在为了它再次独立扫描大体积 AVI。

## Face benchmark 顺序

Face benchmark 与上述两个 Parquet 后处理步骤工程上独立，不需要等 pose-features 完全冻结。最快路线是：本地现在跑 `pose-qc` / `pose-features`，下一开发步骤立即进入 Py-Feat vs LibreFace benchmark。最终 Face backend 确定后，再合成一次视频读取的正式 batch。

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
