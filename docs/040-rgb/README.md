# RGB

> 2026-08-25（Asia/Shanghai）｜`rgb-dev`：RGB 模态已从“保留接口”进入开发与方法验证阶段；本分支基于 `amd-DirectML`。

RGB 当前目标不是直接生成“注意力分数”，而是从正式实验 RGB 视频中提取可审计、可与 NIR/Behavior 对齐的连续行为特征。当前主线固定为三部分：**Face、Pose、Motion**。

## 当前分析主线

| 分支 | 当前工具路线 | 主要输出 | 当前状态 |
|---|---|---|---|
| Face | **Py-Feat vs LibreFace 2.0** | AU、表情；Py-Feat 额外覆盖 head pose、gaze、valence/arousal、FaceMesh 等 | **待 benchmark 后二选一** |
| Pose | **MediaPipe Pose** | 33 个身体关键点及其派生的上半身运动、手臂运动、躯干姿态/稳定性 | **当前默认路线** |
| Motion | **OpenCV Motion Energy** | 全局与人体区域的连续运动量 | **当前默认路线** |

暂不把 YuNet、YOLO Pose、Action Recognition、rPPG、HR/HRV 纳入第一阶段正式 RGB 主链。它们只有在当前三条路线出现明确缺口时再作为候选。

## 时间轴与 FocusWave

RGB 视频原则上**不物理切段**。模型从真实 3 分钟静息 baseline 开始，连续分析到最后一个正式 B block 结束；实验阶段只在输出时间序列中标记，而不是先把视频拆成多个文件。

时间对齐依赖 FocusWave 正式实验输出，而不把 FocusWave 仓库作为 Python 运行依赖直接 import：

- `*_rgb_timestamps.csv`：RGB 每帧 Unix 毫秒时间戳；
- `master_timeline.csv`：baseline、instructions、practice、block start/stop 等实验节点；
- `*_Block1_B_beh.csv` / `*_Block2_B_beh.csv`：trial-level `absolute_onset_time`。

正式实验程序来源以 `kyandi233-dev/FocusWave` 的 **`formaltest` 分支**为准。该分支主程序当前头部记录到 **v3.1.4**；现有正式数据主要为 v3.1.3 / v3.1.4，两者均属于最终 BB（B1/B2）分析口径。后续 RGB audit 会按被试实际日志记录版本，而不是从 FocusWave 当前 default branch 推断。

## 已实现：RGB-1 数据审计

第一版可运行入口已经建立，不调用 Py-Feat、LibreFace 或 MediaPipe，也不修改实验原始数据：

```powershell
python scripts/rgb_analysis.py --stage audit
```

它会从 `configs/rgb_analysis.yaml` 声明的候选数据根发现 `sub-031+` RGB 数据，并检查：

- RGB AVI 是否存在且 OpenCV 可打开；
- 视频分辨率、标称 FPS、标称帧数和标称时长；
- `*_rgb_timestamps.csv` 是否存在、frame index 是否连续、Unix 时间是否单调递增；
- `master_timeline.csv`、Block1/Block2 行为 CSV 是否存在；
- RGB 录制范围是否完整覆盖真实 180 s baseline 开始到第二个正式 block 结束；
- 同一被试是否在多个有效数据根重复出现。

输出默认写到仓库外运行结果目录：

```text
outputs/rgb-dev/audit/
├── rgb_inventory.csv
├── rgb_duplicate_subjects.csv
└── audit_summary.json
```

这一阶段只回答“正式 RGB 数据是否完整、是否能与 FocusWave 时间轴可靠对齐”。只有 audit 通过后，才进入 Motion Energy、MediaPipe Pose 和 Face benchmark。

## 文档入口

- [`041-RGB分析目标与数据流.md`](041-RGB分析目标与数据流.md)：回答“RGB 要分析什么、数据怎么流动、最后得到什么”。
- [`042-面部分析工具与Benchmark.md`](042-面部分析工具与Benchmark.md)：Py-Feat 与 LibreFace 的能力、环境和比较标准。
- [`043-姿态与运动量分析方法.md`](043-姿态与运动量分析方法.md)：MediaPipe Pose 与 Motion Energy 的角色和派生指标。
- [`../050-decisions/053-RGB分析路线与开发边界.md`](../050-decisions/053-RGB分析路线与开发边界.md)：当前路线为何这样选，以及哪些内容尚未冻结。

## 当前工程边界

`rgb-dev` 只表示开发/验证分支，不代表 RGB 已正式冻结或全量完成。当前已建立配置、数据发现、时间轴解析和数据审计入口；Face/Pose/Motion 模型适配、DirectML 导出、QC 阈值、采样率和全量运行参数仍必须经过 pilot/benchmark 后再冻结。
