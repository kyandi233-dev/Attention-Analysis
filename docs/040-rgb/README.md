# RGB

> 2026-08-25（Asia/Shanghai）｜`rgb-dev`：RGB 模态已从“保留接口”进入开发与方法验证阶段；本分支基于 `amd-DirectML`。

> **后续 RGB 开发前优先阅读：** 本页 → [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md) → 当前具体方法文档。尤其在新增模型字段、QC 或全量运行前，必须先检查 044，避免因过早过滤而重新运行昂贵模型。

RGB 当前目标不是直接生成“注意力分数”，而是从正式实验 RGB 视频中提取可审计、可与 NIR/Behavior 对齐的连续行为特征。当前主线固定为三部分：**Face、Pose、Motion**。

## 当前分析主线

| 分支 | 当前工具路线 | 主要输出 | 当前状态 |
|---|---|---|---|
| Face | **Py-Feat vs LibreFace 2.0** | AU、表情；Py-Feat 额外覆盖 head pose、gaze、valence/arousal、FaceMesh 等 | **待 benchmark 后二选一** |
| Pose | **MediaPipe Pose** | 33 个身体关键点及其派生的上半身运动、手臂运动、躯干姿态/稳定性 | **当前默认路线** |
| Motion | **OpenCV Motion Energy** | 全局运动量 + 亮度/帧差 QC | **sub-031 gap 逻辑通过；进入亮度混淆人工 spot-check** |

暂不把 YuNet、YOLO Pose、Action Recognition、rPPG、HR/HRV 纳入第一阶段正式 RGB 主链。它们只有在当前三条路线出现明确缺口时再作为候选。

## 时间轴与 FocusWave

RGB 视频原则上**不物理切段**。模型从真实 3 分钟静息 baseline 开始，连续分析到最后一个正式 B block 结束；实验阶段只在输出时间序列中标记，而不是先把视频拆成多个文件。

时间对齐依赖 FocusWave 正式实验输出，而不把 FocusWave 仓库作为 Python 运行依赖直接 import：

- `*_rgb_timestamps.csv`：RGB 每帧 Unix 毫秒时间戳；
- `master_timeline.csv`：baseline、instructions、practice、block start/stop 等实验节点；
- `*_Block1_B_beh.csv` / `*_Block2_B_beh.csv`：trial-level `absolute_onset_time`、condition、trial/probe 等行为字段。

正式实验程序来源以 `kyandi233-dev/FocusWave` 的 **`formaltest` 分支**为准。当前正式数据主要为 v3.1.3 / v3.1.4，均属于最终 BB（B1/B2）口径；SART 正式 trial 为 250 ms stimulus + 900 ms mask，共 1150 ms。

## 信息保留硬规则

RGB 不以“当前打算分析什么”决定 raw 层保存什么。昂贵推理原则上只跑一次：模型已经能够稳定返回、以后可能需要且无法从现有结果直接恢复的信息，应在第一次推理时尽量完整落盘；QC 先标记、筛选后移；derived feature 和 summary 从 raw 层重建。

完整规则、各模型 raw schema、frame identity、manifest 和开发检查清单见 [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md)。`configs/rgb_analysis.yaml` 的 `retention` 节同时编码了这一开发约束。

## 已实现：RGB-1 数据审计

```powershell
python scripts/rgb_analysis.py --stage audit
```

它检查 RGB AVI、timestamps、FocusWave timeline、B1/B2 行为文件、正式分析时间覆盖和重复被试。frame index 不连续本身不自动排除被试；只要 AVI 实际帧数与 timestamp 行数一一对应、Unix 时间单调且覆盖正式实验时间轴，局部采集 gap 作为 QC/missing 处理。`sub-033` 因此保留用于正式 RGB 分析；`sub-9504` 继续作为配置排除对象。

当前实测：45 个唯一 RGB 被试中，45 个通过基础完整性，1 个（`sub-9504`）按研究口径排除，44 个进入 RGB 可分析队列。

## 已实现：RGB-2 timestamp gap QC

```powershell
python scripts/rgb_analysis.py --stage gaps
```

输出 `D:\_AttentionData\Beijing-RGB\rgb_timestamp_gaps.csv`。当前 `timestamp_gap_warning_ms=100` 只是开发期扫描阈值，不是正式排除标准。每个超过阈值的 gap 保留 AVI/capture 两套 frame identity、Unix 时间、持续时间、phase/block、capture index 缺失数、按时间估计缺失帧数以及 exclusion/source provenance。

2026-08-25 首次全体扫描得到 66 个 >100 ms gap，涉及 21 个被试（其中 20 个非配置排除）。关键例子：`sub-031` 的 13.853 s 与 0.758 s gap 位于 Block2；`sub-033` 的 8.024 s 最大 gap 位于 baseline；`sub-053`、`sub-054`、`sub-167` 的数秒级最大 gap 位于正式分析区间之外。

## 已实现：RGB-3 Motion Energy 单被试 pilot

```powershell
python scripts/rgb_analysis.py --stage motion --subject sub-031
```

输出 `_test/sub-031_motion-test.parquet` 与对应 manifest。逐帧 raw parquet 保留 frame/timestamp identity、phase/block、可可靠映射的 trial/probe 上下文、亮度统计、原始帧差统计、changed-pixel ratio、Motion Energy、dt/gap 与 motion-valid QC。

`sub-031` 实跑：46,479 行，46,474 行有效 Motion；处理约 593.4 s / 78.3 fps；zstd Parquet 约 3.0 MB。4 个 reset 已逐条核对：102 ms（Block1）、101 ms（interblock transition）、13.853 s（Block2）、758 ms（Block2），全部是纯 timestamp gap 且正确写为 Motion missing。

## 已实现：RGB-4 Motion 分布 QC

```powershell
python scripts/rgb_analysis.py --stage motion-qc --subject sub-031
```

输出 `_test/sub-031_motion-qc.json`，只读取已有 Parquet，不重新跑视频。`sub-031` 首次 QC 结果：

- `dt_ms`：P50=32、P90=46、P95=53、P99=70 ms；92.3% 的相邻帧间隔 ≤48 ms，98.6% ≤66 ms；
- `irregular_dt`（当前 >1.5× median）占 7.67%，因此目前只保留为 QC flag，不作为删除规则；
- Motion Energy P50≈0.00148、P99≈0.00220，最大值≈0.0509；
- Motion 与 `abs(gray_mean_delta)` 的相关约 `r=0.642`，提示高全局 Motion 可能受整体亮度/曝光变化污染；
- 最高 Motion 与最大亮度变化集中在 baseline 开始附近的相同一组帧，因此在冻结正式 Motion 指标前必须回看原视频。

`motion_vs_changed_pixel_ratio≈0.753` 不视为独立“混淆证据”，因为 changed-pixel ratio 本身也是从同一帧差图派生的运动测量。

## 已实现：RGB-5 Motion 人工代表帧检查

为避免凭相关系数直接判断高 Motion 的来源，新增轻量 review stage：

```powershell
python scripts/rgb_analysis.py --stage motion-review --subject sub-031
```

它不会重新计算 Motion，只从原视频读取少量代表 frame pair，并在 `_test` 生成：

```text
sub-031_motion-review.png
sub-031_motion-review.json
```

Contact sheet 每行左侧为上一帧、右侧为当前帧，覆盖最高 Motion、最大亮度变化和 P50/P90/P99 的典型 Motion。该步骤只用于人工确认“高 Motion 是真实身体运动还是曝光/全局亮度变化”，不修改 raw parquet，也不作为正式 batch 的额外必需输出。

## 输出目录约定

RGB 分析结果不写入 Git 仓库，统一放到 `D:\_AttentionData\Beijing-RGB`。数据集级 inventory/QC/summary 直接放根目录；pilot/benchmark/review 统一放 `_test/`；只有真正产生正式被试结果时才创建 `sub-XXX/`；内部文件仍重复带被试编号；不额外建立 face/pose/motion/raw/processed 等空套目录。

## 文档入口

- [`041-RGB分析目标与数据流.md`](041-RGB分析目标与数据流.md)
- [`042-面部分析工具与Benchmark.md`](042-面部分析工具与Benchmark.md)
- [`043-姿态与运动量分析方法.md`](043-姿态与运动量分析方法.md)
- [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md)：**开发前必读。**
- [`../050-decisions/053-RGB分析路线与开发边界.md`](../050-decisions/053-RGB分析路线与开发边界.md)

## 当前工程边界

`rgb-dev` 只表示开发/验证分支，不代表 RGB 已正式冻结或全量完成。当前已建立数据审计、timestamp gap QC、行为 trial/probe 映射、Motion Energy pilot、Motion 分布 QC 和代表帧人工 review。Motion 的 gap 逻辑已经通过，但正式 global/body Motion 口径、亮度污染处理和时间抖动参数仍未冻结。Face/Pose 仍需各自 pilot/benchmark。
