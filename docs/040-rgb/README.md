# RGB

> 2026-08-25（Asia/Shanghai）｜`rgb-dev`：RGB 模态已从“保留接口”进入开发与方法验证阶段；本分支基于 `amd-DirectML`。

> **后续 RGB 开发前优先阅读：** 本页 → [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md) → 当前具体方法文档。尤其在新增模型字段、QC 或全量运行前，必须先检查 044，避免因过早过滤而重新运行昂贵模型。

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

## 信息保留硬规则

RGB 不以“当前打算分析什么”决定 raw 层保存什么。昂贵推理原则上只跑一次：模型已经能够稳定返回、以后可能需要且无法从现有结果直接恢复的信息，应在第一次推理时尽量完整落盘；QC 先标记、筛选后移；derived feature 和 summary 从 raw 层重建。

完整规则、各模型 raw schema、frame identity、manifest 和开发检查清单见 [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md)。`configs/rgb_analysis.yaml` 的 `retention` 节同时编码了这一开发约束。

## 已实现：RGB-1 数据审计

```powershell
python scripts/rgb_analysis.py --stage audit
```

它检查 RGB AVI、timestamps、FocusWave timeline、B1/B2 行为文件、正式分析时间覆盖和重复被试。frame index 不连续本身不自动排除被试；只要 AVI 实际帧数与 timestamp 行数一一对应、Unix 时间单调且覆盖正式实验时间轴，局部采集 gap 作为 QC/missing 处理。`sub-033` 因此保留用于正式 RGB 分析并需要 gap QC；`sub-9504` 继续作为配置排除对象。

## 已实现：RGB-2 timestamp gap QC

```powershell
python scripts/rgb_analysis.py --stage gaps
```

输出 `D:\_AttentionData\Beijing-RGB\rgb_timestamp_gaps.csv`。当前 `timestamp_gap_warning_ms=100` 只是开发期扫描阈值，不是正式排除标准。每个超过阈值的 gap 保留：

- AVI 物理 frame position 与 FocusWave capture frame index；
- gap 前后 Unix 时间、持续时间和被试正常 median interval；
- capture index 缺失数与按真实时间估计的缺失帧数；
- gap 前/后/中点所属 phase、block、是否跨 phase；
- 是否位于正式分析区间、是否涉及正式 block；
- subject exclusion/provenance 与原始 source paths。

这样后续 Motion Energy、速度、角速度等依赖相邻帧的指标可以在 gap 后第一帧显式记 missing，而不是把断流伪造成大幅运动。

## 输出目录约定

RGB 分析结果不写入 Git 仓库，统一放到与 `Beijing-NIR` 并列的外部目录：

```text
D:\_AttentionData\
└── Beijing-RGB\
    ├── _test\
    ├── rgb_inventory.csv
    ├── rgb_duplicate_subjects.csv          # 只有检测到重复时才新生成
    ├── audit_summary.json
    ├── rgb_timestamp_gaps.csv
    ├── rgb_batch_summary.csv               # 后续正式 batch 汇总
    ├── sub-031\
    │   ├── sub-031_motion_raw.parquet
    │   ├── sub-031_pose_landmarks.parquet
    │   ├── sub-031_face_raw.parquet
    │   ├── sub-031_rgb_features.parquet
    │   ├── sub-031_rgb_summary.csv
    │   ├── sub-031_qc.csv
    │   └── sub-031_manifest.json
    └── sub-032\
        └── ...
```

结构规则：

- 数据集级 inventory/QC/summary 直接放 `Beijing-RGB` 根目录，不额外套 `audit/`、`formal/` 等层级；
- pilot、benchmark、DirectML 对比等测试输出统一放 `_test/`；
- 只有真正产生某个被试结果时才创建 `sub-XXX/`；
- 每个被试目录内部的文件仍重复带 `sub-XXX_` 前缀；
- 不建立 `face/`、`pose/`、`motion/`、`raw/`、`processed/` 等额外空套子目录；
- RGB 不额外建立 `amd-directml/` 层，实际 backend 写入 manifest/config。

## 文档入口

- [`041-RGB分析目标与数据流.md`](041-RGB分析目标与数据流.md)：RGB 要分析什么、数据怎么流动、最后得到什么。
- [`042-面部分析工具与Benchmark.md`](042-面部分析工具与Benchmark.md)：Py-Feat 与 LibreFace 的能力、环境和比较标准。
- [`043-姿态与运动量分析方法.md`](043-姿态与运动量分析方法.md)：MediaPipe Pose 与 Motion Energy 的角色和派生指标。
- [`044-RGB输出Schema与信息保留原则.md`](044-RGB输出Schema与信息保留原则.md)：**开发前必读；定义 raw 输出、QC、manifest 和“先保留后筛选”原则。**
- [`../050-decisions/053-RGB分析路线与开发边界.md`](../050-decisions/053-RGB分析路线与开发边界.md)：当前路线为何这样选，以及哪些内容尚未冻结。

## 当前工程边界

`rgb-dev` 只表示开发/验证分支，不代表 RGB 已正式冻结或全量完成。当前已建立配置、数据发现、详细时间轴解析、数据审计、timestamp gap QC 和统一输出路径模块；Face/Pose/Motion 模型适配、DirectML 导出、正式 QC 阈值、采样率和全量运行参数仍必须经过 pilot/benchmark 后再冻结。
