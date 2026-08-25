# Attention-Analysis

> 2026-08-26（Asia/Shanghai）｜GitHub 仓库已统一命名为 `Attention-Analysis`；当前默认维护分支为 `nvidia-cuda`，AMD/DirectML 路线使用 `amd-DirectML`。

## 当前 NVIDIA 四分类补跑：先看这里

RTX 5070 工作站当前正式任务不是重新跑完整 YOLO + RITnet，而是**复用既有正式 `eyes.csv` 和原 AVI，只重新运行 RITnet 四分类**，补齐 iris/sclera/background、pupil/iris normalization 和 sparse QC。

从 GitHub 拉取、激活 Conda 环境、检查 RTX 5070 / CUDA / `CUDAExecutionProvider`、检查 `J:/Data`、运行 pytest、dry-run、单被试验收、72 人正式补跑、断点恢复和输出检查的完整终端命令统一放在根目录：

**[`NVIDIA-RITnet全分类补跑使用说明.md`](NVIDIA-RITnet全分类补跑使用说明.md)**

当前 NVIDIA 机器固定信息：

```text
GPU: NVIDIA GeForce RTX 5070
Conda env: D:\conda_envs\eye-ai
repo: D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda
data root: J:\Data
formal NIR output/source root:
D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR
```

本次 full-class 补跑固定：`640×400 / FP32 / b16`，不使用 FP16，不降到 512，不重新运行 YOLO。主 pupil 指标为 `fullclass_pupil_to_iris_diameter_ratio`。

## 当前状态｜2026-08-26

- 历史正式 NIR 全量分析已经完成。
- 当前正式 NIR 主链：FocusWave v3.1.3 phase windows → **逐帧 YOLO26n** → ROI → RITnet batch inference → 指标/QC 输出。
- 由于历史正式 `eyes.csv` 只保存 pupil 派生量，当前新增 post-hoc RITnet full-class extension：复用既有 `frame_idx + ROI`，不重跑 YOLO，只重新运行冻结 RITnet ONNX，保存 background / sclera / iris / pupil、虹膜外轮廓、pupil/iris normalization 与 sparse QC。
- 正式 NIR runtime：`runtime/nir-formal/`；正式输出保存在仓库外独立分析目录。
- 当前正式实验版本为 FocusWave v3.1.3，正式阶段包含 `block1`、`block2` 两个 B block。
- 当前正式 Behavior 分析已经按最终 BB 版本建立：`configs/behavior_formal.yaml` → `scripts/sart_formal_analysis.py` → `src/attention_pipeline/behavior_formal/`。
- NIR ↔ SART trial-level 对齐作为独立下游步骤，通过 NIR `unix_ms` 与行为绝对时间戳映射，不需要再次运行 RITnet。
- 旧 v3.0 BBB 行为分析仍保留独立历史可执行入口，方便以后重跑，但不作为当前正式口径。
- NVIDIA/CUDA 历史全量基线冻结为 tag `nvidia-v1.0.0`；当前 full-class 补全阶段计划冻结 tag `nvidia-v1.2-ritnet-fullclass`。

## 快速入口

| 我想找 | 入口 |
|---|---|
| **RTX 5070 当前 RITnet 四分类正式补跑** | **[`NVIDIA-RITnet全分类补跑使用说明.md`](NVIDIA-RITnet全分类补跑使用说明.md)** |
| RITnet full-class 技术说明 | [`runtime/nir-formal/RITNET_FULLCLASS_EXTENSION.md`](runtime/nir-formal/RITNET_FULLCLASS_EXTENSION.md) |
| NIR formal 运行/故障恢复规则 | [`runtime/nir-formal/RUNBOOK.md`](runtime/nir-formal/RUNBOOK.md) |
| NVIDIA runtime 安装说明 | [`runtime/nir-formal/INSTALL.md`](runtime/nir-formal/INSTALL.md) |
| 项目整体结构与数据流 | [`docs/010-overview/`](docs/010-overview/) |
| NIR 方法、当前入口与历史路线 | [`docs/020-nir/`](docs/020-nir/) |
| 当前 BB 行为分析与历史 BBB | [`docs/030-behavior/`](docs/030-behavior/) |
| NIR ↔ SART 对齐方法 | [`docs/030-behavior/035-NIR与正式SART行为数据对齐分析方法.md`](docs/030-behavior/035-NIR与正式SART行为数据对齐分析方法.md) |
| RGB 保留接口 | [`docs/040-rgb/`](docs/040-rgb/) |
| 技术选择与路线变更原因 | [`docs/050-decisions/`](docs/050-decisions/) |
| 日期型研究过程 | [`docs/工作记录/`](docs/工作记录/) |
| 正式 NIR 运行包 | [`runtime/nir-formal/`](runtime/nir-formal/) |
| 仓库长期工作规则 | [`AGENTS.md`](AGENTS.md) |

完整文档导航见 [`docs/README.md`](docs/README.md)。

## 关键资产关系

```text
datasets/
    ↓
training/
    ↓
trained weights
    ↓
runtime/nir-formal/
    ↓
正式 NIR 全量分析
    ↓
RITnet full-class post-hoc extension
    ↓
NIR × SART downstream alignment/analysis
```

其中：

- `datasets/` 保存冻结训练/标注数据与 provenance；
- `training/` 保存本项目 YOLO 训练过程与结果；
- `runtime/nir-formal/` 保存正式分析所需的冻结 YOLO/RITnet 权重、RITnet 运行源码、配置和执行入口；
- `runtime/nir-formal/run_ritnet_fullclass_batch.py` 是当前 NVIDIA RITnet 四分类 post-hoc 批量补跑入口；
- `scripts/sart_formal_analysis.py` 与 `src/attention_pipeline/behavior_formal/` 是当前 FocusWave v3.1.3 BB 行为分析；
- `scripts/sart_bbb_v3_0_analysis.py` 与 `src/attention_pipeline/behavior_bbb_v3_0/` 是明确标记的历史 BBB 可执行复现；
- `tests/` 保存仓库级自动化测试；`runtime/nir-formal/tests/` 保存正式 runtime 自包含测试；
- 已淘汰的第三方模型源码、历史候选模型和阶段性 artifacts 已从当前 `nvidia-cuda` 删除，历史用途通过 `docs/工作记录/`、决策记录、tags 和 Git 历史追溯。

更完整说明见 [`docs/010-overview/013-仓库资产与复现关系.md`](docs/010-overview/013-仓库资产与复现关系.md)。

## 当前 NVIDIA 正式原始数据根

当前 `nvidia-cuda` 分支对应的 RTX 5070 工作站使用：

```text
J:/Data
```

正式 NIR 与正式 Behavior 配置均以 `J:/Data` 为当前 NVIDIA 数据根。不要为了复用 AMD 机器配置而把本分支改成 AMD 的路径。

典型正式被试目录：

```text
J:/Data/sub-XXX_/
├── beh/
│   ├── master_timeline.csv
│   ├── sub-XXX_Block1_B_beh.csv
│   └── sub-XXX_Block2_B_beh.csv
└── nir/
    ├── sub-XXX_nir.avi
    └── sub-XXX_nir_timestamps.csv
```

最终正式 Behavior/NIR 均从 `sub-031` 及以后进入 v3.1.3 两-block 口径；实际 NVIDIA 队列以当前机器 dry-run 发现结果为准。

## 分支状态

`nvidia-cuda` 是当前 GitHub default，当前 package 为 `1.0.1`；已完成正式全量分析的 `1.0.0` 基线由 tag `nvidia-v1.0.0` 保留。AMD 路线使用 `amd-DirectML`，其硬件后端、数据根和运行参数说明由 AMD 分支自己的文档维护，不从 `nvidia-cuda` 当前本机配置反推。

tracking 时代已冻结为 tag `v0.8-tracking`；正式 NIR 全量完成阶段为 `v0.9-nir-formal`；历史 BBB 行为分析为 `behavior-bbb-v3.0`。旧开发分支已删除，新开发不再从 tracking 路线继续。

## 历史与 provenance

日期型工作记录和历史研究文档不追溯改写。旧文档中的“候选 / 待准入 / 准备全量 / YOLO + tracking + RITnet / BBB”等表述代表当时阶段，不代表当前正式流程。

原根目录《项目总览与架构》的 2026-08-23 快照完整保留在：

[`docs/010-overview/014-2026-08-23项目总览与架构历史快照.md`](docs/010-overview/014-2026-08-23项目总览与架构历史快照.md)
