# Attention-Analysis

> GitHub 仓库当前仍使用历史名称 `attention-pipeline-v2`；项目文档统一使用 `Attention-Analysis`。

## 当前状态｜2026-08-24

- 正式 NIR 全量分析已经完成。
- 当前正式 NIR 主链：FocusWave v3.1.3 phase windows → **逐帧 YOLO26n** → ROI → RITnet batch inference → 指标/QC 输出。
- 正式 NIR runtime：`runtime/nir-formal/`；正式输出保存在仓库外独立分析目录。
- 当前正式实验版本为 FocusWave v3.1.3，正式阶段包含 `block1`、`block2` 两个 B block；旧 v3.0 BBB 行为分析包仅保留为历史结果，不作为当前行为分析口径。
- 当前工作重点：仓库整理、结果复核、资产 provenance、可复现维护，以及按最终正式版本重建行为分析与后续增量眼状态分析。

## 快速入口

| 我想找 | 入口 |
|---|---|
| 项目整体结构与数据流 | [`docs/010-overview/`](docs/010-overview/) |
| NIR 方法、当前入口与历史路线 | [`docs/020-nir/`](docs/020-nir/) |
| 行为分析与历史 SART 结果 | [`docs/030-behavior/`](docs/030-behavior/) |
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
```

其中：

- `datasets/` 保存冻结训练/标注数据与 provenance；
- `training/` 保存本项目 YOLO 训练过程与结果；
- `runtime/nir-formal/` 保存正式分析所需的冻结 YOLO/RITnet 权重、RITnet 运行源码、配置和执行入口；
- `tests/` 保存当前自动化测试代码；
- 已淘汰的第三方模型源码、历史候选模型和阶段性 artifacts 已从当前 `main` 删除，删除原因与历史用途保存在 `docs/工作记录/` 和 Git 历史中。

更完整说明见 [`docs/010-overview/013-仓库资产与复现关系.md`](docs/010-overview/013-仓库资产与复现关系.md)。

## 分支状态

`main` 是当前唯一应继续维护的主线。tracking 时代已冻结历史入口 `history/tracking-era-2026-08`，用于追溯旧版本；新开发不再从旧 tracking 分支继续。

## 历史与 provenance

日期型工作记录和历史研究文档不追溯改写。旧文档中的“候选 / 待准入 / 准备全量 / YOLO + tracking + RITnet / BBB”等表述代表当时阶段，不代表当前正式流程。

原根目录《项目总览与架构》的 2026-08-23 快照完整保留在：

[`docs/010-overview/014-2026-08-23项目总览与架构历史快照.md`](docs/010-overview/014-2026-08-23项目总览与架构历史快照.md)
