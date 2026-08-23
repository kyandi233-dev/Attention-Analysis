# Attention-Analysis

> GitHub 仓库当前仍使用历史名称 `attention-pipeline-v2`；项目文档统一使用 `Attention-Analysis`。

## 当前状态｜2026-08-23

- 正式 NIR 全量分析已经完成，不再处于“准备正式分析 / 等待全量推理”阶段。
- NIR 眼框 YOLO26n 已完成 100 epochs 训练；正式 runtime 使用冻结权重。
- 当前正式 NIR 主链：FocusWave v3.1.3 phase windows → **逐帧 YOLO26n** → ROI → RITnet batch inference → 指标/QC 输出。
- CSRT/KCF 等 ROI tracking 只保留用于诊断和历史复现，不属于当前正式主链。
- 当前正式 runtime：`runtime/nir-formal/`；正式输出保存在仓库外独立分析目录。
- 当前工作重点：仓库整理、结果复核、资产 provenance、可复现维护以及后续增量分析。

## 快速入口

| 我想找 | 入口 |
|---|---|
| 项目整体结构与数据流 | [`docs/010-overview/`](docs/010-overview/) |
| NIR 方法、当前入口与历史路线 | [`docs/020-nir/`](docs/020-nir/) |
| 行为分析与正式 SART | [`docs/030-behavior/`](docs/030-behavior/) |
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
- `training/` 保存本项目模型训练过程与结果；
- `models/external/` 保存第三方算法源码，`models/historical/` 保存历史候选模型资产；
- `runtime/nir-formal/` 保存已经用于正式分析的可迁移运行环境；
- `tests/` 保存自动化测试代码，不等于所有历史实验；
- `artifacts/` 保存已提交的历史评估/QC/审批证据，不是正式全量分析输出。

更完整说明见 [`docs/010-overview/013-仓库资产与复现关系.md`](docs/010-overview/013-仓库资产与复现关系.md)。

## 分支状态

`main` 是当前唯一应继续维护的主线。历史分支 `codex/nir-formal-gpu-v3` 与 `codex/v2-YOLO+Tracking+RInet` 目前仍存在于 GitHub；其中后者包含已经过时的 tracking-current 文档，因此不再整体合并回 `main`。待分支清理后，只保留 `main` 作为长期主线。

## 历史与 provenance

日期型工作记录和历史研究证据不追溯改写。旧文档中的“候选 / 待准入 / 准备全量 / YOLO + tracking + RITnet”等表述代表当时阶段，不代表当前正式流程。

原根目录《项目总览与架构》的 2026-08-23 快照完整保留在：

[`docs/010-overview/014-2026-08-23项目总览与架构历史快照.md`](docs/010-overview/014-2026-08-23项目总览与架构历史快照.md)
