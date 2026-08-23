# Attention-Analysis

> GitHub 仓库当前仍使用历史名称 `attention-pipeline-v2`；项目文档统一使用 `Attention-Analysis`。

## 当前状态｜2026-08-24

- 正式 NIR 全量分析已经完成。
- 当前正式 NIR 主链：FocusWave v3.1.3 phase windows → **逐帧 YOLO26n** → ROI → RITnet batch inference → 指标/QC 输出。
- 正式 NIR runtime：`runtime/nir-formal/`；正式输出保存在仓库外独立分析目录。
- 当前正式实验版本为 FocusWave v3.1.3，正式阶段包含 `block1`、`block2` 两个 B block。
- 当前正式 Behavior 分析已经按最终 BB 版本建立：`configs/behavior_formal.yaml` → `scripts/sart_formal_analysis.py` → `src/attention_pipeline/behavior_formal/`。
- 旧 v3.0 BBB 行为分析仍保留独立历史可执行入口，方便以后重跑，但不作为当前正式口径。
- 当前工作重点：结果复核、资产 provenance、可复现维护、最终 BB 行为分析校验/运行，以及后续眨眼、EAR、PERCLOS 等增量眼状态分析。

## 快速入口

| 我想找 | 入口 |
|---|---|
| 项目整体结构与数据流 | [`docs/010-overview/`](docs/010-overview/) |
| NIR 方法、当前入口与历史路线 | [`docs/020-nir/`](docs/020-nir/) |
| 当前 BB 行为分析与历史 BBB | [`docs/030-behavior/`](docs/030-behavior/) |
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
- `scripts/sart_formal_analysis.py` 与 `src/attention_pipeline/behavior_formal/` 是当前 FocusWave v3.1.3 BB 行为分析；
- `scripts/sart_bbb_v3_0_analysis.py` 与 `src/attention_pipeline/behavior_bbb_v3_0/` 是明确标记的历史 BBB 可执行复现；
- `tests/` 保存当前自动化测试代码；
- 已淘汰的第三方模型源码、历史候选模型和阶段性 artifacts 已从当前 `main` 删除，删除原因与历史用途保存在 `docs/工作记录/` 和 Git 历史中。

更完整说明见 [`docs/010-overview/013-仓库资产与复现关系.md`](docs/010-overview/013-仓库资产与复现关系.md)。

## 数据根

正式实验在两台机器上使用同样的目录结构，当前已确认根目录为：

```text
E:/正式实验
F:/正式实验
```

最终正式 Behavior/NIR 均从 `sub-031` 及以后进入 v3.1.3 两-block 口径。典型正式行为文件为 `sub-XXX_/beh/sub-XXX_Block1_B_beh.csv` 与 `sub-XXX_/beh/sub-XXX_Block2_B_beh.csv`。

## 分支状态

`main` 是当前唯一应继续维护的主线。tracking 时代已冻结历史入口 `history/tracking-era-2026-08`，用于追溯旧版本；新开发不再从旧 tracking 分支继续。

## 历史与 provenance

日期型工作记录和历史研究文档不追溯改写。旧文档中的“候选 / 待准入 / 准备全量 / YOLO + tracking + RITnet / BBB”等表述代表当时阶段，不代表当前正式流程。

原根目录《项目总览与架构》的 2026-08-23 快照完整保留在：

[`docs/010-overview/014-2026-08-23项目总览与架构历史快照.md`](docs/010-overview/014-2026-08-23项目总览与架构历史快照.md)
