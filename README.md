# Attention-Analysis

> 2026-08-24（Asia/Shanghai）｜GitHub 仓库已统一命名为 `Attention-Analysis`；当前默认维护分支为 `nvidia-cuda`，后续 AMD/DirectML 路线使用 `amd-DirectML`。

> 本 checkout 为 `amd-DirectML` package `0.1.1`：YOLO26n 与 RITnet 已改用 ONNX Runtime DirectML，并增加正式完成性校验；NVIDIA/CUDA 基线仍保存在 `nvidia-cuda`。

## 当前状态｜2026-08-24

- 正式 NIR 全量分析已经完成。
- 当前正式 NIR 主链：FocusWave v3.1.3 phase windows → **逐帧 YOLO26n** → ROI → RITnet batch inference → 指标/QC 输出。
- 正式 NIR runtime：`runtime/nir-formal/`；正式输出保存在仓库外独立分析目录。
- 当前正式实验版本为 FocusWave v3.1.3，正式阶段包含 `block1`、`block2` 两个 B block。
- 当前正式 Behavior 分析已经按最终 BB 版本建立：`configs/behavior_formal.yaml` → `scripts/sart_formal_analysis.py` → `src/attention_pipeline/behavior_formal/`。
- 旧 v3.0 BBB 行为分析仍保留独立历史可执行入口，方便以后重跑，但不作为当前正式口径。
- AMD runtime 已用 ONNX Runtime DirectML 替换 Ultralytics/PyTorch CUDA 推理，固定 RITnet batch=16 + FP32，尾批补位后丢弃补位输出，默认输出使用 `amd-directml` 隔离层。

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
- `tests/` 保存仓库级自动化测试；`runtime/nir-formal/tests/` 保存正式 runtime 自包含测试；
- 已淘汰的第三方模型源码、历史候选模型和阶段性 artifacts 已从当前 `nvidia-cuda` 删除，历史用途通过 `docs/工作记录/`、决策记录、tags 和 Git 历史追溯。

更完整说明见 [`docs/010-overview/013-仓库资产与复现关系.md`](docs/010-overview/013-仓库资产与复现关系.md)。

## 正式原始数据根

正式原始数据分布在两个逻辑数据目录：`正式实验` 与 `Data`。它们位于两块外接存储设备上，而 Windows 可能根据连接顺序把两块盘分配为 `E:` 或 `F:`，因此不能把某个逻辑目录与某个盘符永久绑定。

当前 Behavior 与 NIR 配置统一声明四个候选路径：

```text
E:/正式实验
F:/正式实验
E:/Data
F:/Data
```

运行时会忽略不存在的候选根，并在所有有效根中发现被试。若同一被试在多个有效根中出现重复正式数据，程序应拒绝静默选取并报告重复数据。

最终正式 Behavior/NIR 均从 `sub-031` 及以后进入 v3.1.3 两-block 口径。典型正式行为文件为 `sub-XXX_/beh/sub-XXX_Block1_B_beh.csv` 与 `sub-XXX_/beh/sub-XXX_Block2_B_beh.csv`。

## 分支状态

`nvidia-cuda` 是 GitHub default，保存已完成正式全量分析的 NVIDIA/CUDA `1.0.0` 基线。`amd-DirectML` 已从该冻结基线开始实际改造，当前 package 为 `0.1.1`。

tracking 时代已冻结为 tag `v0.8-tracking`；正式 NIR 全量完成阶段为 `v0.9-nir-formal`；历史 BBB 行为分析为 `behavior-bbb-v3.0`。旧开发分支已删除，新开发不再从 tracking 路线继续。

## 历史与 provenance

日期型工作记录和历史研究文档不追溯改写。旧文档中的“候选 / 待准入 / 准备全量 / YOLO + tracking + RITnet / BBB”等表述代表当时阶段，不代表当前正式流程。

原根目录《项目总览与架构》的 2026-08-23 快照完整保留在：

[`docs/010-overview/014-2026-08-23项目总览与架构历史快照.md`](docs/010-overview/014-2026-08-23项目总览与架构历史快照.md)
