# ChatGPT ↔ Codex Luna 协作规则

本文件定义 `Attention-Analysis` 的长期执行分工。它属于 GitHub 工程协作规则，不占用 `docs/020-nir/` 的科研文档编号。

## 职责

### ChatGPT
负责总体研究规划、方法学判断、正式分析代码/配置/测试的设计与撰写、GitHub 提交、科研文档与已验收进度更新。

### Codex Luna
负责访问本地数据、同步指定分支、运行测试和脚本、执行批处理、做明确限定的小量描述性检查，并在当前任务 Issue 中回报结果。

Luna 默认不自行修改总体分析路线、决定 exclusion / QC cutoff（排除/质量阈值）、选择“最显著”的窗口或模型、冻结左右眼融合/PIR 标准化/RT cutoff/omission subtype/模型结构，也不修改 frozen runtime 或 `nir-behavior-v1.2 / schema 2`。

## GitHub 各区域用途

- **Issue（任务单）**：一个明确的科研/工程任务对应一个 Issue。若同一任务需要“运行 → ChatGPT 复核 → 补充检查 → 再复核”等多轮本地执行，可以继续在同一个 Issue 内追加明确指令与 Codex 回复；不必为了每一次命令单独新建 Issue。只有当当前任务已经形成最终结论或验收完成时才关闭；进入新的科研问题或新的独立工程目标时再新建 Issue。
- **Discussion（讨论区）**：保存尚未冻结的方法学讨论；形成正式决定后写回科研文档或 `docs/050-decisions/`。
- **Repository docs（仓库文档）**：保存长期科研计划、数据契约、已接受决策与历史记录。

当前不依赖 GitHub Project（项目看板）作为执行接口。

## Issue 内部协作格式

同一 Issue 内允许多轮执行，但每一轮都应保持单一目的：

1. ChatGPT 明确说明“本轮还缺什么、为什么需要、Codex 具体做什么”；
2. Codex 只执行本轮要求并回复结果，不自行扩展分析；
3. ChatGPT 读取结果并决定是补充同一任务、形成最终决策，还是结束该 Issue；
4. 最终决策和正式筛选/分析规则必须落盘到仓库文档或本地分析输出，不能只留在 Issue 评论里。

避免在同一 Issue 中混入无关的新科研问题，也避免为同一任务的每个小命令创建大量碎片 Issue。

## 科研文档编号规则

科研文档编号按“目录类别编号 + 该目录内顺序号”连续扩展，不因为顺序号超过 9 就跳到下一目录类别。

以 `docs/020-nir/` 为例：

```text
021, 022, ... 029, 210, 211, 212, ...
```

含义是 `02` 类目下的第 1、2、…、9、10、11、12 个文档。第 10 个文档使用 `210`，不写成 `030`，也不额外补前导 0。已有编号不因后续扩展而重排。

同类规则应用于其他 docs 子目录时，以该目录既有类别前缀为准；新增文档前必须先检查该目录现有最大顺序号，避免重复、跳号或误跨类别编号。

## 分支边界

- `amd-DirectML`：AMD 正式/稳定硬件运行主线。
- `nvidia-cuda`：NVIDIA 正式/稳定硬件运行主线。
- `analysis/multimodal-integration`：NIR、Behavior、Probe、Questionnaire 以及后续 RGB 的下游融合分析主线。
- `rgb-amd`：AMD 平台 RGB 开发分支。
- `rgb-nvidia`：NVIDIA 平台 RGB 开发分支。

RGB 分支优先修改 `src/attention_pipeline/rgb/`、RGB configs/scripts、`docs/040-rgb/`；融合分析分支优先修改 cohort / multimodal 模块。双方尽量避免同时修改共享 `README.md`、`AGENTS.md`、`pyproject.toml`、`docs/README.md` 和 `docs/010-overview/`。

## NIR cohort 特别边界

- 当前 44 人是 exploratory/development cohort（探索/开发样本）；最终北京样本为 116 人。
- frozen 输入保持 `nir-behavior-v1.2 / schema 2`。
- 任何 QC、排除、标准化、窗口和模型规则不能因为 44 人中的结果更显著而选择。

## 术语表达

向用户说明变量、代码字段或专业方法时，第一次出现采用“英文术语（中文解释）”形式，例如 `preflight（运行前检查）`、`QC（质量控制）`、`alignment（时间对齐）`。
