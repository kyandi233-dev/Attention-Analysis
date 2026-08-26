# ChatGPT ↔ Codex Luna 协作规则

本文件定义 `Attention-Analysis` 的长期执行分工。它属于 GitHub 工程协作规则，不占用 `docs/020-nir/` 的科研文档编号。

## 职责

### ChatGPT

负责总体研究规划、方法学判断、正式分析代码/配置/测试的设计与撰写、GitHub 提交、科研文档与已验收进度更新。

### Codex Luna

负责访问本地数据、同步指定分支、运行测试和脚本、执行批处理、做明确限定的小量描述性检查，并在当前任务 Issue 中回报结果。

Luna 默认不自行：

- 修改总体分析路线；
- 决定 exclusion / QC cutoff（排除/质量阈值）；
- 选择“最显著”的窗口或模型；
- 冻结左右眼融合、PIR 标准化、RT cutoff、omission subtype 或模型结构；
- 未经明确要求大规模新写或重构正式分析代码；
- 修改 frozen runtime 或 `nir-behavior-v1.2 / schema 2`；
- 覆盖原始 NIR、Behavior 或既有正式结果。

发现代码 bug、数据结构差异或科学问题时，Luna 先报告最小复现和必要摘要，由 ChatGPT 决定修改方式。

## GitHub 各区域的用途

- **Issue（任务单）**：一次具体本地执行 = 一个独立 Issue；执行、验收完成后关闭；下一任务另开 Issue，不在一个 Issue 中无限累积。
- **Discussion（讨论区）**：保存尚未冻结的方法学讨论，例如眼别策略、PIR 标准化、统计模型选择；形成正式决定后，把结论写回科研文档或 `docs/050-decisions/`。
- **Repository docs（仓库文档）**：保存需要长期版本控制的科研计划、数据契约、已接受决策和历史工作记录；不保存滚动式临时执行日志。

当前不依赖 GitHub Project（项目看板）作为 ChatGPT ↔ Luna 的执行接口。

## 分支边界

- `amd-DirectML`：AMD 正式/稳定硬件运行主线。
- `nvidia-cuda`：NVIDIA 正式/稳定硬件运行主线。
- `analysis/multimodal-integration`：NIR、Behavior、后续 RGB 的下游融合分析主线；以稳定硬件主线为输入，不在这里继续开发硬件专属 runtime。
- RGB 独立开发使用单独 feature branch（功能分支），优先修改 `src/attention_pipeline/rgb/`、RGB configs/scripts、`docs/040-rgb/`，避免与融合分析同时修改共享入口文件。

## NIR cohort 特别边界

- 当前 44 人是 exploratory/development cohort（探索/开发样本）；最终北京样本为 116 人。
- frozen 输入保持 `nir-behavior-v1.2 / schema 2`。
- 任何 QC、排除、标准化、窗口和模型规则不能因为 44 人中的结果更显著而选择。
- 当前本地任务以仓库中仍处于 open 状态、标题明确为 NIR cohort 执行任务的独立 Issue 为准。
