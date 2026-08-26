# ChatGPT ↔ Codex Luna 协作规则

本文件定义 `Attention-Analysis` 的长期执行分工。它属于 GitHub 工程协作规则，不占用 `docs/020-nir/` 的科研文档编号。

## 职责

### ChatGPT
负责总体研究规划、方法学判断、正式分析代码/配置/测试的设计与撰写、GitHub 提交、科研文档与已验收进度更新。

### Codex Luna
负责访问本地数据、同步指定分支、运行测试和脚本、执行批处理、做明确限定的小量描述性检查，并在当前任务 Issue 中回报结果。

Luna 默认不自行修改总体分析路线、决定 exclusion / QC cutoff（排除/质量阈值）、选择“最显著”的窗口或模型、冻结左右眼融合/PIR 标准化/RT cutoff/omission subtype/模型结构，也不修改 frozen runtime 或 `nir-behavior-v1.2 / schema 2`。

## GitHub 各区域用途

- **Issue（任务单）**：一次具体本地执行一个独立 Issue；完成并验收后关闭。
- **Discussion（讨论区）**：保存尚未冻结的方法学讨论；形成正式决定后写回科研文档或 `docs/050-decisions/`。
- **Repository docs（仓库文档）**：保存长期科研计划、数据契约、已接受决策与历史记录。

当前不依赖 GitHub Project（项目看板）作为执行接口。

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