# 029｜ChatGPT 与本地 Codex Luna 协作规则

> 2026-08-26（Asia/Shanghai）｜适用于 `analysis/nir-behavior-v2` 与 `nir-behavior-v2 / cohort-44-exploratory`。本文件只定义长期协作职责，不替代 `028-2026-08-26-NIR-cohort44分析实施计划与进度.md` 中的科学分析计划。

## 1. 核心分工

本项目后续采用明确的“设计/编码”和“本地执行”分工。

### ChatGPT 负责

ChatGPT 是当前 NIR cohort 分析的主要规划与代码负责人，负责：

- 总体科学问题、阶段顺序和分析路线；
- QC、标准化、左右眼策略、时间窗、统计模型、稳健性和 ML 边界的设计；
- 对 44 人 exploratory cohort 与未来 116 人 final cohort 的角色划分；
- 代码架构、配置结构、输入/输出 schema 和 provenance 设计；
- 正式 Python / YAML / tests / scripts 的撰写与 GitHub 提交；
- 阅读本地 Codex Luna 返回的小型汇总结果，并据此判断下一步；
- 更新 `028` 中的分析进度、证据、冻结项和未冻结项；
- 对重要方法学决定给出解释，避免只根据显著性结果倒推 QC、标准化或窗口规则。

除非用户明确改变分工，正式 cohort 分析代码默认由 ChatGPT 编写，而不是交给本地 Codex Luna自由设计或大规模开发。

### 本地 Codex Luna 负责

Codex Luna 是本机执行与数据访问代理，主要负责：

- 读取 ChatGPT 已经写入 GitHub 的代码、配置和执行说明；
- 访问 ChatGPT 无法直接访问的本地数据，例如 `D:\_AttentionData`、Behavior 原始目录和现有分析输出；
- 在指定 branch/worktree 中 pull/fetch 最新代码；
- 安装缺失依赖或报告环境冲突；
- 按说明运行测试、preflight、alignment、QC、统计脚本或其他批处理；
- 生成本地 CSV/JSON/PNG/日志和 compact summary；
- 做少量、明确限定的描述性检查，例如计数、分位数、文件存在性、hash、错误定位和小范围 sanity check；
- 把执行结果以简短摘要写入 GitHub Issue #2，或在必要时生成一个供用户上传给 ChatGPT 的小型 review bundle。

Codex Luna 默认**不负责**：

- 自行更改总体分析路线；
- 自行决定 exclusion/QC cutoff；
- 自行选择“最显著”的窗口或模型；
- 自行冻结左右眼融合、PIR 标准化、RT cutoff、omission subtype 或模型结构；
- 在没有 ChatGPT 明确要求时大规模新写/重构 cohort 分析代码；
- 修改 `runtime/nir-formal/` 或 frozen `nir-behavior-v1.2 / schema 2`；
- 覆盖原始 NIR、Behavior 或已有 v1 结果。

如果本地执行暴露出代码 bug、数据结构不一致或新的科学问题，Codex Luna 应停止扩展设计，只报告：现象、最小复现、相关路径/字段、错误信息和建议检查点，由 ChatGPT决定是否修改代码或分析政策。

## 2. GitHub 与本地数据的职责边界

GitHub 保存：

- ChatGPT 撰写的代码、配置、测试和脚本；
- `028` 分析计划与进度；
- 本协作规则；
- Issue #2 中的简短任务/执行结果；
- 非敏感、compact 的 aggregate summary 和 provenance。

本地保存：

- full-class 大 CSV；
- raw Behavior；
- schema-2 alignment 大型产物；
- trial/probe 大表；
- cohort QC/统计输出；
- 大型图像和运行日志。

不为了让 ChatGPT读取而把全量实验数据推到 GitHub。

## 3. 最简单的工作流

后续不要求用户理解 `C44-P1-001` 之类的任务编号。任务编号只是 GitHub 内部追踪标记。

正常流程只有四步：

1. 用户在 ChatGPT 中说要继续当前 cohort 分析。
2. ChatGPT 完成规划/代码修改并把需要本地执行的内容写到 GitHub Issue #2。
3. 用户只需对本地 Codex Luna 说：

```text
读取 Attention-Analysis 仓库 GitHub Issue #2 的最新执行说明，按 029 的职责分工执行。只负责本地运行、测试和小量汇总，不自行改分析计划或大规模写代码。完成后把简短结果回复到 Issue #2。
```

4. Codex Luna 完成后，用户回到 ChatGPT 只需说：

```text
Codex Luna 已执行完成，请读取 Issue #2 结果并继续。
```

ChatGPT 随后自行读取 Issue、commit 和必要的 GitHub 文件，不要求用户复制长日志。

## 4. 什么时候需要用户上传文件

只有当本地 aggregate summary 不足以支持科学判断时，ChatGPT 才要求一个小型 phase review bundle。例如：

- 需要直接看 44 人 PIR valid fraction 分布；
- 需要看左右眼 offset/一致性分布；
- 需要判断某类 omission subtype 的个体分布；
- 需要直接审查关键 QC 图或 compact table。

这时 Codex Luna 应只生成必要的小文件，例如：

```text
phase1_review_bundle/
├── summary.md
├── subject_eye_block_qc.csv
├── selected_distribution_table.csv
└── selected_figures/
```

用户只上传这个 bundle，不上传全量 frame/trial 数据。

## 5. 当前 cohort44 特别边界

- 44 人当前属于 exploratory/development cohort；最终北京 cohort 为 116 人。
- frozen 输入仍为 `nir-behavior-v1.2 / schema 2`。
- 新 cohort 输出根为 `D:\_AttentionData\Beijing-NIR\analysis\nir-behavior-v2\cohort-44-exploratory\`。
- 当前 44/44 有完整 Behavior B1/B2 和四个 NIR eye×block；现有 complete schema-2 alignment 只有 sub-031。
- omission 必须保留 raw program scoring 与 QC-aware subtype 两层，不能把全部 omission 解释为同一种注意漏失。
- OAR 仍是连续 ocular aperture / eye-openness auxiliary feature，不等同于 blink、EAR 或 PERCLOS。
- 任何 QC/exclusion/standardization/window/model rule 都不能因为 44 人中某个结果更显著而被选择。

## 6. 关于 Issue #2 中旧任务编号

Issue #2 早期的 `C44-P1-001` 等编号仅用于最初设计的自动交接格式。自本文件生效后：

- 用户不需要理解或手动指定这些编号；
- ChatGPT 可以继续在 Issue 中使用编号作为内部追踪，但必须同时写清楚自然语言执行说明；
- 本地 Codex Luna 只执行 Issue #2 中由 ChatGPT明确标记为“当前执行说明”的最新内容；
- 若旧评论与本文件的职责分工冲突，以本文件和 Issue #2 最新首页/说明为准。
