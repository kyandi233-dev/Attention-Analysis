# 029｜ChatGPT 与本地 Codex Luna 协作规则

> 2026-08-26（Asia/Shanghai）｜适用于 `analysis/nir-behavior-v2` 与 `nir-behavior-v2 / cohort-44-exploratory`。本文件只定义长期协作职责，不替代 `028-2026-08-26-NIR-cohort44分析实施计划与进度.md` 中的科学分析计划。

## 1. 文件与目录各自负责什么

后续固定分成四层，而且把 ChatGPT 与 Luna 的日常文件物理分目录：

1. `028-2026-08-26-NIR-cohort44分析实施计划与进度.md`：**科学分析总计划与长期进度**。回答“现在分析到哪一步、为什么这样做、哪些规则已经冻结/仍未冻结”。
2. `029-2026-08-26-ChatGPT与Codex-Luna协作规则.md`：**长期职责分工**。回答“ChatGPT 和 Codex Luna 各自负责什么”。
3. `docs/020-nir/chatgpt-control/030-NIR-cohort当前本地执行任务.md`：**ChatGPT → Luna 当前任务**。只由 ChatGPT 写，Luna 只读。
4. `docs/020-nir/luna-output/031-NIR-cohort最近一次本地执行结果.md`：**Luna → ChatGPT 最新结果**。只由 Luna 写，ChatGPT 只读/验收。

GitHub Issue #2 仅保留为早期联动历史档案，**不再作为日常执行入口，也不要求用户阅读或理解其中旧评论**。

顶层旧路径 `docs/020-nir/030-...` 与 `docs/020-nir/031-...` 只保留兼容跳转说明，不再承载当前内容。

## 2. 核心分工

### ChatGPT 负责

ChatGPT 是当前 NIR cohort（多人样本）分析的主要规划与代码负责人，负责：

- 总体科学问题、阶段顺序和分析路线；
- QC（质量控制）、标准化、左右眼策略、时间窗、统计模型、稳健性和机器学习边界设计；
- 44 人 exploratory/development cohort（探索/开发样本）与未来 116 人 final cohort（最终样本）的角色划分；
- 正式 Python / YAML / tests / scripts 的设计、撰写与 GitHub 提交；
- 输入/输出 schema（字段结构）和 provenance（来源追踪）设计；
- 阅读 Luna 返回的小型本地结果，并据此判断下一步；
- 更新 028 中的分析进度、证据、冻结项和未冻结项；
- 编写/覆盖 `chatgpt-control/030...` 当前执行任务；
- 对方法学决定负责，禁止仅根据结果显著性反向选择 QC、标准化或窗口规则。

除非用户明确改变分工，正式 cohort 分析代码默认由 ChatGPT 编写。

### Codex Luna 负责

Codex Luna 是本机执行与数据访问代理，主要负责：

- 只读取 `docs/020-nir/chatgpt-control/030-NIR-cohort当前本地执行任务.md`；
- 访问 ChatGPT 无法直接访问的本地数据，例如 `D:\_AttentionData`；
- 在指定 branch/worktree 中同步 GitHub 最新代码；
- 按 030 运行测试、preflight（运行前检查）、QC、alignment（时间对齐）、统计脚本或其他批处理；
- 生成本地 CSV/JSON/PNG/日志；
- 做少量、明确限定的描述性检查，例如计数、分位数、文件存在性、hash、错误定位和小范围 sanity check（合理性核验）；
- 执行后只覆盖更新 `docs/020-nir/luna-output/031-NIR-cohort最近一次本地执行结果.md`；
- 需要 ChatGPT直接审查分布时，生成小型 review bundle（复核资料包）供用户上传。

Codex Luna 默认**不负责**：

- 自行改变总体分析路线；
- 自行决定 exclusion/QC cutoff（排除/质量阈值）；
- 自行选择“最显著”的窗口或模型；
- 自行冻结左右眼融合、PIR 标准化、RT cutoff、omission subtype 或模型结构；
- 未被明确要求时大规模新写/重构 cohort 分析代码；
- 修改 `runtime/nir-formal/` 或 frozen `nir-behavior-v1.2 / schema 2`；
- 覆盖原始 NIR、Behavior 或已有 v1 结果；
- 修改 `chatgpt-control/` 中的任何文件。

若本地执行暴露出代码 bug、数据结构不一致或新的科学问题，Luna 应停止扩展设计，只在 `luna-output/031...` 报告现象、最小复现、相关路径/字段和错误信息，由 ChatGPT决定如何修改。

## 3. 最简单的日常流程

以后不再让用户记 Issue 编号或任务编号。

正常流程只有：

```text
用户在 ChatGPT 说“继续”
        ↓
ChatGPT 规划/写代码/更新 028，并覆盖 chatgpt-control/030 当前任务
        ↓
用户对 Luna 说“读取 chatgpt-control/030 并执行”
        ↓
Luna 本地运行，只更新 luna-output/031 最新结果
        ↓
用户回 ChatGPT 说“Luna 执行完了”
        ↓
ChatGPT 直接读取 luna-output/031、代码和必要文件并继续
```

用户给 Luna 的固定一句话：

```text
读取 Attention-Analysis 仓库 `docs/020-nir/chatgpt-control/030-NIR-cohort当前本地执行任务.md`，严格按 029 的职责分工执行。不要修改 `chatgpt-control/`、分析计划或分析代码；完成后只更新 `docs/020-nir/luna-output/031-NIR-cohort最近一次本地执行结果.md`。
```

Luna 完成后，用户回到 ChatGPT 只需说：

```text
Luna 执行完了，继续。
```

## 4. GitHub 与本地数据边界

GitHub 保存：

- ChatGPT 撰写的代码、配置、测试和脚本；
- 028 分析计划与长期进度；
- 029 协作规则；
- `chatgpt-control/030...` 当前本地任务；
- `luna-output/031...` 最近一次本地结果；
- 少量、非敏感的 aggregate summary（汇总结果）和 provenance。

本地保存：

- full-class 大 CSV；
- raw Behavior；
- schema-2 alignment 大型产物；
- trial/probe 大表；
- cohort QC/统计输出；
- 大型图像和运行日志。

不为了让 ChatGPT读取而把全量实验数据推到 GitHub。

## 5. 什么时候需要用户上传文件

只有当 031 的小型汇总不足以支持科学判断时，ChatGPT 才要求 Luna 生成 phase review bundle（阶段复核资料包）。例如需要直接看：

- 44 人 PIR valid fraction（PIR 有效率）分布；
- 左右眼差异/一致性分布；
- omission subtype（遗漏反应子类型）的个体分布；
- 关键 QC 图或 compact table（小型汇总表）。

用户只上传这个小 bundle，不上传逐帧 full-class 或全部 trial 数据。

## 6. 当前 cohort44 特别边界

- 当前 44 人属于 exploratory/development cohort；最终北京 cohort 为 116 人。
- frozen 输入仍为 `nir-behavior-v1.2 / schema 2`。
- 新 cohort 输出根为 `D:\_AttentionData\Beijing-NIR\analysis\nir-behavior-v2\cohort-44-exploratory\`。
- omission 必须保留 raw program scoring（程序原始评分）与 QC-aware subtype（质量控制辅助子类型）两层。
- OAR 仍是连续 ocular aperture / eye-openness auxiliary feature（眼部可见开合度辅助指标），不等同于 blink、EAR 或 PERCLOS。
- 任何 QC/exclusion/standardization/window/model rule 都不能因为 44 人中某个结果更显著而被选择。

## 7. 术语表达规则

ChatGPT 后续向用户说明变量、代码字段或专业方法时，第一次出现必须同时给出中文解释。例如：

- `cohort（队列/多人样本集合）`；
- `inventory（数据清单/盘点）`；
- `preflight（运行前检查）`；
- `QC / Quality Control（质量控制）`；
- `alignment（时间对齐：把 NIR 帧与 trial/probe 的实验时间对应起来）`；
- `robust z-score（稳健标准分：用中位数和 MAD 衡量某个值偏离总体中心的程度）`。

后续新增术语沿用同一原则；不能只给英文缩写而不解释其分析含义。
