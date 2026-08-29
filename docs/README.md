# 文档目录

本目录按“当前职责”组织，而不是同时按模态、方法类型和生命周期平铺。判断当前状态时，从本页进入对应模块；日期型工作记录保留当时语境，不追溯改写。

## 快速导航

| 我想找什么 | 去哪里 |
|---|---|
| 整个项目现在是什么、模块如何连接 | [010-overview/](010-overview/) |
| NIR 当前方法、运行入口与历史路线 | [020-nir/](020-nir/) |
| 当前 BB 行为分析与历史 BBB | [030-behavior/](030-behavior/) |
| RGB 当前分析与开发状态 | [040-rgb/](040-rgb/) |
| 为什么采纳/放弃某条技术路线 | [050-decisions/](050-decisions/) |
| **正式多模态 V2：路径、cohort、pupil-only 与合并合同** | [060-formal-analysis/001-正式多模态V2路径与分析契约.md](060-formal-analysis/001-正式多模态V2路径与分析契约.md) |
| 某一天实际做了什么 | [工作记录/](工作记录/) |

## 当前模块状态

当前正式状态以 `kyandi233-dev/FocusWave-Formal-Analysis` 的证据仓库和本分支 V2 合同为准，不再把旧 README/历史输出中的“完成”字样直接继承为当前科学结论。

- **NIR**：`fullclass-final` producer 与瞳孔几何/QC 资产已存在；当前正式下游以 pupil-only 为主。证据基线已完成 27 个场次的来源/schema/时间/QC 预检，但这不是 44 场正式统计，也不是 27 名独立参与者。旧 PIR/iris_outer 分析只保留历史复现。
- **Behavior**：FocusWave v3.1.3 BB 原始数据与工程骨架可用；V2 新入口可以按外部 cohort/path manifest 做提取和指标落盘。旧 `stats.py` 仍是 session-level，当前正式推断默认阻断，直到实现 repeat-participant-safe 模型。
- **RGB**：已有工程 producer/输出，但 `complete_or_skipped` 不等于科学 QC 通过。正式优先级是主脸、有效观测、眼睑/眨眼，再到头动、姿态、体动与 AU。
- **毫米波**：producer 不在本仓库复制；本分支只接收其 merge-ready 输出。当前正式证据仍有依赖、字段合同与 NaN candidate 阻断，缺失/无效场次不得补零。
- **Cross-modal**：融合统一在 `analysis/multimodal-integration` 进行，采用外部路径注册表、cohort/source manifest、`repeat_participant_id` 分组和统一 merge key。当前没有可发布的正式融合性能或心理效应结果。

## 文档编号规则

顶层目录编号先表达**模块归属**：

- `010-overview/` → overview；核心说明使用 `011–019`；
- `020-nir/` → NIR；核心说明使用 `021–029`；
- `030-behavior/` → Behavior；核心说明使用 `031–039`；
- `040-rgb/` → RGB；核心说明使用 `041–049`；
- `050-decisions/` → decisions；核心说明使用 `051–059`；
- `060-formal-analysis/` → 当前正式下游/融合合同；核心说明使用 `061–069` 或描述性文件名，现有 `001-...` 作为本次迁移入口保留。

因此，核心编号不是全仓库单纯按创建时间连续递增，而是“**十位表示模块，个位表示该模块中的核心阅读顺序**”。日期型历史资料继续使用 `MM-DD-序号-...` 或其他明确日期命名，因为日期本身就是 provenance（来源/过程追踪）。

## GitHub 协作区域与仓库文档的边界

实时协作状态不再占用科研文档编号：

- **Issue（任务单）**：一次具体本地执行一个独立 Issue；完成并验收后关闭，下一任务另开；
- **Discussion（讨论区）**：保存尚未冻结的方法学讨论；形成决定后再写回正式科研文档或 `050-decisions/`；
- **`docs/`**：保存需要长期版本控制的科研计划、数据契约、正式方法/决策和历史工作记录。

当前不依赖 GitHub Project（项目看板）作为 ChatGPT ↔ Codex Luna 的执行接口。长期执行分工见 `.github/CODEX-LUNA-COLLABORATION.md`。

## 信息职责

- `010-overview/`：系统与仓库总览。
- `020-nir/`、`030-behavior/`、`040-rgb/`：对应 producer/模态方法与历史说明。
- `050-decisions/`：为什么这样选、何时被替代。
- `060-formal-analysis/`：当前正式下游的路径、cohort、schema、统计门和融合合同。
- `工作记录/`：当时具体发生了什么。
