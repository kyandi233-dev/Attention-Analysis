# 文档目录

本目录按“当前职责”组织。判断当前状态时，从本页进入对应模块；日期型工作记录保留当时语境，不追溯改写。

当前 branch：`amd-DirectML`。它是 AMD 综合线；当前 AMD RGB active development 在 `rgb-amd`。分支并行与同步规则见 [010-overview/015-并行分支与同步约定.md](010-overview/015-并行分支与同步约定.md)。

## 快速导航

| 我想找什么 | 去哪里 |
|---|---|
| 项目结构、模块关系与分支同步规则 | [010-overview/](010-overview/) |
| NIR 当前方法、运行入口与历史路线 | [020-nir/](020-nir/) |
| 当前 BB Behavior 与历史 BBB | [030-behavior/](030-behavior/) |
| RGB 当前科学路线、正式化状态与运行说明 | [040-rgb/](040-rgb/) |
| 为什么采纳/放弃某条技术路线 | [050-decisions/](050-decisions/) |
| 某一天实际做了什么 | [工作记录/](工作记录/) |

## 当前模块状态

- **NIR**：正式全量分析已完成；AMD DirectML runtime 与 full-class 补充分析资产已建立。
- **Behavior**：FocusWave v3.1.3 B1/B2 两-block 正式实现已建立；旧 v3.0 BBB 保留历史复现。
- **NIR × Behavior**：Unix-ms / trial / probe 对齐、coverage/QC/diagnostics 已建立。
- **RGB / AMD**：Py-Feat 2.1.1 scientific core + DirectML backend 与 15 Hz 已冻结；完整正式单被试 runner 已实现。后续 RGB 新开发优先在 `rgb-amd` 实机验收，再同步回本综合线。
- **Cross-modal**：后续统一进入 `analysis/multimodal-integration` 工作线；各模态先完成正式抽取与 QC。

## 当前状态文档优先级

判断“现在应该怎么做”时，优先看根 `README.md`、本页、对应模态 `README.md`、环境/运行说明、`scripts/README.md` 和当前 decisions/methods。

`docs/工作记录/` 只代表当时发生了什么。旧 `rgb-dev`、旧 backend 候选和阶段性“待完成”表述不代表当前状态。

## 文档编号规则

数字编号主要服务于人阅读说明文档：表达阅读顺序、所属模块和快速定位。例如 `021-...` 属于 NIR，`041-...` 属于 RGB，`051-...` 属于 decisions。

程序文件、配置、模型、数据、测试和普通运行脚本默认不加排序数字；目录入口统一使用 `README.md`。日期型历史工作记录继续保留日期命名，因为日期本身就是 provenance。

## 信息职责

- `010-overview/`：项目现在是什么、怎么连接、并行分支如何同步。
- `020-nir/`、`030-behavior/`、`040-rgb/`：对应模块现在怎么做。
- `050-decisions/`：为什么这样选、何时被替代。
- `工作记录/`：当时具体发生了什么，不追溯改写。

当前方法只保留一个 canonical 说明位置；其他文档需要提及时应链接过去，避免形成互相矛盾的“当前版本”。
