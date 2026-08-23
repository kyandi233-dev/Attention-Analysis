# 文档目录

> 2026-08-23｜文档体系同时保留研究模态划分，以及方法、架构、决策和历史工作记录。

## 研究模态

| 目录 | 内容 | 当前状态 |
|---|---|---|
| [010-nir/](010-nir/README.md) | NIR 检测、分割、时序 QC、训练/评价与历史方法 | 正式全量分析已完成；当前 NIR 核心入口 |
| [020-rgb/](020-rgb/README.md) | RGB / rPPG 保留接口 | 当前未重新启用正式分析 |
| [030-cross-modal/](030-cross-modal/README.md) | NIR + RGB + 行为跨模态接口 | 当前无冻结融合实现 |
| [040-behavior/](040-behavior/README.md) | 08-13 预实验行为证据与实验设计修改依据 | 历史设计证据层 |
| [050-sart-formal/](050-sart-formal/README.md) | 正式 BBB SART 行为分析 | 正式行为分析入口；主要分析已完成 |

## 通用说明

| 目录 | 用途 |
|---|---|
| [060-methods/](060-methods/README.md) | YOLO、tracking、RITnet 等方法说明；正文按 061→062→063 连续编号 |
| [070-architecture/](070-architecture/README.md) | pipeline、代码、模型、runtime 和输出边界；正文从 071 顺延 |
| [080-decisions/](080-decisions/README.md) | 技术路线选择与淘汰理由；正文从 081 顺延 |
| [工作记录/](工作记录/README.md) | 按日期保存计划、执行证据、对话和历史决策 |

## 命名原则

目录入口统一使用 `README.md`，不占用 `00` / `000` 之类的伪序号。只有真正存在连续编号关系的正文才使用数字编号。

历史工作记录不因目录重命名而批量改写；其中出现的旧路径按当时真实状态保留。
