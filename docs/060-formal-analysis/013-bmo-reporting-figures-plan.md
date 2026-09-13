# Behavior / Ocular / Movement 科研绘图与结果汇报线

> 日期：2026-09-13  
> 分支：`codex/reporting-figures-bmo-v1`  
> 状态：`planning_and_audit_only_before_redraw`

本分支只负责 Behavior（行为）、Ocular（眼部）、Movement（动作）的正式结果表达、图件规划、语义映射与绘图实现。不得修改已冻结特征、正式模型公式、正式效应数字、P3、P5、7 场 NIR（近红外）结构性缺失裁决、mmWave/Cardiopulmonary、统一特征注册表、监督学习、LOSO（留一参与者交叉验证）和多模态性能分析。

## 1. 真实输出位置

本分支不新建真实结果根目录。正式 full run 继续写：

```text
D:\Project\厚粲杯\11_数据\_FormalAnalysis\FormalScience\
├─ Behavior\
├─ Ocular\
└─ Movement\
```

每个模态继续使用 `tables/`、`figures/{main,qualification,qc,sensitivity}/`、`manifests/`。

## 2. 第一轮只做计划与审计

在大规模重画前先维护：

- `report_figure_plan.csv`：图件为什么存在、回答什么问题、放正文还是附录；
- `scientific_label_dictionary.csv`：程序字段到正式中文科学语义的唯一映射；
- `existing_figure_audit.csv`：现有图逐张裁决，记录保留/重构/移附录/废弃；
- 本文件：执行边界和本地 agent 交接合同。

正文主图不按 `p < .05` 升级。主图资格来自预冻结科学角色与研究问题。

## 3. Ocular 报告层级冻结

Ocular 的 within-person（人内）效应作为正文主叙事；between-person（人际）效应完整保留正式结果表或附录图。该决定只改变报告组织，不修改任何模型或效应数字。

## 4. 网页端与本地 agent

网页端负责科研叙事、图件计划、标签字典、GitHub 代码与 PR（拉取请求）、回传审查。

本地 agent 负责真实 `FormalScience` 资产枚举、现有图回读、运行固定 SHA、生成真实图、必要时从完全相同冻结模型生成 predicted values（预测值）/ marginal effects（边际效应）、一致性核对、视觉检查和 `_AI_HANDOFF` bundle。

### 阶段 A：立即可执行

- 枚举三模态现有 tables / figures / manifests；
- 逐张回读现有 PNG/SVG；
- 记录字体、裁切、重叠、坐标轴、图例、代码变量暴露问题；
- 记录输入输出 checksum（校验和）和执行环境；
- 不改模型、不生成新派生效应。

### 阶段 B：`report_figure_plan.csv` 审查后执行

只生成已批准图件。需要调整后概率、预测值或边际效应时，只调用原冻结模型生成展示性派生量，并逐项与原正式估计核对。

## 5. 第一轮停止条件

第一轮只在以下内容完成并经研究者审查后进入绘图实现：现有图逐张裁决、三模态结果叙事地图、统一科学语义字典、`report_figure_plan.csv`、正文主图组合、需要本地派生值的图件列表、本地 agent 阶段 A 审计回传。
