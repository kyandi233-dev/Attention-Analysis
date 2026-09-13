# 本地 agent 交接：Behavior / Ocular / Movement 现有科研图件审计

> 日期：2026-09-13  
> 当前阶段：A，仅审计现有真实资产，不重画、不重新估计  
> GitHub 分支：`codex/reporting-figures-bmo-v1`

## 1. 本地权威根目录

```text
D:\Project\厚粲杯\11_数据\_FormalAnalysis\FormalScience
```

只处理：

```text
FormalScience\Behavior
FormalScience\Ocular
FormalScience\Movement
```

不得处理 Cardiopulmonary、监督学习、LOSO（留一参与者交叉验证）或多模态输出。

## 2. 阶段 A 目标

本阶段只建立“当前真实资产究竟有什么、图长什么样、是否符合报告表达要求”的审计证据。不得修改统计模型，不生成新预测值，不自行决定新图件，不覆盖任何现有正式结果。

### 必须完成

1. 递归枚举三模态当前全部 `tables/`、`figures/`、`manifests/` 文件；
2. 读取每个模态现有 `figure_manifest.csv`、figure audit（若存在）和 `science_output_manifest.json`；
3. 回读所有正式 PNG/SVG；
4. 对每张图记录：
   - 实际文件名；
   - 当前 role；
   - 图中是否暴露程序字段；
   - 中文/英文标签是否符合当前报告要求；
   - 是否存在字体缺字；
   - 是否裁切；
   - 标签/点/误差线是否重叠；
   - 图例是否遮挡或有边框；
   - 坐标轴单位是否清楚；
   - 95% CI（95%置信区间）是否可辨；
   - 是否把不同科学问题混在一张图；
   - 是否把 qualification/QC/sensitivity 与主科学效应混画；
   - 是否需要 keep / reconstruct / appendix / retire。
5. 记录三个模态正式表和 manifest 的文件 checksum（校验和）；
6. 记录当前本地 Git 工作区状态。若工作区有未提交修改，不得 `git reset --hard`；后续真正运行代码时必须使用独立 worktree（工作树）；
7. 将审计结果写入不可覆盖 `_AI_HANDOFF` bundle。

## 3. Ocular 额外要求

Ocular 当前真实结果优先以：

```text
2026-09-13_ocular-postfreeze-analysis-4fdb1a2/
```

为准，并核对其执行提交：

```text
4fdb1a22f99afaad604d6023a34796e9663fd4dd
```

不得用历史 Ocular 图或 Formal 旧结果总账覆盖这一 bundle 的真实结果。

当前报告层级已冻结：

- within-person（人内）效应：正文主叙事；
- between-person（人际）效应：正式伴随结果/附录；
- between-person 不得因为放附录而被标为 sensitivity（敏感性）。

## 4. 本阶段禁止事项

- 不重新拟合任何模型；
- 不改变样本或缺失规则；
- 不筛选特征；
- 不根据显著性升级/降级变量；
- 不生成 adjusted probability（调整后概率）、predicted values（预测值）或 marginal effects（边际效应）；
- 不覆盖 `FormalScience` 中现有正式资产；
- 不修改 `report_figure_plan.csv` 的科学问题和角色。

## 5. `_AI_HANDOFF` 输出建议

创建新的不可覆盖目录，例如：

```text
_AI_HANDOFF/2026-09-13_bmo-existing-figure-audit_<sha>/
```

至少包含：

```text
HANDOFF.md
asset_inventory.csv
real_figure_audit.csv
formal_table_inventory.csv
manifest_inventory.csv
checksums_sha256.csv
git_status.txt
execution_environment.txt
preview/
```

`preview/` 中保留必要的低体积预览或联系表，原正式图继续留在 `FormalScience`，不复制或覆盖权威资产。

## 6. HANDOFF.md 必须回答

1. 三个模态当前分别有多少正式表、多少主图、多少 qualification/QC/sensitivity 图；
2. 哪些图存在明显程序字段、英文缩写或无法直接理解的模型行；
3. 哪些图视觉上已经可以继续使用；
4. 哪些图必须重构；
5. 当前真实资产是否与 GitHub `existing_figure_audit.csv` 的初步裁决冲突；
6. 是否发现正式结果数字或样本结构与当前方法文档不一致；
7. 所有执行命令、exit code（退出码）、输入/输出路径、SHA 和 checksum。

如果发现统计结果本身存在科学错误，只报告并停止对应图件，不自行修改分析。
