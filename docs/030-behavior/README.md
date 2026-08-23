# Behavior

这里是 **Attention-Analysis 当前行为分析的唯一文档入口**。

## 当前正式版本

当前正式实验以 FocusWave **v3.1.3** 为准，正式 SART 结构为两个 B block：

```text
B1 → B2
```

当前代码与配置：

- `configs/behavior_formal.yaml`
- `scripts/sart_formal_analysis.py`
- `src/attention_pipeline/behavior_formal/`

当前说明按顺序阅读：

- `031-正式BB行为分析流程.md`
- `032-行为指标定义.md`
- `033-统计分析方法.md`
- `034-行为QC与输出.md`

旧 v3.0 BBB 的三 block 统计、B1↔B3、sub-011~030 队列和旧结果均不属于当前正式分析。

## 历史材料

历史材料统一放在 `history/`，不再与当前说明并排竞争：

- `history/BBB-v3.0/`：2026-08-16 的 BBB 分析计划、报告、图和冻结旧配置；
- `history/preformal/`：最终正式实验确定前的行为实验修改建议、图和统计审计。

旧 BBB 的原工作记录已经移入统一的 `docs/工作记录/08-16-08-SART-v3.0-BBB行为分析工作记录.md`，正文保持原样。完整旧 BBB 可执行代码另由 `history/behavior-bbb-v3.0` 分支冻结。

因此本目录不再设置 `sart-formal/` 子层：`030-behavior/` 本身就是行为模块，当前正式文档直接放在这里；只有退出当前主线的材料进入 `history/`。
