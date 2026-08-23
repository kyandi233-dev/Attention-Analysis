# Behavior

这里是 **Attention-Analysis 当前行为分析的唯一文档入口**。

## 当前正式版本

当前正式实验以 FocusWave **v3.1.3** 为准，正式任务结构为两个 B block：

```text
B1 → B2
```

行为分析代码位于：

```text
src/attention_pipeline/behavior_formal/
```

运行入口：

```text
scripts/sart_formal_analysis.py
```

当前配置：

```text
configs/behavior_formal.yaml
```

详细说明按顺序阅读：

- `031-正式BB行为分析流程.md`
- `032-行为指标定义.md`
- `033-统计分析方法.md`
- `034-行为QC与输出.md`

## 历史版本

`history-bbb-v3.0/` 保存 2026-08-16 阶段的 v3.0 BBB 分析 bundle，包括旧计划、旧报告、PDF、图和当时的工作记录。它们用于历史追溯，不是当前正式分析入口。

旧 BBB 可执行状态另外冻结在 Git 分支：

```text
history/behavior-bbb-v3.0
```

因此 current `main` 不需要同时维护 BBB 和 BB 两套可执行分析。

## 版本边界

当前正式 BB cohort 与 NIR formal runtime 使用同一版本边界：

- FocusWave release：v3.1.3
- 被试编号下限：31
- formal blocks：2
- block：B1、B2

仓库当前没有提交正式行为 CSV，因此被试总数、异常被试、实际每 block trial / No-Go / probe 数不在配置中凭旧 BBB 结果写死。runner 会从正式数据根目录发现被试，并从实际文件做一致性校验。
