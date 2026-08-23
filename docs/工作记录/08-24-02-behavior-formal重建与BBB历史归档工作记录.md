# 08-24-02 behavior_formal 重建与 BBB 历史归档工作记录

## 本次目的

将行为分析从 2026-08-16 阶段的 FocusWave v3.0 **BBB** 实现纠正为最终正式实验 FocusWave v3.1.3 **BB** 实现，并解决 `docs/030-behavior/sart-formal/` 与行为模块重复嵌套的问题。

## 已确认的版本边界

当前正式 NIR runtime 已冻结 FocusWave v3.1.3、formal subject minimum 31、`block1`/`block2` 与 `expected_formal_blocks: 2`。因此旧 `configs/sart_formal.yaml` 中 `sub-011~030 + [B,B,B]`、B3-B1、三水平 Friedman 等均属于 v3.0 BBB 历史分析。

## 历史保护

重建前创建完整冻结分支 `history/behavior-bbb-v3.0`。旧 `docs/030-behavior/sart-formal/` bundle 整体迁为 `docs/030-behavior/history-bbb-v3.0/`，旧 YAML 同时保存历史副本。旧 BBB 若未来需要真正重跑，以冻结分支为准确可执行来源；current main 不维护 BB/BBB 两套 runner。

## 当前行为分析重建

新增 `configs/behavior_formal.yaml`，重建 `src/attention_pipeline/behavior_formal/` 与 `scripts/sart_formal_analysis.py`。正式 block 固定为 B1/B2；不写死 sub-011~030；发现不完整正式被试时不静默跳过；不直接继承 BBB 的 trial/No-Go/probe 硬编码数量；主要 block 推断改为 B2-B1 配对 Wilcoxon + bootstrap CI + Cohen's dz + Holm，并保留 block 内时间趋势、No-Go 前兆与探针辅助分析。

程序输出由旧 `051-* / 000-reports / 040-behavior / 090-manifests` 改为语义文件名。

## 文档结构

current 文档直接位于 `030-behavior/`：`031-正式BB行为分析流程.md`、`032-行为指标定义.md`、`033-统计分析方法.md`、`034-行为QC与输出.md`。历史 BBB bundle 才进入 `history-bbb-v3.0/`。

## 当前限制

仓库不保存正式行为 CSV，因此本次重建只能冻结结构契约和分析逻辑，不能在 GitHub 环境中跑出正式 cohort 结果。正式运行前仍需在数据机确认实际 v3.1.3 behavior CSV schema、每 block trial/No-Go/probe 数、最终 probe 编码语义、实际正式被试与异常/排除清单。若真实数据与当前 reader 假设不一致，应以正式数据和最终实验脚本为准更新 current implementation，不能回退套用旧 BBB 假设。
