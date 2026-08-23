# Behavior｜行为分析

本目录统一保存行为模态的当前说明与历史正式分析资产。

## 当前正式实验版本

当前正式实验以 `runtime/nir-formal/config.yaml` 冻结的 FocusWave v3.1.3 为准：

```text
min_subject_number: 31
expected_formal_blocks: 2
formal phases: block1 + block2
```

也就是说，当前正式行为分析应围绕 **BB 两个正式 B block** 与最终版本被试重新建立，而不能继续沿用 2026-08-16 的 v3.0 BBB / sub-011~030 口径。

## 历史 SART 分析包

`sart-formal/` 当前保存的是已经完成的 **v3.0 BBB 历史分析包**，包括当时的计划、报告、PDF、图表和工作记录。它们反映当时真实执行过的分析，不追溯改写，也不再作为当前最终实验的行为分析结论。

同理：

- `configs/sart_formal.yaml` 当前仍是 v3.0 BBB 历史配置；
- `scripts/sart_formal_analysis.py` 与 `src/attention_pipeline/behavior_formal/` 当前仍包含旧 BBB 分析假设；
- 后续必须按 v3.1.3 BB 数据重新审计并改写后，才能重新称为“当前正式行为分析”。

## 其他历史资产

`014-正式实验修改建议报告.md` / `.pdf` 与 `plots/` 是正式实验前形成的行为证据与修改建议，按历史研究语境保留。

## 目录原则

后续新增当前行为方法说明时，在 `030-behavior/` 顶层使用 `031-`、`032-`、`033-` 等文档编号。程序文件、配置文件和输出文件不因为文档排序而增加数字前缀。

历史报告及其配套图表不追溯改写。正式方法发生变化时新增当前说明或 decision record，而不是覆盖旧报告的历史语境。
