# BBB v3.0 历史行为分析

> Status: historical / superseded / executable

这里保存 2026-08-16 基于 FocusWave v3.0 三个 B block（BBB）的行为分析材料。它们真实反映当时的分析过程，但已经被最终 v3.1.3 两个 B block（BB）实验版本替代，不能作为当前正式行为分析口径。

本目录保留当时的分析计划、分析报告与 PDF、配套图表和冻结历史材料。旧工作记录位于 `docs/工作记录/08-16-08-SART-v3.0-BBB行为分析工作记录.md`。

为满足未来无需重新编写代码即可重跑 BBB 的要求，`main` 还保留了独立、明确标记为历史的可执行实现：

```text
scripts/sart_bbb_v3_0_analysis.py
configs/sart_bbb_v3_0.yaml
src/attention_pipeline/behavior_bbb_v3_0/
```

重跑旧 BBB 时使用上述入口；当前最终 BB 则使用 `scripts/sart_formal_analysis.py`、`configs/behavior_formal.yaml` 和 `src/attention_pipeline/behavior_formal/`。两套实现分离，避免当前 BB 的后续修改改变历史 BBB 的统计逻辑。
