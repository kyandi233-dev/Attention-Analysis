"""Markdown reporting for the current final BB behavior analysis."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from ..config import Config
from . import metrics as fmet
from . import stats as fstat


def _markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "（无可报告结果）"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("–" if pd.isna(value) else f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_report(config: Config, trials: pd.DataFrame, output_path: Path | None = None) -> dict:
    output_path = output_path or (config.path_value("output_root") / "report.md")
    blocks = fmet.formal_block_metrics(config, trials)
    effects = fstat.paired_block_effects(config, blocks)
    probes = fmet.probe_behaviour_link(config, trials)
    probe_stats = fstat.probe_associations(probes)
    summary = blocks.groupby("block_num")[["commission_rate", "omission_rate", "dprime_loglinear", "go_rt_median_ms", "rt_cv"]].mean().reset_index()

    md = f"""# FocusWave v3.1.3 正式行为分析报告

> 当前分析对象：最终正式 **BB** 版本；被试从正式数据根目录自动发现，最低编号由配置定义。旧 v3.0 BBB 结果不参与本报告。

## 数据范围

- FocusWave release：`{config.section('pipeline').get('focuswave_release')}`
- 被试数：{trials['subject'].nunique()}
- 正式 block：B1、B2
- 总 trial：{len(trials)}
- 探针 trial：{int(trials['is_probe'].eq(1).sum())}

## Block 描述统计

{_markdown_table(summary)}

## B2 − B1 被试内比较

{_markdown_table(effects)}

解释口径：`B2_minus_B1_*` 为 B2 减 B1；主检验为配对 Wilcoxon，`cohen_dz` 为配对差值标准化效应量，`wilcoxon_p_holm` 为主指标族的 Holm 校正结果。

## 探针与临近行为

```json
{json.dumps(probe_stats, ensure_ascii=False, indent=2)}
```

## 说明

本报告不把旧 BBB 的 B3-B1、三水平 Friedman 或固定 `sub-011~030` 规则带入最终正式分析。最终被试排除、异常数据处理与跨模态关联应以本次正式数据 QC 和项目 decisions/work records 为依据。
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    return {"report": str(output_path), "subjects": int(trials["subject"].nunique())}
