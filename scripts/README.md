# Scripts

`scripts/` 只保留当前仍值得在主线直接执行或即将重建的仓库级入口。正式 NIR 全量分析不从这里启动，当前正式 NIR 入口是：

```text
runtime/nir-formal/
```

## 当前保留

| 脚本 | 当前定位 |
|---|---|
| `extract_eye_dataset.py` | YOLO 眼框数据集抽帧与 provenance |
| `evaluate_yolo_eye_test.py` | 冻结 YOLO test 的正式评估与审计 |
| `sart_formal_analysis.py` | **旧 v3.0 BBB 行为分析总控，暂留待按最终 v3.1.3 BB 版本重写；当前不可视为正式行为分析入口** |

## 行为分析版本纠偏

当前正式实验 runtime 已冻结：

```text
FocusWave v3.1.3
min_subject_number: 31
expected_formal_blocks: 2
block1 + block2
```

因此旧 `configs/sart_formal.yaml`、`scripts/sart_formal_analysis.py` 和 `src/attention_pipeline/behavior_formal/` 当前仍反映的 v3.0 BBB / sub-011~030 分析口径，只能作为历史实现参考。下一步应依据最终 BB 数据重新定义 cohort、校验契约、block 比较、图表和报告，再恢复其“当前正式行为分析”身份。

旧 `docs/030-behavior/sart-formal/` 报告包不追溯改写，作为 v3.0 BBB 历史结果保留。

## 已清理的历史脚本

2026-08-24 已从 `main` 删除旧 ROI 选型、多算法 benchmark、DeepVOG/MediaPipe Iris、PuRe/PuReST、Gate1 以及一次性历史报告生成脚本。具体删除名单与理由见：

```text
docs/工作记录/08-24-01-scripts与models清理及行为版本纠偏工作记录.md
```

这些历史实现仍可通过 Git 历史和 `history/tracking-era-2026-08` 追溯，不再作为当前主线可执行入口。
