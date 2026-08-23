# Scripts

`scripts/` 只保留当前仍有明确用途的任务入口。正式 NIR 全量分析入口不在这里，而在 `runtime/nir-formal/`。

| 脚本 | 当前用途 |
|---|---|
| `extract_eye_dataset.py` | NIR 眼框数据集抽帧与 provenance |
| `evaluate_yolo_eye_test.py` | YOLO26n frozen test 评估 |
| `sart_formal_analysis.py` | **FocusWave v3.1.3 最终 BB 行为分析入口** |

行为分析默认配置：`configs/behavior_formal.yaml`。

```bash
PYTHONPATH=src python scripts/sart_formal_analysis.py --stage all
```

旧 v3.0 BBB 的计划、报告、图与冻结配置位于 `docs/030-behavior/history/BBB-v3.0/`；完整旧可执行实现冻结在 `history/behavior-bbb-v3.0` 分支。历史 BBB 不在当前 scripts 中长期维护第二套入口。
