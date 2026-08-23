# Scripts

`scripts/` 只保留当前仍有明确用途的任务入口。正式 NIR 全量分析入口不在这里，而在 `runtime/nir-formal/`。

| 脚本 | 当前用途 |
|---|---|
| `extract_eye_dataset.py` | NIR 眼框数据集抽帧与 provenance |
| `evaluate_yolo_eye_test.py` | YOLO26n frozen test 评估 |
| `sart_formal_analysis.py` | **FocusWave v3.1.3 最终 BB 行为分析入口** |

行为分析默认配置：

```text
configs/behavior_formal.yaml
```

示例：

```bash
PYTHONPATH=src python scripts/sart_formal_analysis.py --stage all
```

旧 v3.0 BBB runner 已由当前脚本重建替代；旧可执行状态冻结在 `history/behavior-bbb-v3.0`。
