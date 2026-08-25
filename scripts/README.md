# Scripts

`scripts/` 保留当前仍有明确用途的任务入口，以及少量用户明确要求继续保留、可直接重跑的历史分析入口。正式 NIR 全量分析入口不在这里，而在 `runtime/nir-formal/`。

| 脚本 | 定位 | 用途 |
|---|---|---|
| `extract_eye_dataset.py` | 当前 | NIR 眼框数据集抽帧与 provenance |
| `evaluate_yolo_eye_test.py` | 当前 | YOLO26n frozen test 评估 |
| `sart_formal_analysis.py` | **当前** | FocusWave v3.1.3 最终 BB 行为分析入口 |
| `nir_behavior_alignment.py` | **当前** | frozen full-class NIR × v3.1.3 BB Behavior 的下游 Unix-ms 对齐、trial/probe 窗口特征与 alignment QC |
| `sart_bbb_v3_0_analysis.py` | **历史、可执行** | 2026-08-16 FocusWave v3.0 BBB 行为分析重跑入口 |

当前 BB 行为分析默认配置为 `configs/behavior_formal.yaml`：

```bash
PYTHONPATH=src python scripts/sart_formal_analysis.py --stage all
```

NIR × Behavior 对齐默认配置为 `configs/nir_behavior_alignment.yaml`。当前配置显式只选择 `sub-031` 作为 prototype safety gate：

```bash
PYTHONPATH=src python scripts/nir_behavior_alignment.py --subjects sub-031
```

对齐结果默认写到仓库外 `D:/_AttentionData/Beijing-NIR/analysis/nir-behavior-v1/`，不改写 `runtime/nir-formal/` 或原始 Behavior CSV。prototype schema 验收前不要把配置中的 `subjects.include` 改成空列表全量发现。

旧 BBB 为避免与当前 BB 实现互相覆盖，使用独立配置和独立 Python 包：

```text
configs/sart_bbb_v3_0.yaml
src/attention_pipeline/behavior_bbb_v3_0/
scripts/sart_bbb_v3_0_analysis.py
```

重跑旧 BBB 时使用：

```bash
PYTHONPATH=src python scripts/sart_bbb_v3_0_analysis.py --stage all
```

旧 BBB 的计划、报告和图仍保存在 `docs/030-behavior/history/BBB-v3.0/`；Git 历史分支 `history/behavior-bbb-v3.0` 继续作为完整旧仓库快照。当前正式结果解释只认 v3.1.3 的 BB 管线，历史 BBB 入口不得被误作当前分析。
