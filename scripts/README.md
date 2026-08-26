# Scripts

`scripts/` 保留当前仍有明确用途的任务入口，以及少量用户明确要求继续保留、可直接重跑的历史分析入口。正式 NIR 全量分析入口不在这里，而在 `runtime/nir-formal/`。

| 脚本 | 定位 | 用途 |
|---|---|---|
| `extract_eye_dataset.py` | 当前 | NIR 眼框数据集抽帧与 provenance |
| `evaluate_yolo_eye_test.py` | 当前 | YOLO26n frozen test 评估 |
| `sart_formal_analysis.py` | **当前** | FocusWave v3.1.3 最终 BB 行为分析入口 |
| `nir_behavior_alignment.py` | **当前** | frozen full-class NIR × v3.1.3 BB Behavior 的下游 Unix-ms 对齐、trial/probe 窗口特征、schema-v2 coverage 与 alignment QC |
| `build_stimulus_visual_table.py` | **当前** | 按 FocusWave formaltest 实际绘制规则重建 9×3=27 个正式 SART 画面，计算数字 relative luminance / contrast，并输出报告用完整 PNG/总览图 |
| `sart_bbb_v3_0_analysis.py` | **历史、可执行** | 2026-08-16 FocusWave v3.0 BBB 行为分析重跑入口 |

当前 BB 行为分析默认配置为 `configs/behavior_formal.yaml`：

```bash
PYTHONPATH=src python scripts/sart_formal_analysis.py --stage all
```

NIR × Behavior 对齐默认配置为 `configs/nir_behavior_alignment.yaml`，当前正式下游版本为 **`nir-behavior-v1.2` / schema 2**。sub-031 prototype 已完成验收，schema 2 已冻结；在其余 full-class 尚未完成前，配置仍显式保留 `sub-031` safety gate：

```bash
PYTHONPATH=src python scripts/nir_behavior_alignment.py --subjects sub-031
```

schema 2 额外区分 Block 边界造成的窗口截断与 Block 内部真实 NIR 缺失，并使用 `oar_available_fraction` 表示 OAR 数值存在率；它不是 blink/闭眼质量真值。

对齐结果默认写到仓库外 `D:/_AttentionData/Beijing-NIR/analysis/nir-behavior-v1/`，不改写 `runtime/nir-formal/` 或原始 Behavior CSV。

SART 刺激视觉协变量使用 `configs/stimulus_visual.yaml`。如果本地存在同级 FocusWave checkout 可自动发现素材；否则显式指定 `01-MainProgram/素材`：

```bash
PYTHONPATH=src python scripts/build_stimulus_visual_table.py \
  --materials-dir "D:/path/to/FocusWave/01-MainProgram/素材"
```

`stimulus-visual-v1.2` 默认一次生成：

```text
D:/_AttentionData/Beijing-NIR/analysis/nir-behavior-v1/
├── stimulus_visual_properties.csv
├── stimulus_visual_manifest.json
├── stimulus_visual_overview_full.png
├── stimulus_visual_overview_central.png
└── stimulus_visual_rendered/
    ├── references/
    │   ├── 00_background_screen.png
    │   └── 00_mask_screen.png
    └── conditions/
        ├── 01_mango_size080.png
        ├── ...
        └── 09_no_go_size120.png
```

`conditions/` 中 27 张 PNG 均为北京正式 Surface Pro 2880×1920 的完整任务窗口重建；`overview_full` 是 9×3 完整屏幕缩略总览，`overview_central` 是固定中央 ROI 总览，后者更适合报告正文排版。数字表仍只提供 linear-sRGB relative luminance / RMS contrast，不等同于光度计校准的 cd/m²。完整方法与字段见 `docs/020-nir/025-2026-08-26-SART刺激视觉协变量重建.md`，图片输出见 `026-2026-08-26-SART刺激报告图片输出.md`。

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
