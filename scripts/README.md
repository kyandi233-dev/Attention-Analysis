# Scripts

`scripts/` 保留当前仍有明确用途的任务入口，以及少量用户明确要求继续保留、可直接重跑的历史分析入口。正式 NIR 全量分析入口不在这里，而在 `runtime/nir-formal/`。

| 脚本 | 定位 | 用途 |
|---|---|---|
| `extract_eye_dataset.py` | 当前 | NIR 眼框数据集抽帧与 provenance |
| `evaluate_yolo_eye_test.py` | 当前 | YOLO26n frozen test 评估 |
| `sart_formal_analysis.py` | **当前** | FocusWave v3.1.3 最终 BB 行为分析入口 |
| `sart_bbb_v3_0_analysis.py` | **历史、可执行** | 2026-08-16 FocusWave v3.0 BBB 行为分析重跑入口 |
| `face_export_libreface_onnx.py` | **RGB 开发** | 在 LibreFace reference 环境导出当前 AU / expression / gaze 模型为 ONNX，并记录源权重 hash |
| `face_export_pyfeat_onnx.py` | **RGB 开发** | 在 Py-Feat reference 环境导出 Detectorv2 RetinaFace + multitask scientific core 为 ONNX |
| `face_directml_probe.py` | **RGB 开发** | 在独立 DirectML 环境执行 AMD Gate-1：provider / fallback / batch model-core smoke benchmark |

当前 BB 行为分析默认配置为 `configs/behavior_formal.yaml`：

```bash
PYTHONPATH=src python scripts/sart_formal_analysis.py --stage all
```

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

RGB Face DirectML 当前按 `docs/工作记录/08-26-02-RGB-Face-DirectML-Gate01实现.md` 执行。Gate-1 只验证 ONNX 模型是否能在 `DmlExecutionProvider` 上稳定创建/执行、是否出现 CPU fallback，以及 batch 1/8/16/32 的 model-core 基础吞吐；**它不替代现有 300 帧 reference parity / end-to-end benchmark，也不用于提前冻结 Face backend。**
