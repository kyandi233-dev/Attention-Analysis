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
| `face_directml_probe.py` | **RGB 开发** | AMD Gate-1：provider / fallback / batch model-core smoke benchmark；v0.2 禁止 Python wrapper 静默整体退回 CPU |
| `face_directml_diagnose.py` | **RGB 开发/诊断** | strict-DML 与实际 ORT kernel profiling；用于解释异常 Gate-1，不是正式 pipeline |
| `face_real_prepare_libreface.py` | **RGB 开发/real-300** | 在 LibreFace reference 环境 fresh 执行 alignment + MediaPipe gaze feature CPU 前处理，不重跑 PyTorch heads |
| `face_real_directml.py` | **RGB 开发/real-300** | 在独立 DirectML 环境运行 LibreFace ONNX heads 或完整 Py-Feat raw-frame DirectML pipeline |
| `face_real_parity.py` | **RGB 开发/real-300** | 将新 DML real-300 输出与既有 CPU reference parquet 做 coverage / bbox / scientific outputs parity |

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

## RGB Face DirectML 当前阶段

LibreFace 与 Py-Feat Gate 0/1 均已通过。Gate-1 只证明 ONNX / `DmlExecutionProvider` / fallback / batch 的 model-core 可运行性，**不能替代真实 300 帧 CPU-reference parity 与 raw-frame end-to-end**。

当前下一步固定为同一 `face-continuous/sub-031` 300 帧的 real-input 验证：

1. LibreFace：fresh CPU-side alignment + MediaPipe gaze feature extraction → DirectML AU / expression / gaze heads；
2. Py-Feat：raw RGB → RetinaFace DML → decode/NMS → isotropic square-pad crop → multitask DML；
3. `face_real_parity.py` 与既有 `libreface_*.parquet` / `pyfeat_raw.parquet` 对照；
4. 速度与 parity 分开解释，再决定是否冻结 Face backend。

Real-300 的完整命令与输出约定见 `docs/040-rgb/045-RGB开发环境与运行指令.md` 及最新 `docs/工作记录/08-26-07-RGB-Face-真实300帧DirectML验证实现.md`。
