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
| `face_real_directml_libreface.py` | **RGB 开发/real-300 当前入口** | LibreFace fresh prep → DirectML AU/expression/gaze heads；同时保留 native AU probabilities 与 derived outputs |
| `face_real_directml_pyfeat.py` | **RGB 开发/real-300 当前入口** | raw RGB → RetinaFace DirectML → decode/NMS → square-reflect crop → multitask DirectML；保留 raw + canonical scientific outputs |
| `face_real_parity_v02.py` | **RGB 开发/real-300 当前入口** | 与既有 CPU reference 做 coverage、bbox、68/478 landmarks、AU、emotion、VA、pose、gaze、blendshape/headpose parity |
| `face_real_directml.py` | **RGB 开发/real-300 早期原型** | 首版合并 runner；保留历史，不作为当前推荐入口 |
| `face_real_parity.py` | **RGB 开发/real-300 早期原型** | 首版 parity；保留历史，不作为当前推荐入口 |

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

当前固定使用同一 `face-continuous/sub-031` 300 帧进行 real-input 验证：

1. LibreFace：`face_real_prepare_libreface.py` fresh 执行 CPU-side alignment + MediaPipe gaze features；随后 `face_real_directml_libreface.py` 运行 DirectML heads；
2. Py-Feat：`face_real_directml_pyfeat.py` 从 raw RGB 开始完成 RetinaFace DML → decode/NMS → isotropic square-pad/reflection crop → multitask DML；
3. `face_real_parity_v02.py` 分别与既有 `libreface_*.parquet` / `pyfeat_raw.parquet` 做逐字段 parity；
4. 速度与 parity 分开解释，再结合 coverage、信息完整性和工程复杂度冻结 backend。

### 信息保留原则

real-300 不是为了只生成当前马上要画图的几个变量，而是验证未来正式 pipeline 的完整保存能力。当前入口因此保留：

- LibreFace：fresh alignment/headpose/landmarks、1404 gaze features、AU intensity/detection 原始 probabilities、derived AU、8-class expression scores/label、gaze；
- Py-Feat：RetinaFace decoded bbox/score/5-point landmarks、20 AU probabilities、7 emotion probabilities、V/A、raw + canonical gaze、raw + canonical pose、478 normalized mesh、478 original-frame mesh、dlib-68 compatibility view、52 blendshapes、frame provenance；
- multi-face 不在这一层静默过滤；primary-face 规则继续等 backend 冻结后决定；
- Py-Feat identity branch 继续按 Gate-0 scientific-core 决策排除，不为本项目单被试视频测量增加额外工程成本。

Real-300 的完整命令、输出目录和判定边界见最新工作记录：

```text
docs/工作记录/08-26-07-RGB-Face-真实300帧DirectML验证实现.md
```
