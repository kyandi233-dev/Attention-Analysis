# Scripts

`scripts/` 保留当前仍有明确用途的任务入口，以及少量用户明确要求继续保留、可直接重跑的历史分析入口。正式 NIR 全量分析入口不在这里，而在 `runtime/nir-formal/`。

| 脚本 | 定位 | 用途 |
|---|---|---|
| `extract_eye_dataset.py` | 当前 | NIR 眼框数据集抽帧与 provenance |
| `evaluate_yolo_eye_test.py` | 当前 | YOLO26n frozen test 评估 |
| `sart_formal_analysis.py` | **当前** | FocusWave v3.1.3 最终 BB 行为分析入口 |
| `sart_bbb_v3_0_analysis.py` | **历史、可执行** | 2026-08-16 FocusWave v3.0 BBB 行为分析重跑入口 |
| `face_export_libreface_onnx.py` | **RGB 历史验证** | LibreFace reference 环境导出 AU / expression / gaze ONNX |
| `face_export_pyfeat_onnx.py` | **RGB backend provenance** | Py-Feat Detectorv2 RetinaFace + multitask scientific core ONNX 导出 |
| `face_directml_probe.py` | **RGB 历史验证** | AMD Gate-1 provider / fallback / batch smoke benchmark |
| `face_directml_diagnose.py` | **RGB 历史诊断** | strict-DML 与 ORT kernel profiling；解释 Gate-1 异常 |
| `face_real_prepare_libreface.py` | **RGB real-300 历史验证** | LibreFace fresh CPU prep |
| `face_real_directml_libreface.py` | **RGB real-300 历史验证** | LibreFace DirectML learned heads |
| `face_real_directml_pyfeat.py` | **RGB validated backend runner** | raw RGB/JPEG → RetinaFace DML → crop → multitask DML；保留完整 scientific outputs |
| `face_real_parity_v02.py` | **RGB real-300 历史 parity** | 首次 retention-aware parity；保留 provenance |
| `face_real_parity_v03.py` | **RGB real-300 最终 parity** | 修复 LibreFace schema alignment 与 SciPy 依赖后的最终 parity |
| `face_real_directml.py` | **RGB real-300 早期原型** | 首版合并 runner；保留历史，不作为当前推荐入口 |
| `face_real_parity.py` | **RGB real-300 早期原型** | 首版 parity；保留历史，不作为当前推荐入口 |
| `face_formal_dryrun_sample.py` | **RGB 正式化 dry-run** | timestamp-driven 15 Hz 连续代表窗口抽帧；测试 primary-face / eyelid / schema |
| `face_formal_dryrun_directml.py` | **RGB 正式化 dry-run** | 使用已冻结 Py-Feat DirectML backend 跑 dry-run frames |
| `face_derive_tracking_eyelid.py` | **RGB 正式化 dry-run/derived** | 从已保存 Py-Feat raw 分配 track / primary face，并派生 EAR、aperture/iris、normalized openness；不重跑模型 |

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

## RGB Face 当前正式化阶段

Face backend 已冻结为：

```text
Py-Feat 2.1.1 Detectorv2 scientific core
+ RetinaFace R34 ONNX
+ multitask scientific-core ONNX
+ ONNX Runtime DmlExecutionProvider
```

Real-300 已证明：300/300 coverage、bbox / AU / emotion / V-A / pose / gaze / mesh / blendshape parity 均通过；AMD raw-frame end-to-end≈17.29 fps。LibreFace 2.0 保留 reference/fallback，不进入 44-subject 主全量。

Face 正式采样频率已冻结为 **timestamp-driven 15 Hz**。Pose 保留 10 Hz；Motion 保留原始全帧。

### Representative dry-run

第一批推荐：

```text
sub-031  reference subject
sub-033  capture/timestamp-gap stress subject
```

每人抽取约 4 min 连续窗口：baseline start/end、Block1 middle、interblock middle、Block2 middle；15 Hz 约 3600 sampled frames。

步骤：

```powershell
conda activate "D:\CondaEnvs\attention-rgb"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-dev"

python scripts/face_formal_dryrun_sample.py --subject sub-031
```

然后切 DirectML：

```powershell
conda activate "D:\CondaEnvs\attention-face-directml"

python scripts/face_formal_dryrun_directml.py `
  --sample-dir "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031" `
  --model-dir "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat" `
  --output-dir "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\directml"
```

最后只读取已保存 raw 做 tracking / eyelid derived：

```powershell
python scripts/face_derive_tracking_eyelid.py `
  --raw "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\directml\pyfeat_dml_raw.parquet" `
  --frame-manifest "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\sub-031_face-dryrun_frames.csv" `
  --output-dir "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031\derived"
```

`sub-033` 将上述 `sub-031` 全部替换为 `sub-033` 即可。

### 信息保留原则

正式 raw 继续遵守 `docs/040-rgb/044-RGB输出Schema与信息保留原则.md`：

- 所有检测到的 faces 先保留，不在 raw 里只保存 primary；
- Py-Feat 20 AU、7 emotion、V/A、gaze、pose、478 mesh、68 compatibility landmarks、52 blendshapes 全部保留；
- `eyeBlinkLeft/Right` 属于 native raw；
- EAR、aperture/iris、normalized eye openness 属于 derived，可从 mesh 重建；
- primary-face 先 temporal tracking，再按 Block1+Block2 长期 occupancy 选主轨迹；
- blink event threshold / `perclos80_proxy` 最终 QC 在 representative dry-run 后才冻结。

决策入口：

```text
docs/050-decisions/054-RGB-Face-Backend冻结.md
docs/050-decisions/055-RGB-Face-15Hz采样频率冻结.md
docs/050-decisions/056-RGB-Face-Primary与眼睑派生规则.md
```
