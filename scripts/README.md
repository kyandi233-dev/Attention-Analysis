# Scripts

`scripts/` 保留当前仍有明确用途的任务入口，以及少量用户明确要求继续保留、可直接重跑的历史分析入口。正式 NIR 全量分析入口不在这里，而在 `runtime/nir-formal/`。

## 当前入口索引

| 脚本 | 定位 | 用途 |
|---|---|---|
| `extract_eye_dataset.py` | 当前 | NIR 眼框数据集抽帧与 provenance |
| `evaluate_yolo_eye_test.py` | 当前 | YOLO26n frozen test 评估 |
| `sart_formal_analysis.py` | **当前** | FocusWave v3.1.3 最终 BB 行为分析入口 |
| `nir_behavior_alignment.py` | **当前** | frozen full-class NIR × v3.1.3 BB Behavior 的 Unix-ms 对齐、trial/probe 窗口、coverage/QC/diagnostics |
| `build_stimulus_visual_table.py` | **当前** | 重建正式 SART 画面并生成视觉协变量/报告 PNG |
| `rgb_analysis.py` | **当前，共享 RGB** | RGB audit / timeline / Motion / Pose / Face sampling 与 QC 入口 |
| `face_formal_dryrun_sample.py` | **当前，共享 RGB** | timestamp-driven 15 Hz representative Face dry-run sampling |
| `face_formal_dryrun_directml_v02.py` | **当前，AMD** | 已接受的 direct-AVI + prefetch + RetinaFace B8 → pending multitask B16 DirectML dry-run runner |
| `face_derive_tracking_eyelid_v02.py` | **当前，共享 RGB** | window-aware primary tracking + EAR / aperture-iris / eyeBlink derived |
| `face_qc_visualize_v03.py` | **当前，共享 RGB** | 全脸 478 mesh + eyes/iris + primary/secondary + metrics QC；单层黑字 |
| `face_compare_pyfeat_runs.py` | 当前辅助 | 两次 Py-Feat raw 输出 parity 比较 |
| `face_real_directml_pyfeat.py` / `face_real_parity_v03.py` | AMD 验证资产 | real300 DirectML 与 CPU-reference parity 的历史可复现入口 |
| `face_directml_probe.py` / `face_directml_diagnose.py` | AMD 验证资产 | ONNX Runtime DirectML provider / fallback / batch diagnostics |
| `sart_bbb_v3_0_analysis.py` | **历史、可执行** | 2026-08-16 FocusWave v3.0 BBB 行为分析重跑入口 |

## Behavior

当前 BB 行为分析默认配置为 `configs/behavior_formal.yaml`：

```powershell
$env:PYTHONPATH = "src"
python scripts/sart_formal_analysis.py --stage all
```

正式 Behavior 默认输出已迁至仓库外：

```text
D:\_AttentionData\Beijing-Behavior\formal-v1
```

## NIR × Behavior

NIR × Behavior 对齐默认配置为 `configs/nir_behavior_alignment.yaml`，当前正式下游版本为 `nir-behavior-v1.2` / schema 2。sub-031 prototype 已完成验收；在其余 full-class 尚未完成前，配置仍保留 prototype safety gate。

```powershell
$env:PYTHONPATH = "src"
python scripts/nir_behavior_alignment.py --subjects sub-031
```

schema 2 区分 Block 边界造成的窗口截断与 Block 内部真实 NIR 缺失，并使用 `oar_available_fraction` 表示 OAR 数值存在率；它不是 blink/闭眼质量真值。SART 视觉协变量使用 `configs/stimulus_visual.yaml` 和 `build_stimulus_visual_table.py`。

## AMD RGB 当前入口

RGB 当前正式化工作已经进入 `amd-DirectML`，不需要为了 Face/Pose/Motion 切回 `rgb-dev`。

主 RGB 环境：

```powershell
conda activate "D:\CondaEnvs\attention-rgb"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"
```

Face DirectML 环境：

```powershell
conda activate "D:\CondaEnvs\attention-face-directml"
```

当前 AMD Face 已冻结为 Py-Feat 2.1.1 scientific core + ONNX Runtime DirectML；15 Hz 已冻结。sub-031 第一档优化版本使用 direct AVI + reader/preprocess prefetch + RetinaFace B8 + pending multitask B16，实测约 29.15 fps。

代表性 dry-run 的科学流程为：

```text
face_formal_dryrun_sample.py
→ face_formal_dryrun_directml_v02.py
→ face_derive_tracking_eyelid_v02.py
→ face_qc_visualize_v03.py
```

这里仍是 formal dry-run，不是 44 被试 full-video 正式入口。sub-033 gap stress、blink/`perclos80_proxy` 和 full-video runner 尚待冻结。

AMD RGB 输出统一位于仓库外：

```text
D:\_AttentionData\Beijing-RGB
```

环境/命令细节见 `docs/040-rgb/README.md` 与 `docs/040-rgb/045-RGB开发环境与运行指令.md`。

## 历史 BBB

旧 BBB 为避免与当前 BB 实现互相覆盖，使用独立配置和独立 Python 包：

```text
configs/sart_bbb_v3_0.yaml
src/attention_pipeline/behavior_bbb_v3_0/
scripts/sart_bbb_v3_0_analysis.py
```

旧 BBB 的计划、报告和图仍保存在 `docs/030-behavior/history/BBB-v3.0/`；Git 历史继续保留完整旧仓库快照。当前正式结果解释只认 v3.1.3 的 BB 管线，历史 BBB 入口不得被误作当前分析。