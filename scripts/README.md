# Scripts

`scripts/` 保留当前仍有明确用途的任务入口，以及少量用户明确要求继续保留、可直接重跑的历史分析入口。正式 NIR 全量/补跑入口不在这里，而在 `runtime/nir-formal/`。

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
| `face_derive_tracking_eyelid_v02.py` | **当前，共享 RGB** | window-aware primary tracking + EAR / aperture-iris / eyeBlink derived |
| `face_qc_visualize_v03.py` | **当前，共享 RGB** | 全脸 478 mesh + eyes/iris + primary/secondary + metrics QC；单层黑字 |
| `face_compare_pyfeat_runs.py` | 当前辅助 | 两次 Py-Feat raw 输出 parity 比较 |
| `face_benchmark_pyfeat.py` | reference/验证资产 | Py-Feat 官方/native reference 入口；可用于 CUDA runner 实现前后的语义核对 |
| `sart_bbb_v3_0_analysis.py` | **历史、可执行** | 2026-08-16 FocusWave v3.0 BBB 行为分析重跑入口 |

## Behavior

当前 BB 行为分析默认配置为 `configs/behavior_formal.yaml`：

```powershell
$env:PYTHONPATH = "src"
python scripts/sart_formal_analysis.py --stage all
```

正式 Behavior 默认输出位于仓库外：

```text
D:\Project\厚粲杯\11_数据\02_Attention-Analysis_nvidia-cuda_formal_Behavior
```

## NIR × Behavior

`src/attention_pipeline/nir_behavior/`、`scripts/nir_behavior_alignment.py` 与 stimulus visual 代码已从共享科学层同步进入 `nvidia-cuda`。NVIDIA 配置保持同一科学参数，只使用 NVIDIA 工作站的数据/输出根。

```powershell
$env:PYTHONPATH = "src"
python scripts/nir_behavior_alignment.py --subjects sub-031
```

默认 NIR-Behavior 输出位于：

```text
D:\Project\厚粲杯\11_数据\03_Attention-Analysis_nvidia-cuda_NIR-Behavior
```

## NVIDIA RGB 当前入口与边界

RGB 共享科学层已经进入 `nvidia-cuda`，因此 Motion/Pose/Face sampling、tracking/eyelid derived 和 QC 不需要切换到 `rgb-nvidia-cuda` 才能查看或继续开发。

当前 Face 科学定义：

```text
Py-Feat 2.1.1 Detectorv2 scientific core
+ timestamp-driven 15 Hz
+ RetinaFace / crop / multitask canonical semantics
+ complete raw schema
+ primary tracking / EAR / eyelid derived / QC
```

但 **NVIDIA native PyTorch/CUDA formal Face runner 尚未实现/验收**。因此当前不要在 `nvidia-cuda` 中把任何 DirectML runner 当成 NVIDIA 正式入口，也不要因为共享脚本已经存在就直接启动 RGB Face 全量。

NVIDIA 接下来的 Face 顺序：

```text
实现 native Py-Feat / PyTorch CUDA runner
→ 使用同一 sub-031 3600 timestamp 与 AMD DirectML 做 parity
→ sub-033 gap stress
→ blink / perclos80_proxy 冻结
→ full-video formal runner
```

NVIDIA RGB 默认输出位于仓库外：

```text
D:\Project\厚粲杯\11_数据\04_Attention-Analysis_nvidia-cuda_RGB
```

方法与边界见 `docs/040-rgb/README.md`、`docs/040-rgb/046-NVIDIA-CUDA-RGB运行路线.md` 和 `docs/050-decisions/055-RGB-Face-15Hz采样频率冻结.md`。

## 历史 BBB

旧 BBB 为避免与当前 BB 实现互相覆盖，使用独立配置和独立 Python 包：

```text
configs/sart_bbb_v3_0.yaml
src/attention_pipeline/behavior_bbb_v3_0/
scripts/sart_bbb_v3_0_analysis.py
```

旧 BBB 的计划、报告和图仍保存在 `docs/030-behavior/history/BBB-v3.0/`；Git 历史继续保留完整旧仓库快照。当前正式结果解释只认 v3.1.3 的 BB 管线，历史 BBB 入口不得被误作当前分析。