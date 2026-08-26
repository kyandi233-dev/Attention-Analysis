# Scripts

`scripts/` 保留当前仍有明确用途的任务入口，以及少量需要继续保留、可直接重跑的历史验证入口。正式 NIR runtime 仍位于 `runtime/nir-formal/`。

`amd-DirectML` 是 AMD 综合线；当前 AMD RGB 的 active development branch 为 `rgb-amd`。两条分支属于同一个 Attention-Analysis 项目。分支关系见 `docs/010-overview/015-并行分支与同步约定.md`。

## 当前入口索引

| 脚本 | 定位 | 用途 |
|---|---|---|
| `extract_eye_dataset.py` | 当前 | NIR 眼框数据集抽帧与 provenance |
| `evaluate_yolo_eye_test.py` | 当前 | YOLO26n frozen test 评估 |
| `sart_formal_analysis.py` | 当前 | FocusWave v3.1.3 最终 BB Behavior |
| `nir_behavior_alignment.py` | 当前 | NIR × Behavior Unix-ms / trial / probe 对齐 |
| `build_stimulus_visual_table.py` | 当前 | SART 视觉协变量与报告图 |
| `rgb_analysis.py` | 当前，共享 RGB | RGB audit / gaps / Motion / Pose / sampling / QC 开发入口 |
| `face_formal_prepare.py` | **当前，AMD formal** | 完整正式时间段 timestamp-driven 15 Hz Face frame manifest |
| `rgb_formal_motion_pose.py` | **当前，AMD formal** | 复用已验证 Motion / Pose engine，写正式 subject 输出 |
| `face_formal_directml.py` | **当前，AMD formal** | original AVI → Py-Feat DirectML full-span Face raw |
| `face_formal_derive.py` | **当前，AMD formal** | continuous tracking → primary face → eyelid / openness derived |
| `run_rgb_formal_subject.ps1` | **当前，AMD formal 总控** | 跨两个 Conda 环境一条命令跑完整单被试 |
| `face_formal_dryrun_sample.py` | 验证/provenance | representative 15 Hz dry-run sampling |
| `face_formal_dryrun_directml_v02.py` | 已接受工程基线 | direct AVI + prefetch + RetinaFace B8 + pooled multitask B16 |
| `face_derive_tracking_eyelid_v02.py` | 验证/provenance | dry-run window-aware tracking / eyelid derived |
| `face_qc_visualize_v03.py` | QC | 478 mesh + eyes/iris + primary/secondary + metrics 可视化 |
| `face_real_directml_pyfeat.py` / `face_real_parity_v03.py` | 历史验证资产 | real-300 DirectML / CPU parity |
| `face_directml_probe.py` / `face_directml_diagnose.py` | 历史诊断资产 | ORT DirectML provider / fallback / batch diagnostics |
| `face_export_pyfeat_onnx.py` | backend provenance | Py-Feat RetinaFace + multitask scientific core ONNX export |
| `face_export_libreface_onnx.py` | 历史验证 | LibreFace ONNX export |
| `sart_bbb_v3_0_analysis.py` | 历史、可执行 | FocusWave v3.0 BBB 行为分析重跑 |

## Behavior

当前正式 BB：

```powershell
$env:PYTHONPATH = "src"
python scripts/sart_formal_analysis.py --stage all
```

正式 Behavior 输出位于：

```text
D:\_AttentionData\Beijing-Behavior\formal-v1
```

## NIR × Behavior

当前入口：

```powershell
$env:PYTHONPATH = "src"
python scripts/nir_behavior_alignment.py --subjects sub-031
```

NIR、Behavior、RGB 都属于同一个项目；当前拆 branch 只是为了并行工作。

## AMD RGB 当前正式流程

Face 科学定义：

```text
Py-Feat 2.1.1 Detectorv2 scientific core
+ ONNX Runtime DirectML
+ timestamp-driven 15 Hz
+ RetinaFace B8
+ pooled multitask B16
+ original AVI direct decode
```

第一档工程优化已在 `sub-031` representative dry-run 上 Accepted。现在已经进入正式完整时间段 runner 阶段。

单被试正式链：

```text
face_formal_prepare.py
→ rgb_formal_motion_pose.py
→ face_formal_directml.py
→ face_formal_derive.py
```

当前 active development branch 推荐在：

```text
D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-amd
branch: rgb-amd
```

推荐总控：

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-amd"
$env:ATTENTION_FACE_MODEL_DIR = "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat"

powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_subject.ps1 `
  -Subject sub-031
```

正式输出：

```text
D:\_AttentionData\Beijing-RGB\sub-XXX
```

当前验收顺序：

```text
sub-031 单被试从头到尾实机验收
→ 44 人 batch + resume
→ body_motion_energy
→ blink / perclos80_proxy 最终科学规则
```

时间戳 gap 保留为 QC 信息，不再作为单独阻挡首个全程运行的前置 Gate。

## 历史 BBB

旧 BBB 使用独立配置和包：

```text
configs/sart_bbb_v3_0.yaml
src/attention_pipeline/behavior_bbb_v3_0/
scripts/sart_bbb_v3_0_analysis.py
```

历史结果与工作记录只用于 provenance，不作为当前 v3.1.3 BB 正式口径。
