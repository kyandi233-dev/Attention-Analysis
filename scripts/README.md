# Scripts

`scripts/` 保留当前仍有明确用途的任务入口，以及少量需要继续保留、可直接重跑的历史验证入口。正式 NIR runtime 仍位于 `runtime/nir-formal/`。

当前 branch 为 `rgb-amd`。它是 AMD RGB 的并行工作线，但仍属于统一的 Attention-Analysis 项目；NIR / Behavior / shared docs 可以同步进来。分支关系见 `docs/010-overview/015-并行分支与同步约定.md`。

## 当前入口索引

| 脚本 | 定位 | 用途 |
|---|---|---|
| `sart_formal_analysis.py` | 当前 | FocusWave v3.1.3 最终 BB Behavior |
| `nir_behavior_alignment.py` | 当前 | NIR × Behavior Unix-ms / trial / probe 对齐 |
| `build_stimulus_visual_table.py` | 当前 | SART 视觉协变量与报告图 |
| `rgb_analysis.py` | 当前，共享 RGB | RGB audit / gaps / Motion / Pose / sampling / QC 开发入口 |
| `face_formal_prepare.py` | **当前，AMD formal** | 完整正式时间段 timestamp-driven 15 Hz Face frame manifest；不生成 JPEG |
| `rgb_formal_motion_pose.py` | **当前，AMD formal** | 复用已验证 Motion / Pose engine，写正式 subject 输出 |
| `face_formal_directml.py` | **当前，AMD formal** | original AVI → Py-Feat DirectML full-span Face raw |
| `face_formal_derive.py` | **当前，AMD formal** | continuous tracking → primary face → eyelid / openness derived |
| `run_rgb_formal_subject.ps1` | **当前，AMD formal 总控** | 跨 `attention-rgb` 与 `attention-face-directml` 一条命令跑完单被试 |
| `face_formal_dryrun_sample.py` | 验证/provenance | representative 15 Hz dry-run sampling |
| `face_formal_dryrun_directml_v02.py` | 已接受工程基线 | direct AVI + prefetch + RetinaFace B8 + pooled multitask B16 dry-run |
| `face_derive_tracking_eyelid_v02.py` | 验证/provenance | window-aware dry-run tracking / primary / eyelid |
| `face_qc_visualize_v03.py` | QC | 478 mesh + eyes/iris + primary/secondary + metrics 可视化 |
| `face_real_directml_pyfeat.py` | 历史验证资产 | real-300 DirectML reference runner |
| `face_real_parity_v03.py` | 历史验证资产 | Py-Feat CPU reference ↔ DirectML 最终 parity |
| `face_compare_pyfeat_runs.py` | 历史/辅助 | 两次 Py-Feat raw 输出 parity / A-B |
| `face_directml_probe.py` / `face_directml_diagnose.py` | 历史诊断资产 | ORT DirectML provider / fallback / batch diagnostics |
| `face_export_pyfeat_onnx.py` | backend provenance | Py-Feat RetinaFace + multitask scientific core ONNX export |
| `face_export_libreface_onnx.py` | 历史验证 | LibreFace ONNX export |
| `sart_bbb_v3_0_analysis.py` | 历史、可执行 | FocusWave v3.0 BBB 行为分析重跑 |

## AMD RGB 当前正式流程

当前 Face 科学定义已经冻结：

```text
Py-Feat 2.1.1 Detectorv2 scientific core
+ ONNX Runtime DirectML
+ timestamp-driven 15 Hz
+ RetinaFace B8
+ cross-batch pending multitask B16
+ original AVI direct decode
```

第一档工程优化在 `sub-031` 3600-frame dry-run 上已 Accepted：约 29.15 input frames/s，且与 JPEG95 reference 保持高 parity。历史 dry-run/real300 脚本继续保留，不删除。

现在已经从“代表窗口 dry-run”进入“正式完整时间段 runner”阶段。单被试正式链为：

```text
face_formal_prepare.py
→ rgb_formal_motion_pose.py
→ face_formal_directml.py
→ face_formal_derive.py
```

推荐直接使用总控：

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-amd"

$env:ATTENTION_FACE_MODEL_DIR = "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat"

powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_subject.ps1 `
  -Subject sub-031
```

总控脚本会显式调用：

```text
D:\CondaEnvs\attention-rgb
D:\CondaEnvs\attention-face-directml
```

因此不用人工在每个阶段之间切 Conda 环境。

## 正式输出

正式输出位于：

```text
D:\_AttentionData\Beijing-RGB\sub-XXX\
```

当前正式文件包括：

```text
sub-XXX_face_frames.csv
sub-XXX_face_prepare_manifest.json
sub-XXX_face_raw.parquet
sub-XXX_face_raw_manifest.json
sub-XXX_face_tracks.parquet
sub-XXX_eye_features.parquet
sub-XXX_face_derived_manifest.json
sub-XXX_motion_raw.parquet
sub-XXX_motion_manifest.json
sub-XXX_pose_landmarks.parquet
sub-XXX_pose_manifest.json
sub-XXX_pose_features.parquet
sub-XXX_pose_features_manifest.json
```

文件名重复带 `sub-XXX_`，避免文件复制出 subject 目录后失去被试身份。

## 当前验收顺序

现在优先完成：

```text
sub-031 单被试从头到尾实机验收
→ 修实际出现的 orchestration / environment / output 问题
→ 44 人 batch + resume
```

时间戳 gap 保留为 QC 信息，不再单独阻挡首个全程运行。blink event、`perclos80_proxy`、`body_motion_energy` 仍要继续科学收口，但 expensive Face raw 已完整保留，因此这些派生规则不要求为了修改阈值重新跑 Face inference。

## Behavior / NIR-Behavior

当前 BB Behavior：

```powershell
$env:PYTHONPATH = "src"
python scripts/sart_formal_analysis.py --stage all
```

当前 NIR × Behavior：

```powershell
$env:PYTHONPATH = "src"
python scripts/nir_behavior_alignment.py --subjects sub-031
```

这些模块属于同一个项目，不因为当前 branch 名叫 `rgb-amd` 就被排除。

## 历史 BBB

旧 BBB 使用独立配置和包：

```text
configs/sart_bbb_v3_0.yaml
src/attention_pipeline/behavior_bbb_v3_0/
scripts/sart_bbb_v3_0_analysis.py
```

旧结果与工作记录只用于 provenance，不作为当前 v3.1.3 BB 正式口径。
