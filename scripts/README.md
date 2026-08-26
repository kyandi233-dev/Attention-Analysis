# Scripts

`scripts/` 保留当前仍有明确用途的任务入口，以及少量用户明确要求继续保留、可直接重跑的历史分析入口。正式 NIR 推理入口仍在 `runtime/nir-formal/`；正式 NIR **下游分析**入口位于本目录。

## 当前入口索引

| 脚本 | 定位 | 用途 |
|---|---|---|
| `extract_eye_dataset.py` | 当前 | NIR 眼框数据集抽帧与 provenance |
| `evaluate_yolo_eye_test.py` | 当前 | YOLO26n frozen test 评估 |
| `sart_formal_analysis.py` | **当前** | FocusWave v3.1.3 最终 BB 行为分析入口 |
| `nir_materialize_analysis_ready.py` | **当前 NIR 下游** | production → `10_analysis_ready` |
| `nir_build_analysis_tables.py` | **当前 NIR 下游** | `10_analysis_ready` + Behavior → `11_analysis_tables` |
| `nir_formal_pipeline.py` | **当前 NIR 下游统一入口** | 分阶段运行 `materialize / tables / all`；不会调用 YOLO/RITnet |
| `nir_pipeline_validation.py` | **当前 validation-only** | 读取已完成 `11_analysis_tables` + 只读 `10_analysis_ready`，验证完整 Behavior / PIR / Probe / QC / robustness / visual-covariate / model / code-figure 分析接口，写入 `12_pipeline_validation` |
| `nir_behavior_alignment.py` | **历史 prototype、可执行** | 旧 production-based NIR × Behavior schema-v2 对齐；不再是当前正式主分析入口 |
| `build_stimulus_visual_table.py` | **当前** | 重建正式 SART 画面并生成视觉协变量/报告 PNG |
| `rgb_analysis.py` | **当前，共享 RGB** | RGB audit / timeline / Motion / Pose / Face sampling 与 QC 入口 |
| `face_formal_dryrun_sample.py` | **当前，共享 RGB** | timestamp-driven 15 Hz representative Face dry-run sampling |
| `face_formal_dryrun_directml_v02.py` | **当前，AMD** | direct-AVI + prefetch + RetinaFace B8 + multitask DirectML dry-run runner |
| `face_derive_tracking_eyelid_v02.py` | **当前，共享 RGB** | window-aware primary tracking + EAR / aperture-iris / eyeBlink derived |
| `face_qc_visualize_v03.py` | **当前，共享 RGB** | 全脸 mesh + eyes/iris + primary/secondary + metrics QC |
| `face_compare_pyfeat_runs.py` | 当前辅助 | 两次 Py-Feat raw 输出 parity 比较 |
| `face_real_directml_pyfeat.py` / `face_real_parity_v03.py` | AMD 验证资产 | real300 DirectML 与 CPU-reference parity 历史可复现入口 |
| `face_directml_probe.py` / `face_directml_diagnose.py` | AMD 验证资产 | ONNX Runtime DirectML provider / fallback / batch diagnostics |
| `sart_bbb_v3_0_analysis.py` | **历史、可执行** | 2026-08-16 FocusWave v3.0 BBB 行为分析重跑入口 |

## Behavior

当前 BB 行为分析默认配置为 `configs/behavior_formal.yaml`：

```powershell
$env:PYTHONPATH = "src"
python scripts/sart_formal_analysis.py --stage all
```

正式 Behavior 默认输出：

```text
D:\_AttentionData\Beijing-Behavior\formal-v1
```

## NIR 下游层级

当前完整层级定义为：

```text
NIR production / full-class
        ↓
10_analysis_ready
        ↓
11_analysis_tables
        ↓
12_pipeline_validation      # 当前允许
        ↓
20_formal_statistics        # 当前禁止，待正确 NIR 后进入
```

`nir_formal_pipeline.py` 只管理 downstream derived data，不运行 YOLO / RITnet。

当前已经确认现阶段 NIR/PIR 数值错误，因此 **不要继续剩余被试的 `11_analysis_tables`，也不要进入 `20_formal_statistics`**。现有 completed subjects 仅用于 validation-only 的代码、模型与绘图验收。

## `10_analysis_ready` / `11_analysis_tables`

代表性正式构表入口：

```powershell
python scripts/nir_formal_pipeline.py `
  --stage tables `
  --subjects sub-031
```

但在当前 PIR 数值修正前，不应为了 validation 再运行更多 subject。

具体契约：

```text
docs/020-nir/212-2026-08-27-NIR数据清洗逻辑与正式分析纳入规则.md
docs/020-nir/213-2026-08-27-NIR-analysis-ready数据契约与物化规范.md
docs/020-nir/214-2026-08-27-NIR正式下游分析表数据契约.md
docs/020-nir/215-2026-08-27-NIR正式下游分析管线运行手册.md
```

## 当前完整 `12_pipeline_validation`

当前总方法导航：

```text
docs/020-nir/218-2026-08-27-NIR完整分析管线与统计逻辑总说明.md
```

错误 PIR 条件下的解释边界：

```text
docs/020-nir/217-2026-08-27-NIR错误值条件下下游分析管线验证方案.md
```

运行前同步：

```powershell
git status --short --branch
git fetch origin --prune
git switch analysis/multimodal-integration
git pull --ff-only

conda activate "D:\CondaEnvs\nir-amd"
python -m pip install -e .
```

完整 validation 测试：

```powershell
python -m pytest `
  tests/test_nir_pipeline_validation.py `
  tests/test_nir_probe_validation.py `
  tests/test_nir_pipeline_validation_extended.py `
  tests/test_nir_formal_analysis.py -q
```

然后只运行已有 completed subjects：

```powershell
python scripts/nir_pipeline_validation.py
```

输出：

```text
D:\_AttentionData\Beijing-NIR\analysis\nir-behavior-v2\cohort-44-exploratory\12_pipeline_validation\
├── tables\
├── figures\
├── extension_readiness.json
└── validation_summary.json
```

### 当前 validation 分析逻辑

代码不再只验证 PIR median，而是覆盖五条互相衔接的分析线。

**持续状态层**验证 Block1→Block2、1 s time-on-task、30 s 展示轨迹，以及后续 mixed model / GAMM 所需时间结构。

**Trial / Behavior 层**保留 program scoring，同时分别处理 Go RT、RT-CV、ex-Gaussian、d′/c/β、commission、program omission、clean/ambiguous omission、anticipatory/multiple-keypress，并增加 No-Go 之前若干正确 Go trial 的 RT/PIR precursor trajectory。

**NIR 动态层**并行保留：

```text
median / mean / P10 / P90
MAD / IQR / SD
slope_per_sec
diff_mad
diff_rate_mad_per_sec
```

用于区分状态水平、波动、趋势和短时不稳定性，而不是只看平均 PIR。

**Probe 层**同时分析 `probe_response` raw option、`probe_vigilance`、`probe_rt`、`probe_vigilance_rt`，并连接 pre-10/20/30/60 s 的 PIR 与客观 SART 行为；raw `probe_response` 在正式任务源码语义核验前不得擅自贴文字心理状态标签。

**QC / robustness / confound 层**验证：

```text
binocular_primary / left_primary / right_primary
binocular_strict / left_strict / right_strict
source-mode composition
available duration / boundary truncation / internal coverage / max gap / PIR validity
current + previous stimulus luminance / contrast / visible-area covariates
raw PIR between-person baseline characteristics
questionnaire optional subject-level interface
OAR / RGB extension readiness
```

### 代码生成图

图全部由 Python/Matplotlib 生成，不使用图片生成模型。除原有 Block/time-on-task/trial/Probe/model 图外，当前还包括：

```text
fig07_dynamic_pir_feature_matrix
fig08_trial_multiscale_pir_trajectory
fig09a_nogo_precursor_rt
fig09b_nogo_precursor_pir
fig10a_probe_response_rt
fig10b_probe_vigilance_rt_by_response
fig10c_probe_prebehavior_rt_cv_multiscale
fig10d_probe_prebehavior_ambiguous_omission_multiscale
fig11_advanced_behavior_block_profile
fig12_track_robustness_pre_5s
fig13_source_mode_qc_pre_5s
fig14a_coverage_multidimensional_trial_pre_5s
fig14b_coverage_multidimensional_probe_pre_20s
fig15_visual_luminance_pir
fig16_raw_between_person_pir
```

所有含当前错误 PIR 的 PNG/PDF 必须带固定标记：

```text
PIPELINE VALIDATION ONLY — CURRENT NIR VALUES KNOWN INVALID
```

`extension_readiness.json` 明确记录当前哪些扩展能运行、哪些尚未具备正式数据契约。例如：问卷未配置时应写 `unavailable`；OAR 尚未进入 `10_analysis_ready` 时应写 `blocked_by_analysis_ready_schema`，而不是绕回 production 读取。

当前可以检查代码、join、模型是否拟合、专业图形结构、coverage/source-mode/visual-covariate 接口；禁止解释任何 PIR 方向、p 值、窗口优劣或据此调整 QC 阈值。

## 正确 NIR 修复后的顺序

```text
修正 NIR 数值来源
→ 重建 10_analysis_ready
→ 重建 11_analysis_tables
→ 重跑 12_pipeline_validation
→ 冻结 primary inferential windows / coverage / Probe 语义 / visual covariates / sensitivity set
→ 进入 20_formal_statistics
```

OAR 必须先扩展 `10_analysis_ready` schema；Questionnaire 在 subject-level 连接；RGB 未来通过 subject / Block / absolute time / trial / probe keys 进入 multimodal table。正式统计不得为了方便绕过这些数据层直接读 production。

## AMD RGB 当前入口

RGB 当前正式化工作位于 AMD DirectML 路线。主 RGB 环境：

```powershell
conda activate "D:\CondaEnvs\attention-rgb"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"
```

Face DirectML 环境：

```powershell
conda activate "D:\CondaEnvs\attention-face-directml"
```

代表性 dry-run 科学流程：

```text
face_formal_dryrun_sample.py
→ face_formal_dryrun_directml_v02.py
→ face_derive_tracking_eyelid_v02.py
→ face_qc_visualize_v03.py
```

AMD RGB 输出统一位于：

```text
D:\_AttentionData\Beijing-RGB
```

环境/命令细节见 `docs/040-rgb/README.md` 与 `docs/040-rgb/045-RGB开发环境与运行指令.md`。

## 历史 BBB

旧 BBB 继续使用独立配置、包和入口：

```text
configs/sart_bbb_v3_0.yaml
src/attention_pipeline/behavior_bbb_v3_0/
scripts/sart_bbb_v3_0_analysis.py
```

旧结果和文档继续保留在 `docs/030-behavior/history/BBB-v3.0/`；历史 provenance 不删除、不改写成当前正式版本。