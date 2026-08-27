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
| `nir_pipeline_validation.py` | **当前 validation-only** | core diagnostic validation + publication analysis + Figure 1–10；只写 `12_pipeline_validation` |
| `nir_behavior_alignment.py` | **历史 prototype、可执行** | 旧 production-based NIR × Behavior schema-v2 对齐；不再是当前正式主分析入口 |
| `build_stimulus_visual_table.py` | **当前** | 重建正式 SART 画面并生成视觉协变量/报告 PNG |
| `rgb_analysis.py` | **当前，共享 RGB** | RGB audit / timeline / Motion / Pose / Face sampling 与 QC 入口 |
| `multimodal_pupil_audit.py` | **当前，Issue #22 validation-only** | 只读 NIR–RGB 时间配对、pupil-only/Face+Pose nuisance 审计与身份 provenance 摘要；可通过 `--repeat-registry` 接入外部非 PII 重复被试 registry |
| `multimodal_pupil_correction_pilot.py` | **当前，Issue #22 validation-only** | 比较 M0–M3 的 NIR YOLO eye-bbox / RGB 几何校正候选；只使用无标签测量学指标，默认 baseline-only fit，不读取 Behavior/Probe/ML outcome |
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

科学总导航：

```text
docs/020-nir/218-2026-08-27-NIR完整分析管线与统计逻辑总说明.md
```

论文级 Figure / 补充分析实现：

```text
docs/020-nir/219-2026-08-27-NIR论文级Figure体系与补充分析实现.md
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
  tests/test_nir_publication_suite.py `
  tests/test_nir_formal_analysis.py -q
```

然后只运行已有 completed subjects：

```powershell
python scripts/nir_pipeline_validation.py
```

默认会顺序执行：

```text
core diagnostic validation
        ↓
publication validation
        ↓
Figure 1–10
```

也可以单独运行：

```powershell
python scripts/nir_pipeline_validation.py --core-only
python scripts/nir_pipeline_validation.py --publication-only
```

### 当前完整分析覆盖

**持续状态层**：whole-experiment global PIR、Block1→Block2、1 s time-on-task、Block transition/recovery、个体 slope 与前后半 Block 对照。

**Trial / Behavior 层**：Go RT、RT-CV、ex-Gaussian、d′/c/β、commission、program omission、clean/ambiguous omission、anticipatory/multiple-keypress；同时保留 No-Go 离散 trial-lag precursor。

**真实 continuous event 层**：只读 `10_analysis_ready`，用 `11_analysis_tables` 的 trial/probe onset 对齐，默认按 1 s bin 构造 `-60s→event` 的 No-Go、omission、Probe 连续 PIR trajectory。这里不会绕回 production。

**NIR 动态层**：

```text
median / mean / P10 / P90
MAD / IQR / SD
slope_per_sec
diff_mad
diff_rate_mad_per_sec
```

并增加 feature redundancy、within-person correlation、between-person raw-PIR correlation 与 prespecified window-effect stability。

**Probe 层**：`probe_response` raw option、`probe_vigilance`、`probe_rt`、`probe_vigilance_rt`、pre-10/20/30/60 s Behavior + PIR、continuous pre-Probe trajectory、Probe sequential transition。

**QC / robustness / confound 层**：六条 primary/strict/eye track、source-mode、available duration、boundary truncation、internal coverage、max gap、PIR validity、current/previous stimulus luminance/contrast/visible area、raw PIR between-person baseline、individual heterogeneity。

### 论文级 Figure 1–10

论文候选图不再使用零散 `fig03a/fig04b/...` 作为主输出，而固定为：

```text
Figure01_global_PIR_landscape
Figure02_Block_time_on_task
Figure03_trial_behavior_states
Figure04_error_precursor_trajectories
Figure05_probe_states_trajectories
Figure06_visual_PLR_controls
Figure07_individual_differences
Figure08_feature_structure_multiscale
Figure09_data_quality_coverage
Figure10_robustness_models
```

全部由 Python/Matplotlib 代码生成，不使用图片生成模型。论文级 Figure 固定 17 cm 整版宽度、A/B/C/D panel、统一 Arial（fallback DejaVu Sans）、字号、线宽、图例、坐标轴和画布边距；PDF/SVG 为矢量，PNG/TIFF 为 600 dpi。旧 diagnostic figures 继续保留工程 provenance，但不再当 manuscript Figure 主候选。

输出：

```text
D:\_AttentionData\Beijing-NIR\analysis\nir-behavior-v2\cohort-44-exploratory\12_pipeline_validation\
├── tables\
│   └── publication_analysis\
├── figures\
│   └── publication\
├── extension_readiness.json
├── validation_summary.json
└── publication_suite_summary.json
```

所有含当前错误 PIR 的图必须带：

```text
PIPELINE VALIDATION ONLY — CURRENT NIR VALUES KNOWN INVALID
```

当前可以检查代码、join、真实 event alignment、Figure 版式、coverage/source-mode/visual-covariate 接口；禁止解释任何 PIR 方向、p 值、窗口优劣或据此调整 QC 阈值。

## 正确 NIR 修复后的顺序

```text
修正 NIR 数值来源
→ 重建 10_analysis_ready
→ 重建 11_analysis_tables
→ 重跑 12_pipeline_validation
→ 检查 Figure 1–10 / coverage / Probe 语义 / visual controls / sensitivity set
→ 冻结正式模型与报告顺序
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
