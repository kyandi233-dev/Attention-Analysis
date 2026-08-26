# NIR

## 2026-08-27 当前 NIR 状态

当前 NIR 已经有完整的下游工程骨架，但**现阶段 NIR/PIR 数值已知错误，不能用于科学解释**。因此当前只允许把少量已完成 `11_analysis_tables` 的被试当作占位输入，继续把分析代码、模型接口、QC 与专业绘图调通；禁止把这些 PIR 的方向、差异、相关、p 值解释为结果，也不继续为了错误数值批量构表剩余被试。

当前层级：

```text
NIR extraction / full-class production
        ↓
10_analysis_ready
        ↓
11_analysis_tables
        ↓
12_pipeline_validation      # 当前工作层：结构/模型/图形验证
        ↓
20_formal_statistics        # 当前禁止，待正确 NIR 后进入
```

当前 AMD 44 人仍是 exploratory/development cohort；未来约 116 人 final cohort 应复用冻结后的同一套代码，在独立 snapshot 中执行。

## 当前上位导航

如果要理解“整个 NIR 到底怎么分析”，优先读取：

**[218-2026-08-27-NIR完整分析管线与统计逻辑总说明.md](218-2026-08-27-NIR完整分析管线与统计逻辑总说明.md)**。

它不是指标清单，而是按下面的科学逻辑组织：

```text
逐帧测量是否可信
→ subject×eye continuous analysis-ready signal
→ time-on-task / trial / probe / subject 四个分析尺度
→ Behavior 错误形成方式
→ Probe 主观状态
→ PIR level / variability / slope / instability
→ current/previous visual covariates
→ within-person / between-person 分轨
→ primary / strict / left / right robustness
→ coverage / source-mode QC
→ OAR / questionnaire / RGB 扩展边界
```

## 数据与方法契约

| 文档 | 作用 | 当前地位 |
|---|---|---|
| [212-2026-08-27-NIR数据清洗逻辑与正式分析纳入规则.md](212-2026-08-27-NIR数据清洗逻辑与正式分析纳入规则.md) | primary/strict validity、左右眼、baseline、coverage 上位规则 | 逐帧方法基线 |
| [213-2026-08-27-NIR-analysis-ready数据契约与物化规范.md](213-2026-08-27-NIR-analysis-ready数据契约与物化规范.md) | production → `10_analysis_ready` | analysis-ready 数据契约 |
| [214-2026-08-27-NIR正式下游分析表数据契约.md](214-2026-08-27-NIR正式下游分析表数据契约.md) | `10_analysis_ready` → trial/probe/time-on-task | 下游构表契约 |
| [215-2026-08-27-NIR正式下游分析管线运行手册.md](215-2026-08-27-NIR正式下游分析管线运行手册.md) | 新终端、测试、构表、续跑 | 运行手册 |
| [217-2026-08-27-NIR错误值条件下下游分析管线验证方案.md](217-2026-08-27-NIR错误值条件下下游分析管线验证方案.md) | 错误 PIR 条件下的 validation-only 边界 | 当前验证原则 |
| [218-2026-08-27-NIR完整分析管线与统计逻辑总说明.md](218-2026-08-27-NIR完整分析管线与统计逻辑总说明.md) | 从测量到正式统计的完整分层逻辑 | **当前总导航** |
| [025-2026-08-26-SART刺激视觉协变量重建.md](025-2026-08-26-SART刺激视觉协变量重建.md) | 正式 SART 视觉协变量 | trial-level PLR/confound 控制 |
| [023-2026-08-25-sub031行为按键审计.md](023-2026-08-25-sub031行为按键审计.md) | prestimulus / carry-over / multiple keypress 真实证据 | omission/motor-timing QC 依据 |
| [021-眨眼检测边界与RITnet派生开合度.md](021-眨眼检测边界与RITnet派生开合度.md) | OAR、blink、PERCLOS 边界 | 后续 OAR 扩展依据 |

## `10_analysis_ready`：被试自己的连续有效信号

入口：

```text
scripts/nir_materialize_analysis_ready.py
configs/nir_analysis_ready.yaml
src/attention_pipeline/nir_analysis_ready/
```

这一层做 frame-level validity、subject×eye baseline、left/right preservation、centered PIR、strict parallel track 和 binocular source mode。它仍然是连续时间序列，不切 trial/probe 窗口。

当前错误 PIR snapshot 只能证明这个数据契约能运行，不能证明数值本身科学有效。

## `11_analysis_tables`：把连续信号变成分析单位

入口：

```text
scripts/nir_formal_pipeline.py
scripts/nir_build_analysis_tables.py
configs/nir_formal_analysis.yaml
src/attention_pipeline/nir_formal_analysis/
```

输出：

```text
trial_level
trial_pir_windows
probe_pir_windows
time_on_task_1s
trial_window_coverage
probe_window_coverage
```

这一层已经一次性保存 trial/probe 各候选时间窗、PIR 水平/波动/趋势/短时不稳定性、Behavior 摘要、coverage 和 source mode；后续统计不能再临时回到 frame CSV 自由切窗。

当前批量在用户要求下主动停止。已有 completed subjects 只用于 pipeline validation；PIR 修正前不要继续剩余被试。

## `12_pipeline_validation`：当前允许做的完整分析代码验证

入口：

```text
scripts/nir_pipeline_validation.py
configs/nir_pipeline_validation.yaml
src/attention_pipeline/nir_pipeline_validation/
```

当前 validation 已覆盖：

```text
Block / time-on-task
Go RT / RT-CV / ex-Gaussian
commission / omission / anticipatory
omission QC subtype
No-Go precursor RT/PIR trajectory
probe_response / probe_vigilance
probe_rt / probe_vigilance_rt
probe 前 10/20/30/60s Behavior + PIR
PIR median/mean/MAD/IQR/SD/P10/P90/slope/diff instability
primary/strict + binocular/left/right robustness
source-mode QC
multidimensional coverage QC
current/previous stimulus visual covariates
raw PIR between-person baseline
questionnaire optional interface
OAR / RGB extension readiness
```

主要代码模块：

```text
src/attention_pipeline/nir_pipeline_validation/
├── analysis.py
├── plots.py
├── probe_analysis.py
├── probe_plots.py
├── extended.py
├── extended_models.py
├── extended_plots.py
└── run.py
```

输出位于：

```text
D:/_AttentionData/Beijing-NIR/analysis/nir-behavior-v2/cohort-44-exploratory/12_pipeline_validation/
├── tables/
├── figures/
├── extension_readiness.json
└── validation_summary.json
```

所有当前错误 PIR 相关图都必须带：

```text
PIPELINE VALIDATION ONLY — CURRENT NIR VALUES KNOWN INVALID
```

图形全部由 Python/Matplotlib 代码生成，不使用图片生成模型。

## 当前运行方式

```powershell
git status --short --branch
git pull --ff-only

conda activate "D:\CondaEnvs\nir-amd"
python -m pip install -e .

python -m pytest `
  tests/test_nir_pipeline_validation.py `
  tests/test_nir_probe_validation.py `
  tests/test_nir_pipeline_validation_extended.py `
  tests/test_nir_formal_analysis.py -q

python scripts/nir_pipeline_validation.py
```

在 PIR 修复前，不要为了这一步执行：

```text
python scripts/nir_formal_pipeline.py --stage tables
```

## 正确 NIR 修复后的恢复顺序

```text
修正 NIR 数值来源
→ 重建 10_analysis_ready
→ 重建 11_analysis_tables
→ 重新跑 12_pipeline_validation
→ 冻结 primary window / coverage / visual covariates / Probe 语义 / sensitivity set
→ 才进入 20_formal_statistics
```

OAR 必须先扩展 `10_analysis_ready` schema 再进入下游；问卷按 subject-level 连接；RGB 未来通过 subject/Block/time/trial/probe key 进入 multimodal table。任何正式统计脚本都不得为了方便绕过已冻结的数据层直接读 production。

更早日期文档和 `docs/工作记录/` 继续保留完整历史 provenance，不追溯改写。