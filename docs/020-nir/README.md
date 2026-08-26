# NIR

## 2026-08-27 当前 NIR 状态

正式 NIR production 推理与 full-class extension 已完成；当前工作不再是重新运行 YOLO / RITnet，而是基于 frozen production 构建可复现的正式下游分析管线。

当前主线固定为：

```text
runtime/nir-formal/ frozen production
        ↓
10_analysis_ready
        ↓
11_analysis_tables
        ↓
20_formal_statistics（下一阶段）
```

当前 AMD 本地 44 人仍是 exploratory/development cohort；未来约 116 人北京正式 cohort 应复用同一冻结代码与方法，但写入独立 snapshot，不能把 44 人效应直接作为最终正式样本结果。

## 当前最重要的方法入口

| 文档 | 作用 | 当前地位 |
|---|---|---|
| [022-2026-08-25-NIR正式分析设计与待验证项.md](022-2026-08-25-NIR正式分析设计与待验证项.md) | 总体科学问题、Behavior/Probe/time-on-task/统计路线 | 总体科学设计 |
| [212-2026-08-27-NIR数据清洗逻辑与正式分析纳入规则.md](212-2026-08-27-NIR数据清洗逻辑与正式分析纳入规则.md) | primary/strict validity、左右眼、baseline、coverage 的现行规则 | **逐帧清洗上位规则** |
| [213-2026-08-27-NIR-analysis-ready数据契约与物化规范.md](213-2026-08-27-NIR-analysis-ready数据契约与物化规范.md) | frozen production → `10_analysis_ready` | **正式基础数据层契约** |
| [214-2026-08-27-NIR正式下游分析表数据契约.md](214-2026-08-27-NIR正式下游分析表数据契约.md) | `10_analysis_ready` → trial / probe / time-on-task 分析表 | **正式下游构表契约** |
| [215-2026-08-27-NIR正式下游分析管线运行手册.md](215-2026-08-27-NIR正式下游分析管线运行手册.md) | 新终端、测试、representative、44 人运行与验收命令 | **当前运行手册** |
| [027-44人全量分析数据边界与资料清单.md](027-44人全量分析数据边界与资料清单.md) | 当前 44 人数据边界 | 当前 exploratory 数据边界 |
| [028-2026-08-26-NIR-cohort44分析实施计划与进度.md](028-2026-08-26-NIR-cohort44分析实施计划与进度.md) | cohort 分析阶段与进度 | 当前进度记录 |

## 当前正式代码入口

### 1. Production 推理

```text
runtime/nir-formal/
```

已经存在的 production 不因下游分析需要而重跑。

### 2. Analysis-ready 物化

```text
scripts/nir_materialize_analysis_ready.py
configs/nir_analysis_ready.yaml
src/attention_pipeline/nir_analysis_ready/
```

当前 44 人已完成 `10_analysis_ready` 物化和验收。production/strict 有效率约 74.2%，primary 有效率约 87.2%，且 `strict_not_primary_n = 0`。

### 3. 正式下游分析表

统一入口：

```text
scripts/nir_formal_pipeline.py
```

底层构表入口：

```text
scripts/nir_build_analysis_tables.py
configs/nir_formal_analysis.yaml
src/attention_pipeline/nir_formal_analysis/
```

当前输出目标：

```text
D:/_AttentionData/Beijing-NIR/analysis/nir-behavior-v2/cohort-44-exploratory/
├── 10_analysis_ready/
├── 11_analysis_tables/
└── 20_formal_statistics/   # 后续
```

`11_analysis_tables` 只读 `10_analysis_ready` + formal Behavior，生成：

```text
trial_level
trial_pir_windows
probe_pir_windows
time_on_task_1s
trial_window_coverage
probe_window_coverage
manifest / summary / completion
```

不会直接读取 production NIR，不会重新运行 YOLO / RITnet，也不会在这一层运行显著性模型。

## 当前运行最短路径

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"

git status --short --branch
git fetch origin --prune
git switch analysis/multimodal-integration
git pull --ff-only

conda activate "D:\CondaEnvs\attention-behavior"
python -m pip install -e .

python -m pytest `
  tests/test_nir_analysis_ready.py `
  tests/test_nir_formal_analysis.py `
  tests/test_nir_behavior_alignment.py -q

python scripts/nir_formal_pipeline.py --stage tables
```

第一次验证新版本必须先按 [215](215-2026-08-27-NIR正式下游分析管线运行手册.md) 用 `sub-031` representative 验收，再扩到 44 人。

## PIR 当前正式定义与处理边界

PIR（pupil-to-iris diameter ratio，瞳孔/虹膜直径比）继续使用直径比而非像素面积比：

```text
D_pupil = sqrt(pupil_axis_a × pupil_axis_b)
D_iris  = sqrt(iris_axis_a × iris_axis_b)
PIR     = D_pupil / D_iris
```

primary frame validity 以 212 为准：pupil/iris outer 椭圆拟合有效、pupil center 位于 iris outer、`D_iris > D_pupil`、PIR finite。旧 whole-mask edge gate 不再是主分析 hard exclusion。

左右眼先分别做 subject×eye 跨 Block1+Block2 中位数中心化，再构造 binocular PIR；单眼有效时允许 single-eye fallback，并永久保留 source mode。

## Behavior / Probe / coverage

正式 Behavior 仍为 FocusWave v3.1.3 BB。trial/probe 的绝对时间轴、Block 起点重建、边界截断与内部缺失分离等逻辑继续复用已经验证的 schema-v2 原型。

但 [024-2026-08-26-NIR行为对齐原型与数据契约.md](024-2026-08-26-NIR行为对齐原型与数据契约.md) 与 `scripts/nir_behavior_alignment.py` 现在属于**历史 prototype / provenance**：它们直接读取 production、使用旧 validity 且不融合左右眼，因此不再作为当前正式主分析入口。

coverage 只表示具体 trial/probe/time window 是否具有足够时间代表性，不重新成为 frame-level 清洗规则。当前构表阶段不根据结果冻结 coverage threshold，也不允许根据显著性选择窗口。

## OAR / blink 边界

[021-眨眼检测边界与RITnet派生开合度.md](021-眨眼检测边界与RITnet派生开合度.md) 继续记录 OAR（ocular aperture ratio，眼球可见开合度比率）、blink/PERCLOS 的解释边界。

当前 `10_analysis_ready` 正式主契约只冻结 PIR，因此 `11_analysis_tables` v1 暂只构造 PIR。后续若正式纳入 OAR，应先把 OAR 作为只读 passthrough 字段加入 analysis-ready schema，再由同一构表层读取；正式统计脚本不得绕过 `10_analysis_ready` 临时直读 production。

## 其他仍有效资料

- [025-2026-08-26-SART刺激视觉协变量重建.md](025-2026-08-26-SART刺激视觉协变量重建.md)：27 个正式 SART 画面与视觉协变量。
- [029-2026-08-26-PIR有效性与usable筛选定义.md](029-2026-08-26-PIR有效性与usable筛选定义.md)：production `fullclass_normalization_valid` 的历史真实定义；不再等于 primary 纳入规则。
- [210-2026-08-26-PIR六条gate失败原因QC结果.md](210-2026-08-26-PIR六条gate失败原因QC结果.md)：旧 strict gate QC。
- [211-2026-08-26-左右眼PIR处理与标准化冻结决策.md](211-2026-08-26-左右眼PIR处理与标准化冻结决策.md)：左右眼结构与标准化验证；若与 212 的基础层定义冲突，以 212 为准。

更早的 08-13、08-16、08-17、08-21、08-22 文档和 `docs/工作记录/` 保留完整历史 provenance，不追溯改写为当前状态。
