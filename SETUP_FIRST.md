# SETUP FIRST｜正式分析环境与路径配置（优先读取）

> **这是正式分析仓库的快速入口。换电脑、换磁盘、换终端或运行任何 Behavior / NIR / RGB 下游分析前，先读本文件。完整迁移说明以根目录 `ANALYSIS_SETUP_FIRST.md` 为准，新电脑逐项排错见 `docs/060-formal-analysis/010-新电脑迁移与常见报错检查表_20260830.md`。**
>
> 本文件优先于历史 README 中的机器路径示例。历史记录中的 `D:/`、`E:/`、`F:/`、`J:/` 等路径只代表当时机器，不得复制为正式配置。

## 1. 一条硬规则：环境和机器路径必须分离

正式代码不得硬编码数据盘符。机器差异全部进入本机路径注册表 `configs/paths.local.yaml`（该文件不提交 Git），科学配置只引用 `@path:<key>`。分析环境按模态隔离，不在一个环境里混装所有 GPU/CPU producer 依赖。

另外必须区分 **participant/session identity** 与 **modality availability**：一个已知 participant/session 可以因为设备、录制或文件问题没有 RGB、mmWave 或 NIR。缺模态只应在对应模态记为 missing/not_estimable，不得因此从 Behavior 或其他模态删掉该 session，也不得改变 participant 分组。

## 2. 三个正式下游环境

| 分析 | Conda 环境名 | 创建文件 | 用途 |
|---|---|---|---|
| Behavior | `attention-behavior-formal` | `environments/behavior-formal.yml` | SART、多尺度统计、问卷/身份、图表 |
| NIR pupil-only | `attention-nir-formal` | `environments/nir-pupil-formal.yml` | 读取已完成 NIR 输出，构造 10/11 层、候选验证、模型和图表；**不重跑 YOLO/RITnet** |
| RGB downstream | `attention-rgb-formal` | `environments/rgb-formal.yml` | 读取已保存 Face/Pose/Motion Parquet，派生 Motion QC、Pose confirmation/direction 与 algorithm-defined blink candidates；**PERCLOS/AU/emotion/rPPG/复杂预测当前 deferred** |

推荐直接使用 bootstrap：

```powershell
cd <你的 Attention-Analysis 仓库根目录>
python scripts/setup_formal_environment.py behavior
# 或 nir / rgb
```

手工回退示例：

```powershell
conda env create -f environments/behavior-formal.yml
conda activate attention-behavior-formal
python -m pip install -e .
```

NIR / RGB 时把 yml 和环境名换成对应项。已经存在且确需更新时使用 `python scripts/setup_formal_environment.py <analysis> --update`；不要在另一个分析环境里临时 `pip install` 来解决冲突。

## 3. 必须由你在每台电脑修改的唯一位置

复制当前 version 3 模板：

```powershell
Copy-Item configs/paths.example.yaml configs/paths.local.yaml
```

然后只编辑 `configs/paths.local.yaml`。至少把本机实际使用的键填成真实路径；运行相应 pipeline 前必须补齐其所需键。

关键键：

- `formal_raw_roots`：当前正式原始数据根；
- `cohort_manifest`：正式 session 队列；
- `repeat_registry`：participant/repeat identity 表；
- `questionnaire_derived_data`：问卷派生表；
- `nir_analysis_ready_source_manifest_json`：**当前 staged NIR JSON source manifest**；
- `nir_source_manifest`：**历史 CSV adapter 专用**，不能与上一个键混用；
- `nir_analysis_ready_root`、`nir_analysis_tables_root`：NIR 派生 10/11 层；
- `stimulus_visual_properties`：刺激视觉属性表；
- `rgb_raw_output_root`：已完成 RGB raw-first Face/Pose/Motion 输出；
- `rgb_analysis_ready_root`、`rgb_analysis_tables_root`：RGB 下游派生层；
- `behavior_output_root`：Behavior 正式输出；
- `fusion_output_root`：多模态融合派生输出（当前 disabled/deferred，不作为 release gate）。

设置本机注册表：

```powershell
$env:ATTENTION_ANALYSIS_PATHS_CONFIG = (Resolve-Path configs/paths.local.yaml).Path
```

可写入 PowerShell profile，但**不要把本机绝对路径提交 Git**。

## 4. 新电脑最小检查

```powershell
git fetch origin --prune
git switch codex/formal-analysis-v2-portable
git pull --ff-only
$env:ATTENTION_ANALYSIS_PATHS_CONFIG = (Resolve-Path configs/paths.local.yaml).Path
python -c "import sys; print(sys.executable); print(sys.version)"
python -c "from attention_pipeline.path_registry import load_path_registry; r=load_path_registry(); print(r.path); print(r.data.get('version'))"
python -m pytest -q
```

`pytest` 通过只代表代码测试通过，不代表真实 44-session science release。正式运行前仍需 representative smoke，并检查 manifest、availability/QC 和 failure tables。

## 5. RGB producer 与 downstream 不混装

`rgb-amd` / `rgb-nvidia` 历史分支中的 Py-Feat/LibreFace/MediaPipe/CUDA/DirectML 环境属于**昂贵 raw producer**。当前正式 downstream 不重新调用这些模型；它消费已经保存的 `*_face_raw.parquet`、`*_pose_landmarks.parquet`、`*_motion_raw.parquet`。

某个 governed session 没有 RGB 目录或某个 component 文件时，正式行为是对该 component 记录 `source_missing/not_estimable`，而不是自动重跑 producer，更不是删除 participant/session。

## 6. 当前 NIR 科学边界

NIR 当前正式主信号是 pupil geometry。可靠性不足的 PIR/iris geometry 禁止恢复为正式 endpoint。

生产端已有的 `fullclass_ocular_aperture_ratio_median` / `p90` **不是被禁止的 PIR**：它们作为 visible ocular mask 的 eye-opening QC candidates 保留，但不是 MediaPipe EAR、不是 blink event、不是 PERCLOS，也不自动成为正式生理 endpoint。

因此当前口径是：

```text
PIR / iris geometry: refused
ocular-aperture QC: preserved
ocular-aperture formal endpoint: false
```

## 7. 当前 release 边界

- Behavior：当前正式 science-v3 下游；真实结果仍需真实数据运行验证。
- NIR：pupil-only staged downstream；PIR/iris geometry 禁止，OAR 仅保留 QC；不重跑 YOLO/RITnet。
- RGB：Motion QC、Pose confirmation/direction、algorithm-defined blink candidates；PERCLOS/AU/emotion/rPPG/复杂预测 deferred。
- mmWave：正式 producer/analysis authority 在外部毫米波仓库；缺场次只能记 modality missing/invalid，不能缩减其他模态 cohort。
- Multimodal fusion：当前明确 `disabled_deferred`；future common-available paired subset 必须显式构造并报告覆盖率，不能成为默认 cohort。

## 8. 文档优先级

如果旧文档与当前 formal portable 分支在“环境、机器路径、模态 availability 或当前 release 边界”上冲突，按以下顺序：

```text
ANALYSIS_SETUP_FIRST.md
→ SETUP_FIRST.md
→ configs/paths.example.yaml + 本机 paths.local.yaml
→ configs/README.md
→ docs/060-formal-analysis/009-正式管线修复后完整复审_20260830.md
→ docs/060-formal-analysis/010-新电脑迁移与常见报错检查表_20260830.md
→ 当前 runner/config
→ 历史 README / 工作记录（provenance）
```

历史科学决策和旧硬件运行记录继续保留，但不得覆盖当前 formal portable 的执行入口。
