# SETUP FIRST｜正式分析环境与路径配置（优先读取）

> **这是正式分析仓库的第一入口。换电脑、换磁盘、换终端或运行任何 Behavior / NIR / RGB 下游分析前，先读本文件。**
>
> 本文件优先于历史 README 中的机器路径示例。历史记录中的 `D:/`、`E:/`、`F:/`、`J:/` 等路径只代表当时机器，不得复制为正式配置。

## 1. 一条硬规则：环境和机器路径必须分离

正式代码不得硬编码数据盘符。机器差异全部进入本机路径注册表 `configs/paths.local.yaml`（该文件不提交 Git），科学配置只引用 `@path:<key>`。分析环境按模态隔离，不在一个环境里混装所有 GPU/CPU 依赖。

## 2. 三个正式下游环境

| 分析 | Conda 环境名 | 创建文件 | 用途 |
|---|---|---|---|
| Behavior | `attention-behavior-formal` | `environments/behavior-formal.yml` | SART、多尺度统计、问卷/身份、图表 |
| NIR pupil-only | `attention-nir-formal` | `environments/nir-pupil-formal.yml` | 读取已完成 NIR 输出，构造 10/11 层、候选验证、模型和图表；**不重跑 YOLO/RITnet** |
| RGB downstream | `attention-rgb-formal` | `environments/rgb-formal.yml` | 读取已保存 Face/Pose/Motion Parquet，派生眼睑/眨眼/PERCLOS/头姿/身体与运动指标、时间窗、统计和图表 |

创建示例：

```powershell
cd <你的 Attention-Analysis 仓库根目录>
conda env create -f environments/behavior-formal.yml
conda activate attention-behavior-formal
python -m pip install -e .
```

NIR / RGB 时把 yml 和环境名换成对应项。已经存在时使用 `conda env update -f <yml> --prune`，不要在另一个分析环境里临时 `pip install` 来解决冲突。

## 3. 必须由你在每台电脑修改的唯一位置

复制模板：

```powershell
Copy-Item configs/paths.example.yaml configs/paths.local.yaml
```

然后只编辑 `configs/paths.local.yaml`。至少把本机实际使用的键填成真实路径；没有的模态可以先留空，但运行相应 pipeline 前必须补齐。

关键键：

- `formal_raw_roots`：当前正式原始数据根；
- `cohort_manifest`：正式 session 队列；
- `repeat_registry`：`participant_key / visit_order` 权威身份表；
- `questionnaire_derived_data`：问卷派生表；
- `nir_analysis_ready_root`、`nir_analysis_tables_root`：NIR 派生 10/11 层；
- `stimulus_visual_properties`：刺激视觉属性表；
- `rgb_raw_output_root`：已完成 RGB raw-first Face/Pose/Motion 输出；
- `rgb_analysis_ready_root`、`rgb_analysis_tables_root`：RGB 下游派生层；
- `behavior_output_root`：Behavior 正式输出；
- `fusion_output_root`：多模态融合派生输出（当前 deferred，不作为 release gate）。

设置本机注册表：

```powershell
$env:ATTENTION_ANALYSIS_PATHS_CONFIG = (Resolve-Path configs/paths.local.yaml)
```

可写入 PowerShell profile，但**不要把本机绝对路径提交 Git**。

## 4. 新电脑最小检查

```powershell
git fetch origin --prune
git switch codex/formal-analysis-v2-portable
git pull --ff-only
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pytest -q
```

`pytest` 通过只代表代码测试通过，不代表真实 44-session science release。正式运行前仍需 representative smoke、identity/cohort parity、candidate endpoint freeze。

## 5. RGB producer 与 downstream 不混装

`rgb-amd` / `rgb-nvidia` 历史分支中的 Py-Feat/LibreFace/MediaPipe/CUDA/DirectML 环境属于**昂贵 raw producer**。当前正式 downstream 不重新调用这些模型；它消费已经保存的 `*_face_raw.parquet`、`*_pose_landmarks.parquet`、`*_motion_raw.parquet`。如果未来必须补跑 raw producer，应按对应硬件单独建立 producer 环境，不能污染 `attention-rgb-formal`。

## 6. 当前 release 边界

- Behavior：正式科学代码持续修复；真实数据 endpoint freeze 仍需运行验证。
- NIR：pupil-only downstream；PIR/OAR 禁止恢复；不重跑 YOLO/RITnet。
- RGB：raw producer 资产在历史 RGB 分支；正式 downstream 正迁入本主线。
- Multimodal fusion：当前明确 `deferred_not_release_ready`，见 `docs/090-formal-analysis/MULTIMODAL_DEFERRED.md`。

如果任何旧文档与本文件在“环境、机器路径或当前 release 边界”上冲突，以本文件和当前正式配置为准；历史科学决策仍保留其 provenance。