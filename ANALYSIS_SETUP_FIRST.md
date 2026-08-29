# ANALYSIS_SETUP_FIRST｜正式分析环境与路径注册表（优先读取）

> **优先级：最高。** 在新电脑、全新终端、Codex/ChatGPT 本地执行或正式分析前，先读本文件，再读各模态 README/运行手册。本文件是当前 `codex/formal-analysis-v2-portable` 的环境与机器路径入口；旧 README 中出现的 D:/E:/F:/J:/ 等路径仅是历史机器记录，不是可复制的正式配置。

## 1. 核心原则

1. 不建立一个“大一统”环境。Behavior、NIR pupil-only downstream、RGB downstream 使用彼此隔离的 Conda 环境。
2. 下游科学分析环境不安装昂贵 producer 的 GPU/模型依赖。RGB downstream 只读取已保存 Parquet；NIR downstream 只读取已完成的 pupil-only analysis-ready 数据。
3. **科学配置禁止硬编码机器绝对路径。** 所有会因电脑/硬盘改变的路径只写在本机 `configs/paths.local.yaml`（gitignored）或由 `ATTENTION_ANALYSIS_PATHS_CONFIG` 指向的等价文件。
4. 正式 cohort 始终由 `cohort_manifest` 决定；问卷不能反向定义或删减正式 session。
5. 多模态融合当前 `deferred_not_release_ready`，不要把未冻结的融合脚本当正式入口。

## 2. 新电脑首次配置

在仓库根目录执行。仓库放在哪个盘都可以；下面不假设 D:/ 或任何固定目录。

```powershell
# 1) 克隆并进入仓库（<REPO_PARENT> 必须换成这台电脑实际目录）
cd "<REPO_PARENT>"
git clone https://github.com/kyandi233-dev/Attention-Analysis.git
cd Attention-Analysis
git fetch origin --prune
git switch codex/formal-analysis-v2-portable
git pull --ff-only

# 2) 创建当前任务对应的独立环境（三选一；不要无理由全部创建）
conda env create -f environments/behavior-formal.yml
conda env create -f environments/nir-pupil-formal.yml
conda env create -f environments/rgb-formal.yml
```

环境名：

| 正式任务 | 环境 | 环境文件 |
|---|---|---|
| Behavior / questionnaire / SART formal | `attention-behavior-formal` | `environments/behavior-formal.yml` |
| NIR pupil-only downstream | `attention-nir-formal` | `environments/nir-pupil-formal.yml` |
| RGB saved-output downstream | `attention-rgb-formal` | `environments/rgb-formal.yml` |

RGB producer（Py-Feat/LibreFace/MediaPipe/CUDA/DirectML）属于历史昂贵提取层，**不属于 `attention-rgb-formal`**。只有需要重新做 producer 时才读 `docs/040-rgb/` 的硬件/producer 文档；不能为了做下游统计重新安装整套 producer。

## 3. 唯一需要按机器修改的文件

复制模板：

```powershell
Copy-Item configs/paths.example.yaml configs/paths.local.yaml
```

然后只修改 `configs/paths.local.yaml` 中 `${...}` 对应的值，例如：

```yaml
version: 2
paths:
  formal_raw_roots:
    - "X:/本机正式实验根目录"
  cohort_manifest: "X:/.../cohort_manifest.csv"
  repeat_registry: "X:/.../subject_repeat_registry.csv"
  questionnaire_derived_data: "X:/.../questionnaire_derived_data.csv"

  nir_source_manifest: "X:/.../nir_source_manifest.csv"
  nir_standardized_root: "X:/.../nir_standardized"
  nir_analysis_ready_root: "X:/.../10_analysis_ready"
  nir_analysis_tables_root: "X:/.../11_analysis_tables"
  stimulus_visual_properties: "X:/.../stimulus_visual_properties.csv"

  rgb_raw_output_root: "X:/.../Beijing-RGB"
  rgb_analysis_ready_root: "X:/.../rgb-analysis-ready"
  rgb_analysis_tables_root: "X:/.../rgb-analysis-tables"

  behavior_output_root: "X:/.../behavior-formal"
  mmwave_output_root: "X:/.../mmwave-formal"
  fusion_output_root: "X:/.../fusion-formal"
```

`X:/...` 只是占位示例。**必须改成当前电脑的真实位置；不要把本机绝对路径提交到 Git。**

随后设置：

```powershell
$env:ATTENTION_ANALYSIS_PATHS_CONFIG = (Resolve-Path "configs/paths.local.yaml").Path
```

若希望每个新终端自动生效，可在本机 PowerShell profile/系统环境变量设置该变量；这属于机器配置，不提交 Git。

## 4. 每次新终端的固定检查

```powershell
git status --short --branch
git fetch origin --prune
git switch codex/formal-analysis-v2-portable
git pull --ff-only
$env:ATTENTION_ANALYSIS_PATHS_CONFIG = (Resolve-Path "configs/paths.local.yaml").Path
```

然后只激活本轮需要的环境：

```powershell
conda activate attention-behavior-formal
# 或
conda activate attention-nir-formal
# 或
conda activate attention-rgb-formal
```

验证路径注册表能被读取：

```powershell
python -c "from attention_pipeline.config import load_config; c=load_config('configs/rgb_formal.yaml'); print(c.registry_path('cohort_manifest'))"
```

如果提示未加载 path registry，先检查 `ATTENTION_ANALYSIS_PATHS_CONFIG`。如果路径不存在，修改 `configs/paths.local.yaml`，**不要修改科学配置去适配某一台机器。**

## 5. 当前正式入口与放行状态

- Behavior：`scripts/sart_formal_analysis.py`；代码层持续完善，正式全量前仍需本地 pytest + representative smoke。
- NIR pupil-only：`scripts/nir_formal_pipeline.py` / `scripts/nir_build_analysis_tables.py`；不得重跑 YOLO/RITnet 以补下游统计。
- RGB downstream：消费既有 `*_face_raw.parquet`、`*_pose_landmarks.parquet`、`*_motion_raw.parquet`；昂贵 producer 不重跑。正式下游仍需 synthetic tests + representative smoke 后才允许全 cohort。
- Multimodal fusion：**deferred_not_release_ready**；当前只保留接口与历史资产，不作为正式结果入口。

## 6. 路径与环境审计硬规则

以下情况应 fail closed，而不是猜测：

- science config 出现新的 `D:/`、`E:/`、`F:/`、`J:/` 等机器绝对路径；
- `participant_key` / `visit_order` 缺失却尝试从 `sub-XXX` 编号推断；
- RGB downstream 缺 raw Parquet 却自动触发 producer；
- NIR downstream 缺 analysis-ready 却自动重跑 YOLO/RITnet；
- 多模态融合在单模态 endpoint/QC/identity 尚未冻结时被标记为 release-ready。

## 7. 旧文档如何阅读

仓库保留 AMD/NVIDIA 历史运行记录用于 provenance，其中可能包含旧电脑绝对路径、旧环境名和旧分支名。它们不能覆盖本文件。遇到冲突时，执行优先级为：

```text
ANALYSIS_SETUP_FIRST.md
→ configs/paths.example.yaml + 本机 paths.local.yaml
→ environments/*.yml
→ 当前正式模态 config / runner
→ 历史 README / 工作记录（仅用于追溯）
```
