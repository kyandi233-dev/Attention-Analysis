# ANALYSIS_SETUP_FIRST｜正式分析环境与路径注册表（优先读取）

> **优先级：最高。** 在新电脑、全新终端、Codex/ChatGPT 本地执行或正式分析前，先读本文件，再读各模态 README/运行手册。本文件是当前 `codex/formal-analysis-v2-portable` 的环境与机器路径入口；旧 README 中出现的 D:/E:/F:/J:/ 等路径仅是历史机器记录，不是可复制的正式配置。

## 1. 核心原则

1. 不建立一个“大一统”环境。Behavior、NIR pupil-only downstream、RGB downstream 使用彼此隔离的 Conda 环境。
2. 下游科学分析环境不安装昂贵 producer 的 GPU/模型依赖。RGB downstream 只读取已保存 Parquet；NIR downstream 只读取已完成的 pupil-only analysis-ready 数据。
3. **科学配置禁止硬编码机器绝对路径。** 所有会因电脑/硬盘改变的路径只写在本机 `configs/paths.local.yaml`（gitignored）或由 `ATTENTION_ANALYSIS_PATHS_CONFIG` 指向的等价文件。
4. 正式 cohort 始终由 `cohort_manifest` 决定；问卷不能反向定义或删减正式 session。
5. 多模态融合当前 `deferred_not_release_ready`，不要把未冻结的融合脚本当正式入口。
6. **优先使用 `scripts/setup_formal_environment.py` 创建/检查正式下游环境。** `conda env create` 手工命令只作为 bootstrap 脚本不可用时的回退。

## 2. 新电脑首次配置

仓库可以放在任意磁盘。唯一需要人工替换的是 `<REPO_PARENT>`，它表示当前电脑实际希望存放 Git 仓库的位置。

```powershell
cd "<REPO_PARENT>"
git clone https://github.com/kyandi233-dev/Attention-Analysis.git
cd Attention-Analysis
git fetch origin --prune
git switch codex/formal-analysis-v2-portable
git pull --ff-only
```

### 2.1 推荐：由仓库自动创建对应正式环境

只创建本次实际要做的分析，不要无理由一次创建全部环境：

```powershell
# Behavior / questionnaire / SART formal
python scripts/setup_formal_environment.py behavior

# NIR pupil-only downstream
python scripts/setup_formal_environment.py nir

# RGB preserved-output downstream
python scripts/setup_formal_environment.py rgb
```

bootstrap 会自动完成：

```text
选择分析类型
→ 找到对应 environments/*.yml
→ 检查 conda
→ 若环境不存在则创建
→ 若环境已存在则默认保留，不擅自重建
→ 在目标环境中 pip install -e .
→ import 关键科学依赖做环境自检
→ 检查/创建 configs/paths.local.yaml
→ 若仍有路径占位符则阻止“正式运行已就绪”的判断
```

如果确实需要按当前 YAML 更新既有环境：

```powershell
python scripts/setup_formal_environment.py behavior --update
python scripts/setup_formal_environment.py nir --update
python scripts/setup_formal_environment.py rgb --update
```

`--update` 会调用 `conda env update --prune`，因此不要在无明确原因时随意使用。

环境名和定义文件固定为：

| 正式任务 | 独立环境 | 环境文件 |
|---|---|---|
| Behavior / questionnaire / SART formal | `attention-behavior-formal` | `environments/behavior-formal.yml` |
| NIR pupil-only downstream | `attention-nir-formal` | `environments/nir-pupil-formal.yml` |
| RGB saved-output downstream | `attention-rgb-formal` | `environments/rgb-formal.yml` |

RGB producer（Py-Feat/LibreFace/MediaPipe/CUDA/DirectML）属于昂贵提取层，**不属于 `attention-rgb-formal`**。需要重新做 producer 时才读取 `docs/040-rgb/` 中对应硬件/producer 文档；不能为了下游统计重新安装整套 producer。

### 2.2 手工回退方式

只有 bootstrap 无法使用时才手工执行：

```powershell
conda env create -f environments/behavior-formal.yml
conda env create -f environments/nir-pupil-formal.yml
conda env create -f environments/rgb-formal.yml
```

创建后仍需在对应环境中安装当前 checkout：

```powershell
conda activate attention-behavior-formal
python -m pip install -e .
```

NIR / RGB 同理使用各自环境名。

## 3. 唯一需要按机器修改的文件

bootstrap 在 `configs/paths.local.yaml` 不存在时会从模板创建；也可手工执行：

```powershell
Copy-Item configs/paths.example.yaml configs/paths.local.yaml
```

**必须编辑的是 `configs/paths.local.yaml`，不是正式 science config。** 其中 `${...}` / `X:/...` 都只是占位符，例如：

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

`X:/...` 必须替换成当前电脑真实位置。不要把 `paths.local.yaml` 或本机绝对路径提交到 Git。

设置本终端的路径注册表：

```powershell
$env:ATTENTION_ANALYSIS_PATHS_CONFIG = (Resolve-Path "configs/paths.local.yaml").Path
```

若希望每个新终端自动生效，可在本机 PowerShell profile/系统环境变量中设置该变量；这属于机器配置，不提交 Git。

只检查路径模板、不创建 Conda 环境时可运行：

```powershell
python scripts/setup_formal_environment.py rgb --paths-only
```

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

验证路径注册表：

```powershell
python -c "from attention_pipeline.config import load_config; c=load_config('configs/rgb_formal.yaml'); print(c.registry_path('cohort_manifest'))"
```

如果提示未加载 path registry，先检查 `ATTENTION_ANALYSIS_PATHS_CONFIG`。如果路径不存在，修改 `configs/paths.local.yaml`，**不要修改科学配置去适配某一台机器。**

## 5. 当前正式入口与放行状态

- Behavior：`scripts/sart_formal_analysis.py`；正式全量前仍需 targeted pytest + representative smoke。
- NIR pupil-only：`scripts/nir_formal_pipeline.py` / `scripts/nir_build_analysis_tables.py`；不得重跑 YOLO/RITnet 以补下游统计。
- RGB downstream：消费既有 `*_face_raw.parquet`、`*_pose_landmarks.parquet`、`*_motion_raw.parquet`；昂贵 producer 不重跑。正式下游仍需 synthetic tests + representative smoke 后才允许全 cohort。
- Multimodal fusion：**deferred_not_release_ready**；当前只保留接口与历史资产，不作为正式结果入口。

## 6. 路径与环境审计硬规则

以下情况必须 fail closed，而不是猜测：

- science config 出现新的 `D:/`、`E:/`、`F:/`、`J:/` 等机器绝对路径；
- `configs/paths.local.yaml` 仍含 `${...}` / `X:/...` 占位符却准备跑正式分析；
- `participant_key` / `visit_order` 缺失却尝试从 `sub-XXX` 编号推断；
- RGB downstream 缺 raw Parquet 却自动触发 producer；
- NIR downstream 缺 analysis-ready 却自动重跑 YOLO/RITnet；
- 多模态融合在单模态 endpoint/QC/identity 尚未冻结时被标记为 release-ready。

## 7. 旧文档如何阅读

仓库保留 AMD/NVIDIA 历史运行记录用于 provenance，其中可能包含旧电脑绝对路径、旧环境名和旧分支名。它们不能覆盖本文件。冲突时执行优先级为：

```text
ANALYSIS_SETUP_FIRST.md
→ scripts/setup_formal_environment.py
→ configs/paths.example.yaml + 本机 paths.local.yaml
→ environments/*.yml
→ 当前正式模态 config / runner
→ 历史 README / 工作记录（仅用于追溯）
```
