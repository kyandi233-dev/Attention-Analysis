# ANALYSIS_SETUP_FIRST｜正式分析环境、路径与新电脑迁移入口

> **优先级：最高。** 在新电脑、全新终端、Codex/ChatGPT 本地执行或正式分析前，先读本文件，再读各模态 README/运行手册。本文件是当前 `codex/formal-analysis-v2-portable` 的环境与机器路径入口；旧 README 中出现的 D:/E:/F:/J:/ 等路径仅是历史机器记录，不是可复制的正式配置。

## 1. 核心原则

1. 不建立一个“大一统”环境。Behavior、NIR pupil-only downstream、RGB downstream 使用彼此隔离的 Conda 环境。
2. 下游科学分析环境不安装昂贵 producer 的 GPU/模型依赖。RGB downstream 只读取已保存 Parquet；NIR downstream 只读取已经完成的 fullclass-final / pupil-only 下游资产，不因为统计阶段缺文件而自动重跑 YOLO/RITnet。
3. **科学配置禁止硬编码机器绝对路径。** 所有会因电脑/硬盘改变的路径只写在本机 `configs/paths.local.yaml`（gitignored）或由 `ATTENTION_ANALYSIS_PATHS_CONFIG` 指向的等价文件。
4. 正式 cohort 始终由 `cohort_manifest` 决定。参与者身份与“某模态有没有录到”是两件事：RGB、mmWave、NIR 某场缺失只能记成该模态 `missing/not_estimable`，不能反向删掉 Behavior 或其他模态中的已知 session/participant。
5. 问卷/重复被试登记用于 participant identity 与 visit 信息，不用于决定某模态是否存在。
6. 多模态融合当前 `disabled_deferred`，不要把未冻结的融合脚本当正式入口。
7. **优先使用 `scripts/setup_formal_environment.py` 创建/检查正式下游环境。** `conda env create` 手工命令只作为 bootstrap 脚本不可用时的回退。

## 2. 新电脑首次配置

仓库可以放在任意磁盘。唯一需要人工替换的是 `<REPO_PARENT>`，它表示当前电脑实际希望存放 Git 仓库的位置。

### 2.1 先确认 Git、Conda、PowerShell

```powershell
git --version
conda --version
python --version
```

如果 `conda` 提示“不是内部或外部命令”，先在 Anaconda Prompt/Miniconda Prompt 中运行：

```powershell
conda init powershell
```

然后关闭并重新打开 PowerShell。正式下游环境统一使用 Python 3.11 的环境 YAML；不要拿系统 Python 或其他项目环境直接跑。

### 2.2 Clone 并切到唯一正式下游分支

```powershell
cd "<REPO_PARENT>"
git clone https://github.com/kyandi233-dev/Attention-Analysis.git
cd Attention-Analysis
git fetch origin --prune
git switch codex/formal-analysis-v2-portable
git pull --ff-only
git status --short --branch
git log -1 --oneline
```

如果 `git pull --ff-only` 被本地修改阻断，不要 `reset --hard`。先用 `git status` 查明是不是本机误改了 science config / README / 代码，再决定保留、提交或另存。

### 2.3 推荐：由仓库自动创建对应正式环境

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

### 2.4 手工回退方式

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

NIR / RGB 同理使用各自环境名。若后面出现 `ModuleNotFoundError: attention_pipeline`，第一检查项就是当前环境里是否真的执行过这条 editable install。

## 3. 唯一需要按机器修改的文件

bootstrap 在 `configs/paths.local.yaml` 不存在时会从模板创建；也可手工执行：

```powershell
Copy-Item configs/paths.example.yaml configs/paths.local.yaml
```

**必须编辑的是 `configs/paths.local.yaml`，不是正式 science config。** 当前模板版本为 **3**，示例如下：

```yaml
version: 3
paths:
  formal_raw_roots:
    - "X:/本机正式实验根目录"
  cohort_manifest: "X:/.../cohort_manifest.csv"
  repeat_registry: "X:/.../subject_repeat_registry.csv"
  questionnaire_derived_data: "X:/.../questionnaire_derived_data.csv"

  # 当前 staged NIR 的 JSON source manifest；不要与下面 legacy CSV 混用
  nir_analysis_ready_source_manifest_json: "X:/.../nir_analysis_ready_source_manifest.json"
  nir_source_manifest: "X:/.../nir_source_manifest_legacy.csv"
  nir_standardized_root: "X:/.../nir_standardized"
  nir_analysis_ready_root: "X:/.../10_analysis_ready"
  nir_analysis_tables_root: "X:/.../11_analysis_tables"
  stimulus_visual_properties: "X:/.../stimulus_visual_properties.csv"

  rgb_raw_output_root: "X:/.../Beijing-RGB"
  rgb_analysis_ready_root: "X:/.../rgb-analysis-ready"
  rgb_analysis_tables_root: "X:/.../rgb-analysis-tables"

  behavior_output_root: "X:/.../behavior-formal"
  rgb_output_root: "X:/.../rgb-output"
  mmwave_output_root: "X:/.../mmwave-formal"
  fusion_output_root: "X:/.../fusion-formal"
```

`X:/...` 必须替换成当前电脑真实位置。也可以保留 `${ENV_VAR}` 形式，但运行前相应环境变量必须真实存在；path registry 会对未解析变量 fail closed。不要把 `paths.local.yaml` 或本机绝对路径提交到 Git。

### 3.1 最容易混淆的两个 NIR manifest

- `nir_analysis_ready_source_manifest_json`：**当前权威 staged NIR** 使用，格式是 JSON object + `sessions[]`。
- `nir_source_manifest`：只给历史 `formal_multimodal_analysis.py nir-adapt` CSV adapter 使用。

把这两个指向同一个文件，是新电脑迁移时最容易出现的接口错误之一；JSON/CSV 不能混用。

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
cd "<你的 Attention-Analysis 仓库>"
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

验证当前 Python 与包来源：

```powershell
where.exe python
python -c "import attention_pipeline,sys; print(sys.executable); print(attention_pipeline.__file__)"
```

验证路径注册表：

```powershell
python -c "from attention_pipeline.config import load_config; c=load_config('configs/rgb_formal.yaml'); print(c.registry_path('cohort_manifest'))"
```

如果提示未加载 path registry，先检查：

```powershell
echo $env:ATTENTION_ANALYSIS_PATHS_CONFIG
Test-Path $env:ATTENTION_ANALYSIS_PATHS_CONFIG
```

如果路径不存在，修改 `configs/paths.local.yaml`，**不要修改科学配置去适配某一台机器。**

## 5. 当前正式入口与缺模态规则

- Behavior：`scripts/sart_formal_analysis.py`；cohort 由 manifest 决定，不因 RGB/mmWave/NIR 缺失而缩小。
- NIR pupil-only：`scripts/nir_formal_pipeline.py` / `scripts/nir_build_analysis_tables.py`；不得重跑 YOLO/RITnet 以补下游统计。PIR/iris geometry 禁止；producer 的 ocular-aperture ratio 只保留为眼睛开合 QC，不是 EAR/blink/PERCLOS，也不是自动正式 endpoint。
- RGB downstream：消费既有 `*_face_raw.parquet`、`*_pose_landmarks.parquet`、`*_motion_raw.parquet`。某 session 整个 RGB 目录不存在时，应记录 `rgb_source_absent` / component `not_estimable`，而不是删除这个 session 或触发 producer。
- mmWave：正式处理权威在外部毫米波仓库；结构性缺失必须记为 missing/invalid，不允许填 0 或伪装成功，也不能缩减 Behavior/RGB/NIR cohort。
- Multimodal fusion：**disabled_deferred**；当前只保留接口与历史资产，不作为正式结果入口。未来 paired/complementarity 分析可使用明确的 common-available subset，但必须同时报告各模态覆盖率和原 governed cohort 分母。

## 6. 新电脑最可能遇到的报错与优先检查顺序

| 报错/现象 | 最可能原因 | 先检查什么 |
|---|---|---|
| `conda` 找不到 | PowerShell 未初始化 / PATH 不含 Conda | `conda init powershell`，重开终端 |
| `ModuleNotFoundError: attention_pipeline` | 当前环境没做 `pip install -e .`，或激活错环境 | `where.exe python`、`python -m pip show attention-analysis` |
| `未提供路径注册表` | 没设置 `ATTENTION_ANALYSIS_PATHS_CONFIG` / 没传 `--paths-config` | `echo $env:ATTENTION_ANALYSIS_PATHS_CONFIG` |
| `路径仍包含未解析环境变量` | `paths.local.yaml` 仍有 `${...}` 占位符 | 编辑本机路径或设置对应 env var |
| `路径注册表缺少逻辑路径` | 新旧 `paths.local.yaml` 键不一致 | 对照 `configs/paths.example.yaml` version 3 补齐键 |
| `不支持的路径注册表版本` | 拿了未来/错误 schema | 当前 loader 只支持 1/2/3；按模板重建 |
| NIR JSON 解析失败 / manifest root 不是 object | 把 legacy CSV 误填给 `nir_analysis_ready_source_manifest_json` | 检查两个 NIR manifest 键是否串线 |
| `pyarrow` / Parquet engine 缺失 | 环境未按 YAML 创建或 editable install 中断 | `python -c "import pyarrow; print(pyarrow.__version__)"` |
| RGB 某 session `source_directory_missing` | 该场本来没录上或数据盘未挂载 | 先确认是不是预期缺模态；不要把它当 participant 缺失 |
| RGB `motion_source_missing` / `pose_source_missing` / `blink not_estimable` | 目录存在但某 producer 文件缺失 | 核对 `*_motion_raw.parquet` / pose / face 文件；已完成组件不应被另一个组件阻断 |
| NIR `requested sessions absent from source manifest` | 指定了没有 NIR source 的 session，或 source manifest 不完整 | 检查 manifest availability；不要用 Behavior cohort 强行伪造 NIR 数据 |
| NIR/RGB session 名对不上 | 文件夹名、文件内 `session_id`、cohort ID canonicalization 不一致 | 同时核对目录名、文件名、表内 session 列 |
| questionnaire/repeat registry join 报冲突 | `session_id` 重复、participant_key 映射冲突、拿错版本 CSV | 先核对两个身份表版本，不要从 sub 编号猜 participant |
| `git pull --ff-only` 失败 | 本地有未提交改动或远端发生非快进变化 | `git status`，禁止无脑 `reset --hard`/force push |
| PermissionError / WinError 32 | 文件正被 Excel/Python/同步软件占用 | 关闭占用文件，确认输出目录可写 |
| 路径很长或含中文后某第三方工具失败 | Windows 旧组件/外部 executable 对 Unicode/长路径支持差 | 优先把仓库和运行输出放在较短父目录；不要因此硬改 science config |
| 旧 AMD/NVIDIA README 命令和当前正式入口冲突 | 正在读历史硬件运行记录 | 以本文件 + `configs/README.md` + `docs/060-formal-analysis/` 为准 |

## 7. 正式运行前最低检查

环境和路径能加载，只代表“机器可启动”，不代表正式科学运行已经完成。最低顺序：

```text
1. git / branch / HEAD 正确
2. 对应 Conda 环境正确
3. paths.local.yaml version 3 且无占位符
4. 输入根、cohort、identity registry、当前模态 source 可访问
5. python -m pytest -q 通过（历史本机数据测试可以明确 skip；skip 不是 pass）
6. representative session smoke
7. 检查 QC / failure tables / manifest
8. 才决定是否跑当前可用场次的正式全量
```

“某模态只有 39/44”本身不是代码失败；关键是这 5 场必须被明确标成 unavailable/invalid，并且其他模态仍保留完整 governed cohort。

## 8. 旧文档如何阅读

仓库保留 AMD/NVIDIA 历史运行记录用于 provenance，其中可能包含旧电脑绝对路径、旧环境名和旧分支名。它们不能覆盖本文件。冲突时执行优先级为：

```text
ANALYSIS_SETUP_FIRST.md
→ configs/paths.example.yaml + 本机 paths.local.yaml
→ scripts/setup_formal_environment.py
→ environments/*.yml
→ configs/README.md
→ 当前正式模态 config / runner
→ docs/060-formal-analysis/ 当前审计与迁移文档
→ 历史 README / 工作记录（仅用于追溯）
```
