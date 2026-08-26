# Attention-Analysis｜AMD / DirectML 主线

> 当前长期硬件主线：`amd-DirectML`。本分支统一维护 **NIR + Behavior + NIR-Behavior + RGB**。代码与文档在同一个 Git 仓库中，但不同模态继续使用彼此隔离的 Conda 环境；正式结果全部写到仓库外，Git 更新不会覆盖已经生成的数据。

> 当前 AMD 工作方式：**网页版 ChatGPT 负责规划、审计与 GitHub 代码/文档修改；AMD 本机 Codex 默认负责读取最新仓库、执行命令、检查输出和反馈结果。** 除非明确要求 Codex 修改代码，否则 Codex 不应在仓库内产生本地改动。

## 1. 先确认自己要做哪一种分析

不要打开终端后固定先激活某一个环境。正确顺序是：**先进入仓库并同步 Git → 再按模态激活对应环境 → 再进入对应运行目录。**

| 任务 | Conda 环境 | 主要入口 | 正式/开发状态 |
|---|---|---|---|
| NIR 正式 YOLO + RITnet / RITnet full-class | `D:\CondaEnvs\nir-amd` | `runtime/nir-formal/` | 正式 runtime 已冻结；当前 full-class 从 `sub-032` 继续 |
| Behavior 正式 SART BB | `D:\CondaEnvs\attention-behavior` | `scripts/sart_formal_analysis.py` | 当前正式 v3.1.3 BB 分析 |
| NIR × Behavior 对齐 | `D:\CondaEnvs\attention-behavior` | `scripts/nir_behavior_alignment.py` | schema 2 已建立；当前仍保留 prototype safety gate |
| RGB Motion / Pose / sampling / QC | `D:\CondaEnvs\attention-rgb` | `scripts/rgb_analysis.py` | 科学层已进入主线；正式 full-video runner 尚未全部冻结 |
| RGB Face DirectML 正式化验证 | `D:\CondaEnvs\attention-face-directml` | `scripts/face_formal_dryrun_directml_v02.py` 等 | Py-Feat 2.1.1 + DirectML 已冻结；full-video formal runner 尚待收口 |
| Py-Feat 官方 PyTorch reference / export | `D:\CondaEnvs\attention-face-pyfeat` | `scripts/face_benchmark_pyfeat.py` / export | 参考与导出用途，不是 AMD 正式推理环境 |
| LibreFace 历史 reference / export | `D:\CondaEnvs\attention-face-libreface` | LibreFace scripts | 参考/历史用途，不是当前正式 backend |

## 2. AMD 固定路径与仓库外输出

仓库：

```text
D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML
```

当前原始数据 discovery 使用两块外接盘的候选根；盘符可能在 `E:` / `F:` 间交换：

```text
E:/正式实验
F:/正式实验
E:/Data
F:/Data
```

仓库外正式/分析输出：

```text
NIR formal/full-class:
D:\_AttentionData\Beijing-NIR\amd-directml

Behavior formal:
D:\_AttentionData\Beijing-Behavior\formal-v1

NIR-Behavior:
D:\_AttentionData\Beijing-NIR\analysis\nir-behavior-v1

RGB:
D:\_AttentionData\Beijing-RGB
```

不要把正式结果重新改到仓库内 `outputs/`。仓库内输出目录只允许作为临时/兼容路径，正式结果以以上仓库外路径为准。

## 3. 每次开始工作：先做 Git 检查，再选环境

新 PowerShell / VS Code Terminal 首先执行：

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"

git status --short --branch
git branch --show-current
git fetch origin --prune
git log -1 --oneline
```

正常应位于：

```text
amd-DirectML
```

若工作区干净，再同步网页版 ChatGPT 已提交到 GitHub 的最新代码：

```powershell
git switch amd-DirectML
git pull --ff-only
git status --short --branch
git log -1 --oneline
```

### AMD 当前 ChatGPT ↔ Codex 协作规则

正常工作流：

```text
网页版 ChatGPT
  ↓ 规划 / 审计 / 修改 GitHub
GitHub amd-DirectML
  ↓ Codex 在本机 git pull --ff-only
AMD Codex
  ↓ 执行 / 检查 / 返回日志与结果
网页版 ChatGPT
```

因此 ChatGPT 告诉你“代码已提交”以后，Codex 应先检查：

```powershell
git status --short --branch
```

如果干净，再：

```powershell
git pull --ff-only
```

如果 Codex 意外产生本地修改，**不要直接 pull、不要 reset --hard、不要 force push**。先查看：

```powershell
git diff
git status --short
```

只有你明确要求 Codex 把本地代码改动提交到 GitHub 时才使用：

```powershell
git add <明确需要提交的文件>
git diff --cached
git commit -m "<说明本次修改>"
git push origin amd-DirectML
```

不要使用：

```text
git push --force
git reset --hard
```

除非已经明确知道会删除/覆盖什么并获得你的许可。

## 4. 第一次配置 AMD 环境

已经存在且验证过的环境**不要为了照 README 再重建**。下面只用于新机器、环境丢失或明确需要重建时。

### 4.1 NIR / DirectML

```powershell
conda create -p "D:\CondaEnvs\nir-amd" python=3.11 -y
conda activate "D:\CondaEnvs\nir-amd"

cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML\runtime\nir-formal"
python -m pip install -r requirements.txt

python -m pytest tests -q
python run_pipeline.py check-env
```

当前 NIR 正式组合固定为：YOLO26n `640×640 / FP32 / DirectML / fixed b8`，RITnet `640×400 / FP32 / DirectML / fixed b16`。

### 4.2 Behavior + NIR-Behavior

Behavior 与下游 NIR-Behavior 不需要 GPU，建议共用一个轻量独立环境：

```powershell
conda create -p "D:\CondaEnvs\attention-behavior" python=3.11 -y
conda activate "D:\CondaEnvs\attention-behavior"

cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pyarrow pytest

python -c "import numpy,pandas,scipy,statsmodels,pyarrow; print('behavior env ok')"
python -m pytest tests -q
```

正式结果仍由 `configs/behavior_formal.yaml` 和 `configs/nir_behavior_alignment.yaml` 指向仓库外路径。

### 4.3 RGB 主环境：Motion / Pose / sampling / QC

AMD 已有验证环境：

```text
D:\CondaEnvs\attention-rgb
```

若确实需要从零新建：

```powershell
conda create -p "D:\CondaEnvs\attention-rgb" python=3.11 -y
conda activate "D:\CondaEnvs\attention-rgb"

cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pyarrow pytest mediapipe

python -c "import cv2,numpy,pandas,pyarrow,mediapipe; print('rgb core env ok')"
```

新建环境后先跑 RGB tests / representative pilot，不要直接全量。已有 `attention-rgb` 环境不要仅为了升级依赖而重装；Pose/MediaPipe 版本变化可能影响数值，应先做代表性 parity/QC。

### 4.4 RGB Face DirectML

当前 AMD Face 正式化使用独立 DirectML runtime：

```powershell
conda create -p "D:\CondaEnvs\attention-face-directml" python=3.11 -y
conda activate "D:\CondaEnvs\attention-face-directml"
python -m pip install onnx onnxruntime-directml numpy pandas pyarrow opencv-python pillow
```

检查：

```powershell
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

必须包含：

```text
DmlExecutionProvider
```

该环境不要同时安装普通 `onnxruntime` / `onnxruntime-gpu`。

## 5. NIR：当前正式/补跑操作

### 5.1 每次进入 NIR

```powershell
conda activate "D:\CondaEnvs\nir-amd"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML\runtime\nir-formal"

python run_pipeline.py check-env
python -m pytest tests -q
```

### 5.2 完整正式 runtime 的 dry-run

```powershell
python run_pipeline.py discover --formal-only
python run_formal_batch.py --dry-run
```

只有在明确需要重新运行完整 YOLO + RITnet 时才执行完整 formal；当前已有历史正式结果，不应无原因重跑。

### 5.3 当前 RITnet full-class extension：从 sub-032 继续

```powershell
$subjects = Get-ChildItem "D:\_AttentionData\Beijing-NIR\amd-directml" -Directory |
  ForEach-Object {
    if ($_.Name -match '^(sub-(\d{3}))_formal_') {
      [PSCustomObject]@{ Subject=$matches[1]; Number=[int]$matches[2] }
    }
  } |
  Where-Object { $_.Number -ge 32 } |
  Sort-Object Number |
  Select-Object -ExpandProperty Subject -Unique

$subjectArg = $subjects -join ","
$subjectArg
```

先检查：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "$subjectArg" `
  --device 0 `
  --postprocess-workers 4 `
  --dry-run
```

确认后正式继续：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "$subjectArg" `
  --device 0 `
  --postprocess-workers 4
```

该 extension **不重新运行 YOLO**，复用原正式 `frame_idx + ROI`，不覆盖旧 `eyes.csv`。

## 6. Behavior：正式 SART BB

```powershell
conda activate "D:\CondaEnvs\attention-behavior"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"

python scripts/sart_formal_analysis.py --stage all
```

正式输入来自当前有效的 `E:/F:` 候选根，正式输出：

```text
D:\_AttentionData\Beijing-Behavior\formal-v1
```

运行前建议先检查配置：

```powershell
Select-String -Path ".\configs\behavior_formal.yaml" -Pattern "roots:|output_root|min_subject_number"
```

历史 `scripts/sart_bbb_v3_0_analysis.py` 只用于 v3.0 BBB 复现，**不是当前正式 Behavior**。

## 7. NIR × Behavior 对齐

```powershell
conda activate "D:\CondaEnvs\attention-behavior"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"

python scripts/nir_behavior_alignment.py --subjects sub-031
```

当前 `configs/nir_behavior_alignment.yaml` 仍保留 `sub-031` prototype safety gate；在正式解除 gate 前不要擅自改成全队列。该步骤直接消费已经生成的 NIR full-class + Behavior，**不需要重新运行 RITnet**。

## 8. RGB：先明确当前不是“一条命令全量”

当前 RGB 科学层已经在 `amd-DirectML`，但 Motion / Pose / Face full-video formal runner、blink event 和 `perclos80_proxy` 最终规则尚未全部冻结。因此：

- 可以运行 audit、gap、representative Motion/Pose、Face dry-run、tracking/eyelid、QC；
- **现在不要启动 44 人 RGB 全量**；
- full cohort 必须等 formal runner + completion/resume + QC 冻结后再开始。

### 8.1 RGB audit / timestamp gaps

```powershell
conda activate "D:\CondaEnvs\attention-rgb"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"

python scripts/rgb_analysis.py --stage audit
python scripts/rgb_analysis.py --stage gaps
```

### 8.2 Motion representative

```powershell
python scripts/rgb_analysis.py --stage motion --subject sub-031
python scripts/rgb_analysis.py --stage motion-qc --subject sub-031
python scripts/rgb_analysis.py --stage motion-review --subject sub-031
```

### 8.3 Pose representative

```powershell
python scripts/rgb_analysis.py --stage pose --subject sub-031
python scripts/rgb_analysis.py --stage pose-qc --subject sub-031
python scripts/rgb_analysis.py --stage pose-features --subject sub-031
```

### 8.4 Face 15 Hz representative dry-run

抽取/定位 dry-run windows 使用 RGB 主环境：

```powershell
conda activate "D:\CondaEnvs\attention-rgb"
python scripts/face_formal_dryrun_sample.py --subject sub-031
```

DirectML Face 推理切换到：

```powershell
conda activate "D:\CondaEnvs\attention-face-directml"
```

当前 AMD Face backend 已冻结为 Py-Feat 2.1.1 scientific core + ONNX Runtime DirectML；正式采样率为 timestamp-driven 15 Hz。sub-031 已完成代表性验证，后续仍需 sub-033 gap stress 与 blink/PERCLOS proxy 冻结。

详细 Face 命令、model dir、QC 与历史 Gate 见：

- `docs/040-rgb/`
- `scripts/README.md`
- `docs/050-decisions/054-RGB-Face-Backend冻结.md`
- `docs/050-decisions/055-RGB-Face-15Hz采样频率冻结.md`

## 9. 结果检查原则

每次正式/代表性运行结束都先检查：

1. 命令退出码是否正常；
2. summary / manifest / completion 是否存在；
3. processed rows 是否等于 expected rows；
4. GPU provider 是否确实为目标 provider；
5. QC 图片/视频是否肉眼正常；
6. 仓库 `git status --short --branch` 是否仍然干净；
7. 正式结果是否写在仓库外正确根目录。

不要因为程序完成就直接进入全队列。

## 10. 快速导航

| 内容 | 入口 |
|---|---|
| AMD NIR 安装与正式运行 | `runtime/nir-formal/INSTALL.md` |
| NIR 故障恢复 | `runtime/nir-formal/RUNBOOK.md` |
| RITnet full-class | `runtime/nir-formal/RITNET_FULLCLASS_EXTENSION.md` |
| Behavior | `docs/030-behavior/` |
| NIR × Behavior | `docs/030-behavior/035-NIR与正式SART行为数据对齐分析方法.md` |
| RGB | `docs/040-rgb/` |
| RGB 当前配置 | `configs/rgb_analysis.yaml` |
| 所有 scripts 索引 | `scripts/README.md` |
| 技术决策 | `docs/050-decisions/` |
| 历史工作记录 | `docs/工作记录/` |
| 仓库规则 | `AGENTS.md` |

## 11. 当前边界

- NIR 历史正式全量已完成；当前做 post-hoc full-class extension。
- Behavior 当前正式口径为 FocusWave v3.1.3、两个 B block。
- NIR-Behavior schema 2 已建立，但当前仍保留 prototype safety gate。
- RGB Face backend 和 15 Hz 已冻结；AMD representative engineering optimization 已验证，但 RGB 全模态 full-video formal runner 尚未整体冻结。
- AMD RGB 当前队列约 44 名，以本机实际 discovery 为准。
- `rgb-dev` / `rgb-nvidia-cuda` 仍作为开发期保险分支；长期维护目标仍是 `amd-DirectML` 与 `nvidia-cuda` 两条硬件主线。
- 日期型工作记录和历史 provenance 不追溯改写。