# Attention-Analysis｜AMD / DirectML 主线

> 当前长期硬件主线：`amd-DirectML`。本分支统一维护 **NIR + Behavior + NIR-Behavior + RGB**。代码与文档在同一个 Git 仓库中，但不同模态继续使用彼此隔离的 Conda 环境；正式结果全部写到仓库外，Git 更新不会覆盖已经生成的数据。

> 当前 AMD 工作方式：**网页版 ChatGPT 负责规划、审计、代码与 GitHub 文档修改；AMD 本机 Codex 默认负责读取最新仓库、执行命令、检查输出并把日志/结果反馈回来。** 除非明确要求 Codex 修改代码，否则 Codex 不应在仓库内产生本地代码改动。

> 当前数据盘分工：**AMD 工作站连接的是约 44 名被试的数据盘；NVIDIA RTX 5070 工作站连接剩余约 72 名被试的数据盘。** 因此 AMD representative 继续以本机实际存在的 `sub-031` 等为准，不要把 NVIDIA 的 `sub-130` 队列配置复制到 AMD。

---

## 1. 先确认自己要做哪一种分析

不要打开终端以后固定激活某一个“大一统环境”。正确顺序是：

```text
网页版 ChatGPT 规划 / 修改 GitHub
→ AMD Codex 进入本地仓库
→ git status / pull --ff-only
→ 按模态激活对应环境
→ 环境与路径检查
→ dry-run / representative / tests
→ 正式执行
→ Codex 检查输出并反馈结果
```

| 任务 | AMD Conda 环境 | 主要入口 | 当前状态 |
|---|---|---|---|
| NIR 正式 YOLO + RITnet / RITnet full-class | `D:\CondaEnvs\nir-amd` | `runtime/nir-formal/` | 正式 runtime 已冻结；当前 full-class 从 `sub-032` 继续 |
| Behavior 正式 SART BB | `D:\CondaEnvs\attention-behavior` | `scripts/sart_formal_analysis.py` | 当前正式 v3.1.3 BB |
| NIR × Behavior 对齐 | `D:\CondaEnvs\attention-behavior` | `scripts/nir_behavior_alignment.py` | schema 2 已建立；当前仍保留 prototype safety gate |
| RGB Motion / Pose / sampling / QC | `D:\CondaEnvs\attention-rgb` | `scripts/rgb_analysis.py` | 科学层已进入主线；full-video formal 尚未整体冻结 |
| RGB Face DirectML | `D:\CondaEnvs\attention-face-directml` | `scripts/face_formal_dryrun_directml_v02.py` | Py-Feat 2.1.1 scientific core + DirectML 已验证；正式 full-video runner 待收口 |
| Py-Feat CPU/PyTorch reference / ONNX export | `D:\CondaEnvs\attention-face-pyfeat` | Py-Feat benchmark/export scripts | 参考与导出环境，不是 AMD 正式推理环境 |
| LibreFace 历史 reference | `D:\CondaEnvs\attention-face-libreface` | LibreFace scripts | 历史/参考，不是当前正式 backend |

---

## 2. AMD 固定路径与当前 44 人数据盘

### 2.1 仓库

```text
D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML
```

### 2.2 当前原始数据 discovery roots

AMD 当前连接的正式数据盘可能在 `E:` / `F:` 间交换，因此配置保留：

```text
E:/正式实验
F:/正式实验
E:/Data
F:/Data
```

当前 AMD 队列约 44 名，**实际运行以本机 discovery 结果为准**；不要按 NVIDIA 72 名队列硬编码 AMD subject list。

### 2.3 仓库外正式/分析输出

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

不要把正式结果重新改到仓库内 `outputs/`。Git 更新只管理代码/文档，不管理已经生成的正式数据。

---

## 3. 第一次没有本地仓库：clone

如果当前仓库已经存在，跳过本节。

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan"

git clone https://github.com/kyandi233-dev/Attention-Analysis.git Attention-Analysis-amd-DirectML
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"

git switch amd-DirectML
git pull --ff-only
git status --short --branch
git log -1 --oneline
```

正常分支：

```text
amd-DirectML
```

---

## 4. 每次开始工作：Codex 先检查 Git，再选环境

AMD 本机 Codex 每次新会话/新终端先执行：

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"

git status --short --branch
git branch --show-current
git fetch origin --prune
git log -1 --oneline
```

若工作区干净，再同步网页版 ChatGPT 已提交的最新代码：

```powershell
git switch amd-DirectML
git pull --ff-only
git status --short --branch
git log -1 --oneline
```

### 4.1 ChatGPT ↔ GitHub ↔ Codex 的标准协作链

```text
网页版 ChatGPT
  ↓ 规划 / 审计 / 代码与文档修改
GitHub amd-DirectML
  ↓
AMD Codex: git status → git pull --ff-only
  ↓
Codex 执行 / 检查 / 读取结果
  ↓
把日志、summary、manifest、QC 结果反馈给网页版 ChatGPT
```

**默认情况下 Codex 不负责擅自改代码。** ChatGPT 说“GitHub 已更新”以后，Codex 的职责是先 pull，再执行。

### 4.2 Codex 发现本地修改怎么办

不要直接：

```text
git reset --hard
git push --force
```

先看：

```powershell
git diff
git status --short
```

只有你明确要求 Codex 把本地代码修改提交时才：

```powershell
git add <明确需要提交的文件>
git diff --cached
git commit -m "<说明本次修改>"
git pull --ff-only
git push origin amd-DirectML
```

如果 `pull --ff-only` 因本地修改阻塞，先把修改内容反馈给网页版 ChatGPT/你本人，不要自行覆盖。

---

# 5. 第一次配置 AMD Conda 环境

已经存在且验证过的环境**不要因为 README 有创建命令就重建**。下面只用于新机器、环境丢失或明确要求重建。

## 5.1 NIR / DirectML：`nir-amd`

```powershell
conda create -p "D:\CondaEnvs\nir-amd" python=3.11 -y
conda activate "D:\CondaEnvs\nir-amd"

cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML\runtime\nir-formal"
python -m pip install -r requirements.txt

python -m pytest tests -q
python run_pipeline.py check-env
```

当前正式 NIR 工程参数：

```text
YOLO26n: 640×640 / FP32 / DirectML / fixed b8
RITnet:  640×400 / FP32 / DirectML / fixed b16
```

---

## 5.2 Behavior + NIR-Behavior：`attention-behavior`

```powershell
conda create -p "D:\CondaEnvs\attention-behavior" python=3.11 -y
conda activate "D:\CondaEnvs\attention-behavior"

cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pyarrow pytest
```

验证：

```powershell
python -c "import numpy,pandas,scipy,statsmodels,pyarrow; print('behavior env ok')"
python -m pytest `
  tests/test_behavior.py `
  tests/test_behavior_formal_bb.py `
  tests/test_behavior_phase2.py `
  tests/test_behavior_phase3.py `
  tests/test_behavior_phase4.py `
  tests/test_behavior_reporting.py `
  tests/test_nir_behavior_alignment.py -q
```

Behavior 与 NIR-Behavior 不需要 GPU，因此共用这一环境即可。

---

## 5.3 RGB core：`attention-rgb`

该环境负责 Motion、Pose、Face sampling、tracking/eyelid derived、QC。

已有验证环境：

```text
D:\CondaEnvs\attention-rgb
```

如确实需要从零新建：

```powershell
conda create -p "D:\CondaEnvs\attention-rgb" python=3.11 -y
conda activate "D:\CondaEnvs\attention-rgb"

cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pyarrow pytest mediapipe
```

验证：

```powershell
python -c "import cv2,numpy,pandas,pyarrow,mediapipe; print('rgb core env ok'); print('mediapipe=',mediapipe.__version__)"
python -m pytest `
  tests/test_rgb_discover.py `
  tests/test_rgb_gaps.py `
  tests/test_rgb_motion.py `
  tests/test_rgb_motion_qc.py `
  tests/test_rgb_motion_review.py `
  tests/test_rgb_paths.py `
  tests/test_rgb_pose.py `
  tests/test_rgb_pose_features.py `
  tests/test_rgb_pose_qc.py `
  tests/test_rgb_timeline.py -q
```

已有 `attention-rgb` 不要为了追最新版依赖而无理由重装；Pose/MediaPipe 版本变化可能改变数值，应先做 representative QC。

---

## 5.4 RGB Face DirectML：`attention-face-directml`

```powershell
conda create -p "D:\CondaEnvs\attention-face-directml" python=3.11 -y
conda activate "D:\CondaEnvs\attention-face-directml"
python -m pip install onnx onnxruntime-directml numpy pandas pyarrow opencv-python pillow
```

验证：

```powershell
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

必须包含：

```text
DmlExecutionProvider
```

不要同时安装普通 `onnxruntime` / `onnxruntime-gpu`。

当前 AMD Face DirectML model root：

```text
D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat
```

---

# 6. NIR：当前正式/补跑操作

## 6.1 每次进入 NIR

```powershell
conda activate "D:\CondaEnvs\nir-amd"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML\runtime\nir-formal"

python run_pipeline.py check-env
python -m pytest tests -q
```

## 6.2 完整 formal runtime dry-run

```powershell
python run_pipeline.py discover --formal-only
python run_formal_batch.py --dry-run
```

已有历史正式结果，不应无原因重新跑完整 YOLO + RITnet。

## 6.3 当前 RITnet full-class：从 sub-032 继续

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

先 dry-run：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "$subjectArg" `
  --device 0 `
  --postprocess-workers 4 `
  --dry-run
```

确认后：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "$subjectArg" `
  --device 0 `
  --postprocess-workers 4
```

该 extension 复用既有 `frame_idx + ROI`，**不重跑 YOLO、不覆盖旧 `eyes.csv`**。

---

# 7. Behavior：正式 SART BB

```powershell
conda activate "D:\CondaEnvs\attention-behavior"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"
```

先检查配置：

```powershell
Select-String -Path ".\configs\behavior_formal.yaml" -Pattern "roots:|output_root|min_subject_number"
```

正式执行：

```powershell
python scripts/sart_formal_analysis.py --stage all
```

正式输出：

```text
D:\_AttentionData\Beijing-Behavior\formal-v1
```

历史 `scripts/sart_bbb_v3_0_analysis.py` 仅用于 v3.0 BBB 复现，不是当前正式 Behavior。

---

# 8. NIR × Behavior 对齐

```powershell
conda activate "D:\CondaEnvs\attention-behavior"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"

python -m pytest tests/test_nir_behavior_alignment.py -q
python scripts/nir_behavior_alignment.py --subjects sub-031
```

当前 `configs/nir_behavior_alignment.yaml` 仍保留 `sub-031` prototype safety gate；在正式解除 gate 前不要擅自扩展到 44 人。

该步骤直接消费已经生成的 NIR full-class + Behavior，**不需要重新运行 RITnet**。

输出：

```text
D:\_AttentionData\Beijing-NIR\analysis\nir-behavior-v1
```

---

# 9. RGB：当前不是“一条命令全量”

当前 RGB scientific layer 已经在 `amd-DirectML`，但 Motion / Pose / Face full-video formal runner、blink event 和 `perclos80_proxy` 最终规则尚未全部冻结。

现在允许：

- audit / gaps；
- representative Motion / Pose；
- Face 15 Hz dry-run；
- tracking / eyelid；
- QC；
- engineering/parity 检查。

**现在不要启动 AMD 44 人 RGB full cohort。**

---

## 9.1 RGB audit / gaps

```powershell
conda activate "D:\CondaEnvs\attention-rgb"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"

python scripts/rgb_analysis.py --stage audit
python scripts/rgb_analysis.py --stage gaps
```

---

## 9.2 Motion representative：sub-031

```powershell
python scripts/rgb_analysis.py --stage motion --subject sub-031
python scripts/rgb_analysis.py --stage motion-qc --subject sub-031
python scripts/rgb_analysis.py --stage motion-review --subject sub-031
```

---

## 9.3 Pose representative：sub-031

```powershell
python scripts/rgb_analysis.py --stage pose --subject sub-031
python scripts/rgb_analysis.py --stage pose-qc --subject sub-031
python scripts/rgb_analysis.py --stage pose-features --subject sub-031
```

---

# 10. RGB Face：AMD DirectML representative

## 10.1 生成 timestamp-driven 15 Hz dry-run sample

```powershell
conda activate "D:\CondaEnvs\attention-rgb"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"

python scripts/face_formal_dryrun_sample.py --subject sub-031
```

默认 sample：

```text
D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031
```

## 10.2 DirectML optimized v02

```powershell
conda activate "D:\CondaEnvs\attention-face-directml"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"

$SAMPLE_DIR = "D:\_AttentionData\Beijing-RGB\_test\face-formal-dryrun\sub-031"
$MODEL_DIR = "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat"

python scripts/face_formal_dryrun_directml_v02.py `
  --sample-dir "$SAMPLE_DIR" `
  --model-dir "$MODEL_DIR" `
  --output-dir "$SAMPLE_DIR\directml-v02" `
  --retinaface-batch 8 `
  --multitask-batch 16 `
  --prefetch-batches 2
```

当前第一档优化保持 scientific core 不变：direct AVI decode + prefetch + RetinaFace B8 + multitask B16。

## 10.3 tracking / eyelid

```powershell
conda activate "D:\CondaEnvs\attention-rgb"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"

python scripts/face_derive_tracking_eyelid_v02.py `
  --raw "$SAMPLE_DIR\directml-v02\pyfeat_dml_raw.parquet" `
  --frame-manifest "$SAMPLE_DIR\sub-031_face-dryrun_frames.csv" `
  --output-dir "$SAMPLE_DIR\derived"
```

## 10.4 QC

```powershell
python scripts/face_qc_visualize_v03.py `
  --tracks "$SAMPLE_DIR\derived\face_tracks.parquet" `
  --eye "$SAMPLE_DIR\derived\eye_features.parquet" `
  --sample-manifest "$SAMPLE_DIR\sub-031_face-dryrun_manifest.json" `
  --frame-manifest "$SAMPLE_DIR\sub-031_face-dryrun_frames.csv" `
  --output-dir "$SAMPLE_DIR\qc" `
  --fps 15
```

QC 读取已保存 raw/derived + 原 AVI，不重新推理。

AMD 当前已完成 `sub-031` representative Face 验证；后续 gap-stress、blink/PERCLOS 和 full-video orchestration 继续收口。NVIDIA 改用 `sub-130` 是因为两机数据盘不同，不代表 AMD representative 要跟着改。

---

# 11. 每次运行结束：Codex 检查什么

Codex 不应只汇报“命令跑完了”。至少检查：

1. 命令退出码；
2. summary / manifest / completion 是否存在；
3. processed rows / expected rows；
4. GPU provider 是否为目标 provider；
5. Face/Motion/Pose coverage / missing / multi-face / gaps；
6. QC 图片/视频是否可读且肉眼正常；
7. 输出是否在仓库外正确目录；
8. `git status --short --branch` 是否仍然干净；
9. 把关键 JSON/CSV summary 或日志片段反馈给网页版 ChatGPT，再决定下一步。

---

# 12. 当前 AMD 与 NVIDIA 数据分工

```text
AMD 数据盘：约 44 名
NVIDIA 数据盘：剩余约 72 名
```

两台机器并不是用同一块正式数据盘，因此：

- AMD representative 可以是 `sub-031`；
- NVIDIA representative 改为 `sub-130`；
- 不同被试不能称为逐帧 cross-device parity；
- 真正需要同帧 AMD↔NVIDIA parity 时，只复制少量 representative sample，不移动完整数据盘。

当前共同 scientific anchor：

```text
Py-Feat 2.1.1 Detectorv2 scientific core
```

AMD 已有 CPU-reference ↔ DirectML parity；NVIDIA 将做 `sub-130` CPU-reference ↔ CUDA parity。

---

# 13. 最短日常操作清单

## NIR

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"
git status --short --branch
git pull --ff-only
conda activate "D:\CondaEnvs\nir-amd"
cd runtime\nir-formal
python run_pipeline.py check-env
```

## Behavior

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"
git status --short --branch
git pull --ff-only
conda activate "D:\CondaEnvs\attention-behavior"
python scripts/sart_formal_analysis.py --stage all
```

## RGB core

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"
git status --short --branch
git pull --ff-only
conda activate "D:\CondaEnvs\attention-rgb"
python scripts/rgb_analysis.py --stage audit
```

## Face DirectML representative

```powershell
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"
git status --short --branch
git pull --ff-only

conda activate "D:\CondaEnvs\attention-rgb"
python scripts/face_formal_dryrun_sample.py --subject sub-031

conda activate "D:\CondaEnvs\attention-face-directml"
```

后续使用第 10 节 DirectML v02 命令。

---

# 14. 快速导航

| 内容 | 入口 |
|---|---|
| AMD NIR 安装与正式运行 | `runtime/nir-formal/INSTALL.md` |
| NIR 故障恢复 | `runtime/nir-formal/RUNBOOK.md` |
| RITnet full-class | `runtime/nir-formal/RITNET_FULLCLASS_EXTENSION.md` |
| Behavior | `docs/030-behavior/` |
| NIR × Behavior | `docs/030-behavior/035-NIR与正式SART行为数据对齐分析方法.md` |
| RGB | `docs/040-rgb/` |
| RGB 当前配置 | `configs/rgb_analysis.yaml` |
| scripts 索引 | `scripts/README.md` |
| 技术决策 | `docs/050-decisions/` |
| 历史工作记录 | `docs/工作记录/` |
| 仓库规则 | `AGENTS.md` |

---

# 15. 当前边界

- NIR 历史正式全量已完成；当前做 post-hoc full-class extension。
- Behavior 当前正式口径为 FocusWave v3.1.3、两个 B block。
- NIR-Behavior schema 2 已建立，但仍有 prototype safety gate。
- RGB Face backend / 15 Hz / AMD DirectML first-tier optimization 已验证。
- RGB 全模态 full-video formal runner、gap stress、blink event、`perclos80_proxy` 尚未全部冻结。
- **现在不要启动 AMD 44 人 RGB full cohort。**
- `rgb-dev` / `rgb-nvidia-cuda` 仍作为开发期保险分支，长期维护目标仍是 `amd-DirectML` 与 `nvidia-cuda`。
- 日期型工作记录和历史 provenance 不追溯改写。
