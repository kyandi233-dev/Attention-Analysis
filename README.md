# Attention-Analysis｜NVIDIA / CUDA 主线

> 当前长期硬件主线：`nvidia-cuda`。本分支统一维护 **NIR + Behavior + NIR-Behavior + RGB**。代码和文档在同一个 Git 仓库中，但不同模态使用彼此隔离的 Conda 环境；正式结果全部写到仓库外，Git 更新不会覆盖已经生成的数据。

> 当前数据盘分工：**NVIDIA RTX 5070 工作站连接的是剩余约 72 名被试的数据盘；AMD 工作站连接另一块约 44 名被试的数据盘。** NVIDIA 当前原始数据根为 `J:\Data`。因此 NVIDIA RGB representative Face Gate 使用 `sub-130`，不要再默认使用 AMD 盘上的 `sub-031` / `sub-033`。

---

## 1. 先判断你要做哪一种分析

不要为了方便把所有任务塞进同一个 Python 环境。正确顺序是：

```text
进入仓库并检查 Git
→ 拉取最新 nvidia-cuda
→ 根据任务激活对应 Conda 环境
→ 检查数据根 / GPU / 配置
→ dry-run / tests
→ 正式运行
```

| 任务 | NVIDIA Conda 环境 | 主要入口 | 当前状态 |
|---|---|---|---|
| NIR 正式 CUDA runtime / RITnet full-class | `D:\CondaEnvs\nir-nvidia` | `runtime/nir-formal/` | 当前正式 launcher 固定使用该 Conda 解释器；队列进度见 `docs/020-nir/NIR_STATUS_TODAY.md` |
| Behavior 正式 SART BB | `D:\conda_envs\attention-behavior` | `scripts/sart_formal_analysis.py` | **需要首次创建**；正式 v3.1.3 BB |
| NIR × Behavior 对齐 | `D:\conda_envs\attention-behavior` | `scripts/nir_behavior_alignment.py` | 与 Behavior 共用环境；schema 2 已建立，当前 config 仍有 prototype gate |
| RGB Motion / Pose / sampling / QC | `D:\conda_envs\attention-rgb` | `scripts/rgb_analysis.py` | **需要首次创建**；共享科学层已进入主线 |
| RGB Face Py-Feat native CUDA | `D:\conda_envs\attention-face-cuda` | `scripts/face_formal_dryrun_cuda.py` | **需要首次创建**；sub-130 3600-frame Gate 待实机运行 |

NIR 已有环境不要为了统一命名而重建。Behavior、RGB core、Face CUDA 从现在开始固定使用以上三个路径；建好一次以后不要每次重建。

---

## 2. NVIDIA 固定路径

### 2.1 仓库

```text
D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda
```

### 2.2 当前 NVIDIA 原始数据根

```text
J:\Data
```

典型正式被试：

```text
J:\Data\sub-XXX_\
├── beh\
│   ├── master_timeline.csv
│   ├── sub-XXX_Block1_B_beh.csv
│   └── sub-XXX_Block2_B_beh.csv
├── nir\
│   ├── sub-XXX_nir.avi
│   └── sub-XXX_nir_timestamps.csv
└── rgb\
    ├── sub-XXX_rgb.avi
    └── sub-XXX_rgb_timestamps.csv
```

### 2.3 仓库外输出

```text
NIR formal/full-class:
D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR

Behavior formal:
D:\Project\厚粲杯\11_数据\02_Attention-Analysis_nvidia-cuda_formal_Behavior

NIR-Behavior:
D:\Project\厚粲杯\11_数据\03_Attention-Analysis_nvidia-cuda_NIR-Behavior

RGB:
D:\Project\厚粲杯\11_数据\04_Attention-Analysis_nvidia-cuda_RGB
```

不要把正式结果改回仓库内 `outputs/`。Git branch switch / pull / merge 不应成为正式数据管理手段。

---

## 3. 第一次没有本地仓库：clone

如果仓库已经存在，跳过本节。

```powershell
cd "D:\Project\厚粲杯\08_算法"

git clone https://github.com/kyandi233-dev/Attention-Analysis.git 01_Attention-Analysis_nvidia-cuda
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"

git switch nvidia-cuda
git pull --ff-only
git status --short --branch
git log -1 --oneline
```

正常当前分支必须是：

```text
nvidia-cuda
```

---

## 4. 每次打开新终端：先同步 Git，再激活环境

先进入仓库：

```powershell
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"

git status --short --branch
git branch --show-current
git fetch origin --prune
git log -1 --oneline
```

如果工作区干净：

```powershell
git switch nvidia-cuda
git pull --ff-only
git status --short --branch
git log -1 --oneline
```

### 如果 `git status` 有本地修改

不要直接：

```text
git reset --hard
git push --force
```

先检查：

```powershell
git diff
git status --short
```

如果这些修改是需要保留并提交的本机代码修改：

```powershell
git add <明确需要提交的文件>
git diff --cached
git commit -m "<说明本次修改>"
git push origin nvidia-cuda
```

如果只是运行产生的正式数据，它们本来就应该在仓库外，不应出现在 Git working tree。

### 每次准备 push 前建议再做一次

```powershell
git status --short --branch
git diff --cached
git pull --ff-only
git push origin nvidia-cuda
```

不要对长期硬件主线做 force push。

---

# 5. 第一次搭建 Conda 环境

## 5.1 NIR：沿用已有 `nir-nvidia`

当前 RTX 5070 已经有：

```text
D:\CondaEnvs\nir-nvidia
```

激活：

```powershell
conda activate "D:\CondaEnvs\nir-nvidia"
where.exe python
python --version
```

验证 NVIDIA / PyTorch / ORT CUDA：

```powershell
nvidia-smi
python -c "import torch; print('torch=',torch.__version__); print('cuda=',torch.cuda.is_available()); print('torch_cuda=',torch.version.cuda); print('gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python -c "import onnxruntime as ort; print('ort=',ort.__version__); print('providers=',ort.get_available_providers())"
```

NIR full-class 必须看到：

```text
CUDAExecutionProvider
```

完整 NIR 环境重建只在 `eye-ai` 丢失时执行，具体按：

- `runtime/nir-formal/INSTALL.md`
- `NVIDIA-RITnet全分类补跑使用说明.md`

不要为了 Behavior/RGB 往 `eye-ai` 里继续塞依赖。

---

## 5.2 Behavior：首次创建 `attention-behavior`

Behavior 不需要 GPU。首次创建：

```powershell
conda create -p "D:\conda_envs\attention-behavior" python=3.11 -y
conda activate "D:\conda_envs\attention-behavior"

cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pyarrow pytest
```

验证：

```powershell
where.exe python
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

确认当前配置仍指向 NVIDIA 数据盘/仓库外输出：

```powershell
Select-String -Path ".\configs\behavior_formal.yaml" -Pattern "J:/Data|output_root|min_subject_number"
Select-String -Path ".\configs\nir_behavior_alignment.yaml" -Pattern "nir_source_roots|output_root|include"
```

以后每次只需要：

```powershell
conda activate "D:\conda_envs\attention-behavior"
```

不需要重新 `conda create`。

---

## 5.3 RGB core：首次创建 `attention-rgb`

这个环境负责：

- RGB audit；
- timestamp gaps；
- Motion；
- Pose；
- Face timestamp sampling；
- tracking / eyelid derived；
- QC rendering。

首次创建：

```powershell
conda create -p "D:\conda_envs\attention-rgb" python=3.11 -y
conda activate "D:\conda_envs\attention-rgb"

cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pyarrow pytest mediapipe
```

验证：

```powershell
where.exe python
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

MediaPipe Pose model由 pipeline 在需要时按配置下载到 test/model 路径。**正式全量前必须记录 NVIDIA 当前 MediaPipe 版本，并与 AMD 已验收行为做 representative QC；不要因为 `pip install -U` 随意升级以后直接全量。**

---

## 5.4 RGB Face CUDA：首次创建 `attention-face-cuda`

这个环境只负责 Py-Feat 2.1.1 native PyTorch CUDA Face 推理，不承担 Pose/Motion。

Py-Feat 2.1.1 要求 Python >= 3.11。当前项目固定：

```text
py-feat == 2.1.1
```

首次创建：

```powershell
conda create -p "D:\conda_envs\attention-face-cuda" python=3.11 -y
conda activate "D:\conda_envs\attention-face-cuda"

python -m pip install --upgrade pip
```

先安装 NVIDIA CUDA PyTorch。RTX 5070 使用官方 CUDA wheel；当前项目建议优先使用 CUDA 13.0 wheel：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

再安装冻结 Py-Feat 和输出依赖：

```powershell
python -m pip install "py-feat==2.1.1" pandas pyarrow opencv-python pillow
```

让该环境能读取本仓库包，但不要让 editable install 重写已装好的 CUDA Torch：

```powershell
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
python -m pip install -e . --no-deps
```

验证：

```powershell
nvidia-smi
python -c "import torch; print('torch=',torch.__version__); print('cuda=',torch.cuda.is_available()); print('torch_cuda=',torch.version.cuda); print('gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python -c "import feat, importlib.metadata as m; print('py-feat=',m.version('py-feat'))"
python -c "import pandas,pyarrow,cv2; print('face cuda env ok')"
```

必须满足：

```text
torch.cuda.is_available() == True
py-feat == 2.1.1
GPU == NVIDIA GeForce RTX 5070
```

如果官方 PyTorch 安装页面未来不再提供 `cu130`，不要自行换成 CPU wheel；使用当时官方仍支持 RTX 5070 的 CUDA wheel，并把 `torch.__version__ / torch.version.cuda / GPU` 记录进运行 manifest。

---

# 6. NIR：RTX 5070 当前正式 full-class 补跑

每次进入：

```powershell
conda activate "D:\CondaEnvs\nir-nvidia"
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda\runtime\nir-formal"
```

检查：

```powershell
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
Test-Path "J:\Data"
python -m pytest tests -q
python run_pipeline.py check-env
```

当前任务只补 RITnet 四分类，**不重跑 YOLO**。

先 dry-run：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --device 0 `
  --postprocess-workers 4 `
  --dry-run
```

dry-run discovery 应以当前 NVIDIA 数据盘约 72 名队列为准，不要用 AMD 44 名被试列表覆盖它。

首次新代码验收可选一名实际存在的 NVIDIA 被试：

```powershell
$TEST_SUBJECT = "sub-XXX"
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --subjects $TEST_SUBJECT `
  --device 0 `
  --postprocess-workers 4 `
  --validate-pupil
```

验收后正式全队列：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --device 0 `
  --postprocess-workers 4
```

完整恢复/lock/completion 规则见：

**[`NVIDIA-RITnet全分类补跑使用说明.md`](NVIDIA-RITnet全分类补跑使用说明.md)**

---

# 7. Behavior：正式 SART BB

当前 NVIDIA Behavior 环境首次建立完成以后：

```powershell
conda activate "D:\conda_envs\attention-behavior"
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
```

运行前检查数据盘与配置：

```powershell
Test-Path "J:\Data"
Select-String -Path ".\configs\behavior_formal.yaml" -Pattern "J:/Data|output_root|min_subject_number"
```

正式执行：

```powershell
python scripts/sart_formal_analysis.py --stage all
```

正式输出：

```text
D:\Project\厚粲杯\11_数据\02_Attention-Analysis_nvidia-cuda_formal_Behavior
```

历史：

```text
scripts/sart_bbb_v3_0_analysis.py
```

只用于 FocusWave v3.0 BBB 历史复现，**不是当前正式 Behavior**。

如果未来需要只处理一部分 NVIDIA 被试，应优先使用脚本/配置实际支持的 include/subject 参数，而不是临时搬动 J 盘目录。

---

# 8. NIR × Behavior 对齐

使用 Behavior 环境：

```powershell
conda activate "D:\conda_envs\attention-behavior"
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
```

先跑测试：

```powershell
python -m pytest tests/test_nir_behavior_alignment.py -q
```

当前配置：

```text
configs/nir_behavior_alignment.yaml
```

当前仍保留历史 prototype `include: sub-031` safety gate。由于 NVIDIA 当前正式数据盘与 AMD 数据盘分离，**不要直接把下面命令误当成 72 人正式运行入口**：

```powershell
python scripts/nir_behavior_alignment.py --subjects sub-031
```

在正式解除/改写该 safety gate 前，NVIDIA 只把这一模块视为“代码和科学层已同步、正式队列尚未重新冻结”。解除 gate 后，输出固定到：

```text
D:\Project\厚粲杯\11_数据\03_Attention-Analysis_nvidia-cuda_NIR-Behavior
```

该步骤消费已经完成的 NIR full-class + Behavior，不需要再次运行 RITnet。

---

# 9. RGB：当前使用方式

当前 RGB scientific layer 已进入 `nvidia-cuda`，但**还不是一条可以直接跑 72 人全量的 formal runner**。目前允许：

- audit；
- gaps；
- representative Motion；
- representative Pose；
- Face 15 Hz dry-run；
- native CPU/CUDA parity；
- tracking / eyelid；
- QC。

在 Face parity、gap stress、blink/PERCLOS proxy 和 full-video completion/resume 收口前，不启动 72 人 RGB 全量。

---

## 9.1 RGB audit / gaps

```powershell
conda activate "D:\conda_envs\attention-rgb"
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"

python scripts/rgb_analysis.py --stage audit
python scripts/rgb_analysis.py --stage gaps
```

输出根：

```text
D:\Project\厚粲杯\11_数据\04_Attention-Analysis_nvidia-cuda_RGB
```

---

## 9.2 Motion representative：sub-130

```powershell
conda activate "D:\conda_envs\attention-rgb"
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"

python scripts/rgb_analysis.py --stage motion --subject sub-130
python scripts/rgb_analysis.py --stage motion-qc --subject sub-130
python scripts/rgb_analysis.py --stage motion-review --subject sub-130
```

Motion 正式 full-FPS 与 body-motion integration 仍待最终 full-video orchestration。

---

## 9.3 Pose representative：sub-130

```powershell
conda activate "D:\conda_envs\attention-rgb"
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"

python scripts/rgb_analysis.py --stage pose --subject sub-130
python scripts/rgb_analysis.py --stage pose-qc --subject sub-130
python scripts/rgb_analysis.py --stage pose-features --subject sub-130
```

Pose 当前 scientific cadence 为 10 Hz；代表性结果需要检查肩部有效率、多人检测、时间 gap 和画外 landmark 行为。

---

# 10. RGB Face：sub-130 3600-frame CUDA Gate

## 10.1 第一步：生成 sub-130 15 Hz dry-run sample

使用 RGB core 环境：

```powershell
conda activate "D:\conda_envs\attention-rgb"
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"

python scripts/face_formal_dryrun_sample.py --subject sub-130
```

当前配置五个时间窗合计约 240 秒 × 15 Hz，预期约：

```text
3600 frames
```

默认 sample 目录：

```text
D:\Project\厚粲杯\11_数据\04_Attention-Analysis_nvidia-cuda_RGB\_test\face-formal-dryrun\sub-130
```

先检查：

```powershell
Get-ChildItem "D:\Project\厚粲杯\11_数据\04_Attention-Analysis_nvidia-cuda_RGB\_test\face-formal-dryrun\sub-130"
```

应该至少看到 `sub-130_face-dryrun_frames.csv`、manifest 和 frames 目录。

## 10.2 第二步：native Py-Feat CPU reference

**这一阶段的目的不是重新做 CPU 性能 benchmark，而是给 sub-130 CUDA 建同输入 scientific reference。**

CPU reference runner 的 dry-run-compatible 入口/比较器仍需在当前 CUDA Gate 收口中完成；在 README 标记为 pending 的情况下，不要用不同 subject 的 AMD parquet 代替。

验证原则：

```text
同一 sub-130 3600 sample
CPU reference ↔ CUDA
```

而不是：

```text
sub-031 AMD ↔ sub-130 NVIDIA
```

## 10.3 第三步：native PyTorch CUDA

激活 Face CUDA 环境：

```powershell
conda activate "D:\conda_envs\attention-face-cuda"
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"

nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python -c "import importlib.metadata as m; print(m.version('py-feat'))"
```

运行：

```powershell
$SAMPLE_DIR = "D:\Project\厚粲杯\11_数据\04_Attention-Analysis_nvidia-cuda_RGB\_test\face-formal-dryrun\sub-130"
$CUDA_DIR = "$SAMPLE_DIR\cuda"

python scripts/face_formal_dryrun_cuda.py `
  --sample-dir "$SAMPLE_DIR" `
  --output-dir "$CUDA_DIR" `
  --batch-size 8 `
  --num-workers 0 `
  --device cuda
```

第一轮 `batch-size=8` 只是安全 Gate 起点，不代表 RTX 5070 最终最优 batch。先通过科学 parity，再单独 benchmark 8/16/更高候选。

主要输出：

```text
cuda\pyfeat_cuda_raw.parquet
cuda\pyfeat_cuda_columns.json
cuda\pyfeat_cuda_dryrun_manifest.json
```

manifest 必须确认：

- `py-feat == 2.1.1`；
- `cuda_available == true`；
- GPU 对应 RTX 5070；
- expected input 约 3600；
- multi-face/no-face 保留；
- identity scientific branch 关闭；
- CUDA peak memory / throughput 已记录。

## 10.4 第四步：tracking / eyelid

CUDA raw parity 通过以后，切回 RGB core 环境：

```powershell
conda activate "D:\conda_envs\attention-rgb"
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"

$SAMPLE_DIR = "D:\Project\厚粲杯\11_数据\04_Attention-Analysis_nvidia-cuda_RGB\_test\face-formal-dryrun\sub-130"

python scripts/face_derive_tracking_eyelid_v02.py `
  --raw "$SAMPLE_DIR\cuda\pyfeat_cuda_raw.parquet" `
  --frame-manifest "$SAMPLE_DIR\sub-130_face-dryrun_frames.csv" `
  --output-dir "$SAMPLE_DIR\derived"
```

主要输出：

```text
derived\face_tracks.parquet
derived\eye_features.parquet
derived\tracking_eyelid_summary.json
```

## 10.5 第五步：QC 图片 / 视频

```powershell
python scripts/face_qc_visualize_v03.py `
  --tracks "$SAMPLE_DIR\derived\face_tracks.parquet" `
  --eye "$SAMPLE_DIR\derived\eye_features.parquet" `
  --sample-manifest "$SAMPLE_DIR\sub-130_face-dryrun_manifest.json" `
  --frame-manifest "$SAMPLE_DIR\sub-130_face-dryrun_frames.csv" `
  --output-dir "$SAMPLE_DIR\qc" `
  --fps 15
```

QC 从已经保存的 raw/derived + 原 AVI 生成，不重新跑 Py-Feat。

---

# 11. NVIDIA 与 AMD 的 Face parity 现在怎样理解

当前两台机器连接不同数据盘，所以不能拿不同被试做逐帧 cross-device parity。

正确证据链：

```text
AMD 已完成：Py-Feat CPU reference ↔ ONNX Runtime DirectML
NVIDIA 待完成：sub-130 Py-Feat CPU reference ↔ native PyTorch CUDA
```

两条链共同锚定：

```text
Py-Feat 2.1.1 Detectorv2 scientific core
```

若之后确实需要 AMD↔NVIDIA 完全相同帧的 row-wise parity，只复制 representative sample 到另一台机器即可，不需要移动 44/72 人完整数据盘。

---

# 12. 当前不要做的事

在正式冻结前不要：

- 用 `sub-031` 作为 NVIDIA 默认 RGB representative；
- 把 AMD 44 名被试列表复制成 NVIDIA 72 名正式队列；
- 把不同被试的 Face 输出称为逐帧 parity；
- 直接启动 NVIDIA RGB 72 人 full cohort；
- 为了省空间删掉 raw mesh/AU/blendshape/multi-face/provenance；
- 在 `eye-ai` 里同时安装 Behavior/RGB/Py-Feat 一整套依赖；
- force push / reset --hard；
- 删除 `rgb-dev` / `rgb-nvidia-cuda`，直到 CUDA Face、gap stress、blink/PERCLOS、full-video formal runner 完成收口。

---

# 13. 最短日常操作清单

## NIR

```powershell
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
git status --short --branch
git pull --ff-only
conda activate "D:\CondaEnvs\nir-nvidia"
cd runtime\nir-formal
python run_ritnet_fullclass_batch.py --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" --device 0 --postprocess-workers 4 --dry-run
```

## Behavior

```powershell
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
git status --short --branch
git pull --ff-only
conda activate "D:\conda_envs\attention-behavior"
python scripts/sart_formal_analysis.py --stage all
```

## RGB core

```powershell
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
git status --short --branch
git pull --ff-only
conda activate "D:\conda_envs\attention-rgb"
python scripts/rgb_analysis.py --stage audit
```

## Face CUDA representative

```powershell
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
git status --short --branch
git pull --ff-only

conda activate "D:\conda_envs\attention-rgb"
python scripts/face_formal_dryrun_sample.py --subject sub-130

conda activate "D:\conda_envs\attention-face-cuda"
$SAMPLE_DIR = "D:\Project\厚粲杯\11_数据\04_Attention-Analysis_nvidia-cuda_RGB\_test\face-formal-dryrun\sub-130"
python scripts/face_formal_dryrun_cuda.py --sample-dir "$SAMPLE_DIR" --output-dir "$SAMPLE_DIR\cuda" --batch-size 8 --num-workers 0 --device cuda
```

---

# 14. 文档入口

| 需要的信息 | 文档 |
|---|---|
| NVIDIA RITnet full-class 当前操作 | [`NVIDIA-RITnet全分类补跑使用说明.md`](NVIDIA-RITnet全分类补跑使用说明.md) |
| NIR runtime | [`runtime/nir-formal/`](runtime/nir-formal/) |
| Behavior 正式方法 | [`docs/030-behavior/`](docs/030-behavior/) |
| NIR × Behavior | `src/attention_pipeline/nir_behavior/` + `scripts/nir_behavior_alignment.py` |
| RGB 总方法 | [`docs/040-rgb/`](docs/040-rgb/) |
| NVIDIA CUDA RGB 路线 | [`docs/040-rgb/046-NVIDIA-CUDA-RGB运行路线.md`](docs/040-rgb/046-NVIDIA-CUDA-RGB运行路线.md) |
| scientific retention/schema | [`docs/040-rgb/044-RGB输出Schema与信息保留原则.md`](docs/040-rgb/044-RGB输出Schema与信息保留原则.md) |
| 技术决策 | [`docs/050-decisions/`](docs/050-decisions/) |
| 工作记录/provenance | [`docs/工作记录/`](docs/工作记录/) |
| 仓库长期规则 | [`AGENTS.md`](AGENTS.md) |

历史工作记录保留当时表述，不追溯改写；当前实际运行优先看本 README、当前 config 和对应 runtime/runbook。
