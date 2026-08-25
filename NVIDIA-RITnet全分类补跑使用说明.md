# NVIDIA RTX 5060｜RITnet 四分类正式补跑使用说明

> 当前分支：`nvidia-cuda`  
> 当前任务：在**不重新运行 YOLO** 的前提下，复用既有正式 NIR `eyes.csv` 中已经保存的 `frame_idx + ROI`，只重新运行冻结的 RITnet ONNX，补齐 `background / sclera / iris / pupil` 四分类、虹膜几何、`fullclass_pupil_to_iris_diameter_ratio`、结构 QC 与 sparse QC 图片。  
> 目标机器：NVIDIA GeForce **RTX 5060**。  
> 当前 NVIDIA 原始数据根：`J:/Data`。  
> 本页是当前 NVIDIA 四分类补跑的**首要操作入口**；历史完整 NIR 主链、故障恢复和安装细节仍分别见 `runtime/nir-formal/RUNBOOK.md`、`INSTALL.md` 与 `RITNET_FULLCLASS_EXTENSION.md`。

---

## 0. 当前冻结边界

这次补跑只做 RITnet full-class extension，不改变已经完成的正式 YOLO/ROI 结果。

固定参数：

```text
GPU: RTX 5060
source data root: J:/Data
RITnet ONNX: models/ritnet-b16-fp32.onnx
RITnet input: 640 × 400
precision: FP32
batch size: 16
analysis size: 320 × 160
classes: 0 background / 1 sclera / 2 iris / 3 pupil
primary pupil metric: fullclass_pupil_to_iris_diameter_ratio
postprocess workers: 4
QC stride: 3000 frames + phase anchors + bounded anomaly samples
```

明确不做：

- 不重新跑 YOLO；
- 不重新确定 eye bbox / ROI；
- 不使用 FP16；
- 不改成 512 或更低 RITnet 输入分辨率；
- 不覆盖或删除旧 `eyes.csv`、旧正式 NIR 结果；
- 不为了这次补全去运行原来的完整 `run_formal_batch.py`；
- 不把 `J:/Data` 改成 AMD 电脑使用的路径。

当前 full-class 新输出使用版本化、带被试编号的文件名，因此与历史正式结果并存。

---

## 1. 当前 NVIDIA 机器的路径约定

本页以下命令按当前 RTX 5060 机器的实际项目结构编写。

```text
Conda 环境：
D:\conda_envs\eye-ai

仓库根目录：
D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda

正式原始数据根：
J:\Data

既有 NVIDIA 正式 NIR 输出根 / 本次 full-class source-run 根：
D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR
```

如果以后机器目录发生变化，应修改本页本机路径或使用命令行参数，不要偷偷修改冻结科研参数。

---

# 2. 每次开始工作：打开终端并激活环境

推荐使用 **Anaconda PowerShell Prompt**，或者已经完成 Conda 初始化的 VS Code PowerShell 终端。

先激活当前项目已经配置好的环境：

```powershell
conda activate D:\conda_envs\eye-ai
```

确认当前 Python 确实来自这个环境：

```powershell
where.exe python
python --version
```

正常情况下，`where.exe python` 的首个 Python 应位于：

```text
D:\conda_envs\eye-ai\
```

如果不是，先解决环境激活问题，不要继续跑正式数据。

---

# 3. 从 GitHub 拉取当前 `nvidia-cuda`

## 3.1 仓库已经存在：正常更新流程

进入仓库根目录：

```powershell
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
```

先检查本地状态：

```powershell
git status --short --branch
```

如果这里出现你自己尚未提交的修改，**不要直接 pull 覆盖**。先确认这些修改是什么，再决定提交、暂存或处理。

如果工作区干净，执行：

```powershell
git switch nvidia-cuda
git pull --ff-only
git log -1 --oneline
git status --short --branch
```

`git pull --ff-only` 的意义是：只允许安全的 fast-forward 更新。如果本地与远端已经分叉，它会停止，而不是自动做一个不透明的 merge。

如果已经冻结并推送 tag，也可以查看当前 HEAD 是否位于正式 full-class tag：

```powershell
git tag --points-at HEAD
```

当前计划冻结 tag：

```text
nvidia-v1.2-ritnet-fullclass
```

## 3.2 只有在仓库完全不存在时才 clone

进入算法目录：

```powershell
cd "D:\Project\厚粲杯\08_算法"
```

clone：

```powershell
git clone https://github.com/kyandi233-dev/Attention-Analysis.git 01_Attention-Analysis_nvidia-cuda
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
git switch nvidia-cuda
git pull --ff-only
```

已经存在仓库时不要重复 clone 第二份。

---

# 4. 进入正式 NIR runtime

从仓库根目录进入：

```powershell
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda\runtime\nir-formal"
```

后续本页绝大多数 Python 命令都在这个目录执行。

---

# 5. GPU / CUDA / ONNX Runtime 检查

## 5.1 NVIDIA 驱动与 GPU

```powershell
nvidia-smi
```

必须能看到目标 GPU，当前机器应为：

```text
NVIDIA GeForce RTX 5060
```

如果 `nvidia-smi` 本身失败，先不要继续 Python 分析。

## 5.2 PyTorch CUDA 检查

虽然这次 full-class extension 使用 ONNX Runtime CUDA，但当前正式环境仍保留 PyTorch CUDA 主链，因此一起检查：

```powershell
python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('torch_cuda=', torch.version.cuda); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

硬门：

```text
cuda_available = True
```

GPU 名称应对应 RTX 5060。

## 5.3 ONNX Runtime CUDA 检查

本次 RITnet full-class 正式补跑真正使用的是 `CUDAExecutionProvider`，所以必须单独检查：

```powershell
python -c "import onnxruntime as ort; print('onnxruntime=', ort.__version__); print('providers=', ort.get_available_providers())"
```

输出中必须包含：

```text
CUDAExecutionProvider
```

如果报：

```text
ModuleNotFoundError: No module named 'onnxruntime'
```

只在确认当前环境确实缺失时安装项目已冻结使用的 GPU 版 ORT：

```powershell
python -m pip install onnxruntime-gpu==1.24.4
```

然后重新运行 provider 检查。

如果安装后仍没有 `CUDAExecutionProvider`，**停止正式补跑**，先处理 CUDA/ORT 环境，不允许静默落到 CPU。

---

# 6. 检查数据盘和正式输出根

确认 `J:/Data` 当前挂载：

```powershell
Test-Path "J:\Data"
```

应返回：

```text
True
```

查看 J 盘上发现的被试目录数量：

```powershell
Get-ChildItem "J:\Data" -Directory -Filter "sub-*_*" | Measure-Object
```

这里只是人工快速检查；最终应以程序 dry-run 发现结果为准。

确认既有正式 NIR 输出根存在：

```powershell
Test-Path "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR"
```

应返回：

```text
True
```

同时检查当前 NVIDIA config 没被误改成其他机器路径：

```powershell
Select-String -Path ".\config.yaml" -Pattern "J:/Data"
```

应能看到：

```text
- "J:/Data"
```

如果这里不是 `J:/Data`，不要开始正式补跑。

---

# 7. 运行测试和 runtime 环境检查

先运行正式 runtime 自包含测试：

```powershell
python -m pytest tests -q
```

应全部通过；如果有失败，先看失败测试，不要直接开始 72 人全量。

再运行正式 runtime 环境检查：

```powershell
python run_pipeline.py check-env
```

这里用于确认当前 NVIDIA GPU/CUDA、PyTorch、Ultralytics、OpenCV 和冻结模型资产能够被正式 runtime 识别。

注意：`check-env` 通过不等于 ONNX Runtime CUDA 一定正常，所以第 5.3 节的 `CUDAExecutionProvider` 检查仍然必须单独做。

---

# 8. 先 dry-run：禁止直接从零开始 72 人全量

本次补跑的入口不是 `run_formal_batch.py`，而是：

```text
run_ritnet_fullclass_batch.py
```

执行 dry-run：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --device 0 `
  --postprocess-workers 4 `
  --dry-run
```

这里要检查：

1. 被试编号是否就是 RTX 5060 这台机器当前负责的 NVIDIA 队列；当前预期约 72 名，但以实际 dry-run 输出为准；
2. source run 是否来自既有正式 NVIDIA NIR 输出；
3. 没有 duplicate；
4. 没有错误数据根；
5. 没有把 AMD 电脑负责的输出目录当成 source；
6. `sub-9504` 等明确排除项没有被错误纳入。

如果 dry-run 人数、编号或 source 路径明显不对，停止，不要开始正式计算。

---

# 9. RTX 5060 首次只跑 1 名完整被试做真机验收

因为这套 full-class 代码虽然已经通过仓库 CI，AMD 端也已经完成完整 reference + fast-qc 验证，但 RTX 5060 的 CUDAExecutionProvider 仍需要一次目标机器完整实测。

先从 dry-run 输出里选一个**确实在 NVIDIA 队列中的正常完整被试**。不要凭空使用一个不在本机队列中的编号。

例如把实际编号填入：

```powershell
$TEST_SUBJECT = "sub-XXX"
```

然后首次 CUDA 验收建议开启 pupil parity 审计：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --subjects $TEST_SUBJECT `
  --device 0 `
  --postprocess-workers 4 `
  --validate-pupil
```

`--validate-pupil` 会额外请求 pupil probability，并把新 ONNX hard-label pupil 与 source 旧 pupil 做审计，因此这一人的速度不代表最终 production 的最快速度。

NVIDIA 历史 source pupil 主要来自 PyTorch CUDA，而新 full-class pupil/iris 都来自冻结 ONNX hard labels；因此 parity 是审计信息，不要求凭空设定一个未经冻结的“100%相等阈值”。如果出现系统性、大范围几何差异，则停止并排查；零星边界差异应结合 QC 图和 summary 判断。

---

# 10. 单被试完成后检查哪些文件

设置输出根：

```powershell
$OUTPUT_ROOT = "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR"
```

查找 completion：

```powershell
Get-ChildItem $OUTPUT_ROOT -Recurse -Filter "${TEST_SUBJECT}_ritnet_fullclass_v1-2-fast-qc_completion.json"
```

查找 summary：

```powershell
Get-ChildItem $OUTPUT_ROOT -Recurse -Filter "${TEST_SUBJECT}_ritnet_fullclass_v1-2-fast-qc_summary.json"
```

读取 summary 的关键指标：

```powershell
$SUMMARY_FILE = Get-ChildItem $OUTPUT_ROOT -Recurse -Filter "${TEST_SUBJECT}_ritnet_fullclass_v1-2-fast-qc_summary.json" | Select-Object -First 1
$SUMMARY = Get-Content $SUMMARY_FILE.FullName -Raw | ConvertFrom-Json
$SUMMARY | Select-Object subject, processed_rows, decoded_frames, elapsed_sec, roi_per_sec, normalization_valid_fraction, pupil_parity_ok_fraction, timing_gpu_ms
```

查找 sparse QC 文件夹：

```powershell
Get-ChildItem $OUTPUT_ROOT -Recurse -Directory -Filter "${TEST_SUBJECT}_ritnet_fullclass_v1-2-fast-qc_qc"
```

检查 QC index：

```powershell
Get-ChildItem $OUTPUT_ROOT -Recurse -Filter "${TEST_SUBJECT}_ritnet_fullclass_v1-2-fast-qc_qc_index.csv"
```

至少人工打开几组：

```text
*_labels.png
*_overlay.png
```

确认 background / sclera / iris / pupil 的分类位置合理。

### 单被试正式放行条件

至少同时满足：

```text
completion status = complete
processed_rows == expected_rows
CSV / summary / manifest / completion / qc_index 均存在
QC labels / overlay 能正常打开
CUDAExecutionProvider 未发生 CPU fallback
没有系统性的 pupil / iris 几何异常
elapsed_sec / roi_per_sec 已记录
```

---

# 11. 正式跑 NVIDIA 全部队列

单被试验收通过后，正式 production **去掉 `--validate-pupil`**：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --device 0 `
  --postprocess-workers 4
```

这是当前 NVIDIA 正式 full-class 补跑命令。

production 路径会：

```text
复用旧 source eyes.csv + 原 AVI
→ 按 frame_idx 重建同一 ROI
→ 640×400 / FP32 / b16 RITnet ONNX CUDA
→ labels-only production
→ 四分类结构量
→ pupil + iris outer geometry
→ fullclass_pupil_to_iris_diameter_ratio
→ structural QC
→ sparse labels/overlay QC
→ subject-numbered versioned outputs
```

这一步不会重跑 YOLO。

---

# 12. 全量运行时如何看 GPU

可以另开一个 PowerShell 窗口，不需要启动第二条分析进程，只做监控。

```powershell
nvidia-smi -l 5
```

它会每 5 秒刷新一次 GPU 状态。

查看本机 Python 进程：

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, CPU, StartTime, Path
```

不要因为某一瞬间 GPU 利用率掉到 0% 就立刻结束任务；视频 decode、CPU preprocess、后处理和写盘会与 CUDA inference 交替/流水运行。

---

# 13. 中断以后怎么恢复

正常原则：**重新执行同一条 full-class batch 命令即可。**

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --device 0 `
  --postprocess-workers 4
```

batch 会依据 full-class completion contract 判断已有输出；合法完整的被试应被跳过，未完成/无效结果重新处理。

不要为了“从断点继续”手工把 JSON 状态改成 `complete`。

不要默认使用 `--force`。

只有你已经明确知道为什么要覆盖一个**有效 complete** 的 full-class 结果时，才考虑强制重跑，并应先记录原因。

---

# 14. 输出文件说明

以 `sub-XXX` 为例，每名被试的新版本输出至少包括：

```text
sub-XXX_ritnet_fullclass_v1-2-fast-qc.csv
sub-XXX_ritnet_fullclass_v1-2-fast-qc_summary.json
sub-XXX_ritnet_fullclass_v1-2-fast-qc_manifest.json
sub-XXX_ritnet_fullclass_v1-2-fast-qc_completion.json
sub-XXX_ritnet_fullclass_v1-2-fast-qc_qc_index.csv
sub-XXX_ritnet_fullclass_v1-2-fast-qc_qc\
```

其中：

- `csv`：逐眼、逐帧正式 full-class 数值；
- `summary.json`：速度、行数、normalization coverage、分类比例、CPU/GPU 分阶段耗时；
- `manifest.json`：模型 SHA、source 文件 SHA、运行参数与 provenance；
- `completion.json`：正式完成状态与完整性 contract；
- `qc_index.csv`：所有 sparse QC 图片对应的 subject / phase / frame / unix_ms / eye / reason；
- `qc\`：`labels.png` 与 `overlay.png`。

后续正式 pupil 主分析变量：

```text
fullclass_pupil_to_iris_diameter_ratio
```

正式分析优先使用：

```text
fullclass_normalization_valid == True
```

的帧，再进行左右眼整合、baseline normalization、时间窗/trial/probe 对齐等下游分析。

---

# 15. NIR 与 SART 行为数据怎么衔接

不要为了行为 trial 对齐再次运行 RITnet。

full-class CSV 已保留：

```text
phase
phase_segment
frame_idx
video_time_ms
unix_ms
phase_time_ms
```

正式 SART 行为文件保存在每名被试的：

```text
J:\Data\sub-XXX_\beh\
```

包括正式 Block1 / Block2 行为 CSV 以及 `master_timeline.csv` 等。

后续 NIR ↔ SART alignment 应作为独立下游分析，通过 NIR `unix_ms` 与行为绝对时间戳映射，不把 trial 信息硬写回 RITnet 推理流程。

方法说明见：

```text
docs/030-behavior/035-NIR与正式SART行为数据对齐分析方法.md
```

---

# 16. 当前速度预算

AMD fast-qc 已有完整实测参考，但 RTX 5060 的最终时间以第 9 节首名完整被试的：

```text
elapsed_sec
roi_per_sec
```

为准。

首名完成后可以直接估算：

```text
NVIDIA 72 人理论总耗时 ≈ 单人 elapsed_sec × 72
```

两台电脑并行时，整个 116 人补跑的墙钟时间由较慢的一台决定，不应把两台机器耗时简单相加。

---

# 17. 常见错误与禁止操作

### `CUDAExecutionProvider` 不存在

停止正式跑。检查当前环境、`onnxruntime-gpu`、NVIDIA driver/CUDA compatibility。禁止默认 CPU fallback。

### `J:\Data` 不存在

检查硬盘是否正确挂载。不要为了让程序“先跑起来”把 NVIDIA config 改成 AMD 电脑路径。

### dry-run 人数或被试编号不对

停止。先检查 source output、数据根和被试分工。

### 某个被试已经 `complete`

正常 batch 应跳过。不要无理由加 `--force`。

### 看到旧 `eyes.csv` 已经有 pupil，是否还要 full-class？

要。旧正式输出只保留 pupil 派生量；当前补跑是为了恢复完整 RITnet 四分类和 iris normalization。

### 需要运行 `run_formal_batch.py` 吗？

这次不需要。那是完整 YOLO + ROI + RITnet 正式主链；当前任务只补 RITnet full-class，入口是 `run_ritnet_fullclass_batch.py`。

---

# 18. 最短正式操作清单

如果环境和仓库都已经配置完成，日常最短流程如下。**仍建议逐段执行并检查输出，不要盲目整块粘贴。**

```powershell
# 1) 激活环境
conda activate D:\conda_envs\eye-ai

# 2) 更新仓库
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
git status --short --branch
git switch nvidia-cuda
git pull --ff-only
git log -1 --oneline

# 3) 进入 runtime
cd runtime\nir-formal

# 4) GPU / CUDA / ORT 检查
nvidia-smi
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python -c "import onnxruntime as ort; print('providers=', ort.get_available_providers())"

# 5) 数据根与 config
Test-Path "J:\Data"
Select-String -Path ".\config.yaml" -Pattern "J:/Data"

# 6) 测试
python -m pytest tests -q
python run_pipeline.py check-env

# 7) dry-run
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --device 0 `
  --postprocess-workers 4 `
  --dry-run

# 8) 首次 RTX 5060 真机验收：把 sub-XXX 换成 dry-run 中实际存在的一名完整被试
$TEST_SUBJECT = "sub-XXX"
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --subjects $TEST_SUBJECT `
  --device 0 `
  --postprocess-workers 4 `
  --validate-pupil

# 9) 单人验收通过后正式全量；不要加 --validate-pupil
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --device 0 `
  --postprocess-workers 4
```

---

## 19. 相关文档

```text
README.md
runtime/nir-formal/RITNET_FULLCLASS_EXTENSION.md
runtime/nir-formal/RUNBOOK.md
runtime/nir-formal/INSTALL.md
docs/020-nir/08-25-01-NIR-RITnet全分类补全与瞳孔分析方法.md
docs/030-behavior/035-NIR与正式SART行为数据对齐分析方法.md
docs/工作记录/08-25-NVIDIA-RITnet全分类补全同步工作记录.md
```

当前任务优先按本页执行；遇到 formal 生命周期、锁、历史完整主链等问题再进入对应 runbook/installation 文档。
