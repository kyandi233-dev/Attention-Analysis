# NVIDIA RTX 5070｜RITnet 四分类正式补跑使用说明

> 当前分支：`nvidia-cuda`  
> 当前任务：在**不重新运行 YOLO** 的前提下，复用既有正式 NIR `eyes.csv` 中保存的 `frame_idx + ROI`，只重新运行冻结的 RITnet ONNX，补齐 `background / sclera / iris / pupil` 四分类、虹膜几何、`fullclass_pupil_to_iris_diameter_ratio`、结构 QC 与 sparse QC 图片。  
> 目标机器：NVIDIA GeForce **RTX 5070**。  
> 当前 NVIDIA 原始数据根：`J:/Data`。  
> 本页是当前 NVIDIA 四分类补跑的首要操作入口；历史完整 NIR 主链、故障恢复和安装细节见 `runtime/nir-formal/RUNBOOK.md`、`INSTALL.md` 与 `RITNET_FULLCLASS_EXTENSION.md`。

---

## 0. 当前冻结边界

这次补跑只做 RITnet full-class extension，不改变已经完成的正式 YOLO/ROI 结果。

```text
GPU: NVIDIA GeForce RTX 5070
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

明确不做：不重新跑 YOLO；不重新确定 eye bbox / ROI；不使用 FP16；不改成 512 或更低 RITnet 输入分辨率；不覆盖或删除旧 `eyes.csv` 和历史正式 NIR 结果；不把 `J:/Data` 改成 AMD 电脑路径。

---

## 1. 当前 RTX 5070 工作站路径

```text
Conda 环境：
D:\CondaEnvs\nir-nvidia

仓库根目录：
D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda

正式原始数据根：
J:\Data

既有 NVIDIA 正式 NIR 输出根 / 本次 full-class source-run 根：
D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR
```

---

## 2. 每次开始工作：打开 PowerShell 并激活环境

推荐 Anaconda PowerShell Prompt 或已经完成 Conda 初始化的 VS Code PowerShell。

```powershell
conda activate D:\CondaEnvs\nir-nvidia
where.exe python
python --version
```

`where.exe python` 的首个 Python 应位于 `D:\CondaEnvs\nir-nvidia\`。正式 launcher 会直接
调用 `D:\CondaEnvs\nir-nvidia\python.exe`，并在子进程继承的 PATH 中注入 Conda/NVIDIA DLL 目录。

---

## 3. 从 GitHub 拉取最新 `nvidia-cuda`

### 3.1 本地仓库已经存在

```powershell
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
git status --short --branch
```

如果工作区存在自己尚未提交的修改，不要直接覆盖。工作区干净后：

```powershell
git switch nvidia-cuda
git pull --ff-only
git log -1 --oneline
git status --short --branch
```

如已冻结 tag，可检查：

```powershell
git tag --points-at HEAD
```

当前计划 tag：

```text
nvidia-v1.2-ritnet-fullclass
```

### 3.2 本地完全没有仓库时

```powershell
cd "D:\Project\厚粲杯\08_算法"
git clone https://github.com/kyandi233-dev/Attention-Analysis.git 01_Attention-Analysis_nvidia-cuda
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
git switch nvidia-cuda
git pull --ff-only
```

---

## 4. 进入正式 NIR runtime

```powershell
cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda\runtime\nir-formal"
```

后续 Python 命令默认都在这个目录执行。

---

## 5. GPU / CUDA / ONNX Runtime 检查

### 5.1 NVIDIA GPU

```powershell
nvidia-smi
```

应识别：

```text
NVIDIA GeForce RTX 5070
```

### 5.2 PyTorch CUDA

```powershell
python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('torch_cuda=', torch.version.cuda); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

必须满足：

```text
cuda_available = True
```

GPU 名称应对应 RTX 5070。

### 5.3 ONNX Runtime CUDAExecutionProvider

本次 full-class 补跑实际使用 ONNX Runtime CUDA，因此必须单独确认：

```powershell
python -c "import onnxruntime as ort; print('onnxruntime=', ort.__version__); print('providers=', ort.get_available_providers())"
```

必须包含：

```text
CUDAExecutionProvider
```

若当前环境确实缺少 GPU 版 ORT：

```powershell
python -m pip install onnxruntime-gpu==1.24.4
```

安装后重新检查 provider。若仍没有 `CUDAExecutionProvider`，停止正式补跑，不能静默落到 CPU。

---

## 6. 检查数据盘和正式输出根

```powershell
Test-Path "J:\Data"
Test-Path "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR"
```

两项都应为 `True`。

人工查看 J 盘被试目录：

```powershell
Get-ChildItem "J:\Data" -Directory -Filter "sub-*_*" | Measure-Object
```

检查 NVIDIA config 仍然使用 `J:/Data`：

```powershell
Select-String -Path ".\config.yaml" -Pattern "J:/Data"
```

应看到：

```text
- "J:/Data"
```

---

## 7. 运行代码测试和环境检查

```powershell
python -m pytest tests -q
python run_pipeline.py check-env
```

若 pytest 有失败，先处理失败项。`check-env` 用于检查 NVIDIA GPU/CUDA、PyTorch、Ultralytics、OpenCV 和冻结模型；它不能替代上一节的 `CUDAExecutionProvider` 检查。

---

## 8. 先做 full-class dry-run

当前任务入口是 `run_ritnet_fullclass_batch.py`，不是重新运行完整 YOLO+RITnet 的 `run_formal_batch.py`。

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --device 0 `
  --postprocess-workers 4 `
  --dry-run
```

检查：被试编号是否为 RTX 5070 机器当前负责的 NVIDIA 队列；当前预期约 72 名但以 dry-run 为准；source run 是否来自既有 NVIDIA 正式 NIR 输出；无 duplicate；无错误数据根；`sub-9504` 等排除项没有被错误纳入。

---

## 9. RTX 5070 首次只跑 1 名完整被试验收

从 dry-run 输出中选一名实际存在的完整被试：

```powershell
$TEST_SUBJECT = "sub-XXX"
```

首次建议开启 pupil parity 审计：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --subjects $TEST_SUBJECT `
  --device 0 `
  --postprocess-workers 4 `
  --validate-pupil
```

`--validate-pupil` 会额外请求 pupil probability，并审计新 ONNX hard-label pupil 与 source 旧 pupil；这一人的速度不代表最终 production 速度。

---

## 10. 单被试完成后检查结果

```powershell
$OUTPUT_ROOT = "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR"
```

找 completion：

```powershell
Get-ChildItem $OUTPUT_ROOT -Recurse -Filter "${TEST_SUBJECT}_ritnet_fullclass_v1-2-fast-qc_completion.json"
```

找 summary：

```powershell
Get-ChildItem $OUTPUT_ROOT -Recurse -Filter "${TEST_SUBJECT}_ritnet_fullclass_v1-2-fast-qc_summary.json"
```

读取关键统计：

```powershell
$SUMMARY_FILE = Get-ChildItem $OUTPUT_ROOT -Recurse -Filter "${TEST_SUBJECT}_ritnet_fullclass_v1-2-fast-qc_summary.json" | Select-Object -First 1
$SUMMARY = Get-Content $SUMMARY_FILE.FullName -Raw | ConvertFrom-Json
$SUMMARY | Select-Object subject, processed_rows, decoded_frames, elapsed_sec, roi_per_sec, normalization_valid_fraction, pupil_parity_ok_fraction, timing_gpu_ms
```

检查 sparse QC：

```powershell
Get-ChildItem $OUTPUT_ROOT -Recurse -Directory -Filter "${TEST_SUBJECT}_ritnet_fullclass_v1-2-fast-qc_qc"
Get-ChildItem $OUTPUT_ROOT -Recurse -Filter "${TEST_SUBJECT}_ritnet_fullclass_v1-2-fast-qc_qc_index.csv"
```

人工打开若干 `*_labels.png` 和 `*_overlay.png`，确认 background / sclera / iris / pupil 分类位置合理。

验收最低条件：completion 为 `complete`；`processed_rows == expected_rows`；CSV/summary/manifest/completion/QC index 存在；QC 图片正常；无 CUDA provider fallback；parity 没有系统性异常。

---

## 11. 正式跑 NVIDIA 全队列

单被试通过后，production **去掉 `--validate-pupil`**：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --device 0 `
  --postprocess-workers 4
```

production 使用 labels-only CUDA inference，不重新跑 YOLO。

运行时可另开终端观察：

```powershell
nvidia-smi
```

不要同时启动第二个写向相同 subject/output 的 full-class 批次。

---

## 12. 中断、失败和恢复

正常重新执行同一 production 命令即可由 completion/manifest 逻辑跳过已完成结果并继续未完成项。不要手工修改 completion，也不要删除旧正式结果。

如需只重跑一个明确有问题的被试，先确认原因、旧进程已退出，再按该脚本实际支持的参数运行。除非明确知道要覆盖什么，不要随意使用 `--force`。

完整的 `.run.lock`、进程、失败状态和恢复判断见：

```text
runtime/nir-formal/RUNBOOK.md
```

---

## 13. 本次新输出

以 `sub-031` 为例：

```text
sub-031_ritnet_fullclass_v1-2-fast-qc.csv
sub-031_ritnet_fullclass_v1-2-fast-qc_summary.json
sub-031_ritnet_fullclass_v1-2-fast-qc_manifest.json
sub-031_ritnet_fullclass_v1-2-fast-qc_completion.json
sub-031_ritnet_fullclass_v1-2-fast-qc_qc_index.csv
sub-031_ritnet_fullclass_v1-2-fast-qc_qc\
```

主瞳孔指标：

```text
fullclass_pupil_to_iris_diameter_ratio
```

正式 normalized pupil 分析优先使用：

```text
fullclass_normalization_valid == True
```

原 `eyes.csv` 和历史正式输出保留，不覆盖。

---

## 14. 行为数据与 NIR 后续衔接

full-class CSV 原样保留：

```text
phase
phase_segment
frame_idx
video_time_ms
unix_ms
phase_time_ms
```

后续 SART trial-level 对齐通过 NIR `unix_ms` 与行为绝对时间戳完成，是独立下游步骤；不需要为了行为对齐再次运行 RITnet。

方法文档：

```text
docs/030-behavior/035-NIR与正式SART行为数据对齐分析方法.md
```

---

## 15. 最短正式操作清单

已经确认本机环境存在时，按以下顺序执行：

```powershell
conda activate D:\CondaEnvs\nir-nvidia

cd "D:\Project\厚粲杯\08_算法\01_Attention-Analysis_nvidia-cuda"
git switch nvidia-cuda
git pull --ff-only
git log -1 --oneline

cd runtime\nir-formal

nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
Test-Path "J:\Data"
Select-String -Path ".\config.yaml" -Pattern "J:/Data"

python -m pytest tests -q
python run_pipeline.py check-env

python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --device 0 `
  --postprocess-workers 4 `
  --dry-run
```

第一次 RTX 5070 真机验收：

```powershell
$TEST_SUBJECT = "sub-XXX"
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --subjects $TEST_SUBJECT `
  --device 0 `
  --postprocess-workers 4 `
  --validate-pupil
```

验收通过后正式全量：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" `
  --device 0 `
  --postprocess-workers 4
```
