# INSTALL｜NIR Formal Runtime（NVIDIA / CUDA v8）

本文件是 `runtime/nir-formal/` 在 NVIDIA/CUDA Windows 机器上的安装入口。当前分支为 `nvidia-cuda-v8`；final full-class scientific/core 与 `amd-DirectML` v8 共用同一套 ROI、RITnet preprocessing、pupil/uncertainty/temporal 公式、schema、QC 与 completion 契约，执行后端改为 ONNX Runtime `CUDAExecutionProvider`。

当前 final full-class **不重新执行历史 YOLO producer**。它消费已经完成的历史 formal run：`completion.json + frames.csv + eyes.csv + 原始 NIR AVI`。

## 1. 获取 NVIDIA v8 分支

新 clone：

```powershell
git clone https://github.com/kyandi233-dev/Attention-Analysis.git
cd Attention-Analysis
git fetch origin --prune
git switch nvidia-cuda-v8
git pull --ff-only
git status --short --branch
git rev-parse HEAD
```

已有仓库：

```powershell
git fetch origin --prune
git switch nvidia-cuda-v8
git pull --ff-only
git status --short --branch
git rev-parse HEAD
```

正式 runner 要求 clean code worktree；不要用 `git reset --hard` 掩盖未知本地修改。

## 2. 创建独立 NVIDIA 环境

推荐 Python 3.11：

```powershell
conda create -p D:\CondaEnvs\nir-nvidia python=3.11 -y
conda activate D:\CondaEnvs\nir-nvidia
```

进入 runtime 并安装依赖：

```powershell
cd runtime\nir-formal
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

当前关键依赖：

```text
onnxruntime-gpu==1.24.4
opencv-contrib-python>=4.10
numpy>=1.26
pandas>=2.2
PyYAML>=6.0
```

final full-class 不依赖 PyTorch 或 Ultralytics 做推理；历史 YOLO bbox 已存在于 `eyes.csv`。

## 3. 检查 CUDA provider

```powershell
nvidia-smi
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

必须看到：

```text
CUDAExecutionProvider
```

`cuda_runtime.py` 会：

- 只创建 CUDAExecutionProvider session；
- 要求 CUDA 为 primary provider；
- 设置 `session.disable_cpu_ep_fallback=1`；
- 调用 `session.disable_fallback()`；
- 固定 `use_tf32=0`；
- CUDA 不可用时 fail closed。

因此不要把 `CPUExecutionProvider` 当作 CUDA 环境安装成功的替代证据。

## 4. 检查 final 资产

至少需要：

```text
models/ritnet-b16-fp32-uncertainty.onnx
models/ritnet-b16-fp32-uncertainty.onnx.data
cuda_runtime.py
ritnet_fullclass_final_runtime.py
run_ritnet_fullclass_extension.py
run_ritnet_fullclass_batch.py
config.yaml
```

历史 producer/reference 资产可以继续保留，但 final full-class 不会因此重新跑 YOLO。

## 5. 跑代码测试

```powershell
python -m pytest tests -q
```

仓库根目录 CI 还会跑 portable baseline tests。CPU CI 可以验证 contract、公式、checkpoint、I/O、CUDA fail-closed mock 等，但无法代替真实 NVIDIA GPU smoke。

## 6. 准备历史 formal source

final batch 的 `--output` 必须指向已经存在历史 formal run 的输出根。每个候选 run 至少应有严格有效的 completion 和：

```text
sub-XXX_formal_*/
├── completion.json
├── frames.csv
└── eyes.csv
```

对应原始 NIR AVI 必须仍可从 recorded source/provenance 找到。

先只做 source selection：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "<NVIDIA 历史 formal 输出根>" `
  --subjects "sub-XXX" `
  --device 0 `
  --dry-run
```

确认选中 run、eyes SHA、frames SHA 与 subject 正确后再正式执行：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "<NVIDIA 历史 formal 输出根>" `
  --subjects "sub-XXX" `
  --device 0
```

单 run：

```powershell
python run_ritnet_fullclass_extension.py `
  --run-dir "<历史 formal run 目录>" `
  --config config.yaml `
  --device 0
```

## 7. checkpoint 与 AMD 隔离

NVIDIA v8 work identity 记录：

```text
execution_backend = onnxruntime-cuda
execution_provider = CUDAExecutionProvider
```

CUDA checkpoint 只能在 execution identity 一致时恢复。DirectML checkpoint 或未记录 execution identity 的旧 checkpoint不会被 NVIDIA v8 静默接续。这样避免一个 subject 的 numeric rows 混合两个后端。

checkpoint 仍是临时 interruption-recovery 数据；最终科研完成状态只由严格有效的 `completion.json` 决定。

## 8. 最终输出与大小限制

输出结构：

```text
<历史 formal 输出根>\ritnet-fullclass-final\sub-XXX\
├── data\eye_metrics.csv
├── data\frame_coverage.csv
├── qc\images\*.png
├── qc\qc_index.csv
├── qc\qc_pixel_evidence.npz
├── summary.json
├── manifest.json
└── completion.json
```

每被试 final directory 必须 ≤1 GiB。旧 `.csv.gz`、半完成 summary/manifest/QC 在没有有效 completion 时会阻止自动覆盖；先人工确认并归档，不自动删除。

## 9. 实机最小验收顺序

在正式全量前按以下顺序：

```text
1. git status / exact HEAD
2. pytest tests -q
3. CUDAExecutionProvider 可用
4. final ONNX 文件存在
5. batch --dry-run source selection 正确
6. 一个真实 subject / 短范围或可控 smoke 的 CUDA 实机验证
7. 检查 manifest execution_backend/provider
8. 检查 plain CSV / bounded QC / completion / <=1 GiB
9. 再进入 cohort full run
```

详细科学输出、checkpoint、QC 和 transfer benchmark 说明见 `README.md`。
