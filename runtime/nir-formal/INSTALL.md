# INSTALL｜NIR Formal Runtime

本文件是 `runtime/nir-formal/` 在 AMD/DirectML Windows 机器上的安装入口。当前正式分支为 `amd-DirectML`，package version 为 `0.2.0`，正式组合为 **YOLO b8 + RITnet b16**。

新机器安装完成后，正式批量运行前必须继续读取 `RUNBOOK_V1.md`。特别是：当前 `config.yaml` 冻结为两个正式 B block 的 v3.1.3 scope；若实际数据是三 block/BBB 或其他 site/protocol，不得仅因为环境安装成功就直接全量运行，必须先通过 protocol compatibility gate。

## 1. 获取当前分支

```powershell
git clone https://github.com/kyandi233-dev/Attention-Analysis.git
cd Attention-Analysis
git switch amd-DirectML
```

如果仓库已经存在：

```powershell
git switch amd-DirectML
git pull --ff-only
```

正式运行时还必须记录 exact Git commit，不能只记录移动的分支名。

## 2. 创建独立环境

推荐 Python 3.11 的独立 Conda 环境。当前已验证环境为：

```powershell
conda create -p D:\CondaEnvs\nir-amd python=3.11 -y
conda activate D:\CondaEnvs\nir-amd
```

进入 runtime 安装依赖：

```powershell
cd runtime\nir-formal
pip install -r requirements.txt
```

`requirements.txt` 使用 ONNX Runtime DirectML；正式推理不需要 Ultralytics、PyTorch 或 CUDA。DirectML 需要 Windows 10 1903+ 与 DirectX 12 可用 GPU/驱动。

## 3. 检查冻结资产

正式运行至少需要：

```text
models/nir-eye-yolo26n-best.onnx
models/nir-eye-yolo26n-best-b8.onnx
models/ritnet-b16-fp32.onnx
models/ritnet-b16-fp32.onnx.data
directml_runtime.py
run_formal_batched.py
run_formal_batch.py
config.yaml
```

其中原 `nir-eye-yolo26n-best.onnx` 作为 b1 reference/diagnostic 资产保留；`nir-eye-yolo26n-best-b8.onnx` 才是 v0.2.0 正式 YOLO。

然后执行：

```powershell
python -m pytest tests -q
python run_pipeline.py check-env
```

DirectML 不可用时必须失败，不允许整个 session 静默退回纯 CPU。

## 4. 挂载正式数据

正式数据逻辑目录为 `正式实验` 与 `Data`；两块外接盘盘符可能在 `E:` / `F:` 之间交换。`config.yaml` 已声明：

```text
E:/正式实验
F:/正式实验
E:/Data
F:/Data
```

程序忽略不存在的候选根；若同一被试同时出现在多个有效根，会直接报告 duplicate。

挂载数据盘后先检查：

```powershell
python run_pipeline.py discover --formal-only
python run_formal_batch.py --dry-run
```

这一步只说明环境/数据发现是否工作，不等于实际 protocol 已被批准。发现结果必须与 `RUNBOOK_V1.md` 的 protocol compatibility gate 一起审查。

## 5. 正式运行

仅当 dry-run 和 protocol gate 都通过后：

```powershell
python run_formal_batch.py
```

只跑少量被试：

```powershell
python run_formal_batch.py --subjects sub-031,sub-033
```

显式重跑：

```powershell
python run_formal_batch.py --subjects sub-031 --force
```

v0.2.0 输出目录示例：

```text
sub-031_formal_v3.1.3_yolo-b8_ritnet-b16_fp32
```

## 6. 单被试正式运行

v0.2.0 的正式单被试入口是：

```powershell
python run_formal_batched.py `
  --video "<实际盘符>:\<数据根>\sub-033_\nir\sub-033_nir.avi" `
  --device 0
```

`run_pipeline.py` 继续保留 diagnostic / discover / check-env 和历史兼容功能；正式全量不再通过旧的逐帧 YOLO formal 路径执行。

## 7. AMD v0.2.0 版本边界

正式参数：

```text
YOLO26n: 640×640, FP32, DirectML, fixed batch=8, every frame
RITnet:  640×400, FP32, DirectML, fixed batch=16
analysis geometry: 320×160
tracking: none
```

YOLO 尾批和 RITnet 尾批都只在固定 ONNX 输入所需时复制最后一个真实样本补齐，padding 输出被丢弃；正式结果仍是一帧一条 frame identity，不跳帧。

本版本的同机同段完整 benchmark（sub-031，1800 帧）约为 30.50 FPS；旧正式运行约 20.21 FPS。性能数字只描述该测试硬件和数据段，不保证跨设备一致。

## 8. 安装完成后的下一份文档

不要直接从本文件跳到全量生产。继续读取：

1. `README.md`：当前 NIR 科研/算法口径；
2. `RUNBOOK_V1.md`：第二台机器操作、protocol gate、provenance 和中央交付。
