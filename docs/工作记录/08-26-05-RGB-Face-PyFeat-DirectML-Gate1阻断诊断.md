# 08-26-05｜RGB Face Py-Feat DirectML Gate 1 阻断诊断

> 2026-08-26｜分支：`rgb-dev`｜承接 `08-26-04-RGB-Face-LibreFace-DirectML-Gate1通过.md`。本记录保存 Py-Feat 2.1.1 Gate 0/1 的实机结果与当前阻断项；不重跑 CPU benchmark，不据此冻结 Face backend。

## 1. Gate 0：PASS

Py-Feat 2.1.1 reference 环境成功导出：

- `pyfeat211_retinaface_r34.onnx`；
- `pyfeat211_multitask_scientific_core.onnx`；
- `pyfeat211_onnx_export_manifest.json`。

导出环境：Python 3.11.15、torch 2.13.0+cpu、onnx 1.16.2、py-feat 2.1.1、opset 17。RetinaFace 固定输入为现有共同 300 帧的 720×1280，两个 ONNX 与源权重均已记录 SHA256。Gate 0 没有调用 `Detectorv2.detect()`，没有重复既有 300 帧 CPU reference。

## 2. Gate 1：RetinaFace 子模型通过

ONNX Runtime 1.24.4，available providers 为 `DmlExecutionProvider`、`CPUExecutionProvider`。

RetinaFace R34：

| batch | status | images/s | DML kernels | CPU kernels | fallback |
|---:|---|---:|---:|---:|---|
| 1 | ok | 79.83 | 13 | 0 | 否 |
| 8 | ok | 85.05 | 13 | 0 | 否 |
| 16 | ok | 81.98 | 13 | 0 | 否 |
| 32 | error | — | — | — | 未判定 |

batch 8 当前最高，约 85.05 frame/s；batch 16 反而略降。真实 720×1280 RetinaFace 后续默认候选因此是 batch 8，而不是机械沿用 LibreFace 的 batch 16。

batch 32 返回：

```text
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd3 in position 362: invalid continuation byte
```

`face_directml_probe.py` 对 profile JSON 的 UTF-8 读取异常已有内部捕获，因此该错误不是普通 profile parse error。当前先把 RetinaFace batch 32 记作不可用/待诊断；batch 8 已满足后续工程候选，不把 batch 32 作为当前阻断项。

## 3. Gate 1 阻断：multitask scientific core 完全没有进入 DML

`pyfeat211_multitask_scientific_core.onnx` 四个 batch 都能输出有限数值，但 session 实际只剩：

```text
CPUExecutionProvider
```

四个 batch 的 profile 均为：

```text
cpu_kernel_events = 8255
dml_kernel_events = 0
cpu_fallback_observed = true
```

因此这不是少量 unsupported node 的混合执行，而是当前完整 scientific core 没有被 `DmlExecutionProvider` 接管。当前 model-core 速度约：batch 1=21.88、8=26.81、16=24.66、32=23.04 images/s；这些数字属于 CPU fallback，不得记作 AMD GPU 性能。

当前 Py-Feat Gate 1 结论：

- RetinaFace DML：**PASS（batch 1/8/16）**；
- multitask scientific core DML：**BLOCKED**；
- Py-Feat 整体 Gate 1：**未通过**；
- 暂不进入 Py-Feat 300 帧真实输入 DirectML parity/end-to-end。

## 4. 为什么先诊断，不直接改模型

ONNX Runtime 的正常 EP 机制会把目标 EP 不支持的节点默认分配给 CPU。仅看到 `status=ok` 不代表 DML 执行。当前 profile 已证明 multitask graph 全部落在 CPU，因此下一步要找出 provider partition 的阻断原因，而不是盲目改 opset、删科学输出或重做 CPU benchmark。

新增 `scripts/face_directml_diagnose.py`：

1. 记录 ONNX graph 的 operator inventory；
2. 先建立正常 `DmlExecutionProvider + CPUExecutionProvider` session；
3. 再只请求 `DmlExecutionProvider`，并设置官方 ORT config `session.disable_cpu_ep_fallback=1`；
4. 如果存在必须依赖 CPU 的节点，strict session 应在创建或执行阶段失败，并保存异常；
5. 可用 `--verbose` 在终端打开 ORT native verbose log，配合 operator inventory 定位阻断算子/子图。

这一步不重新导出模型、不消费 300 帧视频，也不改变 scientific output schema。

## 5. 下一步命令

在现有 DirectML 环境运行：

```powershell
conda activate "D:\CondaEnvs\attention-face-directml"
cd "D:\aaawork\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-rgb-dev"
git pull --ff-only

python scripts/face_directml_diagnose.py `
  --model "D:\_AttentionData\Beijing-RGB\_test\face-directml\models\pyfeat\pyfeat211_multitask_scientific_core.onnx" `
  --batch-size 1 `
  --verbose `
  --output "D:\_AttentionData\Beijing-RGB\_test\face-directml\sub-031\pyfeat_multitask_dml_diagnostic.json"
```

优先只诊断 batch 1。provider compatibility 与 batch throughput 是两个问题；如果 batch 1 都不能 strict-DML，就没有必要先测试更大 batch。

拿到 diagnostic 后再决定是否需要：

- 只调整 ONNX export；
- 将 scientific core 按子模块/输出支路拆分导出定位阻断 branch；
- 对极少数 DML 不支持的算子做等价 graph rewrite；
- 或确认当前 Py-Feat 2.1.1 multitask architecture 不适合 DirectML。

任何修复都必须保持 Py-Feat 2.1.1 CPU reference 的科学输出定义，并在修复后重新做 Gate 1 + 300 帧逐字段 parity；不能为了 GPU 可跑而静默删字段或更换测量语义。

## 6. Strict diagnostic v0.1 实机结果与修复

首次 `rgb-face-directml-diagnostic-v0.1` 实机运行记录到：

- `fallback_allowed`：`status=ok`，session providers 为 DML + CPU；
- `strict_dml`：请求 providers 只有 `DmlExecutionProvider`，但最终 `session_providers` 仍出现 `DmlExecutionProvider` + `CPUExecutionProvider`，并且 `status=ok`。

这不能解释为 multitask 已经全 DML。原因是 ORT Python 存在两层不同 fallback：

1. ORT core 的 graph-node CPU EP fallback；
2. Python `InferenceSession` constructor 的 provider fallback。

v0.1 只通过 `session.disable_cpu_ep_fallback=1` 禁掉了第 1 层；当 strict session creation 因 DML 无法完整覆盖图而失败时，Python wrapper 默认 `enable_fallback=1` 可能捕获异常并重新用 fallback provider 创建 session。因此 v0.1 无法把 `status=ok` 当作 strict-DML 证据。

修复后的 `rgb-face-directml-diagnostic-v0.2` 在 strict 模式同时：

```text
session.disable_cpu_ep_fallback = 1
enable_fallback = 0
providers = [DmlExecutionProvider]
```

即同时关闭 graph-node fallback 与 Python constructor fallback。对应代码 commit：`1c811ab`（`fix(rgb): disable Python EP fallback in strict DML diagnostic`）。

因此 v0.1 diagnostic 只作为“诊断工具自身发现双层 fallback”的工作记录，不用于判断 Py-Feat multitask DML compatibility。

## 7. Strict diagnostic v0.2：provider 列表仍不足以判断实际执行

v0.2 实机运行后，strict 模式已经明确记录：

```text
requested_providers = [DmlExecutionProvider]
python_wrapper_fallback_enabled = false
status = ok
```

但 `session.get_providers()` 仍返回：

```text
DmlExecutionProvider
CPUExecutionProvider
```

这个结果修正了前一轮的一个过强推断：**不能仅凭 `session.get_providers()` 中出现 CPU 就认定发生了 CPU fallback。** ORT Python 的 `get_providers()` 表示 session 中注册的 execution providers，不是“本次 graph 节点实际由哪些 EP 执行”的执行证据。v0.2 因此既不能证明全 DML，也不能证明 CPU fallback。

为了避免继续用 provider 注册列表推断执行，diagnostic 升级到 v0.3：

1. fallback-allowed 和 strict 两种模式都开启 ORT profiling；
2. 实际统计 profile 中 `DmlExecutionProvider` 与 `CPUExecutionProvider` kernel events；
3. strict 模式继续保持 `session.disable_cpu_ep_fallback=1`、`enable_fallback=0`、只请求 DML；
4. 最终以 profile 的实际 kernel provider 为判据，而不是 `session.get_providers()`；
5. 如果 normal 模式为 CPU、strict 模式却能 DML-only，需要先解释 provider partition / session-option 行为，再决定是否修改科学模型。

对应代码 commit：`ae8ee92`（`fix(rgb): profile actual providers in strict DML diagnostic`）。

因此当前下一步不是拆模型，而是先运行 v0.3，取得同一 ONNX 在 normal 与 strict 条件下的实际 provider kernel 证据。