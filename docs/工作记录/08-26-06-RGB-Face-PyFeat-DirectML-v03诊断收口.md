# 08-26-06｜RGB Face Py-Feat DirectML v0.3 诊断收口

> 2026-08-26｜分支：`rgb-dev`｜承接 `08-26-05-RGB-Face-PyFeat-DirectML-Gate1阻断诊断.md`。本记录保留前序异常诊断的历史结论，同时记录 v0.3 profiling 对 Py-Feat multitask DirectML compatibility 的修正证据。

## 1. v0.3 实机结果

对既有 `pyfeat211_multitask_scientific_core.onnx`、batch=1 运行 `rgb-face-directml-diagnostic-v0.3`。环境：Python 3.11.15、ONNX Runtime 1.24.4，available providers 为 `DmlExecutionProvider`、`CPUExecutionProvider`。

### fallback-allowed 模式

- requested providers：DML + CPU；
- status：ok；
- 7 类 scientific outputs 全部 finite；
- ORT profile：`DmlExecutionProvider=1` fused kernel event；
- `CPUExecutionProvider=0`；
- `cpu_execution_observed=false`。

### strict-DML 模式

strict 模式同时设置：

```text
providers = [DmlExecutionProvider]
session.disable_cpu_ep_fallback = 1
Python InferenceSession wrapper fallback = disabled
```

实机结果：

- status：ok；
- 7 类 scientific outputs 全部 finite；
- ORT profile：`DmlExecutionProvider=1` fused kernel event；
- `CPUExecutionProvider=0`；
- `cpu_execution_observed=false`。

因此可以确定：**当前 Py-Feat 2.1.1 multitask scientific core ONNX 本身能够在该 AMD / DirectML runtime 上完整执行，batch=1 下没有观察到 CPU kernel。** 前一版 Gate 1 中“multitask 完全落到 CPU”不能再解释为模型架构不兼容 DirectML。

## 2. 对最初 Gate 1 CPU-only 结果的修正解释

第一版 `face_directml_probe.py` 允许 Python `InferenceSession` provider-level fallback。如果 DML session 创建失败，Python wrapper 可以整体重建 CPU session，从而得到 `session_providers=[CPUExecutionProvider]` 且仍然 `status=ok`。

原始 Py-Feat Gate 1 的执行顺序还包括先测试 RetinaFace batch 1/8/16/32；其中 RetinaFace batch 32 随后异常。因此“batch 32 / DirectML 资源或 EP 状态异常导致后续 multitask DML session 创建失败，再被 wrapper 静默重建成 CPU session”是当前主要工程假设，但**尚未被直接证明**，不得写成确定因果。

能够确定的是：v0.3 在独立后续进程中，normal 与 strict 两种模式均由 DML profile 证明确实执行，无 CPU kernel。

## 3. Probe 修复

`scripts/face_directml_probe.py` 升级为 `rgb-face-directml-probe-v0.2`：

- 继续注册 DML + CPU，以便 ORT graph-level unsupported node 可以由 profile 明确暴露；
- 关闭 Python wrapper provider fallback：`enable_fallback=0` + `session.disable_fallback()`；
- 如果 DML session 本身创建失败，直接记录 error，不再静默重建 CPU-only session；
- `session.get_providers()` 只作为注册 provider 信息，实际执行以 profile kernel counts 为准。

对应 commit：`638fc61`（`fix(rgb): prevent silent ORT session fallback in DirectML probe`）。

## 4. 当前 Gate 状态

- Py-Feat Gate 0：PASS；
- RetinaFace DirectML compatibility：PASS，原测试 batch 1/8/16 均有 DML kernel、0 CPU kernel；batch 32 单独记为异常/不作为候选；
- multitask scientific core DirectML compatibility：**v0.3 batch=1 strict-DML 已证明 PASS**；
- Py-Feat 整体 Gate 1：还差一次使用修复后 probe 的干净 batch 吞吐复测后收口，不再属于“模型不支持 DirectML”的阻断状态。

## 5. 下一步

不重跑 CPU benchmark，也不重新导出 ONNX。只使用修复后的 probe 做小范围干净复测：

1. RetinaFace：batch 1/8/16；不再把已异常的 batch 32 放在同一次进程前段；
2. multitask：batch 1/8/16；
3. 若均为 DML kernel >0、CPU kernel=0，则 Py-Feat Gate 1 正式 PASS；
4. 然后再与 LibreFace 一起进入同一 300 帧真实输入 parity / end-to-end；
5. 如果后续工程复杂度或真实速度明显不利，可以基于实证选择 LibreFace，不需要为了保留 Py-Feat 而强行增加正式 pipeline 复杂度。

本记录不冻结 Face backend。

## 6. 修复后 Gate 1 干净复测：PASS

随后使用 `rgb-face-directml-probe-v0.2`，将 RetinaFace 与 multitask 分成两个独立 Python 进程，只测试 batch 1/8/16，不再测试已异常的 RetinaFace batch 32。

### RetinaFace R34

| batch | images/s | ms/image | DML kernels | CPU kernels | status |
|---:|---:|---:|---:|---:|---|
| 1 | 78.09 | 12.8054 | 13 | 0 | ok |
| 8 | 83.07 | 12.0386 | 13 | 0 | ok |
| 16 | 82.70 | 12.0917 | 13 | 0 | ok |

三个 batch 全部输出 finite，profile 中仅观察到 `DmlExecutionProvider` kernel，`cpu_fallback_observed=false`。当前吞吐仍以 batch 8 最优，batch 8 与 16 非常接近，因此后续真实 pipeline 优先以 batch 8 作为 RetinaFace 候选。

### Multitask scientific core

| batch | images/s | ms/image | DML kernels | CPU kernels | status |
|---:|---:|---:|---:|---:|---|
| 1 | 141.46 | 7.0694 | 13 | 0 | ok |
| 8 | 236.33 | 4.2313 | 13 | 0 | ok |
| 16 | 256.88 | 3.8929 | 13 | 0 | ok |

三个 batch 均输出 AU、emotion、V/A、gaze、pose、478 mesh、52 blendshapes，所有输出 `finite_fraction=1.0`；profile 中只观察到 DML kernel，CPU kernel=0。batch 16 在本次 model-core probe 中吞吐最高。

因此截至本轮：

- Py-Feat Gate 0：**PASS**；
- RetinaFace Gate 1：**PASS（batch 1/8/16）**；
- multitask scientific core Gate 1：**PASS（batch 1/8/16）**；
- Py-Feat 整体 DirectML Gate 1：**正式 PASS**；
- 旧 probe 中 multitask CPU-only 结果保留为历史诊断记录，但不再作为当前 compatibility 结论。

这些数字仍然只属于 synthetic model-core benchmark，不能和既有 300 帧 CPU end-to-end 速度直接比较，也不能据此冻结 Face backend。

## 7. 下一阶段：真实 300 帧 parity + end-to-end

LibreFace 与 Py-Feat 现在都已经通过 Gate 0/1。下一步停止 DirectML compatibility 调试，进入同一批既有连续 300 帧的真实输入验证：

1. LibreFace：复刻既有 alignment / MediaPipe landmark 特征与 ONNX heads，逐字段对照现有 CPU reference；
2. Py-Feat：RetinaFace DML → exact prior/decode/NMS → 1.2 isotropic square/reflection crop → 224 ImageNet preprocess → multitask DML → canonical pose/gaze/mesh postprocess；
3. parity 与速度分开判断，至少报告 face coverage、bbox、AU、emotion、V/A、gaze、pose、mesh、blendshape、missing/multi-face、finite fraction；
4. 单独测 raw-frame end-to-end wall time，不能用 model-core synthetic images/s 代替；
5. 最后结合信息完整性、工程复杂度、visual sanity check 与 AMD 实际吞吐冻结正式 Face backend / fps / primary-face 规则。

本轮没有重跑 CPU benchmark，没有删除或覆盖既有历史输出。