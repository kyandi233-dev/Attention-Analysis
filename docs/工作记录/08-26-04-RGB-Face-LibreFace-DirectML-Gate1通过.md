# 08-26-04｜RGB Face LibreFace DirectML Gate 1 通过

> 2026-08-26｜分支：`rgb-dev`｜承接 `08-26-03-RGB-Face-DirectML首次实机导出修复.md`。本记录只保存 LibreFace ONNX Runtime DirectML Gate 1 的实机结果；不改写既有 CPU benchmark，也不据此冻结 Face backend。

## 1. Gate 1 结论

LibreFace 当前 Python reference 已完成 Gate 0 导出后，在 Windows AMD 开发机使用独立 `attention-face-directml` 环境执行 `scripts/face_directml_probe.py`。

运行环境：

- Python 3.11.15；
- ONNX Runtime 1.24.4；
- available providers：`DmlExecutionProvider`、`CPUExecutionProvider`；
- device_id=0；
- `enable_mem_pattern=false`；
- `execution_mode=ORT_SEQUENTIAL`；
- graph optimization=`ORT_ENABLE_ALL`；
- synthetic random input；
- warmup=3；timed iterations=10；
- batch=1 / 8 / 16 / 32。

三个模型、四个 batch 共 12 个组合全部 `status=ok`，所有输出 `finite_fraction=1.0`。ORT profile 中每个组合均只记录 `DmlExecutionProvider` kernel event，`cpu_kernel_events=0`、`cpu_fallback_observed=false`。因此当前 LibreFace 三个 ONNX learned core 在该 AMD / DirectML runtime 上没有观察到 CPU fallback，**Gate 1 正式 PASS**。

## 2. Model-core 实测

### AU joint

| batch | images/s | ms/image | CPU fallback |
|---:|---:|---:|---|
| 1 | 883.38 | 1.1320 | 否 |
| 8 | 2078.46 | 0.4811 | 否 |
| 16 | 2739.33 | 0.3651 | 否 |
| 32 | 2826.53 | 0.3538 | 否 |

batch 16 → 32 的 model-core throughput 仅再增加约 3.2%，说明 CNN 主干在 batch 16 附近已经接近该测试条件下的吞吐拐点。

### Expression

| batch | images/s | ms/image | CPU fallback |
|---:|---:|---:|---|
| 1 | 684.14 | 1.4617 | 否 |
| 8 | 1965.67 | 0.5087 | 否 |
| 16 | 2241.07 | 0.4462 | 否 |
| 32 | 2290.74 | 0.4365 | 否 |

batch 16 → 32 只再增加约 2.2%，同样显示 batch 16 已接近吞吐平台。

### Gaze MLP

| batch | images/s | ms/image | CPU fallback |
|---:|---:|---:|---|
| 1 | 4502.27 | 0.2221 | 否 |
| 8 | 24961.00 | 0.0401 | 否 |
| 16 | 50983.02 | 0.0196 | 否 |
| 32 | 104742.89 | 0.00955 | 否 |

该 MLP 极小，model-core 数字很高，但正式 LibreFace gaze 仍需要 CPU 侧 MediaPipe landmark / feature extraction。因此该数值不能代表 raw-frame gaze end-to-end，也不用于单独选择正式 batch。

## 3. 当前解释边界

Gate 1 只证明：

1. 当前导出的 AU / expression / gaze learned graphs 能在 `DmlExecutionProvider` 上执行；
2. 这次 profile 没有观察到 CPU kernel fallback；
3. batch 1/8/16/32 均可运行；
4. AU 与 expression 在 batch 16 后收益已经很小。

Gate 1 **不能证明**：

- ONNX 数值与现有 PyTorch CPU reference 已经逐字段 parity；
- MediaPipe alignment、landmark/gaze feature extraction 等 CPU 前处理已经被 GPU 化；
- synthetic model-core speed 等于 300 帧 raw-frame end-to-end speed；
- LibreFace 已经优于 Py-Feat；
- Face backend 可以冻结。

因此当前只把 batch 16 视为真实 pipeline 的默认候选拐点，batch 32 保留为 secondary throughput candidate；最终 batch 要结合真实 300 帧 end-to-end、内存和 parity 后再决定。

## 4. 下一步

为保持两候选验证阶段对称，**暂不直接进入 LibreFace 300 帧真实输入**。下一步先完成 Py-Feat：

1. `attention-face-pyfeat` Gate 0：从 Py-Feat 2.1.1 当前 reference 导出 RetinaFace R34 + Detectorv2 multitask scientific core；
2. `attention-face-directml` Gate 1：同样测试 batch 1/8/16/32、provider、CPU fallback 与 model-core；
3. 两边 Gate 0/1 都通过后，再实现同一批既有 300 帧的真实输入 DirectML runner；
4. 与既有 CPU parquet 做逐字段 parity，速度与 parity 分开报告；
5. 最后结合 coverage、raw schema、visual sanity check 冻结 backend / fps / primary-face。

本轮没有运行新的 CPU benchmark，没有删除或覆盖历史输出。