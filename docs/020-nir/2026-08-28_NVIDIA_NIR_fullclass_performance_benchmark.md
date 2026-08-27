# NVIDIA NIR full-class 性能压力与吞吐基准

## 状态

`PARTIAL`：安全可用的 batch16 吞吐已测得；冻结 ONNX 是固定 batch16，无法在
不改模型/科学逻辑的前提下测试 batch24/32/40/48；未执行完整单被试 wall-time。

本轮未启动 71 人队列、未重跑 YOLO、未修改正式 `config.yaml`、未覆盖或移动任何
正式 completion。benchmark 输出只打印 JSON，不写 formal completion 或正式结果目录。

## Reuse Gate 与输入

- 仓库没有可复用的 NIR full-class benchmark/timing 工具；复用现有
  `RitnetFullClassFinalRuntime`、canonical ROI、source eyes 与 metric adapter。
- 代表被试：`sub-059`，合法完整 formal source，当前没有 full-class final completion，
  使用 `J:/Data/sub-059_/nir/sub-059_nir.avi`。
- 固定真实片段：同一 source 的前 1,024 eye rows、529 个 frame；每档使用完全相同
  的真实 AVI/ROI 输入。
- 路径：`.venv_nir_gpu\Scripts\python.exe`，RTX 5070，FP32，640×400，
  `CUDAExecutionProvider`。

## 实测结果

| batch | eyes/s | frames/s | GPU avg | GPU P95 | VRAM peak / total | 余量 | CPU peak | RAM peak | wall time | 稳定性 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 65.04 | 33.61 | 73.1% | 99% | 11.24 / 11.94 GiB | 5.8% | 930.9% | 1.70 GiB | 15.74 s / 1,024 eyes | PASS；无 OOM/error |
| 24 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 未执行 | 固定 ONNX `[16,1,400,640]`，安全拒绝 |
| 32 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 未执行 | 固定 ONNX，安全拒绝 |
| 40 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 未执行 | 固定 ONNX，安全拒绝 |
| 48 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 未执行 | 固定 ONNX，安全拒绝 |

GPU memory total/peak 来自 benchmark 期间 `nvidia-smi` 采样；CPU/RAM 为 benchmark
进程采样。I/O 瓶颈未观察到，provider 实际为 `CUDAExecutionProvider`。输出完整性检查
通过：1,024 labels、1,024 次 hard-metric 后处理，label shape/value 合同有效。

## 参数判断

- 推荐 `batch_size = 16`：这是唯一与冻结模型兼容的 batch；其余档位不能安全执行。
- 推荐 worker：保持当前 launcher 默认；实际 `run_ritnet_fullclass_batch.py` 不接受
  `--postprocess-workers`，因此没有安全 worker 参数可压测，不做临时重构。
- subject 并行：否。当前 batch16 GPU 采样已达高位且显存余量只有约 5.8%，不满足
  并行的 VRAM/RAM/I/O 安全条件。
- 主要瓶颈：A/E 混合（CUDA 推理与 CPU decode/ROI/preprocess/postprocess 共同供给）；
  未观察到 J: 盘 I/O 成为主瓶颈。更高 batch 的模型契约和显存安全余量首先构成停止门。

## 71 人时间估计

安全 source 扫描得到 71 个合法 formal source，共 2,864,864 frames / 5,547,232 eyes；
其中 3 个已有 final completion，实际待处理为 68 个。按实测 65.04 eyes/s（等价约
33.61 frames/s）并加入 subject 切换、打开/关闭、finalization、QC、磁盘写入和波动：

| 范围 | 乐观 | 典型 | 保守 |
|---|---:|---:|---:|
| 71 人（含 3 个严格 skip） | 24.5 h | 26.1 h | 30.8 h |
| 剩余 68 人（不含已有 completion） | 23.5 h | 25.0 h | 29.5 h |

典型普通被试按约 78,130 eyes / 40,350 frames 估计，约 **22 分钟**；这是吞吐估计，
不是完整单被试 wall-time 实测。由于 5.8% 显存余量低于目标安全线，保守值额外保留
更大的长时波动与恢复空间。

## 科学结果一致性与正式启动结论

本轮没有可安全运行的 batch>16 候选，因此没有发生 batch 间科学输出差异；batch16
自身的 frame/eye 输入计数、label 合同、指标后处理和 CUDA provider 检查通过。不存在
“最佳 batch”与 baseline 的数值对照证据，不能把本轮称为优化后的生产参数验收。

结论：可以把 **batch16 / 单 subject 串行** 作为当前安全候选，但在补充完整单被试
wall-time、确认长时显存稳定性并评估余量前，不建议直接开始正式 71 人队列。
