# NVIDIA NIR full-class：旧版 vs 当前同口径性能对照

## 状态

`PARTIAL`：shared-path 对照和旧 formal workflow 证据已完成；两者的模型输出与
后处理范围不同，不能把 micro benchmark 差异宣称为纯 RITnet 算法 speedup。当前
未启动 71 人队列、未重跑 YOLO、未改正式配置或覆盖 completion。

## Canonical 旧实现

Git history 与 NIR 文档将 `324413c` 的
`runtime/nir-formal/ritnet_onnx_runtime.py` 确认为此前正式使用的 ORT CUDA RITnet
路径；旧 formal 的真实 `sub-059` manifest 也记录 `pytorch-cuda`、FP32、batch16。
旧模型为 `ritnet-b16-fp32.onnx`（SHA-256
`1933F44F483B350E17249A37B4A2EBE8B5E32F83FC8C1EB1A21C27E96477E621`）。当前 full-class
使用 `ritnet-b16-fp32-uncertainty.onnx`（SHA-256
`599F79B89CAE455CE3BF412DD28F00438140D2F739D585C2E2223681B03EAE6D`）。

文档同时明确了口径差异：旧版输出两类结果并做 pupil-only contour/ellipse 后处理；
当前版输出五类结果，直接 640×400 推理后执行 full-class metrics、source-backed
mask、uncertainty/temporal/QC 逻辑。因此下表是“同输入 shared-path mixed-scope
对照”，不是纯同算法对照。

## 第一组：同一真实输入的 micro/pipeline benchmark

两版均使用 `sub-059` 同一 AVI、前 1,024 eye rows、529 frames、RTX 5070、同一
`.venv_nir_gpu`、FP32、batch16、相同 warm-up 规则和真实路径：视频读取 → canonical
ROI → resize/preprocess → CUDA RITnet inference → 必要后处理。两版均不包含 YOLO，
均不生成正式 QC 文件。

| version | commit / implementation | model / outputs | input / precision | batch | eyes / frames | wall time | eyes/s | frames/s | GPU avg / P95 | VRAM peak / total | CPU peak | RAM peak | provider |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| old | `324413c` `ritnet_onnx_runtime.py` | `ritnet-b16-fp32.onnx`, 2 outputs | 640×400 / FP32 | 16 | 1,024 / 529 | 9.90 s | 103.48 | 53.46 | 80.0% / 98% | 10.45 / 11.94 GiB | 372% | 1.40 GiB | CUDAExecutionProvider |
| current | `2a988f7` benchmarked current full-class runtime | uncertainty ONNX, 5 outputs | 640×400 / FP32 | 16 | 1,024 / 529 | 15.74 s | 65.04 | 33.61 | 73.1% / 99% | 11.24 / 11.94 GiB | 930.9% | 1.70 GiB | CUDAExecutionProvider |

机械计算结果：当前/旧 `eyes/s = 0.629`，即当前 shared-path 吞吐约 **低 37.1%**；
同一 1,024-eye wall time 约增加 **59.1%**。由于模型输出结构、后处理和 metrics
范围不同，这不是可正式归因于“RITnet 算法变慢”的纯性能结论；准确结论是：在同一
真实输入上，当前 full-class mixed-scope pipeline 较旧 pupil-only pipeline 更慢。

## 第二组：实际生产 workflow 对照

### 旧 formal NIR（真实历史证据）

`sub-059` 的历史 `summary.json` 记录：39,411 frames、`elapsed_sec=1065.536`、
`processing_fps=36.987`，并包含 YOLO/ROI/RITnet 的完整 formal 产物写入；旧路径会
重新执行逐帧 YOLO，再执行 RITnet，并生成旧 formal outputs。该时间是实际 runtime，
不是文件修改时间推断。

### 当前 full-class 补跑

当前路径复用已完成的 historical `eyes.csv` 与 AVI，不重跑 YOLO；执行 full-class
RITnet、全量 metrics、压缩数据表、QC index 与 sparse pixel evidence。当前没有运行
完整 `sub-059` wall-time；按同一真实片段的 65.04 eyes/s 外推其 77,725 eyes：

- 纯 measured-throughput：约 19.9 min；
- 加入 finalization、QC、写盘和波动的工作估计：约 **21.9 min**。

旧真实时间约 **17.8 min**，所以当前 workflow 的约 21.9 min 只能报告为 estimate，
约比旧 workflow 高 **23%**，不能称为算法加速。没有可靠证据把“省掉 YOLO”的时间
单独拆成净节省，因为当前 full-class 增加了五输出、full-class metrics、source-backed
分析域和 QC 写入成本。

## 结论

1. **RITnet/shared-path 性能**：旧 103.48 eyes/s，当前 65.04 eyes/s；mixed-scope
   变化约 -37.1%，结论为当前路径更慢，但不能归因成纯算法回归。
2. **实际生产 workflow**：旧 sub-059 为 17.8 min 实测；当前约 21.9 min 估计。当前
   的主要工作流变化是省掉 YOLO 重跑，但被更重的 full-class 输出/metrics/QC 抵消，
   因此目前没有可证实的生产时间节省。
3. 未发现 OOM、CUDA/provider 错误或输入/输出完整性错误；当前显存余量约 5.8%，
   不满足更高 batch 或 subject 并行的安全线。
4. 当前不建议启动正式 71 人队列；应先决定是否接受当前 full-class 的 mixed-scope
   成本，或另行设计不改变科学结果的 runtime 优化。
