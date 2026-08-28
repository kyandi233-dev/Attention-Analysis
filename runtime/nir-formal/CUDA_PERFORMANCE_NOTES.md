# NVIDIA CUDA v8 性能继承与验收说明

本文只记录 NVIDIA/CUDA 执行层性能设计，不修改 NIR v8 的科学定义。当前正式科学契约仍由 shared v8 core、schema、ROI、uncertainty、temporal、QC 与 completion 代码决定。

## 1. 对照的“上次稳定版本”

本次性能回查使用 `nvidia-cuda` 在 2026-08-27/28 这轮大改之前的稳定快照作为主要参考：

```text
8aabb49cf034849954013701b302b9f4c4b8d6f9
```

该快照属于当时 NVIDIA/CUDA `1.0.1` 稳定运行包。它与今天之后在旧 `nvidia-cuda` 上出现的 full-class/lean/CUDA 实验性修改不是同一个比较边界。

## 2. 稳定版曾经做过的有效加速

### 2.1 历史 formal producer：YOLO b8 + RITnet b16

稳定 NVIDIA producer 曾把逐帧 YOLO 调度改为 bounded batch=8，并保持 RITnet fixed batch=16。历史文档记录的 representative benchmark 大约为：

```text
sub-031 / 1800 frames
优化后约 30.50 FPS
旧正式运行约 20.21 FPS
```

这个数字描述的是当时的完整 historical formal producer，不是当前 final full-class 的速度。

当前 v8 final full-class **不重跑 YOLO**，直接消费已经保存的 `eyes.csv` bbox，因此 YOLO b8 不属于当前 final full-class 的可再优化项。历史 `yolo_batch_size` 只作为 source provenance 保留。

### 2.2 stable full-class：fixed b16 + labels-only transfer

稳定 full-class CUDA adapter 使用固定 b16；尾批复制最后一个真实 ROI 补齐固定输入，然后丢弃 padding output。

production 在未开启 pupil validation 时只向 CPU 请求 hard `labels`，不返回 pupil probability。这显著减少了 CUDA→CPU 输出传输。

### 2.3 stable full-class：三阶段 overlap

稳定 full-class 的执行结构是：

```text
producer worker=1
    decode / ROI / preprocess batch N+1

CUDA main thread
    infer batch N

postprocess workers=4
    summarize batch N-1 eyes
```

旧实现按 eye 把后处理任务提交给四线程 pool，并允许后处理队列跨 batch 滞后，因此 CPU prepare、CUDA infer 和 CPU postprocess 可以重叠。

### 2.4 CUDA session fail-closed

稳定 CUDA session 已经使用：

```text
CUDAExecutionProvider primary
ORT_ENABLE_ALL
session.disable_cpu_ep_fallback = 1
session.disable_fallback()
use_tf32 = 0
FP32
```

没有发现稳定版还存在额外的 CUDA provider 参数、IO binding、pinned-memory 或其他隐藏 session 加速设置。

## 3. 当前 NVIDIA v8 已继承什么

当前 `nvidia-cuda-v8` 保留：

```text
fixed RITnet b16
fixed-shape tail padding + discard
CUDAExecutionProvider primary
CPU EP/runtime fallback disabled
use_tf32=0
FP32
producer worker=1
producer ↔ CUDA ↔ summary overlap
ordered checkpoint consumption
```

当前 v8 还比旧 stable full-class 多做了这些工程优化：

- production 不再返回完整 max-probability / margin / entropy 三张 map；
- cohort 只请求 `labels + class_probability`；
- 三项 uncertainty mean 从 `class_probability` 懒派生，不持久化完整 uncertainty maps；
- 完整输出结构/概率合法性只做有界周期 validation，而不是每批重复完整扫描；
- plain CSV、lean checkpoint payload、bounded QC 与 final integrity 流程已经收敛。

## 4. 本次从 stable 恢复的 NVIDIA-specific 调度

AMD v8 基线默认：

```text
summary_workers = 2
max_pending_summaries = 2
```

stable NVIDIA full-class 默认后处理为 4 workers。当前 CUDA GPU 预期比 DirectML 更快，因此继续使用 AMD 的 2-worker 默认会更容易暴露 CPU summary 阶段。

`nvidia-cuda-v8` 已恢复为：

```text
summary_workers = 4
max_pending_summaries = 4
```

这两个参数只控制 CPU scheduling。结果仍按 batch/ordinal 顺序消费并写入 checkpoint，不修改 hard metrics、soft fractions、uncertainty、temporal、schema 或 source identity。

## 5. 为什么不能直接恢复 stable labels-only production

当前 v8 正式保留：

```text
4 soft class fractions
ocular max-probability mean
ocular top1-top2 margin mean
ocular entropy mean
```

这些量不能从 hard labels 唯一恢复。当前 cohort 因此必须获得四类 `class_probability`，其固定 b16 tensor 为：

```text
[16,4,400,640] float32
约 62.5 MiB / call
```

所以旧 stable 的 labels-only 模式现在只能作为**传输性能参考**，不能直接替换正式 v8，否则会丢科学信息。

## 6. 当前隔离 benchmark

运行：

```powershell
python benchmark_ritnet_final_output_transfer.py `
  --run-dir "<严格完成的 historical formal run>" `
  --config config.yaml `
  --device 0
```

benchmark 使用同一个真实 b16 tensor、同一个 final ONNX、同一个 CUDA session，交错比较：

```text
labels_only_stable_reference
cohort_current            # labels + class_probability
full_five_output
```

输出会直接给出：

- 每 call 返回 bytes / MiB；
- mean / median / p95 latency；
- current v8 相对 stable labels-only 的额外 MiB；
- current/stable returned-byte ratio；
- median/p95 额外耗时与倍率；
- labels parity；
- class_probability 在 compact/full 请求模式下的 exact parity。

默认 JSON：

```text
outputs/nvidia-cuda/ritnet-final-output-transfer.json
```

该 benchmark 不写正式科研结果、不修改 SQLite checkpoint。

## 7. 下一步优化边界

只有 NVIDIA 实机 benchmark 证明 `class_probability` 回传造成**显著且可重复的 wall-time 成本**，才继续开发 compact/scalar-output ONNX candidate。

候选方向必须满足：

1. RITnet weights/logits/preprocessing 完全不变；
2. hard labels 与当前 final ONNX 一致；
3. 对无 padding ROI，在 graph 内计算 four soft fractions、ocular count 和 three ocular means；
4. 对 padded ROI 保留当前 source-valid `class_probability` fallback，不能让 synthetic padding 进入科学分母；
5. candidate 先作为 benchmark/qualification 路径存在；
6. 逐值 parity、padded fallback、CUDA 实机速度与 CI contract 全部通过前，不进入 production work identity。

不允许为了追求 stable labels-only 的速度删除 soft fractions、uncertainty means、padding exclusion、pupil geometry、temporal QC 或其他当前 v8 正式信息。

## 8. 实机验收时应同时看什么

不要只看总 FPS。代表被试运行时至少记录：

```text
producer_total_ms
preprocess_ms
session_run_ms
output_validation_ms
gpu_and_transfer_ms
hard_metric_ms
uncertainty_ms
summary_total_ms
```

如果 CUDA inference 已明显快于 summary，则 4-way scheduling 的收益应体现在 GPU 等待减少；如果 `cohort_current` 相对 labels-only 的 median/p95 差距明显，则再进入 scalar-output candidate 阶段。

历史 stable 的 30.50 FPS 只能用于理解 NVIDIA producer 曾经的优化量级，不能作为当前 full-class v8 的直接通过阈值。
