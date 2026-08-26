# RGB 临时分支吸收结果｜2026-08-26

来源临时分支：

```text
codex/rgb-nvidia-formal-pipeline-v1
```

RGB provenance commit：

```text
51d17c9a6b7db7a1114380910bb111db38293512
```

本报告只记录 RGB 资产裁决。NIR commit `8c32eaaf31f2404646103986df11c64a3c37c9f2` 继续由独立 NIR 工作线处理，本轮没有 merge/cherry-pick 临时分支，也没有删除临时分支。

## 1. 总体结论

`rgb-nvidia` 已把临时分支中仍有价值的 RGB 内容按当前正式架构逐项吸收/裁决，而不是整支 merge。

当前 NVIDIA 正式架构：

```text
15 Hz Face frame grid preparation
        ↓
Motion full FPS ─┐
Pose 10 Hz      ├─ parallel raw extraction
Face 15 Hz CUDA ┘
        ↓
raw-only validator
        ↓
tracking / eyelid / Pose features / blink / PERCLOS / QC downstream
```

NVIDIA Face executor 使用 **Py-Feat 2.1.1 native PyTorch CUDA**，没有复制 AMD ONNX Runtime DirectML executor。

## 2. 资产裁决

| 临时分支资产 | rgb-amd | rgb-nvidia | 裁决 |
|---|---|---|---|
| Parquet stable nullable schema | 当前 AMD formal Face/Motion/Pose raw 主要为整表写出，临时分支的 streaming `null` 首 chunk 故障路径不直接适用 | **absorbed** | `face_formal_cuda.py` streaming writer 显式固定 nullable numeric/int/bool/string dtype；保留 `ported-from: 51d17c9a...` |
| schema regression test | AMD 当前正式 writer 不依赖临时 streaming writer | **absorbed / migrated** | `tests/test_rgb_formal_schema.py` 直接测试当前 CUDA Face writer：首 chunk numeric 全空、下一 chunk 出现 double 时仍可 append |
| Pose nullable handling | AMD Pose 正式 raw 为整表落盘 | **already covered by current non-streaming Pose formal writer** | 不复制旧 Pose patch；未来若改 streaming 再复用 stable schema rule |
| Face sample nullable handling | AMD 15 Hz frame manifest + final Face raw writer 有自己的 schema | **superseded by formal frame grid + CUDA raw writer** | 不复制旧 sample patch；保留“缺失不提前删、frame identity 不丢”原则 |
| `rgb_formal_audit_v1.py` | 已有正式 audit | **already superseded** | 不新增第二套 audit 入口 |
| `face_formal_reference.py` | N/A | **absorbed** | 用于 sub-130 native CPU reference ↔ native CUDA parity |
| `rgb_formal_full_runner_v1.py` | reference / superseded | **reference / superseded** | 不作为正式入口；single-pass decode 仅保留为设计参考 |
| `047-RGB当前暂停状态_20260826.md` | provenance only | provenance only | partial outputs 不构成科学结果，也不用于 resume 完成判定 |

## 3. 最新 AMD → NVIDIA 同步裁决

2026-08-26 AMD `rgb-amd` 又完成了 raw-first 性能优化和 shared-decode 实验。本次再次审查这些变化，NVIDIA 的同步规则如下。

### A. raw-first / derived 后移

**已同步。** NVIDIA 正式完成只看：

```text
motion_raw.parquet
pose_landmarks.parquet
face_raw.parquet
```

tracking、主脸选择、EAR/眼睑、Pose features、blink/PERCLOS、QC、统计聚合继续后移。

### B. 三条 raw 并行

**已同步。** `scripts/run_rgb_formal_subject.ps1` 默认并行启动 Motion / Pose / Face，并按 raw + complete manifest 做 resume/skip。

### C. AMD B32/B64

**不直接移植。** AMD B32/B64 表示：

```text
RetinaFace ONNX batch = 32
multitask ONNX batch = 64
```

NVIDIA native Py-Feat `Detectorv2.detect()` 只有一个端到端 batch，因此正式配置仍是：

```yaml
native_cuda_batch: 16
native_cuda_batch_candidates: [16, 32, 64]
```

候选值仅用于 RTX 5070 benchmark，不自动选择，也不把 AMD 的两段 batch 解释为 NVIDIA 的同义参数。

### D. AMD 0.5 前置 RetinaFace anchor 过滤

**N/A，不移植。** 这是 AMD 自己调用 RetinaFace ONNX 时的 executor 优化。NVIDIA 使用 Py-Feat native `Detectorv2`，正式调用仍由：

```python
face_detection_threshold=0.5
```

交给 Detectorv2 内部处理，不绕过 Py-Feat 原生 detector pipeline。

### E. AMD RetinaFace CPU/GPU 手工流水

**N/A，不机械移植。** AMD 手工拆开 RetinaFace ONNX、NMS/crop、multitask ONNX 才能做该流水。NVIDIA `Detectorv2.detect()` 是端到端 native PyTorch/Py-Feat 调用；在没有 CUDA profiler 证据前，不把内部阶段强行拆开，以免改变正式 Py-Feat 行为。

### F. shared single-decode

**继续不作为正式默认。** AMD 实机测试显示，把 Motion/Pose/Face 绑定到一个共享视频生产者后会受到最慢消费者反压，墙钟吞吐不一定优于三条独立 reader。因此 NVIDIA 当前保持：

```text
Motion reader ─┐
Pose reader   ├─ parallel
Face reader   ┘
```

除非 RTX 5070 的实际 benchmark 给出相反证据。

## 4. 当前 NVIDIA 正式资产

### Formal timestamp grid

```text
src/attention_pipeline/rgb/face_formal.py
scripts/face_formal_prepare.py
```

baseline start → Block2 end，timestamp-driven 15 Hz，不提前抽 JPEG；保留 frame/capture/unix_ms/phase/behavior/gap provenance。

### Motion / Pose raw formal wrapper

```text
scripts/rgb_formal_motion_pose.py
```

只生成：

```text
motion_raw.parquet
pose_landmarks.parquet
```

Pose features 后移。

### Native CUDA Face formal runner

```text
scripts/face_formal_cuda.py
```

固定执行合同：

```text
py-feat == 2.1.1
Detectorv2(device="cuda", identity_model=None)
torch.cuda.is_available() == True
data_type="tensor"
num_workers=0
pin_memory=False
face_detection_threshold=0.5
```

禁止 silent CPU fallback。

### Raw validator

```text
scripts/rgb_formal_validate.py
```

额外要求 NVIDIA Face manifest：

```text
execution_backend = pytorch_cuda
device = cuda / cuda:<index>
```

### Stable schema regression protection

```text
tests/test_rgb_formal_schema.py
```

直接针对当前 CUDA streaming writer。

### CPU reference

```text
scripts/face_formal_reference.py
```

用于 NVIDIA representative CPU↔CUDA parity。

## 5. 当前正式 runner

单被试：

```text
scripts/run_rgb_formal_subject.ps1
```

Face executor：

```text
scripts/face_formal_cuda.py
```

最终完整性检查：

```text
scripts/rgb_formal_validate.py
```

cohort：

```text
scripts/run_rgb_formal_cohort.ps1
```

## 6. 当前测试状态

代码级回归保护已存在：

```text
tests/test_rgb_formal_schema.py
```

由于当前聊天环境没有 RTX 5070 / `D:\conda_envs\attention-face-cuda`，不能宣称 CUDA full-span 或 schema test 已经在 NVIDIA 工作站通过。

正式 cohort 前仍必须在 NVIDIA 工作站执行：

1. `tests/test_rgb_formal_schema.py`；
2. `torch.cuda.is_available()` / Py-Feat 2.1.1 环境检查；
3. sub-130 native CPU ↔ CUDA representative parity；
4. sub-130 full-span raw extraction；
5. final validator；
6. CUDA `native_cuda_batch` 16/32/64 的 throughput / peak-memory benchmark；
7. 冻结正式 NVIDIA batch 后再批准 cohort。

## 7. 是否还有唯一有效 RGB 代码只存在于临时分支

当前裁决：**没有仍需作为正式执行入口而只能存在于临时分支的唯一 RGB 代码。**

- stable schema rule → 已迁移到当前 CUDA writer/test；
- CPU reference → 已吸收；
- audit → 正式 audit 已覆盖；
- single-pass runner → 明确仅作设计参考；
- 047 → provenance only。

## 8. 删除临时分支的当前结论

```text
现在：不要删除 codex/rgb-nvidia-formal-pipeline-v1
```

RGB 资产已经完成裁决，但仍应等待：

- NIR `8c32eaaf...` 独立任务最终去向；
- NVIDIA sub-130 实机验证；
- 确认没有新的修复只落在临时分支。

满足交接文档全部 Gate 后，再单独删除临时分支。
