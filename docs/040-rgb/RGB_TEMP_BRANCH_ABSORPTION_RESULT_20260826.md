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

NVIDIA 当前正式架构已经从早期“single-pass draft + derived blocking”调整为：

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
| Parquet stable nullable schema | 当前 AMD formal Face/Motion/Pose raw 主要为整表写出，临时分支的 streaming `null` 首 chunk 故障路径不直接适用 | **absorbed** | 新 `face_formal_cuda.py` streaming writer 在首个 schema 建立前显式固定 nullable numeric/int/bool/string dtype；保留 provenance `ported-from: 51d17c9a...` |
| schema regression test | AMD 当前正式 writer 不依赖临时 streaming writer；未为了形式一致强行复制旧测试 | **absorbed / migrated** | 新 `tests/test_rgb_formal_schema.py` 直接测试当前 CUDA Face writer：首 chunk numeric 全空、下一 chunk 出现 double 时仍可 append，读回保持 numeric |
| Pose nullable handling | 当前 AMD Pose 正式 raw 为整表落盘；不需要复制临时分支旧 streaming patch | **already covered by current non-streaming Pose formal writer** | 不复制旧 Pose patch；若未来改成 streaming，再复用 stable schema rule |
| Face sample nullable handling | 当前 AMD 15 Hz frame manifest + final Face raw writer 已有自己的 schema | **superseded by new formal frame grid + CUDA raw writer** | 不复制旧 sample patch；保留相同“缺失不提前删、frame identity 不丢”原则 |
| `rgb_formal_audit_v1.py` | 已有 `scripts/rgb_analysis.py --stage audit` | **already superseded** | 不新增第二套 audit 入口；检查思想继续由现有 audit/inventory 承担 |
| `face_formal_reference.py` | N/A | **absorbed** | 已加入 `rgb-nvidia`，用于 sub-130 native CPU reference ↔ native CUDA parity |
| `rgb_formal_full_runner_v1.py` | reference / superseded | **reference / superseded** | 不作为正式入口；single-pass decode 设计保留为历史参考，当前默认采用 Motion/Pose/Face 三条独立 reader 并行，以避免把三个消费者绑在一个生产者上 |
| `047-RGB当前暂停状态_20260826.md` | provenance only | provenance only | 失败发生于工程 serialization；partial outputs 不构成科学结果，也不用于 resume 完成判定 |

## 3. 当前新增/更新的 NVIDIA 正式资产

### Formal timestamp grid

```text
src/attention_pipeline/rgb/face_formal.py
scripts/face_formal_prepare.py
```

作用：baseline start → Block2 end，timestamp-driven 15 Hz，不提前抽 JPEG；完整保留 frame/capture/unix_ms/phase/behavior/gap provenance。

### Motion / Pose raw formal wrapper

```text
scripts/rgb_formal_motion_pose.py
```

只生成正式 raw：

```text
motion_raw.parquet
pose_landmarks.parquet
```

Pose features 后移。

### Native CUDA Face formal runner

```text
scripts/face_formal_cuda.py
```

固定执行逻辑：

```text
py-feat == 2.1.1
Detectorv2(device="cuda", identity_model=None)
torch.cuda.is_available() == True
data_type="tensor"
face_detection_threshold=0.5
```

禁止 silent CPU fallback。

NVIDIA native Detectorv2 使用一个端到端 `native_cuda_batch`，不把 AMD DirectML 的 RetinaFace batch / multitask batch 当作 NVIDIA executor 参数。

### Raw validator

```text
scripts/rgb_formal_validate.py
```

正式完成只要求：

```text
Motion raw
Pose landmark raw
Py-Feat Face raw
```

并额外要求 NVIDIA Face manifest：

```text
execution_backend = pytorch_cuda
device = cuda / cuda:<index>
```

### Orchestration

```text
scripts/run_rgb_formal_subject.ps1
scripts/run_rgb_formal_cohort.ps1
```

单被试默认三线并行；完成分支可 resume/skip。cohort runner 已实现代码，但在 RTX 5070 sub-130 Gate 完成前不批准正式全量。

### Stable schema regression protection

```text
tests/test_rgb_formal_schema.py
```

直接针对当前 CUDA streaming writer，而不是测试一个已经废弃的临时 runner。

### CPU reference

```text
scripts/face_formal_reference.py
```

从临时分支吸收，用于 NVIDIA representative CPU↔CUDA parity。

## 4. 关键实现 commits

本轮 `rgb-nvidia` 关键提交：

```text
9e6445b0b798dfeab0f65c5d155fa3f39e3839d0  formal Face timestamp grid
919d88e65139130ce925b5cd43d6fcabe9f23425  Face prepare entrypoint
91c1ac7ba7e1814f7962c066baa6e95c49aa0422  raw-only Motion/Pose wrapper
0864fedef9fa11f96ad16f851ce79b3e46eb482d  native PyTorch/CUDA formal Face runner
013822646a52828d8c5bd97ff7d7d46d2fb6dfeb  raw-first NVIDIA validator
6fad04292bd8a082c6a501a797a4642114c57c6a  parallel subject orchestrator
3976c845e323236af77f94b33fd36b3d1e0b06d5  cohort resume runner
18af6faa707308fbc018b42109e12fec72f41856  CPU Face reference helper
bbfea5a300acef20ca6da025b3e3c88d17e393b7  stable schema regression test
cdd1d02523c7f50d12f71a417c693faf5c15fdbc  native CUDA/raw-first config
```

后续文档/配置修正提交继续位于同一 `rgb-nvidia` 分支；以该分支最新 HEAD 为准。

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

### 已建立的代码级回归保护

```text
tests/test_rgb_formal_schema.py
```

### 尚未在本轮聊天环境中执行的实机 Gate

由于当前没有 RTX 5070 / `D:\conda_envs\attention-face-cuda` 的实际执行环境，本轮只能完成仓库代码与调用路径同步，**不能宣称 CUDA full-span 或 schema test 已经在 NVIDIA 工作站通过**。

在正式 cohort 前必须在 NVIDIA 工作站执行：

1. `tests/test_rgb_formal_schema.py`；
2. `torch.cuda.is_available()` / Py-Feat 2.1.1 环境检查；
3. sub-130 native CPU ↔ CUDA representative parity；
4. sub-130 full-span raw extraction；
5. final validator；
6. CUDA batch / peak GPU memory / throughput 记录。

## 7. 是否还有唯一有效 RGB 代码只存在于临时分支

当前裁决：**没有仍需作为正式执行入口而只能存在于临时分支的唯一 RGB 代码。**

- stable schema rule → 已迁移到当前 CUDA writer/test；
- CPU reference → 已吸收；
- audit → 现有正式 audit 已覆盖；
- single-pass runner → 明确仅作设计参考，不作为当前正式入口；
- 047 → provenance only。

但这不等于现在就可以删除临时分支。删除仍需满足交接文档中包括 NIR 独立处理、实机 Gate 和最终 provenance 记录在内的全部条件。

## 8. 删除临时分支的当前结论

```text
现在：不要删除 codex/rgb-nvidia-formal-pipeline-v1
```

RGB 代码资产已经完成裁决，但仍应等待：

- NIR `8c32eaaf...` 独立任务最终去向；
- NVIDIA sub-130 实机验证；
- 确认没有新的修复只落在临时分支。

满足全部 Gate 后，再单独执行临时分支删除操作。
