# NIR Formal Runtime（NVIDIA / CUDA v8）

这是 `Attention-Analysis` 的 NVIDIA/CUDA 最终 NIR full-class runtime。该分支直接从已经修复并验收的 AMD v8 scientific/core 基线创建；**ROI、RITnet preprocessing、pupil/uncertainty/temporal 公式、schema、QC、completion 与 source-selection 规则保持同一份 v8 契约，NVIDIA 分支只替换执行后端与机器配置。**

历史 NVIDIA formal producer 已经完成 YOLO 眼睛检测；当前最终管线严格复用历史 `eyes.csv` 的 YOLO bbox 与 `frames.csv`，从原始 NIR AVI 重建 fixed-1.6 ROI 后执行 RITnet，**不会重新跑 YOLO**。

当前正式核心：

```text
fullclass-final-core-v8-interface-safe-plain-csv
EYE_METRICS_SCHEMA_VERSION = 6
FRAME_COVERAGE_SCHEMA_VERSION = 2
execution_backend = onnxruntime-cuda
execution_provider = CUDAExecutionProvider
RITnet = FP32 / fixed b16 / 640×400
```

最终新增 full-class 输出仍要求每被试 ≤1 GiB；不全量落盘 segmentation/probability map。

---

## 1. 分支与环境

当前 NVIDIA v8 分支：

```text
nvidia-cuda-v8
```

它以 `amd-DirectML` 的 v8 final core 为基线，不从旧 `nvidia-cuda` 反向合并 full-class 逻辑。旧 `nvidia-cuda` 仅作为 NVIDIA 环境、历史 producer 与 CUDA provider 实现的 provenance/reference。

新终端建议：

```powershell
cd "<Attention-Analysis NVIDIA 本地仓库>"
git fetch origin --prune
git switch nvidia-cuda-v8
git pull --ff-only
git status --short --branch

conda activate "<NVIDIA NIR conda 环境>"
cd runtime\nir-formal
```

安装当前 NVIDIA runtime 依赖：

```powershell
python -m pip install -r requirements.txt
```

核心 ONNX Runtime 依赖固定为：

```text
onnxruntime-gpu==1.24.4
```

正式 runner 强制 clean Git worktree。不要为了通过检查直接 `git reset --hard`；先确认本地修改来源。

---

## 2. CUDA 执行约束

当前 `cuda_runtime.py` 明确要求：

- `CUDAExecutionProvider` 必须可用；
- CUDA 必须成为 primary provider；
- ONNX Runtime CPU EP fallback 被禁用；
- runtime fallback 被禁用；
- `use_tf32=0`，避免无意引入 TF32 路径；
- `--device 0` 等价于 `cuda:0`；
- CUDA 初始化失败时直接 fail closed，不静默退回 CPU。

可先检查实际 provider：

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

输出中必须包含：

```text
CUDAExecutionProvider
```

---

## 3. 最终数据流

```text
历史 NVIDIA formal completion / frames.csv / eyes.csv
        ↓ 严格验证 source identity
历史 YOLO bbox + 原始 NIR AVI
        ↓ 不重跑 YOLO
fixed 1.6 ROI + 必要 replicate padding
        ↓
640×400
        ↓
同一 final RITnet ONNX / FP32 / fixed b16
        ↓ CUDAExecutionProvider
hard 4-class + 临时 class_probability
        ↓ 只统计真实 source-backed pixels
pupil-only geometry
+ four soft fractions
+ 3 ocular uncertainty means
+ padding/QC facts
        ↓
CUDA-isolated SQLite interruption checkpoint
        ↓
temporal facts + frame coverage
        ↓
plain CSV + bounded QC
        ↓
summary + manifest + completion + ≤1 GiB
```

人工 padding 可以作为网络输入上下文，但 padding 像素本身不进入正式 hard/soft/uncertainty 科学分母。

---

## 4. 当前正式科学输出

保留：

- hard background / sclera / iris / pupil count 与 fraction；
- `iris_outer` / `ocular` union count 与 fraction；
- four-class soft fractions；
- pupil connected components / fragmentation；
- pupil ellipse / center / axes / area / diameter；
- valid-source / padding QC；
- ocular max-probability mean；
- ocular top1-top2 margin mean；
- ocular entropy mean；
- temporal delta / jump QC；
- historical YOLO provenance；
- frame coverage；
- bounded QC evidence。

不再作为正式输出：

```text
iris ellipse
iris_outer ellipse
PIR / pupil-to-iris ratio
OAR / ocular aperture ratio
cohort percentile uncertainty
cohort boundary-band uncertainty
cohort low-probability threshold fields
full hard-label store
full probability-map store
```

iris 仍保留为四分类类别，但不再作为几何归一化标尺。

---

## 5. 当前 ONNX / runtime contract

正式 cohort 推理只请求：

```text
labels              uint8   [16,400,640]
class_probability   float32 [16,4,400,640]
```

三项 ocular uncertainty mean 从 `class_probability` 在 CPU summary workers 中直接派生；production fast path 不持久化完整 max/margin/entropy map。完整五输出只用于 qualification / bounded sparse QC。

正式模型：

```text
models/ritnet-b16-fp32-uncertainty.onnx
models/ritnet-b16-fp32-uncertainty.onnx.data
```

NVIDIA v8 不重新训练 RITnet，也不使用另一套科学模型；与 AMD v8 共享相同模型内容/hash 约束。

---

## 6. checkpoint 后端隔离

NVIDIA v8 的 work identity 显式记录：

```text
execution_backend = onnxruntime-cuda
execution_provider = CUDAExecutionProvider
```

这两个字段属于 resume-critical identity。含义是：

```text
同一 CUDA checkpoint + Git/config/scheduling 非数值漂移
    → 仍需 source-prefix + payload 校验后才允许恢复

DirectML checkpoint / 未标记 execution identity 的旧 checkpoint
    → 不允许静默在 CUDA 上接着算
```

这样可以避免同一被试前半段由 DirectML、后半段由 CUDA 计算后混在一个 SQLite/final artifact 中。

完整 checkpoint 恢复时不会初始化 CUDA session，也不会重新跑全量 RITnet；它只在身份、source prefix 和 payload contract 均通过后继续 temporal/CSV/QC/finalization。

---

## 7. 代码测试

```powershell
python -m pytest tests -q
```

当前回归测试覆盖 shared v8 science，包括：fixed 1.6 ROI/padding、padding exclusion、pupil-only geometry、four soft fractions、three ocular means、uncertainty parity、component/boundary parity、temporal、coverage、plain CSV、bounded QC、completion integrity、checkpoint migration/prefix/payload guard。

NVIDIA 分支另外明确测试：

- CUDA device parser；
- 无 CUDA provider 时 fail closed；
- CPU EP/runtime fallback 禁用；
- `use_tf32=0`；
- batched YOLO tail padding；
- final RITnet runtime 绑定 `CUDAExecutionProvider`；
- CUDA 与 DirectML checkpoint execution identity 不允许混续。

CI 的 CPU runner 无法证明真实 NVIDIA GPU 推理正确，因此 CI 通过之后仍需要目标 NVIDIA 机器做 provider smoke / real-frame parity。

---

## 8. 历史 source 与运行入口

旧 NVIDIA formal producer 的数据发现逻辑曾支持：

```text
E:/正式实验
F:/正式实验
E:/Data
F:/Data
J:/Data
```

当前 `config.yaml` 保留这些候选根用于 NVIDIA 机器迁移；实际 final full-class batch 最重要的是 `--output` 指向**已经完成历史 formal producer 的输出根**，其中应存在：

```text
sub-XXX_formal_*/
├── completion.json
├── frames.csv
└── eyes.csv
```

先 dry-run：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "<NVIDIA 历史 formal 输出根>" `
  --subjects "sub-XXX" `
  --device 0 `
  --dry-run
```

确认 source selection 后正式运行：

```powershell
python run_ritnet_fullclass_batch.py `
  --output "<NVIDIA 历史 formal 输出根>" `
  --subjects "sub-XXX" `
  --device 0
```

单个历史 run：

```powershell
python run_ritnet_fullclass_extension.py `
  --run-dir "<对应历史 formal run 目录>" `
  --config config.yaml `
  --device 0
```

当前 final full-class 不存在这些旧/废弃参数：

```text
--chunk-rows
--compression
--postprocess-workers
--validate-pupil
--allow-model-mismatch
```

---

## 9. 最终输出结构

final output 位于所选历史 formal 输出根的 sibling final directory：

```text
<历史 formal 输出根>\ritnet-fullclass-final\sub-XXX\
├── data\
│   ├── eye_metrics.csv
│   └── frame_coverage.csv
├── qc\
│   ├── images\
│   │   └── *.png
│   ├── qc_index.csv
│   └── qc_pixel_evidence.npz
├── summary.json
├── manifest.json
└── completion.json
```

只有 `completion.json` 严格验证通过才算完成。

不再生成：

```text
eye_metrics.csv.gz
frame_coverage.csv.gz
```

旧 `.csv.gz` 或半完成 QC/metadata 如果残留在没有有效 completion 的 subject 目录，preflight 会拒绝自动混用；先人工确认并归档，不自动删除。

---

## 10. bounded QC 与 completion

当前上限：

```text
qc_interval_sec = 30
qc_anomaly_max_per_reason = 5
qc_image_max_count = 80
qc_pixel_evidence_max_eyes = 16
qc_artifact_budget_bytes = 268435456
final_output_limit_bytes = 1073741824
```

QC composite 仅画 pupil ellipse，不画 iris ellipse；少量选中帧允许 labels-only / sparse five-output CUDA 推理用于可复核 evidence，这不是重新跑 cohort。

发布 `completion.json` 前必须验证 plain eye/frame CSV、QC、artifact SHA256/size、source selection/work identity，以及整个 subject final directory ≤1 GiB。

---

## 11. CUDA output-transfer benchmark

当前 b16 cohort 每次返回的 `class_probability` 为：

```text
[16,4,400,640] float32 ≈ 62.5 MiB/call
```

隔离 benchmark：

```powershell
python benchmark_ritnet_final_output_transfer.py `
  --run-dir "<任一严格完成的 historical formal run>" `
  --config config.yaml `
  --device 0
```

默认结果：

```text
outputs/nvidia-cuda/ritnet-final-output-transfer.json
```

它交错比较 labels-only、当前 labels+class_probability、all-five-output，并在计时前要求 labels 与 class_probability parity。该 benchmark 不写正式科研输出、不修改 checkpoint。

任何未来 scalar-output ONNX 优化都必须先在 NVIDIA 实机完成与当前 CUDA v8 的逐值/parity 验证，不能直接替换当前正式模型。
