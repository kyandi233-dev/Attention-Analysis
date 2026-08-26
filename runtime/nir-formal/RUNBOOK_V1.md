# NIR Formal Runtime Runbook V1

状态：`ACTIVE / AMD_DIRECTML_TEAMMATE_HANDOFF`

本文件用于第二台 Windows/AMD 机器实际接手 `Attention-Analysis` NIR formal runtime。它补充 `INSTALL.md` 和 `README.md`：`INSTALL.md` 负责安装，`README.md` 负责科研/算法口径，本文件负责“拿到机器后按什么顺序做、什么时候必须停止、最后交给中央仓库什么”。

## 1. 仓库职责

本仓库只负责本机 NIR/RGB producer/runtime。最终科学定义、identity reconciliation、cohort、grouped folds、跨机器合并和最终统计结论由中央仓库：

`greenboo26/focuswave-multimodal-attention-analysis`

管理。

第二台机器不得独立重定义 label、window、global participant ID、global fold 或跨站点最终 AUC/p-value。

## 2. 当前 AMD 正式 runtime

分支：`amd-DirectML`

当前 package：`0.2.0`

正式 AMD NIR 组合：

```text
YOLO26n: 640x640, FP32, DirectML, fixed batch=8, every frame
RITnet:  640x400, FP32, DirectML, fixed batch=16
analysis geometry: 320x160
tracking: none
```

正式模型：

```text
models/nir-eye-yolo26n-best-b8.onnx
models/ritnet-b16-fp32.onnx
models/ritnet-b16-fp32.onnx.data
```

`models/nir-eye-yolo26n-best.onnx` 为 b1 reference/diagnostic，不是 v0.2.0 正式 YOLO producer。

正式运行必须记录 exact Git commit，而不仅仅记录 `amd-DirectML` 分支名。

## 3. 新机器安装

先按 `INSTALL.md` 创建 Python 3.11 独立环境并安装：

```powershell
cd runtime\nir-formal
pip install -r requirements.txt
```

当前依赖核心为 `onnxruntime-directml`。AMD formal inference 不要求 CUDA、PyTorch 或 Ultralytics。

安装后必须先执行：

```powershell
python -m pytest tests -q
python run_pipeline.py check-env
```

DirectML 不可用时视为正式环境失败，不允许整场 session 静默退回纯 CPU。

## 4. 数据发现

当前 `config.yaml` 声明候选根：

```text
E:/正式实验
F:/正式实验
E:/Data
F:/Data
```

正式发现前执行：

```powershell
python run_pipeline.py discover --formal-only
python run_formal_batch.py --dry-run
```

检查：

- 实际发现了哪些 session；
- 是否存在同一 subject 在多个有效根重复；
- 视频路径是否唯一；
- subject 编号、文件结构和输出计划是否符合预期。

发现 duplicate 时必须停止并报告，不允许自动选一份。

## 5. Protocol compatibility gate

这是正式运行前最重要的 gate。

当前 `runtime/nir-formal/config.yaml` 明确冻结为：

```text
focuswave_release: v3.1.3
expected_formal_blocks: 2
phases:
  baseline
  instructions
  practice
  block1
  block2
```

因此当前 config 不是一个“任何 FocusWave 数据都能直接跑”的配置。

如果实际机器上的正式数据属于三 block/BBB 或其他 site/protocol：

1. 先根据 behavior 文件、master timeline、phase marker 核验真实协议；
2. 在没有经过 review 的 site/protocol-specific NIR config/adapter 之前，不执行 formal cohort batch；
3. 不允许只把 `expected_formal_blocks` 从 2 改成 3；
4. 不允许把历史 behavior BBB config 直接复制成 NIR config；
5. adapter 必须明确第三 block 的 phase window、probe/timeline 语义、输出 provenance，并保持 YOLO/RITnet/QC 科学定义不变；
6. adapter/ref 经过中央项目 review 后才能正式全量生产。

程序“成功跑完”不等于 protocol 正确。协议错误的正式输出视为无效。

在 protocol 未通过 gate 时，可以执行：环境检查、数据发现、dry-run、timeline/protocol audit；不得把这些输出标记为 formal NIR result。

## 6. 正式运行

仅在环境、模型资产、数据发现和 protocol gate 均通过后：

```powershell
python run_formal_batch.py
```

少量 subject：

```powershell
python run_formal_batch.py --subjects sub-031,sub-033
```

显式重跑：

```powershell
python run_formal_batch.py --subjects sub-031 --force
```

单被试正式入口：

```powershell
python run_formal_batched.py `
  --video "<actual-root>:\<data-root>\sub-033_\nir\sub-033_nir.avi" `
  --device 0
```

## 7. 每个正式输出至少保留

当前 runtime 典型产物包括：

```text
frames.csv
eyes.csv
summary.json
run_manifest.json
phase_windows.json
completion.json
overlays/
```

不得只保留最终一个 summary 数字。frame identity、QC、phase window 和 run provenance 都是后续中央审计需要的证据。

## 8. 必须记录的 provenance

至少记录：

- exact `Attention-Analysis` Git commit；
- branch/backend：AMD DirectML；
- runtime package version；
- Python/onnxruntime-directml 版本；
- config hash；
- YOLO/RITnet model hash；
- input session/video identity；
- protocol/site；
- phase/window definition；
- output manifest/completion 状态。

移动分支名本身不能替代 exact commit。

## 9. 交回中央仓库

本仓库的正式产物是本机 sensor-derived/QC package，不是最终跨站点统计结论。

完成本机 NIR production 后：

1. 按中央仓库 `docs/canonical/TEAMMATE_ONBOARDING_V1.md` 检查 machine/site/protocol；
2. 按中央 contract 转成标准 derived/QC 和 machine package；
3. 将 exact external producer commit/config/model provenance 写入 manifest；
4. 由中央仓库做 identity reconciliation、cohort/fold 和最终 inference。

禁止：

- 在这里自己平均 AMD/NVIDIA AUC；
- 自己冻结 global participant ID；
- 用不同 site 的不同科学定义拼表；
- 将 raw AVI 或 participant-level private output 提交 Git。

## 10. 完成交接的判定

第二台机器只有同时满足以下条件，才算“环境和 runtime 接手完成”：

- `amd-DirectML` 正确 checkout，并记录 exact commit；
- Python 3.11/DirectML 环境通过；
- formal model assets 完整；
- tests + `check-env` 通过；
- `discover --formal-only` 和 dry-run 已人工/AI 审核；
- actual site/protocol 已确认；
- 当前 config 与 actual protocol 兼容，或已有经过 review 的 protocol adapter；
- 首个正式 subject 输出包含完整 manifest/QC/phase evidence；
- 尚未进行任何未经中央批准的科学参数修改。

完成后再进行全量 NIR production。