# 08-25 NVIDIA RITnet 全分类补全同步工作记录

## 1. 背景

正式 NIR 已经完成，但历史 `eyes.csv` 仅将 RITnet pupil 类转成正式数值变量。AMD 分支已完成 post-hoc full-class 修复并在 `sub-031` 上验证，因此本次将同一分析口径同步到 `nvidia-cuda`，同时补齐 NIR↔行为数据对齐方法文档。

本次 NVIDIA 实际目标机器 GPU 为 **NVIDIA GeForce RTX 5060**。此前对话/部分旧说明中把这台机器记成 RTX 4060，属于硬件型号记录错误；本次同步后的性能评估与后续正式运行均以 RTX 5060 为准。

本次原则：不删除、不覆盖历史正式结果；不重新跑 YOLO；不改变 RITnet 输入分辨率或 FP 精度。

## 2. 冻结方法

```text
RITnet ONNX: ritnet-b16-fp32.onnx
SHA256: 1933f44f483b350e17249a37b4a2ebe8b5e32f83fc8c1eb1a21c27e96477e621
input: 640×400
precision: FP32
batch: 16
analysis size: 320×160
classes: 0 background / 1 sclera / 2 iris / 3 pupil
primary pupil metric: fullclass_pupil_to_iris_diameter_ratio
QC stride: 3000 frames + phase anchors + bounded anomaly samples
```

NVIDIA extension 使用 ONNX Runtime CUDA。为了避免主指标混用旧 PyTorch pupil 与新 ONNX iris，full-class 新输出中 pupil 与 iris 都由同一 ONNX hard-label map 计算。旧 pupil 数据继续保留在 source `eyes.csv` 中。

## 3. 同步内容

新增/同步 runtime：

```text
runtime/nir-formal/ritnet_fullclass_contract.py
runtime/nir-formal/ritnet_fullclass_metrics.py
runtime/nir-formal/ritnet_fullclass_qc.py
runtime/nir-formal/ritnet_fullclass_runtime.py
runtime/nir-formal/run_ritnet_fullclass_extension.py
runtime/nir-formal/run_ritnet_fullclass_batch.py
runtime/nir-formal/RITNET_FULLCLASS_EXTENSION.md
runtime/nir-formal/tests/test_ritnet_fullclass_metrics.py
```

新增方法文档：

```text
docs/020-nir/08-25-01-NIR-RITnet全分类补全与瞳孔分析方法.md
docs/030-behavior/035-NIR与正式SART行为数据对齐分析方法.md
```

既有行为文档 `031`～`034` 不覆盖、不重写，继续作为行为指标、统计和 QC 的正式定义。

## 4. 输出版本

full-class extension version：

```text
ritnet-fullclass-v1.2-fast-qc
```

建议 Git tag：

```text
nvidia-v1.2-ritnet-fullclass
```

该 tag 应指向本次 NVIDIA 同步后的最终通过 CI 的 commit。若连接工具无法直接创建 Git tag，在 NVIDIA 本地拉取最终 commit 后执行：

```powershell
git tag -a nvidia-v1.2-ritnet-fullclass -m "NVIDIA CUDA RITnet full-class v1.2 fast QC"
git push origin nvidia-v1.2-ritnet-fullclass
```

## 5. AMD 已有验证依据

AMD `sub-031` reference full-class：

```text
81830 eye rows
pupil parity OK: 81824
mismatch: 6
parity fraction: 0.9999267
```

AMD fast-qc production：

```text
81830 eye rows
elapsed: 616.26 s
throughput: 132.78 ROI/s
QC images: 154
```

这验证了“复用 source frame_idx + ROI 坐标、只重跑 RITnet”的修复路线以及 sparse QC 输出。NVIDIA RTX 5060 使用不同 execution provider，因此在 72 人全量前仍应对 1 名完整被试执行一次 CUDA parity + speed benchmark。

## 6. NVIDIA 全量前检查

```powershell
git switch nvidia-cuda
git pull --ff-only
cd runtime\nir-formal
python -m pytest tests -q
python run_ritnet_fullclass_batch.py --output "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" --device 0 --postprocess-workers 4 --dry-run
```

然后先对一名完整被试运行 `--validate-pupil`。检查：

```text
status=complete
processed_rows == expected_rows
pupil parity
normalization valid fraction
QC labels/overlay
elapsed_sec / roi_per_sec
timing_cpu_work_ms / timing_gpu_ms
```

验收后正式全量去掉 `--validate-pupil`。

## 7. 时间预算原则

AMD 44 名可用实际 fast-qc `sub-031` 速度作初步预算；NVIDIA RTX 5060 的 72 名在新 CUDA full-class 代码完成首名实测前不写死未经测量的单人耗时。首名 NVIDIA 被试结束后直接用：

```text
72人理论耗时 = 单人 elapsed_sec × 72
```

按 AMD `sub-031` 的 616.26 秒作为保守同速上界，44 人约 7.5 小时；RTX 5060 只跑 RITnet full-class、跳过 YOLO，预期单人约 7–10 分钟时，72 人约 8.4–12 小时。两台电脑并行运行时，总墙钟时间由较慢的一台决定，当前应预留约 9–12 小时，并在 RTX 5060 首名 benchmark 后按实测值收紧预算。
