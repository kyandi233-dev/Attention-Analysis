# 052｜AMD/DirectML 推理后端与固定批策略

**Status:** Accepted  
**Date:** 2026-08-24  
**Branch:** `amd-DirectML`  
**Base:** `e63675ad15c17db6ea2ac7a3bb1c1ac6fc106e06`

## 决策

AMD 分支将 YOLO26n 与 RITnet 推理从 Ultralytics/PyTorch CUDA 替换为 ONNX Runtime DirectML。YOLO 使用固定 FP32 `[1,3,640,640] → [1,300,6]` 图。RITnet 主干使用固定 FP32 `[16,1,400,640]` 输入和 external data；为避免每批向 CPU 回传 62.5 MiB 四通道 logits，图内追加确定性的 ArgMax、Cast、Softmax、Gather 与 Squeeze，输出 UINT8 四分类标签 `[16,400,640]` 和 FP32 pupil probability `[16,400,640]`。

RITnet 所有调用都以 batch=16 执行。尾批不足 16 时，复制最后一个真实 ROI 补齐，推理后丢弃补位 slot 的输出；不将补位写入 CSV。

压缩输出仍保留每个像素的完整 background/sclera/iris/pupil argmax 标签，因此不会阻断后续从 `sclera ∪ iris ∪ pupil` 验证候选眼裂开合度。它不新增第二次 RITnet forward，也不把候选 openness 误写为本版本已验证的 blink。

## 不变的科研口径

- FocusWave v3.1.3 phase windows 与 `sub-031+` 边界；
- 逐帧 YOLO、confidence=0.40、ROI 扩展与 320×160 分析坐标；
- RITnet 640×400 输入与 FP32；
- `frames.csv` / `eyes.csv` schema、missing/QC 语义与 phase 字段。

## DirectML 失败边界

session 创建前必须检查 `DmlExecutionProvider`；创建后 DML 必须是首选 provider，并调用 `disable_fallback()` 禁止运行时整个 session 改用 CPU。如 DML 不存在、初始化失败或最终未成为首选 provider，立即中止。

ORT DirectML 图仍会保留 `CPUExecutionProvider` 处理少量不受 DML 支持的形状/控制节点；若强制禁止任何 CPU EP 节点，当前两个图会在初始化时失败。本决策禁止的是 DirectML 不可用后的整体纯 CPU 静默回退，不伪装成“图中绝对零 CPU 节点”。

## 资产与输出隔离

AMD runtime 只保留 `.onnx` 与 `.onnx.data`，不保留 NVIDIA runtime 的 `.pt/.pkl` 权重。默认输出路径含 `amd-directml`，避免与 NVIDIA 运行目录和 `skip_completed` 状态冲突。

## 验证依据

RX 6750 GRE / `D:\CondaEnvs\nir-amd` 已完成两个 ONNX 的 DirectML forward。接入 pipeline 后，`sub-031` block1 前 600 帧轻量端到端验证完成：600 帧、1187 只眼、RITnet CSV batch 全为 16，尾批实际为 3 只眼并补位 13；输出显式标记为 smoke 截断结果。

同一片段从原四通道 logits 输出改为压缩输出后，处理时间从 52.92 s 降到 44.66 s，吞吐从 11.34 FPS 提高到 13.44 FPS；除耗时字段外，只有 `pupil_confidence` 存在最大 `2.38e-7` 的 FP32 舍入差，其余 CSV 科研字段一致。
