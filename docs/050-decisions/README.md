# Decisions｜决策记录

本目录记录“为什么最终这样做”，不重复方法说明，也不覆盖日期型工作记录。

## 规则

- decision 文档使用 `051-`、`052-`、`053-`……编号，便于按顺序阅读和快速定位。
- 每个决策应写明状态，例如 `Accepted`、`Superseded`、`Deprecated` 或 `Rejected`。
- 当后续方案替代旧方案时，保留旧 decision，并用 `Superseded by` 指向新文件；不要把旧决定改得像从未发生。
- 原始实验过程、失败样本和当天执行细节继续留在 `../工作记录/`。

## 当前决策入口

- `051-NIR正式路线与ROI-Tracking状态.md`：明确正式 NIR 使用逐帧 YOLO，CSRT/KCF ROI tracking 仅保留诊断/历史复现身份。
- `052-AMD-DirectML推理后端与固定批策略.md`：记录 ONNX Runtime DirectML、固定 RITnet batch=16/FP32、provider 失败策略和 AMD 输出隔离。

后续可继续记录眼框检测路线、RITnet 选择、眨眼/EAR/PERCLOS 方法、QC 口径、RGB 启停以及跨模态升级条件等关键决定。
