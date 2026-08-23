# 051｜NIR 正式路线与 ROI Tracking 状态

**Status:** Accepted  
**Date:** 2026-08-23  
**Supersedes:** 2026-08-21 阶段的 `YOLO + CSRT/KCF tracking + RITnet` 候选设计

## 决策

当前正式 NIR 主链定义为：

```text
FocusWave v3.1.3 phase windows
    ↓
逐帧 YOLO26n 眼框检测
    ↓
单眼 ROI
    ↓
RITnet batch inference
    ↓
指标 / QC / phase 输出
```

CSRT/KCF 等 ROI tracking **不是当前正式分析环节**。相关实现继续保留，仅用于诊断、历史复现或未来重新评估，不应再在当前 README、方法说明或架构图中写成正式主链。

## 已确认事实

- 正式 NIR 全量分析已经完成。
- 当前正式 runtime 为 `runtime/nir-formal/`。
- 根 `README.md`、`AGENTS.md` 与当前 runtime 口径均明确正式模式采用逐帧 YOLO。
- 历史文档中关于“周期性 YOLO + tracking”的内容代表 2026-08-21/08-22 的候选或联调阶段，保留其历史语境即可。

## 当前证据边界

当前主分支能够确认“最终正式模式没有采用 ROI tracking”，但仓库中尚未找到一份完整、独立的原始 decision record，能够精确说明当时从 tracking 候选切换到逐帧 YOLO 的全部实证理由。因此本文件**不补写或猜测未记录的原因**。

如果后续找到当时的完整运行记录、性能比较或正式批处理 manifest，可在新的 decision 或本文件的证据补充区记录，并保持“历史候选 → 最终正式路线”的时间关系。

## 文档影响

- `docs/020-nir/`：只把逐帧 YOLO → ROI → RITnet 写为当前正式方法；tracking 只能作为历史背景出现。
- `docs/010-overview/`：架构图使用正式主链，不把 tracking 画入 current pipeline。
- `runtime/nir-formal/`：是正式运行事实的最高优先级入口之一。
- `docs/工作记录/`：历史 tracking 讨论原样保留，不追溯改写。
