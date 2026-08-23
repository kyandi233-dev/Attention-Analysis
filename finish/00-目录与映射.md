# 00｜finish —— 已完成工作产物归档

## 当前定位（2026-08-23）

`finish/` 是 **2026-08-16 及更早阶段** 已完成过程证据的历史归档区，不是当前正式 NIR 全量输出目录，也不是当前 active runtime。

目录中的 Gate1、preview、ROI compare 和 smoke test 解释了当时为什么从完整人脸 ROI 路线继续转向专用眼部 ROI。后来项目已经发展到 NIR 专用 YOLO26n + tracking + RITnet，并完成正式全量分析，因此下面“新 ROI 定稿后”“下一轮”等措辞应按 08-16 的历史时间点理解。

当前活动/复现入口分别为：

```text
runtime/README.md
runtime/nir-yolo-tracking-ritnet-v1/
yolotrain/README.md
models/README.md
docs/010-nir/00-目录与映射.md
```

---

## 2026-08-16 历史归档状态

> 08-16（Asia/Shanghai）｜当时只把结论已定、不会继续修改的过程证据放入 `finish/`；仍会复测或作为下一轮输入的文件保留在 `artifacts/`。

### 已归档

| 目录 | 内容 | 当时结论/用途 |
|---|---|---|
| `gate1-24eyes/` | 12帧/24眼审批包与人工标注 | 审批门1已完成；配置当时指向此处 |
| `preview/` | 最早12眼网页预览 | 已被gate1包取代，仅作历史界面证据 |
| `roi-compare/` | 早期离散全视频ROI测速调查 | 已被Block1 60时点入围检查取代 |
| `roi-selection-smoke-sub011/` | Block1起始约0.2秒冒烟 | 首次发现正式输入不是完整人脸 |
| `roi-selection-smoke3s-sub011/` | Block1前3秒三后端冒烟与源图检查 | 确认MediaPipe/YuNet失检、YOLO单眼假脸 |

这些目录当时是移动归档，不是删除；当前仍按历史证据保留在 `finish/`。

### 当时仍在活动产物区的目录

| 历史路径 | 当时保留原因 | 2026-08-23 当前说明 |
|---|---|---|
| `artifacts/truth-528/` | 人工真值，后续任何新瞳孔算法仍需复测 | 当前分支未在 `artifacts/` 保留该目录 |
| `artifacts/benchmark-single/` | 修复前历史对照，不能当最终指标 | 当前分支未保留 |
| `artifacts/benchmark-axis-fix-review/` | 轴角修复后的阶段4/4b权威复核 | 当前分支未保留 |
| `artifacts/sequence-44x121/` | 修复前历史连续序列源 | 当前分支未保留 |
| `artifacts/sequence-adapter-review/` | 新PuReST适配层复核证据 | 当前分支未保留 |
| `artifacts/roi-selection-sub011-block1-2min/` | 当时正式ROI阻断证据；60时点/12组图为下一轮专用眼部ROI比较基线 | 当前分支未保留 |

“当前分支未保留”只描述 Git 当前实体，不否认这些 artifact 当时真实生成过；相关执行证据继续保存在工作记录。

### 当时代码、模型与 runtime 计划

08-16 当时记录：

- 活动代码继续留在 `scripts/` 和 `src/attention_pipeline/nir/`，不放入 `finish/`；
- 当时三个人脸 ROI 候选均淘汰，因此计划不把它们复制进生产 `models/`；
- 当时设想“新 ROI 定稿后，`models/` 放完整生产链实际使用模型”；
- 当时设想 `runtime/` 后续保存环境锁、wheel、模型 SHA-256 和重建说明。

这些计划后来部分实现并发生演化：当前 `runtime/nir-yolo-tracking-ritnet-v1/` 已经冻结 YOLO26n 与 RITnet 的可移植实现及模型，且正式 NIR 全量分析已经完成。因此本节仅作为架构演化历史保留。
