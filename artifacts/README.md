# 00｜finish —— 已完成工作产物归档

> 08-16（Asia/Shanghai）｜只放结论已定、不会继续修改的过程证据；仍会复测或作为下一轮输入的文件保留在`artifacts/`。

## 已归档

| 目录 | 内容 | 结论/用途 |
|---|---|---|
| `gate1-24eyes/` | 12帧/24眼审批包与人工标注 | 审批门1已完成；配置已指向此处 |
| `preview/` | 最早12眼网页预览 | 已被gate1包取代，仅作历史界面证据 |
| `roi-compare/` | 早期离散全视频ROI测速调查 | 已被Block1 60时点入围检查取代 |
| `roi-selection-smoke-sub011/` | Block1起始约0.2秒冒烟 | 首次发现正式输入不是完整人脸 |
| `roi-selection-smoke3s-sub011/` | Block1前3秒三后端冒烟与源图检查 | 确认MediaPipe/YuNet失检、YOLO单眼假脸 |

这些目录是移动归档，不是删除；如需恢复可按本表移回`artifacts/`。

## 仍在活动产物区

| 目录 | 保留原因 |
|---|---|
| `artifacts/truth-528/` | 人工真值，后续任何新瞳孔算法仍需复测 |
| `artifacts/benchmark-single/` | 修复前历史对照，不能当最终指标 |
| `artifacts/benchmark-axis-fix-review/` | 轴角修复后的阶段4/4b权威复核 |
| `artifacts/sequence-44x121/` | 修复前历史连续序列源 |
| `artifacts/sequence-adapter-review/` | 新PuReST适配层复核证据 |
| `artifacts/roi-selection-sub011-block1-2min/` | 当前正式ROI阻断证据；60时点/12组图是下一轮专用眼部ROI的比较基线 |

## 代码、模型与runtime

- 活动代码继续留在`scripts/`和`src/attention_pipeline/nir/`，不放入finish。
- 当前三个人脸ROI候选均淘汰，因此不把它们复制进v2生产`models/`。
- 新ROI定稿后，`models/`放完整生产链实际使用模型；ROI候选只复制最终入选者。
- `runtime/`届时保存双环境锁、PyPupilEXT wheel、模型/wheel SHA-256和重建说明。
