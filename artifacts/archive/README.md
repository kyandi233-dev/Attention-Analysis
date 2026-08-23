# archive｜已封存历史证据

## 当前定位

`artifacts/archive/` 原名 `finish/`，保存 **2026-08-16 及更早阶段** 已经结论确定、主要用于历史追溯的过程证据。它不是当前正式 NIR 全量输出目录，也不是当前 runtime。

这些 Gate1、preview、ROI compare 和 smoke test 解释了当时为什么从完整人脸 ROI 路线继续转向专用眼部 ROI。后来项目已经发展到 NIR 专用 YOLO26n + tracking + RITnet，并完成正式全量分析。

## 已封存内容

| 目录 | 内容 | 历史用途 |
|---|---|---|
| `gate1-24eyes/` | 12 帧 / 24 眼审批包与人工标注 | 审批门 1 证据 |
| `preview/` | 最早 12 眼网页预览 | 后被 Gate1 包取代；保留界面证据 |
| `roi-compare/` | 早期离散全视频 ROI 测速调查 | 后被 Block1 60 时点入围检查取代 |
| `roi-selection-smoke-sub011/` | Block1 起始约 0.2 秒冒烟 | 首次发现正式输入不是完整人脸 |
| `roi-selection-smoke3s-sub011/` | Block1 前 3 秒三后端冒烟与源图检查 | 确认 MediaPipe / YuNet 失检、旧 YOLO 单眼假脸 |

这些目录是历史证据归档，不代表当前正式技术路线。

## 与其他 artifacts 的关系

2026-08-16 时，另一些当时仍活动的 artifacts 包括 `truth-528`、benchmark、sequence、ROI selection 等。当前分支未全部保留这些目录；其执行证据仍存在于工作记录中。

“当前分支未保留”只描述 Git 当前实体，不否认当时真实生成过。若以后从原始备份找回，应以原文件恢复，不根据工作记录人工重建。

## 当前复现入口

```text
runtime/NIR-formal/
yolotrain/README.md
models/README.md
docs/010-nir/README.md
```

本目录只承担历史 archive 角色。
