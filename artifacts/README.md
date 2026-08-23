# Artifacts

`artifacts/` 保存已经提交到仓库、需要长期保留的评估与审计证据。这里不是正式全量分析输出目录，也不表示这些方法仍是当前生产路线。

当前目录主要是 2026-08 中旬 ROI / Gate / smoke test 阶段形成的历史证据：

| 目录 | 内容 | 当前定位 |
|---|---|---|
| `gate1-24eyes/` | 12 帧 / 24 眼审批包与人工标注 | 历史 Gate1 证据 |
| `preview/` | 最早 12 眼网页预览 | 历史界面证据 |
| `roi-compare/` | 早期离散全视频 ROI 测速调查 | 历史比较证据 |
| `roi-selection-smoke-sub011/` | Block1 起始约 0.2 秒冒烟 | 历史 ROI 输入域检查 |
| `roi-selection-smoke3s-sub011/` | Block1 前 3 秒三后端冒烟与源图检查 | 历史 ROI 失败证据 |

这些内容原先位于 `finish/`。本次只做目录语义重命名，没有删除其中任何研究证据。

当前正式 NIR pipeline 已转为 YOLO26n 眼框 + RITnet；正式全量分析结果仍应保存在仓库外的独立输出目录。
