# artifacts｜历史证据产物索引

## 当前状态（2026-08-23）

`artifacts/` 的设计职责是保存项目开发和方法选择过程中产生的**有限、可复现、具有审计价值的历史证据产物**，例如 Gate/QC、benchmark、ROI selection、sequence review、smoke test 和轻量模型评价结果。

但当前 Git 分支实际核验时，`artifacts/` 目录只保留本 `README.md`；早期工作记录中引用的多个 artifact 目录当前已经不在这里。特别是 2026-08-22 YOLO held-out test 曾生成：

```text
artifacts/yolo-eye-evaluation/yolo26n_eye_100epoch/
```

其中包含 `overall_metrics.*`、逐图/逐被试结果、failure index、native test 和 `run_manifest.json`，但当前分支没有保留该目录。

因此需要区分：

- **历史记录证明某 artifact 曾生成**；
- **当前 Git 分支是否仍保存该 artifact**。

当前不能仅依据旧工作记录中的路径假定对应目录仍存在。

## 与正式全量输出的边界

`artifacts/` 不是正式全量分析输出目录。正式被试的大体量中间结果与最终结果应继续保存在仓库外部，避免把数十/数百个视频对应的大量 CSV、ROI、overlay 提交到 Git。

但轻量 provenance 文件具有较高复现价值，例如：

```text
run_manifest.json
summary.json
最终参数摘要
模型 SHA256
```

如果之后从实际全量输出盘找回这些文件，可以讨论把轻量副本补回 Git，而无需搬回全部正式输出。

## 历史 artifacts 去向

2026-08-16 时，一部分“结论已定、不再继续修改”的过程证据被移动到：

```text
finish/
```

例如 Gate1、preview、早期 ROI compare 和 ROI smoke test。`finish/00-目录与映射.md` 保留当时的归档映射。

另一些当时仍活动的 artifacts（truth-528、benchmark、sequence、ROI selection 等）曾被工作记录引用，但当前分支已不在 `artifacts/` 中。这里不根据文字记录人工重建其内容；如有旧备份，应以原文件恢复。

## 为什么不预设多层目录

若将来重新补回有限 artifacts，仍优先保持路径简单。只有当实际产物数量足够多时，再讨论是否拆成 `qc/`、`benchmark/`、`selection/` 等层级，避免为了形式整齐再次破坏历史引用。

## 保留原则

1. 历史 artifacts 不因正式路线已经确定而自动删除。
2. 不覆盖或人工重造原始实验结果和复核证据。
3. 新生成的大体量正式输出默认不进入 Git。
4. 找回历史 artifact 时优先按原路径恢复，并在工作记录中注明来源。
5. 关键轻量 provenance 可以单独保留，以闭合复现链。
6. 任何删除历史 artifact 的操作都需要用户明确同意。
