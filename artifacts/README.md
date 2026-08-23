# artifacts｜可审计证据产物

## artifacts 是什么

这里的 **artifact** 指研究/运行过程中生成、用于复核和追溯的“证据产物”，不是源码本身。例如：

- Gate / QC 预览；
- benchmark 结果；
- ROI selection / smoke test；
- review 图与机器可读评价结果；
- `run_manifest.json`、`summary.json`、模型 SHA256 等轻量 provenance。

它与几个相邻概念的区别是：

```text
src/、scripts/      → 代码
models/、runtime/   → 模型与冻结运行资产
datasets/           → 数据集定义/训练数据资产
artifacts/          → 代码运行后产生的、值得保留的审计证据
正式全量输出盘       → 大体量正式结果，不直接塞进 Git
```

## 当前结构

```text
artifacts/
├── README.md
└── archive/
```

`archive/` 就是原来的根目录 `finish/`。原 `finish/` 实际保存的是已经结论确定、主要用于历史追溯的 Gate1、preview、ROI compare、smoke test 等证据；它本质上属于 artifacts 的“已封存部分”，因此现在统一为：

```text
artifacts/archive/
```

这样不再让 `finish/` 与 `artifacts/` 看起来像两个含义不明、彼此竞争的根目录。

## 与正式全量输出的边界

`artifacts/` 不是正式全量分析输出目录。正式被试的大体量 CSV、ROI、overlay、视频级中间结果继续保存在仓库外部。

如果之后从正式全量运行电脑找回以下轻量文件，可以作为 provenance 补回 Git：

```text
run_manifest.json
summary.json
最终参数摘要
模型 SHA256
```

## 历史缺口

早期工作记录曾引用 `truth-528`、benchmark、sequence、ROI selection、YOLO held-out test machine-readable evaluation 等 artifact；其中一部分当前分支已不保存。历史记录证明“当时生成过”与“当前 Git 中仍存在”是两件事，不能根据文字记录人工重建原文件。

## 保留原则

1. 已存在的历史 artifact 不因正式路线确定而自动删除。
2. `archive/` 保存已经封存、仍有 provenance 价值的历史证据。
3. 不覆盖或人工重造原始实验结果和复核证据。
4. 大体量正式输出默认不进入 Git；轻量 provenance 可以进入。
5. 任何删除历史 artifact 的操作仍需明确批准。
