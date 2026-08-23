# 08-23-05｜artifacts保留价值与Archive边界审计工作记录

> 文档性质：仓库整理 / 历史输出资产审计  
> 日期：08-23  
> 当天序号：05  
> 当前维护主线：`main`  
> 状态：已完成第一轮分类，尚未执行删除或目录迁移

## 一、审计目的

本轮重新判断 `artifacts/` 的真实语义，并回答：

1. `artifacts/` 是否应该整体改名为 `archive/`；
2. `datasets/`、`training/` 既然也是过程资产，为什么不一起归档；
3. Gate / QC / benchmark / preview / smoke 分别是什么意思；
4. 当前 `artifacts/` 中哪些内容值得长期保留，哪些只是可重新生成的阶段性输出。

## 二、核心结论

**当前不建议把 `artifacts/` 整体机械改名为 `archive/`。**

原因是当前目录混合了两种不同价值的内容：

- 人工标注 / 人工审批形成、难以无损重新生成的研究证据；
- 由脚本快速生成、可以通过原始数据和 Git 历史重新得到的阶段性输出。

把二者全部放进 `archive/` 并不能解决语义混乱。

## 三、为什么 datasets / training 不能按“过程文件”一起归档

虽然 `datasets/` 和 `training/` 也产生于研究过程，但它们仍然是当前正式模型 provenance 的直接组成：

```text
datasets/
→ 固定 train / val / test
→ training/
→ YOLO best.pt
→ runtime 冻结副本
→ 正式 NIR 全量分析
```

因此它们仍属于“当前正式结果为什么是现在这个样子”的有效来源，而不是已经退出当前工作的历史资产。

与之不同，当前 `artifacts/` 主要来自已经结束的 08 月中旬 ROI / Gate / smoke 候选阶段。

## 四、当前 artifacts 逐项审计

### 1. `gate1-24eyes/`

当前可见内容包括：

- `annotations_lxy.csv` 人工标注；
- contact sheet；
- review manifest CSV / JSON；
- review HTML；
- ROI / context 图。

这部分包含人工审批和人工标注，不是单纯重新跑脚本即可无损恢复。

**建议：长期保留。**

### 2. `preview/`

这是更早的小样本审核 / 网页预览输出，结构与后续 `gate1-24eyes/` 高度相似。

**建议：可删候选。**

前提：关键结论已由后续 Gate1 和工作记录覆盖。

### 3. `roi-compare/`

主要保存若干离散帧比较 JPG 和 `timeline.csv`，用于早期 ROI 方法视觉比较。

这些输出可由原始数据和历史脚本重新生成，对应方法也已退出正式路线。

**建议：可删候选。**

### 4. `roi-selection-smoke-sub011/`

极短时段的 MediaPipe / YOLO-face / YuNet ROI 冒烟测试输出。

**建议：可删候选。**

### 5. `roi-selection-smoke3s-sub011/`

Block1 前约 3 秒的三后端冒烟和 source-check，用于确认完整人脸 ROI 方法与正式双眼特写输入域不匹配。

关键结论已经进入工作记录和后续路线选择。

**建议：可删候选。**

## 五、术语说明

### Gate

准入检查点。预先设定一组条件，只有通过后某条方法路线才继续推进。

### QC

Quality Control，质量控制。检查数据或模型输出是否可用、是否存在失败、异常、缺失或明显不合理结果。

### Benchmark

在相同数据和指标条件下，对多个算法 / 参数方案进行系统性能比较。

### Preview

少量预览输出，用于人工快速查看图像、ROI、网页或中间结果是否符合预期。

### Smoke test

冒烟测试。用极少量数据先快速确认程序、依赖、数据输入域或基本逻辑是否存在明显问题，不等于正式性能评估。

这些词描述研究开发活动，不意味着仓库必须长期建立同名一级目录。

## 六、archive 的边界

如果未来建立 `archive/`，其语义应严格是：

> 已经退出当前工作，但因为历史价值仍决定长期保留的资产。

它不应该成为“凡是过程文件都往里扔”的垃圾桶。

本轮审计后，当前 `artifacts/` 中明显最符合长期保存要求的是 `gate1-24eyes/`。

如果用户批准删除其余可重新生成内容，后续可以考虑把剩余长期证据改成更准确的：

```text
evidence/
└── nir-gate1-24eyes/
```

比 `archive/` 更能说明“这是研究证据，而不是一切历史文件”。

## 七、本轮修改

已更新：

```text
artifacts/README.md
```

补充：

- datasets / training 与历史输出的区别；
- 每个 artifacts 子目录的实际用途；
- Gate / QC / benchmark / preview / smoke 的含义；
- 各目录保留建议；
- archive 与 evidence 的边界。

本轮没有删除、移动任何 artifacts 文件。

## 八、当前待授权删除候选

如后续用户明确批准，可优先考虑删除：

```text
artifacts/preview/
artifacts/roi-compare/
artifacts/roi-selection-smoke-sub011/
artifacts/roi-selection-smoke3s-sub011/
```

当前建议保留：

```text
artifacts/gate1-24eyes/
```

工作记录不参与上述删除。
