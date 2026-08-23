# behavior_formal BB 重建与历史归档工作记录

> 08-24-02｜正式行为分析版本纠偏与目录收口

## 背景

仓库中原 `behavior_formal`、旧 `configs/sart_formal.yaml` 和旧 `docs/030-behavior/sart-formal/` 来自 2026-08-16 的 FocusWave v3.0 BBB（三个 B block）分析。最终正式实验已经更新为 FocusWave v3.1.3，正式 SART 为两个 B block；正式 NIR runtime 同样冻结 `block1 / block2` 和 `expected_formal_blocks: 2`。

因此旧 BBB 不能继续作为 current formal behavior。

## 已完成

1. `src/attention_pipeline/behavior_formal/` 已重建为最终 v3.1.3 BB 的 current implementation。
2. `scripts/sart_formal_analysis.py` 已改为最终 BB 行为分析入口，默认读取 `configs/behavior_formal.yaml`。
3. `configs/behavior_formal.yaml` 已建立：正式被试从编号 31 起自动发现，仅接受 B1/B2；旧 BBB 固定 trial / No-Go / probe 数不直接继承。
4. 当前统计主比较由旧的 B1/B2/B3 三水平结构改为 B1↔B2 被试内比较，并保留 block 内时间趋势、RT 变异性、SDT、No-Go 前兆和探针关联。
5. `docs/030-behavior/` 直接作为当前行为模块，不再设置 current `sart-formal/` 子层。
6. 旧 BBB 计划、报告、图和旧配置统一归档到 `docs/030-behavior/history/BBB-v3.0/`。
7. 旧 BBB 内部 `002-工作记录.md` 移回统一工作记录目录，改名为 `08-16-08-SART-v3.0-BBB行为分析工作记录.md`，正文不改写。
8. 最终正式实验确定前的 `014-正式实验修改建议报告` 与配套 plots 归档到 `docs/030-behavior/history/preformal/`。
9. 完整旧 BBB 可执行代码另由 `history/behavior-bbb-v3.0` 分支冻结，避免在 current `src/` 中并存两套 active behavior implementation。

## 当前目录语义

```text
docs/030-behavior/
├── README.md
├── 031-正式BB行为分析流程.md
├── 032-行为指标定义.md
├── 033-统计分析方法.md
├── 034-行为QC与输出.md
└── history/
    ├── BBB-v3.0/
    └── preformal/
```

当前说明直接放在行为模块根目录；`history/` 只保存退出 current 主线、但仍有 provenance 价值的材料。

## 仍待核验

最终 v3.1.3 每 block 的固定 trial 数、No-Go 数、probe 数，以及最终探针文本语义，不从旧 BBB 自动继承。后续应依据最终 FocusWave v3.1.3 任务源码或正式数据重新核验，再决定是否冻结到 current config。
