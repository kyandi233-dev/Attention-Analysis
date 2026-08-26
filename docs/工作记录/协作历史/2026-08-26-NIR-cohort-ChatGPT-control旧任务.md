# 030｜NIR cohort 当前本地执行任务

> OWNER：ChatGPT。此目录 `chatgpt-control/` 只保存由 ChatGPT 编写和维护的当前执行任务；Codex Luna 只读，不在此目录写文件。

## 当前阶段

Phase 1｜44 人 cohort（多人样本）preflight（运行前检查）+ QC（质量控制）。

已完成：

- 44 人 inventory（数据盘点）；
- local contract（本地数据结构说明）；
- ChatGPT 已读取并验收上述盘点；
- Phase 1 cohort QC 代码第一版；
- `sub-031` smoke test（单被试小规模试运行）：PASS；
- `compileall`：PASS；
- `pytest`：3 passed。

当前任务：**运行完整 44 人 Phase 1 preflight/QC，生成小型复核资料包。**

## Luna 本轮只负责什么

1. 同步 `analysis/nir-behavior-v2` 最新代码；
2. 运行完整 44 人 Phase 1 QC；
3. 检查是否 44 人全部成功、理论 176 个 subject×eye×block（被试×眼别×实验区段）质量单元是否生成；
4. 生成一个小型 `phase1_review_bundle` 供 ChatGPT 直接审查 44 人质量分布；
5. 只更新 `docs/020-nir/luna-output/031-NIR-cohort最近一次本地执行结果.md` 并 push。

Luna **不要**自行：修改分析代码、设置排除阈值、删除被试/眼睛、决定 PIR 标准化方案、运行其余 43 人 schema-2 alignment（时间对齐）、进入 Phase 2。

## 执行步骤

在 cohort 专用 worktree 中：

```powershell
git fetch origin --prune
git switch analysis/nir-behavior-v2
git pull --ff-only

$env:PYTHONPATH = "src"
python -m compileall src/attention_pipeline/nir_behavior_cohort scripts/nir_behavior_cohort_qc.py
python -m pytest -q tests/test_nir_behavior_cohort_qc.py

python scripts/nir_behavior_cohort_qc.py
```

默认正式输出根：

```text
D:\_AttentionData\Beijing-NIR\analysis\nir-behavior-v2\cohort-44-exploratory\
```

如果 Git 同步、测试或正式运行失败，不修改代码；把最小错误写入 Luna 输出文件后停止。

## 成功后检查

至少确认：

- `subjects_preflight_ok = 44`；
- `subjects_preflight_failed = 0`；
- `subject_eye_block_qc.csv` 理论应为 176 行；
- `behavior_cohort_qc.csv` 理论应为 88 行；
- `cohort_discovery.csv` 为 44 人；
- 4 份 106 列、40 份 107 列的 schema（字段结构）差异仍被正确识别；
- 未自动应用 exclusion cutoff（排除阈值）；
- 没有运行其余 43 人 alignment。

## Phase 1 review bundle（复核资料包）

请创建：

```text
D:\_AttentionData\Beijing-NIR\analysis\nir-behavior-v2\cohort-44-exploratory\phase1_review_bundle\
```

只复制/整理以下小文件，不复制逐帧 full-class 数据：

```text
cohort_preflight_summary.json
cohort_discovery.csv
subject_eye_block_qc.csv
subject_qc.csv
behavior_cohort_qc.csv
cohort_anomaly_flags.csv
cohort_manifest.json
README.md
```

`README.md` 只需写：运行 branch/HEAD、运行时间、44 人成功/失败数、176/88 行是否满足预期、warnings/errors，以及这些结果仍是描述性 QC、没有排除任何被试。

## 完成后的唯一 GitHub 写入

Luna 只更新：

```text
docs/020-nir/luna-output/031-NIR-cohort最近一次本地执行结果.md
```

写入简短结果后 commit + push 到 `analysis/nir-behavior-v2`。除 `luna-output/031...` 外，不修改仓库其他文件。
