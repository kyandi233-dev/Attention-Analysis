# 031｜NIR cohort 最近一次本地执行结果

> OWNER：Codex Luna。此目录 `luna-output/` 只保存 Luna 的本地执行回报；ChatGPT 负责读取和验收，不在这里写分析计划或任务说明。

## 最近一次结果

任务：Phase 1 `sub-031` smoke test（单被试小规模试运行）

状态：**PASS**

- branch/head：`analysis/nir-behavior-v2` / `0abfe79e6c9ac4108b9b3108a9785e0d41823540`
- stable merge：success，无冲突
- `compileall`：PASS
- `pytest`：PASS，3 passed
- `subjects_preflight_ok / failed`：1 / 0
- `eye_block_qc_rows`：4
- `behavior_qc_rows`：2
- `fullclass_column_count_distribution`：106 列 × 1
- `fullclass_header_signature_count`：1
- B1-left / B1-right / B2-left / B2-right rows：17,468 / 16,930 / 16,621 / 16,144
- `cohort_manifest.json`：已生成
- `analysis_config_snapshot.yaml`：已生成
- pipeline errors：0
- 4 行 anomaly flags 只是 4 个 eye×block 的描述性复核记录，不代表 4 个异常或排除。
- pytest 有一条既有 requests/urllib3 依赖版本 warning；未影响本轮结果。
- 未修改分析代码或分析计划；未运行其余 43 人 alignment（时间对齐）。

本次结果已由 ChatGPT 验收。下一任务只读取：

```text
docs/020-nir/chatgpt-control/030-NIR-cohort当前本地执行任务.md
```
