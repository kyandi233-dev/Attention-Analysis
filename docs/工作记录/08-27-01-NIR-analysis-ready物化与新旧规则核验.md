# 08-27-01｜NIR analysis-ready 物化与新旧规则核验

## 开始记录

### 背景

GitHub Issue #18 要求在 `analysis/multimodal-integration` 分支，以 frozen `ritnet-fullclass-v1.2-fast-qc` production CSV 为输入，物化 44 人 exploratory cohort 的 analysis-ready NIR 数据层，并核验 primary 与 strict validity、baseline、双眼 source mode、时间配对和 provenance。Issue 最新补充要求在全量物化前先确认 44 个 source header 均包含 `phase_time_ms`，不得自行合成该字段。

### 目标

完成指定测试、44 人 source header 检查和 analysis-ready 物化；报告物化完整性、production/primary/strict 纳入率、subject×eye×Block 分布、重点低 usable subject 变化、双眼 source mode、时间/重复键/provenance 核验，并直接回复 Issue #18。不得运行正式效应模型、Behavior/Probe 分析或修改 production CSV。

### 步骤与风险

1. 核验可用 checkout 的 Git 状态并同步 `analysis/multimodal-integration`。
2. 使用 `D:\CondaEnvs\nir-amd` 运行新增单元测试。
3. 仅读 44 个 source CSV header，确认 `phase_time_ms` 完整性。
4. 在输出目录不存在的前提下运行指定物化脚本。
5. 读取派生结果，验证行数、validity 子集关系、时间配对、重复键、source provenance 和重点被试变化。
6. 记录 warning/error；若发现契约异常，不自行修正或覆盖结果。

### 校验边界

- 只物化 `block1/block2`。
- `pir_valid_primary` 只使用核心几何条件与有限 PIR；whole-mask edge、fragmentation、pupil confidence、blur 仅作 QC/敏感性信息。
- `pir_valid_strict` 保持 production `fullclass_normalization_valid` 的可比性，并要求 strict 为 primary 子集。
- 不删除任何 subject/eye/Block/frame，不把 44 人 exploratory 结果解释为最终北京 cohort。

## 完成记录

### 总结

Issue #18 的 analysis-ready exploratory 物化已经完成。使用 `analysis/multimodal-integration` 分支当前代码和 `D:\CondaEnvs\nir-amd` 环境，44/44 个 production full-class subject 成功读取，未运行 RITnet/YOLO、Behavior/Probe 或正式统计模型。

### 原计划

按开始记录中的六步执行：同步分支；先运行新增测试；仅读 source header 核验 `phase_time_ms`；物化 analysis-ready 层；核验纳入率、时间配对、重复键和 provenance；将结果回复 Issue #18。

### 执行与决策过程

1. 当前可用 checkout 为 `C:\Users\goven\.codex\worktrees\6c85\Attention-Analysis-multimodal-integration`，分支为 `analysis/multimodal-integration`。先执行 `git fetch origin --prune` 和 `git pull --ff-only`，快进至 `843e1a467b6a464dd5d9773ed73caae605299f61`。未触碰损坏的 AMD checkout，也未改动 production/runtime/source 代码。
2. 在全量物化前运行 `pytest tests/test_nir_analysis_ready.py -q`，结果 `5 passed`。随后仅读 44 个 source CSV header：44/44 包含 `phase_time_ms`，无缺失，因此未从 `unix_ms` 或 `video_time_ms` 合成该字段。
3. 按 Issue #18 命令运行 `scripts/nir_materialize_analysis_ready.py --config configs/nir_analysis_ready.yaml`。输出目录原先不存在，未使用覆盖选项。物化结果为 44 个 subject、2,919,835 条正式 eye rows、1,488,332 条宽表时间点；`subject_eye_block_inclusion.csv` 为 176 行且键唯一，`subject_eye_baselines.csv` 为 88 行且键唯一，44 个 frame-level 文件均生成。
4. 全局 frame-level validity：production `fullclass_normalization_valid` 为 2,166,700/2,919,835（74.206%）；primary 为 2,546,863/2,919,835（87.226%）；strict 与 production 相同（74.206%）。primary 相对 production 恢复 380,163 行，占全部 13.020%，占原 production invalid 的 50.477%；`strict_not_primary_n=0`。
5. primary binocular source mode：1,150,214 binocular（77.282%）、155,788 left-only（10.467%）、90,647 right-only（6.091%）、91,683 missing（6.160%）。strict mode 对应为 838,858（56.362%）、286,838（19.272%）、202,146（13.582%）、160,490（10.783%）。这些是当前数据层的 source mode 描述，不是排除规则。
6. subject×eye×Block 的 primary-valid fraction 分位数为 min 3.018%、P05 41.942%、P10 65.780%、P25 86.489%、median 94.021%、P75 97.027%、P90 98.904%、P95 99.438%、max 99.783%。recovered-vs-production fraction 分位数为 min 0、P05 0.016%、P10 0.105%、P25 0.534%、median 2.859%、P75 12.150%、P90 48.113%、P95 62.744%、max 98.527%。
7. 预先列出的 10 个重点 subject 均已输出到 `known_low_usable_subject_changes.csv`（40 行）。其四个 eye×Block 单元的 primary fraction 范围分别为：sub-035 92.4–98.3%、sub-036 38.4–96.3%、sub-047 3.0–92.2%、sub-050 76.0–99.7%、sub-150 14.2–96.8%、sub-153 71.0–98.7%、sub-163 65.7–90.6%、sub-165 78.0–98.6%、sub-171 96.7–99.4%、sub-176 62.9–97.8%。这只是异常/低 usable 识别，不构成 subject 排除决定。
8. 左右眼时间配对由物化器按 `subject/phase/phase_segment/frame_idx/eye` 唯一键和 `unix_ms/video_time_ms/phase_time_ms` ≤1 ms 一致性检查；全量物化没有触发异常。派生宽表时间键无重复。44 个 source CSV、completion marker 全部存在；物化 manifest 标记 `production_read_only=true`；物化后 source CSV 的 size 和 mtime 与记录值全部一致，未发现源文件被修改。

### Warning / error

- 测试无 error：`5 passed`。
- 物化无 subject failure：44/44 成功，`subject_load_failures.csv` 为 0 行。
- pandas 输出了两类 `FutureWarning`（空项拼接和 object dtype `fillna` 的未来行为），未导致 traceback、失败或结果中断；本次未修改代码处理该 warning。
- 未发现 `phase_time_ms` 缺失、时间配对异常、重复键或 source provenance size/mtime 不一致。
- `source_csv_sha256` 未填充是当前配置 `hash_fullclass_csv=false` 的既定行为；completion marker SHA256 和 source size/mtime 仍被记录。

### 最终决策结果

本次仅完成 exploratory analysis-ready 数据层物化与新旧 validity 描述性核验。primary 与 strict 均保留为并行口径；未设置阈值、未删除 subject/eye/Block、未把结果解释为最终 cohort，也未进入 Behavior 联结、窗口 coverage、时间-on-task 或正式显著性分析。

### 已完成 / 未完成 / 待确认事项

- 已完成：测试、44 个 header 的 `phase_time_ms` 检查、analysis-ready 物化、validity/binocular/重点 subject/时间和 provenance 核验、Issue #18 结果回复。
- 未完成：正式分析计划下的最终筛选、Behavior/Probe 联结、窗口级 coverage/concordance 和统计模型；这些不属于 Issue #18 范围。
- 待确认：由后续正式分析计划决定 primary/strict 的报告方式、是否进行敏感性分析以及任何 subject/eye/Block 纳入解释。

### 运行环境与产物

- Python：3.11.15（Anaconda，`D:\CondaEnvs\nir-amd`）；numpy 2.4.6；pandas 2.3.3；scipy、statsmodels、scikit-learn、pyarrow 未安装，本任务未依赖它们。
- 运行时粗略资源：约 5.0/15.8 GB RAM 可用；D: 盘约 255.4 GB 可用。
- 输出目录：`D:\_AttentionData\Beijing-NIR\analysis\nir-behavior-v2\cohort-44-exploratory\10_analysis_ready`。
- 主要产物：44 个 `frame_level/sub-XXX/sub-XXX_nir_analysis_ready.csv`、44 个 subject baseline 文件、`subject_eye_baselines.csv`、`subject_eye_block_inclusion.csv`、`known_low_usable_subject_changes.csv`、`cohort_inclusion_summary.json`、`analysis_ready_manifest.json`、`source_files.csv`、`subject_load_failures.csv`。
