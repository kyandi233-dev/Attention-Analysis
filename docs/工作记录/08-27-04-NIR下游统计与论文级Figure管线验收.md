# 08-27-04｜NIR 下游统计与论文级 Figure 管线验收

## 总结

本记录用于追踪 `analysis/multimodal-integration` 分支上的 NIR 下游 validation-only 验收。当前 PIR 数值已知错误，本次只验证数据读取、schema/join、时间对齐、指标接口、模型 smoke test、QC、robustness、论文级 Figure、输出结构和输入不可变性；不解释任何 PIR 方向、差异、效应量或 p 值。

## 原计划

1. 在独立 worktree 核验目标分支与 HEAD，保护用户正在进行的 AMD/DirectML 上游工作。
2. 阅读当前 NIR 下游设计、配置、入口、源码和测试契约。
3. 使用已有 `D:\CondaEnvs\nir-amd` 环境执行静态检查和指定测试；仅修复 downstream validation 范围内的真实代码问题。
4. 先运行 `sub-031`，再运行当前已有 validated completion 的 subjects；只写外部 `12_pipeline_validation`。
5. 程序化核验表、JSON、Figure 1–10 与 S1–S4 的格式、DPI、尺寸、布局元数据，并检查 `10_analysis_ready`、`11_analysis_tables` 和 production 未被修改。
6. 复核分支内容后提交并报告 commit、状态和仍需人工查看项。

## 执行与决策过程

- 当前调用工作区为 `Attention-Analysis-rgb-amd`，分支 `rgb-amd`，工作区干净；没有切换该工作区。
- 已执行 `git fetch origin --prune`，确认 `origin/analysis/multimodal-integration` 存在。
- 已在同级建立独立 worktree `Attention-Analysis-analysis-validation`，分支为 `analysis/multimodal-integration`。
- 当前验证 worktree 初始 HEAD 为 `c4f3163b85e4b02b1df016f029e65ce5a3458aab`，初始状态干净。
- 预期同级路径 `Attention-Analysis-amd-DirectML` 当前不存在，因此没有可被本次操作修改的该名称 worktree；原调用工作区保持不变。
- 验证输入固定为只读 `10_analysis_ready` 与 `11_analysis_tables`；唯一允许的运行输出为 `12_pipeline_validation`。

## 风险与校验

- 任何 `10_analysis_ready`、`11_analysis_tables`、production/runtime 或正式统计目录变化都视为边界错误并停止。
- 不运行 YOLO/RITnet，不构建新的 `11_analysis_tables`，不进入 `20_formal_statistics`。
- 不根据当前错误 PIR 的结果选择窗口、track 或 QC threshold。
- Figure 视觉质量若当前环境不能直接肉眼查看，只报告程序化 QC，不声称已完成视觉验收。

## 最终决策结果

验收完成。`D:\CondaEnvs\nir-amd\python.exe` 的 compileall 通过；用户指定的 6 个测试文件共 24 项测试全部通过（24 passed, 0 failed, 0 skipped）。测试和运行仅产生已知的 pandas `stack` FutureWarning，以及 statsmodels 在当前验证样本上部分混合模型的奇异协方差/边界收敛警告。

先完成 `sub-031` smoke test，再按完成标记自动发现并完成 `sub-031`–`sub-041` 共 11 个被试的全量 validation-only 运行。core、publication、supplementary 三个 summary 均为 `complete`，并保留 `validation_only=true`、`nir_values_known_invalid=true`。`10_analysis_ready`、`11_analysis_tables` 和 `D:\_AttentionData\Beijing-NIR\amd-directml` 运行前后文件元数据均未改变，分别为 95、100、7772 个文件；没有新增或改写 11 级分析表。

输出包含 core 169 个文件、主 Figure 1–10 的 40 个文件和 supplementary Figure S1–S4 的 16 个文件；主/补充 Figure 均输出 PDF、SVG、PNG、TIFF，栅格宽度为 4015 px、有效 600 dpi，向量文件非空。程序化检查确认 trial outcome、program omission QC subtype、全局 block 时间连续性、10 个动态特征、6 条 robustness track、source-mode/coverage 及 publication 表接口均符合当前配置。当前 199 个 program omissions 的 subtype 为 clean 136、carryover-associated 56、prestimulus-associated 7，未见缺失；这些是结构/QC 结果，不是科学结论。

本次唯一的 downstream 代码修复位于 `src/attention_pipeline/nir_pipeline_validation/publication_figures.py`：压缩 Figure 3 omission subtype 标签、Figure 6–9 的长标签/边距，并将 Figure 10 smoke-test 系数图改为明确标注的稀疏 symlog 刻度，使异常宽的验证性区间可见但不伪装成普通线性论文效果图。模型 smoke test 仍有 16 个 complete、3 个 `LinAlgError: Singular matrix`（probe response option/vigilance、probe response option/pre-probe Go RT、probe response RT）；这些失败需作为当前数据结构/数值条件的 engineering status 保留，不能作科学解释。

OAR 状态仍为 `blocked_by_analysis_ready_schema`，questionnaire 状态为 unavailable，visual covariates 可用（27 行），raw between-person PIR 接口可用。没有运行 YOLO/RITnet，没有修改 production、10、11 或正式统计目录，没有生成正式科学结论。

## 已完成/未完成/待确认事项

- 已完成：目标分支与独立 worktree 路由；设计文档、配置、入口、源码和测试契约复核；`nir-amd` 环境安装/compileall；sub-031 smoke test；11 被试全量 validation-only 运行；表、JSON、Figure、DPI/尺寸、时间轴、输入不可变性和 Git diff 审计；Figure 视觉复核及布局修复。
- 已完成：主分支当前改动限定为 publication Figure layout/QC 修复和本工作记录；待提交到 `analysis/multimodal-integration`，不触及 `amd-DirectML` 或 `rgb-amd` 调用工作区。
- 未完成：不把当前输出送入 `20_formal_statistics`，不修复已知无效 PIR，不处理 OAR schema、questionnaire 缺失或 3 个 singular smoke models；这些属于后续明确范围。
- 待确认：正式论文使用前仍需在最终排版环境打开 PDF/SVG 做一次人工版面复核，并在合法、有效的 PIR/analysis-ready 数据准备后重新验证；当前 Figure 和表只能作为 validation artifact。
