# 正式 SART 行为分析计划

> 2026-08-16｜厚璨杯 attention-pipeline-v2｜基于 `E:/正式实验` 正式行为数据，制定并已执行的 SART 行为分析计划（数据口径/指标/统计/图表/输出/校验/审批门）。

## 背景与目标

正式实验为 v3.0 **BBB** 设计（2026-08-14 定稿，3×B block × 432 试次、48 No-Go/block、10 双题探针/block），共 20 名真实被试（sub-011~030）+ 试采 sub-9504。本计划**单独提取 SART 行为数据**并完成分析，覆盖数据维度（试次/block/时间窗/探针）与统计（主效应/交互/回归/相关），输出图文报告。

## 已核实的数据与程序事实

- **设计**：BBB × 432 试次（24 周期 × 18 位置；9 分钟/block）；48 No-Go/block（11.1%）；10 探针/block。
- **序列**：约束随机（前 10 全 Go、任意 27 连续 ≥3 No-Go、相邻不重复）→ 校验不硬编码 No-Go 位置。
- **时基**：`absolute_onset_time`=Unix 毫秒（窗口锚点）；`block_onset_time`=block 内相对毫秒。
- **探针**：Q1 注意状态 4 分类（**1=完全专注→4=大脑空白**，名义，不做均值；用户已确认）；Q2 警觉度 4 点（1=极困倦→4=极清醒，有序）。
- **预判标记**：`prestimulus_press_ms`（刺激上屏前死区按键）区分预判型/走神型。
- **异常**：`sub-015` 完全无反应（144/144 No-Go 误按、仅 2 次正确 Go）→ 主推断 n=19 + 敏感性 n=20；`sub-025` RT 均值 751ms、`sub-029` 漏报 22.3% → 保留+标记。

## 分析口径（冻结）

1. **RT 永不静默删除**；<100/<150/>1000/>1150 只作 QC 标注。
2. **d′/c/β 用 loglinear 校正**：hit=正确 Go，FA=No-Go 误按；c=−(zH+zF)/2（c<0 宽松）；β=f(zH)/f(zF)（c 为主 β 为辅）。
3. **RT-CV/RT-SD 只在 block/bin/长窗**（计数门控 ≥20）。
4. **探针 Q1 保持名义**，逐类报告，不做均值；Q2 有序处理。
5. **错误成分分轨**：Go 准确率 ≠ No-Go 抑制 ≠ d′ ≠ 反应标准。
6. **推断单位=被试**；bootstrap seed=20260816；Holm 校正。

## 统计方法

- **主效应（block）**：Friedman + Kendall's W → 配对 Wilcoxon（Holm）→ 20k 自助法 CI → AnovaRM 佐证。指标：commission/omission/d′/c/β/RT中位/rt_cv/ex-Gaussian τ。
- **交互（block × 周期bin）**：2 路重复测量 AnovaRM + MixedLM `rt ~ C(block)*C(bin) + (1|subject)`；解释规则 = 交互显著 + ≥14/19 被试方向一致。
- **回归**：(a) block 内 RT 漂移 MixedLM `rt ~ cycle_num + (1|subject)` + block×cycle；(b) 探针 Q2 → 行为（被试内 Spearman→Wilcoxon、二分化）、Q1 逐类对比；(c) No-Go 前 RT 偏移（lags −4..−1，Holm）+ 事件级 GEE 二项；(d) 预判按键。
- **相关**：速度-准确权衡（d′×RT-CV / RT中位）、跨 block 一致性（B1↔B3 ρ）、指标相关矩阵。
- **时间窗**：30/60/90/120s × 步长10s × nogo{6,8,12} 滑窗（Jeffreys CI）；探针前/后 30/60/90s 窗。
- **变异性/SDT**：RT-SD、rt_cv、ex-Gaussian（μ/σ/τ，τ=慢尾=注意滑脱）、c、β。

## 图表清单（20 张，每个分析族 ≥1 图，中文 dpi=180）

| # | 图 | 分析 |
|---|---|---|
| 01-03 | 数据完整性热图 / RT分布+ECDF / RT区间组成 | 数据质量 |
| 04-05 | Block主效应轨迹 / B1-B3配对变化 | 主效应 |
| 06-07 | Block×bin交互 / 周期内趋势 | 交互 |
| 08-10 | RT漂移混合模型 / 错误前RT轨迹 / 预判按键 | 回归 |
| 11-13 | 探针Q1注意状态 / Q2警觉度 / 探针与行为 | 探针 |
| 14-16 | 相关热图 / 跨block一致性 / 速度-准确权衡 | 相关 |
| 17-19 | 窗口证据状态 / 窗内轨迹 / 探针前后窗 | 时间窗 |
| 20 | 刺激尺寸效应 | 尺寸探索 |

## 输出与校验

- 输出：`D:/_AttentionData/output-v2/050-sart-formal/`（040-behavior CSVs + 000-reports 图/报告 + 090-manifests）。
- 校验：20×1296 试次、每 block 432/48/10 精确匹配、探针位置跨被试一致、时间戳 delta≤25ms、fail-fast。
- 审批门：Gate A 计划获批 → 提取/指标/统计/图表/报告完成 → **停止复核**。

## 已知风险

sub-015 主 n=19 双轨；Q1 类别失衡（1 占 64.5%）逐格报 n；小样本禁单 p 断言配 bootstrap；No-Go 非周期均匀用实际机会数；混合模型收敛失败退化被试内聚合；窗口锚点用 absolute_onset_time；短 bin rt_cv 计数门控；ex-Gaussian 优化器无效参数裁剪。
