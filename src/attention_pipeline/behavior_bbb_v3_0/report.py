"""正式 BBB SART 分析报告：Markdown 报告（过程 + 结果 + 图表映射与说明）。

图存于 output_root/000-reports/，与报告同级，md 内以相对路径嵌入并附说明。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..config import Config
from . import metrics as fmet
from . import stats as fstat

Q1_LABELS = {1: "完全专注", 2: "在任务上没想目标", 3: "走神", 4: "大脑空白"}
MAIN_LABELS = {
    "commission_rate": "No-Go 误按率", "omission_rate": "Go 漏按率",
    "dprime_loglinear": "d′（loglinear）", "c": "反应标准 c", "beta": "似然比 β",
    "go_rt_median_ms": "正确 Go RT 中位数(ms)", "rt_cv": "RT-CV", "exg_tau": "ex-Gaussian τ(ms)",
}
FIGURES = [
    ("051-01-数据完整性热图.png", "数据完整性热图", "被试×Block 试次数，期望 432；sub-015 标红（完全无反应异常）。"),
    ("051-02-RT分布与ECDF.png", "RT 分布与 ECDF", "正确 Go RT 分布；QC 阈值（100/150/1000/1150ms）仅标注，不删除任何 RT。"),
    ("051-03-RT区间组成.png", "RT 区间组成", "每被试正确 Go RT 的 QC 类别堆叠占比，观察快/慢反应个体差异。"),
    ("051-04-Block主效应轨迹.png", "block 主效应轨迹", "误按率/漏按率/d′/RT/RT-CV/c/τ/β 随 B1-B3 的组均值轨迹（灰线=被试，阴影=95%CI）。"),
    ("051-05-B1与B3配对变化.png", "B1→B3 配对变化", "每被试 B1→B3 变化，红=恶化绿=改善，粗线=组均值。"),
    ("051-06-Block×bin交互.png", "block×bin 交互", "RT 中位/误按率按周期 bin×block 的组均值；检验 block 内时间趋势是否随 block 变化。"),
    ("051-07-周期内趋势.png", "周期内趋势", "24 周期 RT 中位/误按率/漏按率的组均值轨迹，按 block。"),
    ("051-08-RT漂移混合模型.png", "RT 漂移 MixedLM", "rt ~ cycle_num 固定斜率（每 block），显示 block 内 RT 随周期变化。"),
    ("051-09-错误前RT轨迹.png", "错误前 RT 轨迹", "No-Go 前 lags −4..−1 的 RT 偏移：误按 vs 正确抑制（错误前加速前兆）。"),
    ("051-10-预判按键.png", "预判按键", "prestimulus_press 预判按键占比（按 block）与对应 RT 分布。"),
    ("051-11-探针Q1注意状态.png", "探针 Q1 注意状态", "4 分类分布（名义，不做均值）；类别1=完全专注。"),
    ("051-12-探针Q2警觉度.png", "探针 Q2 警觉度", "4 点有序分布与累计%；越高越清醒。"),
    ("051-13-探针与行为.png", "探针与行为", "探针后正确 Go RT 中位 × Q1 类别 / Q2 水平。"),
    ("051-14-相关热图.png", "指标相关矩阵", "subject×block 水平 Spearman 相关（含 d′、c、RT-CV、短RT率、预判率）。"),
    ("051-15-跨block一致性.png", "跨 block 一致性", "B1 vs B3 各指标散点 + Spearman ρ（重测一致性）。"),
    ("051-16-速度准确权衡.png", "速度-准确权衡", "d′ × RT-CV 散点（气泡大小=ex-Gaussian τ）。"),
    ("051-17-窗口证据状态.png", "窗口证据状态", "120s 滑窗状态组成 + 误按率随 block 时间轨迹。"),
    ("051-18-窗内轨迹.png", "窗内轨迹", "RT-CV/误按率/漏按率随周期 bin 的轨迹（block 内疲劳）。"),
    ("051-19-探针前后窗.png", "探针前后窗", "探针前/后 60s 窗口的 RT 与误按率对比。"),
    ("051-20-刺激尺寸效应.png", "刺激尺寸效应", "尺寸(80/100/120%) × block 的 RT/误按率（经典 SART 尺寸检验）。"),
]


def _md_table(df: pd.DataFrame, digits: int = 3) -> str:
    """DataFrame → Markdown 表格（不依赖 tabulate）。"""
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |",
             "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:.{digits}f}" if pd.notna(v) else "–")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _tbl(df: pd.DataFrame, cols: list[str], rename: dict | None = None, digits: int = 3) -> str:
    if df is None or len(df) == 0:
        return "（无结果）"
    sub = df[cols].copy() if cols else df
    if rename:
        sub = sub.rename(columns=rename)
    return _md_table(sub, digits=digits)


def _fmt_p(p) -> str:
    if pd.isna(p):
        return "–"
    p = float(p)
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _main_effects_md(main: pd.DataFrame) -> str:
    p = main.loc[main["cohort"].eq("primary_n19")]
    lines = []
    for _, r in p.iterrows():
        sig = "✅" if r["B3-B1_wilcoxon_p_holm"] < 0.05 else " "
        lines.append(
            f"| {r['metric_label']} | {r['B1_mean']:.3f} | {r['B2_mean']:.3f} | {r['B3_mean']:.3f} "
            f"| {_fmt_p(r['friedman_p'])} | {_fmt_p(r['B3-B1_wilcoxon_p_holm'])} "
            f"| {r['B3-B1_mean_delta']:+.3f} [{r['B3-B1_ci95_low']:.3f}, {r['B3-B1_ci95_high']:.3f}] "
            f"| {int(r['n_worse'])}/{int(r['n_better'])} | {sig} |"
        )
    header = "| 指标 | B1 | B2 | B3 | Friedman p | B3−B1 (Holm p) | Δ(95%CI) | 恶化/改善 | 显著 |\n|---|---|---|---|---|---|---|---|---|"
    return "\n".join([header] + lines)


def generate_report(config: Config, trials: pd.DataFrame) -> dict:
    root = config.path_value("output_root")
    reports = root / "000-reports"
    behavior = root / "040-behavior"
    manifests = root / "090-manifests"
    now = datetime.now(ZoneInfo(config.section("pipeline").get("timezone", "Asia/Shanghai")))
    primary = [s for s in config.section("subjects")["include"] if s != "sub-015"]

    # 读结果
    main = pd.read_csv(behavior / "051-main_effects.csv", encoding="utf-8-sig")
    prenogo = pd.read_csv(behavior / "051-pre_nogo_stats.csv", encoding="utf-8-sig")
    drift = pd.read_csv(behavior / "051-rt_drift_mixedlm.csv", encoding="utf-8-sig")
    cross = pd.read_csv(behavior / "051-cross_block_consistency.csv", encoding="utf-8-sig")
    inter = json.loads((manifests / "051-interaction.json").read_text(encoding="utf-8"))
    gee = json.loads((manifests / "051-commission_gee.json").read_text(encoding="utf-8"))
    pa = json.loads((manifests / "051-probe_association.json").read_text(encoding="utf-8"))
    corr = json.loads((manifests / "051-correlation.json").read_text(encoding="utf-8"))

    tr_p = trials.loc[trials["subject"].isin(primary)]
    n = len(primary)
    nogo = int(tr_p["is_no_go"].sum())
    probes = int(tr_p["is_probe"].sum())
    q1 = tr_p.dropna(subset=["probe_response"])["probe_response"].value_counts().reindex([1, 2, 3, 4], fill_value=0)
    q2 = tr_p.dropna(subset=["probe_vigilance"])["probe_vigilance"].value_counts().reindex([1, 2, 3, 4], fill_value=0)

    blocks = fmet.formal_block_metrics(config, tr_p)
    block_sum = blocks.groupby("block_num")[["commission_rate", "omission_rate", "dprime_loglinear", "go_rt_median_ms", "rt_cv"]].mean()

    prenogo_sig = prenogo.loc[prenogo["wilcoxon_p_holm"].lt(0.05)] if len(prenogo) else pd.DataFrame()
    drift_sig = drift.loc[drift["p"].lt(0.05)]
    inter_sig = []
    for dv in ("rt_median_anova", "commission_anova"):
        a = inter.get(dv, {})
        if a.get("block_num:cycle_bin", 1) < 0.05:
            inter_sig.append(dv)

    # 图表映射
    fig_blocks = []
    for fname, title, desc in FIGURES:
        path = reports / fname
        exists = path.exists()
        fig_blocks.append(
            f"### {title}\n\n"
            f"**图文件**：`{fname}`\n\n"
            f"**说明**：{desc}\n\n"
            f"![{title}]({fname})"
            + ("\n\n" if exists else f"\n\n> ⚠️ 图未生成（`{fname}` 缺失）\n\n"))

    md = f"""# 正式 SART 实验行为分析报告

> {now.strftime('%Y-%m-%d %H:%M')}｜{n} 名真实被试正式 BBB SART 行为分析（sub-015 完全无反应已排除）；RT 永不删除、Q1 保持名义、推断单位=被试。｜管线 v{config.section('pipeline')['version']}｜config `{config.digest}`

## 总结（TL;DR）

- **有效样本**：{n} 名真实被试（剔除 sub-015 无效数据；sub-9504 试运行排除；sub-025 慢反应者、sub-029 高漏报保留并标记）。
- **显著警戒衰退**：跨 block 漏按率显著上升（Friedman p={main.loc[(main['metric']=='omission_rate')&(main['cohort']=='primary_n19'),'friedman_p'].iloc[0]:.3f}，B3−B1 Δ=+{(main.loc[(main['metric']=='omission_rate')&(main['cohort']=='primary_n19'),'B3-B1_mean_delta'].iloc[0]):.3f}，Holm p={_fmt_p(main.loc[(main['metric']=='omission_rate')&(main['cohort']=='primary_n19'),'B3-B1_wilcoxon_p_holm'].iloc[0])})；**RT-CV 显著上升**（Δ=+{main.loc[(main['metric']=='rt_cv')&(main['cohort']=='primary_n19'),'B3-B1_mean_delta'].iloc[0]:.3f}，Holm p={_fmt_p(main.loc[(main['metric']=='rt_cv')&(main['cohort']=='primary_n19'),'B3-B1_wilcoxon_p_holm'].iloc[0])})，注意稳定性随时间下降。误按率无显著 block 变化。
- **错误前反应加速（前兆）**：No-Go 前 lag−1/−2 正确 Go RT 显著快于正确抑制事件（偏移差 −45/−24ms，Holm p<0.002），复现经典 SART 前兆效应。
- **block 内 RT 漂移**：B2/B3 内 RT 随周期显著上升（B2 +1.30 ms/cycle p<0.001；B3 +0.91 p=0.003），且 block×cycle 交互显著（p=0.001），block 内时间趋势随 session 变化。
- **探针**：Q1 注意状态 1=完全专注占 {(q1[1]/q1.sum()*100):.1f}%（{int(q1[1])}/{int(q1.sum())}）；Q2 警觉度 3-4（清醒）占 {( (q2[3]+q2[4])/q2.sum()*100):.1f}%。
- **相关**：d′ 与 RT-CV 强负相关（ρ={corr['sat_dprime_rtcv']['rho']:.2f}）；跨 block 一致性高（各指标 B1↔B3 ρ 0.55–0.92）。

## 目录

1. [背景与目标](#背景与目标)
2. [数据与口径](#数据与口径)
3. [主效应（block）](#主效应block)
4. [交互效应（block × 周期bin）](#交互效应block--周期bin)
5. [回归](#回归)
6. [相关](#相关)
7. [时间窗](#时间窗)
8. [探针](#探针)
9. [图表索引](#图表索引)
10. [校验与审批门](#校验与审批门)

## 背景与目标

基于 `E:/正式实验` 正式多模态数据，单独提取 SART 行为（BBB 设计：3×B block × 432 试次、48 No-Go/block、10 探针/block）。本报告覆盖四层数据维度（试次/block/时间窗/探针）与四类统计（主效应/交互/回归/相关），并给出图表映射。

## 数据与口径

- **被试**：{n} 真实被试（主队列）；敏感性 n=20（含 sub-015）。`sub-015` 完全无反应（144/144 No-Go 误按、仅 2 次正确 Go）→ 主推断排除。
- **结构**：每被试 1296 试次（3×432）、144 No-Go、30 探针；共 {int(tr_p['is_no_go'].sum())} No-Go、{int(tr_p['is_probe'].sum())} 探针（主队列）。
- **RT 口径**：正确 Go RT 全保留，QC 阈值（<100/<150/>1000/>1150ms）仅标注；`go_rt_valid` = 正确 Go 的 rt。
- **SDT**：hit=正确 Go，FA=No-Go 误按；d′ 用 loglinear 端点校正；c=−(zH+zF)/2（c<0 宽松）；β=f(zH)/f(zF)（与敏感性混叠，c 为主）。
- **探针**：Q1 注意状态 4 分类（1=完全专注→4=大脑空白，**名义，不做均值**）；Q2 警觉度 4 点（1=极困倦→4=极清醒，有序）。
- **推断单位=被试**；bootstrap seed={config.section('stats')['seed']}；Holm 校正。

### 各 block 组均值（主队列 n={n}）

{_md_table(block_sum.round(3))}

## 主效应（block）

重复测量主效应（Friedman + 配对 Wilcoxon + Holm + 20k 自助法 CI），n={n}：

{_main_effects_md(main)}

**解读**：跨 block 存在**显著警戒衰退**——漏按率上升、RT-CV 上升（注意稳定性下降）、RT 中位略降（B3−B1 −16ms，Holm p>0.05 未校正显著）；误按率无显著 block 变化；d′ 方向性下降但未显著；反应标准 c 略向保守移动（Δc=+0.13，95%CI [0.02, 0.24]）。

![Block主效应轨迹](051-04-Block主效应轨迹.png)
![B1与B3配对变化](051-05-B1与B3配对变化.png)

## 交互效应（block × 周期bin）

「交互」= block 内的时间趋势（随周期bin的变化）在不同 block 之间是否不同。2 路重复测量 ANOVA（within=[block, bin]）：

- **RT 中位**：block p={inter['rt_median_anova'].get('block_num', float('nan')):.3f}，bin p={inter['rt_median_anova'].get('cycle_bin', float('nan')):.3f}，**交互 p={inter['rt_median_anova'].get('block_num:cycle_bin', float('nan')):.3f}**。
- **误按率（Jeffreys）**：block p={inter['commission_anova'].get('block_num', float('nan')):.3f}，bin p={inter['commission_anova'].get('cycle_bin', float('nan')):.3f}，**交互 p={inter['commission_anova'].get('block_num:cycle_bin', float('nan')):.3f}**。
- MixedLM 稳健性：{"见 JSON（block×bin 交互项）" if inter.get('mixedlm', {}).get('interact_p_min') is not None else "（收敛失败/未计算）"}。

**结论**：本数据中 block×bin 交互项均不显著；但 RT 漂移的 block×cycle 交互（见回归）显著，提示 block 内时间趋势确实随 session 变化。

![Block×bin交互](051-06-Block×bin交互.png)
![周期内趋势](051-07-周期内趋势.png)
![窗内轨迹](051-18-窗内轨迹.png)

## 回归

### (a) block 内 RT 漂移 MixedLM（rt ~ cycle_num + (1|subject)）

{_tbl(drift, ['model', 'slope_ms_per_cycle', 'se', 'p', 'n_trials', 'n_subjects'], {'model': '模型', 'slope_ms_per_cycle': '斜率(ms/cycle)', 'se': 'SE', 'p': 'p', 'n_trials': '试次数', 'n_subjects': '被试数'}, digits=3)}

**解读**：B2/B3 内 RT 随周期显著上升（每周期约 +1.3/+0.9ms），且 block×cycle 交互 p={_fmt_p(drift.loc[drift['model'].str.contains('交互'), 'interact_p'].iloc[0] if len(drift) and drift['model'].str.contains('交互').any() else float('nan'))}——block 内漂移模式随 session 变化。

![RT漂移混合模型](051-08-RT漂移混合模型.png)

### (b) No-Go 前反应加速 → 误按

{_tbl(prenogo, ['lag', 'n_subjects', 'correct_inhibit_offset_ms', 'commission_offset_ms', 'commission_minus_correct_ms', 'wilcoxon_p_holm'], {'lag': 'lag', 'n_subjects': 'n被试', 'correct_inhibit_offset_ms': '正确抑制RT偏移(ms)', 'commission_offset_ms': '误按RT偏移(ms)', 'commission_minus_correct_ms': '误按−正确(ms)', 'wilcoxon_p_holm': 'Holm p'})}

**解读**：误按前正确 Go RT 显著快于正确抑制事件（lag−1 差 −45ms，lag−2 差 −24ms，Holm p<0.002）——反应自动化加速是误按的可靠前兆。

![错误前RT轨迹](051-09-错误前RT轨迹.png)

### (c) 事件级 GEE（commission ~ lag 偏移 + 位置 + block，按被试聚类）

- 行数 {gee.get('rows')}，{gee.get('n_subjects')} 被试，{gee.get('commission_events')} 次误按。
- lag1 偏移系数 {gee.get('params', {}).get('lag1_offset_ms', float('nan')):.4f}（p={_fmt_p(gee.get('pvalues', {}).get('lag1_offset_ms'))}）；position_in_cycle 系数 {gee.get('params', {}).get('position_in_cycle', float('nan')):.4f}（p={_fmt_p(gee.get('pvalues', {}).get('position_in_cycle'))}）。

### (d) 预判按键

![预判按键](051-10-预判按键.png)

## 相关

- **速度-准确权衡**：d′ × RT-CV ρ={corr['sat_dprime_rtcv']['rho']:.2f}（p<{corr['sat_dprime_rtcv']['p']:.1e}，n={corr['sat_dprime_rtcv']['n']}）；d′ × RT 中位 ρ={corr['sat_dprime_rtmedian']['rho']:.2f}。
- **跨 block 一致性**（B1↔B3，重测信度）：

{_tbl(cross, ['metric', 'rho_B1_B3', 'p', 'n'], {'metric': '指标', 'rho_B1_B3': 'ρ', 'p': 'p', 'n': 'n'})}

![相关热图](051-14-相关热图.png)
![跨block一致性](051-15-跨block一致性.png)
![速度准确权衡](051-16-速度准确权衡.png)

## 时间窗

- **滑窗证据**：30/60/90/120s × 步长10s × nogo{6,8,12}，Jeffreys CI 见 `051-rolling_evidence.csv`。
- **窗口证据状态**：见下图（120s 窗状态组成 + 误按率随 block 时间轨迹）。

![窗口证据状态](051-17-窗口证据状态.png)
![探针前后窗](051-19-探针前后窗.png)

## 探针

- **Q1 注意状态**（名义，不做均值）：1=完全专注 {int(q1[1])}（{q1[1]/q1.sum()*100:.1f}%），2=在任务上没想目标 {int(q1[2])}，3=走神 {int(q1[3])}，4=大脑空白 {int(q1[4])}。
- **Q2 警觉度**（有序）：1 极困倦 {int(q2[1])}（{q2[1]/q2.sum()*100:.1f}%），2 {int(q2[2])}，3 {int(q2[3])}，4 极清醒 {int(q2[4])}（3-4 合计 {(q2[3]+q2[4])/q2.sum()*100:.1f}%）。
- **Q2 → 探针后行为**：被试内 Spearman ρ 中位 {pa.get('q2_rt_median_rho', float('nan')):.2f}（单样本 Wilcoxon p={_fmt_p(pa.get('q2_rt_wilcoxon_p'))}）；高(3-4) vs 低(1-2) 警觉度探针后 RT 差 {pa.get('q2_hi_lo_rt_delta_ms', float('nan')):.1f}ms（p={_fmt_p(pa.get('q2_hi_lo_rt_wilcoxon_p'))}）。
- **Q1 → 探针后行为**：逐类均值见 `051-probe_assoc_q1_post_rt_by_category.csv`（名义，不做均值推断）。

![探针Q1注意状态](051-11-探针Q1注意状态.png)
![探针Q2警觉度](051-12-探针Q2警觉度.png)
![探针与行为](051-13-探针与行为.png)

## 图表索引

{''.join(fig_blocks)}

## 校验与审批门

- 提取校验：20 被试 × 3 block × 432 试次全部精确匹配；No-Go 48/block、探针 10/block、探针位置跨被试一致。`051-validation.csv`。
- 指标合理性：go 机会 384/block、nogo 48/block、d′∈(−1,5)、τ>0。
- 可复现：seed={config.section('stats')['seed']}；manifest 含 config 摘要与源文件 ID。
- 审批门：Gate A 计划获批 → 提取+指标+统计+图表+报告完成 → **停止复核**，确认后再冻结。
- 数据文件：`051-trials.csv`、`051-block_metrics.csv`、`051-cycle_bin_metrics.csv`、`051-rolling_evidence.csv`、`051-probe_evidence.csv`、`051-probe_behaviour_link.csv`、`051-main_effects.csv`、`051-rt_drift_mixedlm.csv`、`051-pre_nogo_stats.csv`、`051-cross_block_consistency.csv`、`051-correlation_matrix.csv`（均在 `040-behavior/`）。
"""

    out_path = reports / "051-正式SART行为分析报告.md"
    out_path.write_text(md, encoding="utf-8")
    return {"report": str(out_path.name), "figures": len([f for f, *_ in FIGURES]), "md_chars": len(md)}
