# NIR v1 最小科学特征层与 dry-run QC 报告

状态：`NIR_V1_MINIMAL_SCIENTIFIC_FIX`，等待科学 review。本文档只新增 feature/QC layer，不修改已冻结的 71-session cohort、coverage 分层、timestamp recovery、sub-099、CUDA runtime、matched cohort provenance 或正式统计结论。

## 1. 冻结输入与分层

输入直接使用既有 71-session / 1420-probe cohort 表，未重新发现 session，也未重跑 NIR v1：

- primary：1174 probes，coverage ≥ 0.80；
- sensitivity-only：38 probes，0.50 ≤ coverage < 0.80；
- excluded：208 probes，coverage < 0.50。

脚本在运行时重新计算这三个标签仅用于一致性断言，若不是 1174/38/208 会直接失败，不会自动修正 cohort。

## 2. 正式最小特征

正式 pupil measure 是 full-class `fullclass_pupil_to_iris_diameter_ratio`（PIR），旧 `pupil_equiv_diameter` 不再作为 primary pupil feature，仅用于相关性审计。每个 subject/session × probe × window 输出左右眼独立字段：PIR median、MAD（median absolute deviation，中位绝对离差）、robust slope、valid fraction，以及 QC/missingness 计数。

fused 值是“每只可用眼先汇总，再对左右眼 summary 取中位数”，不是把左右眼 frame 混池，因此左右眼身份不会静默丢失。`left_*`、`right_*`、`n_eyes_with_pir` 和 `eyes_both_available` 均保留。

仅使用 10 s、20 s sensitivity window 和 30 s primary window。每个 window 限制在当前 block，输出 requested/available bounds、block-boundary truncation 和 previous-probe crossing 字段。本次 10/20/30 s 的 boundary truncation 与 previous-probe crossing 均为 0，字段仍保留供后续 session 审计。

RITnet failure、ROI clipped、segmentation failure、PIR invalid 只作为 QC/missingness，不解释为 blink 或 eye closure。OAR 仅作为 secondary exploratory 数值保留，未命名为 blink/PERCLOS。

## 3. Dry-run 结果

产出共 4260 行，即 71 sessions × 20 probes × 3 windows。

| 项目 | 10 s | 20 s | 30 s |
|---|---:|---:|---:|
| 至少一只眼有 PIR | 1408/1420 | 1409/1420 | 1413/1420 |
| 双眼均有 PIR | 1355/1420 | 1377/1420 | 1384/1420 |
| block boundary truncation | 0 | 0 | 0 |
| previous-probe crossing | 0 | 0 | 0 |

30 s PIR fused median（n=1413）分布：min=0.1292，q01=0.2140，q05=0.2462，q25=0.2842，median=0.3227，q75=0.3623，q95=0.4428，q99=0.5918，max=0.7454，mean=0.3306，SD=0.0670。按本次 audit-only 规则 `[0, 2]` 检查，明显异常数为 0；该规则不是新的纳入阈值。

在不改变 coverage tier 的前提下，30 s primary tier 的 1174 个 probe 中 PIR fused feature 可用 1174 个，双眼均可用 1155 个。其余 30 s PIR 缺失集中在 sensitivity/excluded rows，不会被本层静默改写为 primary 纳入或排除。

30 s 至少一次有效 PIR 的左右眼 probe 数分别为 left=1409、right=1388；左右眼有效率以每眼 window 内 `fullclass_normalization_valid` 且 PIR 有限为准，并保留每行 valid fraction。每个 subject 均有 20 个 30 s probe rows，min=max=20，0 个 subject 偏离 20，因此没有 subject probe-count imbalance。

新 PIR fused median 与同 window、同 probe 的旧 absolute `pupil_equiv_diameter` median 的相关性：

| window | n | Pearson r | Spearman ρ |
|---|---:|---:|---:|
| 10 s | 1408 | 0.5031 | 0.6604 |
| 20 s | 1409 | 0.4896 | 0.6615 |
| 30 s | 1413 | 0.4992 | 0.6715 |

相关性仅说明两种测量在本 dry-run 中的关系，不把旧 diameter 恢复为 primary measure，也不替代科学 review。

## 4. 行为 probe source 语义

已核验 v3.1.3 最终 task source 使用的两张 probe PNG。`probe_response`：1=完全专注于分拣任务，2=关注实验本身但没有聚焦于分拣任务，3=在想与实验无关的事情，4=大脑空白、没有明确想法。`probe_vigilance`：1=非常困倦，2=比较困倦，3=比较清醒，4=非常清醒。feature 表同时保留 `probe_response_raw` 与 `probe_vigilance_raw`，未覆盖原始数值。

## 5. Subject-aware 与未来 ML 边界

输出包含 `subject`、`block`、`time_on_task_sec`、`probe_order_in_block` 和 `probe_order`，并为每个 window 的 fused PIR median 提供 `pir_subject_mean` 与 `pir_within_subject_deviation`。当前值是 dry-run 描述性分解，不能直接作为机器学习输入：任何 future cross-validation 中，subject centering 参数必须只由 training fold 计算，再应用到 held-out subject，禁止使用本报告预计算的全 cohort 参数造成 leakage。

## 6. 可复现性与产物边界

版本化代码：[scripts/nir_v1_scientific_feature_extraction.py](../../../../scripts/nir_v1_scientific_feature_extraction.py)。测试：[tests/test_nir_v1_scientific_feature_extraction.py](../../../../tests/test_nir_v1_scientific_feature_extraction.py)。

运行命令：

```powershell
python scripts/nir_v1_scientific_feature_extraction.py --cohort-input "D:\Project\厚粲杯\11_数据\derived\formal_nir_probe_windows_unfiltered_v3\nir_probe_windows_unfiltered.csv" --nir-root "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR" --data-root "J:\Data" --output-root "D:\Project\厚粲杯\11_数据\derived\nir_v1_scientific_fix_v1"
```

source Git commit：`7fe8641eafdc82f6add18ecf28e7e3801ac1cfbd`。本地 derived 输出目录为 `D:\Project\厚粲杯\11_数据\derived\nir_v1_scientific_fix_v1`，其中包含 `nir_v1_probe_features.csv`、`subject_summary.csv`、`audit_summary.json` 和 `provenance_manifest.json`。manifest 记录输入/输出行数、命令和 SHA-256；大型 probe-level CSV 与原始/隐私数据不提交 GitHub。

## 7. 结论边界

本层已具备供 review 的、可对齐的 PIR feature/QC 产物，但尚未修改正式统计结果。只有在本报告通过科学 review 后，才可决定是否开始正式 NIR 统计。
