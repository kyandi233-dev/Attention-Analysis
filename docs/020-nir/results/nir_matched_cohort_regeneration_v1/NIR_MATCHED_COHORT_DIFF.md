# NIR matched cohort regeneration

状态：`repo_closeout_review_ready`

本报告同步本地已经完成并独立 review 的 matched cohort 聚合审计结果。它只保存 cohort 级计数、差异、生成 provenance（可追溯来源）和本地派生产物 hash，不上传原始 AVI、行为 CSV、subject-level 数据或大型 probe-level 表。

## Cohort diff

| 指标 | old cohort | new cohort | delta |
|---|---:|---:|---:|
| matched sessions | 68 | 71 | +3 |
| unique probes / `(subject, probe_id)` | 1,360 | 1,420 | +60 |
| primary coverage `>=0.80` | 未在旧 v1 matrix 中记录 | 1,174 | not comparable |
| sensitivity-inclusive coverage `>=0.50` | 未在旧 v1 matrix 中记录 | 1,212 | not comparable |
| sensitivity-only coverage `0.50-<0.80` | 未在旧 v1 matrix 中记录 | 38 | not comparable |
| excluded coverage `<0.50` | 未在旧 v1 matrix 中记录 | 208 | not comparable |

Added sessions are `sub-067`, `sub-100`, and `sub-178`. No previous session was removed. `sub-067` is present because its completed NIR fullclass output and behavior alignment are available; its separate mmWave usability status is not used as a NIR cohort inclusion rule.

Each recovered session contributes 20 unique `(subject, probe_id)` probes. The alignment output value `probe rows = 160` is not 160 statistical probes: it is 20 probes multiplied by 8 probe-window rows per probe in the alignment table.

## Recovered sessions

| session | unique probes | primary `>=0.80` | sensitivity-inclusive `>=0.50` | sensitivity-only `0.50-<0.80` | excluded `<0.50` |
|---|---:|---:|---:|---:|---:|
| sub-100 | 20 | 10 | 10 | 0 | 10 |
| sub-178 | 20 | 17 | 17 | 0 | 3 |

The sensitivity-inclusive count includes primary probes. For both recovered sessions there were no probes in the intermediate sensitivity-only band; the remaining probes were below 0.50.

## The excluded 72nd source session

`sub-099` remains the only mounted/formal source session outside the 71-session cohort. Its formal NIR run failed at initialization because `J:\Data\sub-099_\beh\master_timeline.csv` is missing. It therefore has no completed formal recovery, fullclass, or alignment chain. This is a timeline input blocker, not a claim of AVI frame loss. Sub-099 recovery was intentionally not processed in this closeout.

## Reproducibility and provenance

- Repository source commit: `a26e2f8ccd2f2f8fba5af1d90b6bdc85ee499d7f` (`nvidia-cuda`), the reviewed source state before this closeout.
- Historical generation commands actually used, without NIR rerun:
  - `python D:\Project\厚粲杯\11_数据\derived\build_formal_nir_probe_windows_unfiltered_v1.py --output D:\Project\厚粲杯\11_数据\derived\formal_nir_probe_windows_unfiltered_v3`
  - `python D:\Project\厚粲杯\11_数据\derived\build_formal_nir_probe_quality_tiers_v1.py --source D:\Project\厚粲杯\11_数据\derived\formal_nir_probe_windows_unfiltered_v3\nir_probe_windows_unfiltered.csv --output D:\Project\厚粲杯\11_数据\derived\formal_nir_probe_windows_quality_tiered_v3`
- Version-controlled reusable generator added in this closeout: `scripts/nir_matched_cohort_regeneration_v1.py`. It preserves the existing 30 s window, `>=0.80` primary tier, `0.50-<0.80` sensitivity-only tier, `<0.50` exclusion tier, and `(subject, probe_id)` key. It was syntax-checked but deliberately not executed in this closeout, because the cohort was already regenerated and the user explicitly prohibited regeneration.
- Source formal NIR root: `D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR`; completed fullclass sessions: 71; completed formal recovery sessions used for the new table: 71.
- Old aggregate input: `D:\Project\厚粲杯\11_数据\derived\nir_69session_final_probe_analysis_v1\analysis_probe_matrix_30s.csv`, 1,360 rows and 68 sessions.
- New aggregate input: `D:\Project\厚粲杯\11_数据\derived\formal_nir_probe_windows_unfiltered_v3\nir_probe_windows_unfiltered.csv`, 1,420 rows and 71 sessions.
- New tiered output: `D:\Project\厚粲杯\11_数据\derived\formal_nir_probe_windows_quality_tiered_v3\nir_probe_windows_quality_tiered.csv`, 1,420 rows and 71 sessions.
- New audit outputs: `nir_probe_window_build_audit.csv` has 71 session rows; `nir_probe_quality_tier_subject_summary.csv` has 71 session rows; the tier count is 1,174 primary, 38 sensitivity-only, and 208 excluded.

### SHA-256 of local derived outputs

| artifact | SHA-256 |
|---|---|
| `formal_nir_probe_windows_unfiltered_v3/nir_probe_windows_unfiltered.csv` | `6F9BDFE1A2B512A987F6A2F43181A25E75943606253C7AE4A1CE6FE8D762B4DF` |
| `formal_nir_probe_windows_unfiltered_v3/nir_probe_window_build_audit.csv` | `A37CF61D4BFB543C12C419B93557C975037CB762732450F63C7085215D43E4A5` |
| `formal_nir_probe_windows_unfiltered_v3/summary.json` | `48D95B8BB2296F381394C12B904F4516A6C35931FCAFC857FA54F7FAC4459BC8` |
| `formal_nir_probe_windows_quality_tiered_v3/nir_probe_windows_quality_tiered.csv` | `A9301F1B343CB91BEFA997229185F346EA18EF21696DBF1601779D6F747AFDC8` |
| `formal_nir_probe_windows_quality_tiered_v3/nir_probe_quality_tier_counts.csv` | `FB3BE9E6B6402E247053B91A976B22FCF910BF88D26B8D0335E195136B54CC3E` |
| `formal_nir_probe_windows_quality_tiered_v3/summary.json` | `9953B4F189A3255B596CF5B74E7589B7D93E64E9180D58E6A8FB091B8316C5FE` |

The existing 68-session/1,360-probe NIR v1 analysis was not rerun. No NIR v1 model, threshold, or scientific definition was changed.
