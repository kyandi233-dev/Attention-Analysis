# NIR v1 formal statistics v1

状态：`NIR_V1_FORMAL_STATS_V1`。本报告是第一版正式 NIR scientific result，不修改 cohort、底层 runtime 或既有 NIR v1 结果。

## Analysis definition

Primary analytic set = `nir_quality_tier == primary`, 30 s window, n=1174 probes. Primary pupil measure = fused full-class PIR. Subject means and within-person deviations were recomputed inside this analytic set before modeling. Repeated probes were handled with subject-clustered exchangeable GEE logistic regression because stable binomial mixed-effects fitting was not available in the current environment.

Outcome: fully task-focused = response 1; not fully task-focused = responses 2/3/4. The latter are not relabeled as a single mind-wandering category. Predictors: PIR within-person deviation, PIR MAD, PIR robust slope, block and time in block; continuous predictors are standardized for odds-ratio presentation.

## Main result

Analytic n=1174, subjects=69. Response counts: {1.0: 885, 2.0: 190, 3.0: 40, 4.0: 59}.

- pir_within_subject_deviation: OR=1.111, 95% CI [0.9684, 1.275], p=0.1329
- pir_fused_pir_mad: OR=0.8877, 95% CI [0.7447, 1.058], p=0.1841
- pir_fused_pir_robust_slope_per_s: OR=1.066, 95% CI [0.9547, 1.191], p=0.2556
- block: OR=0.7104, 95% CI [0.5553, 0.9087], p=0.006491
- time_on_task_sec: OR=0.7699, 95% CI [0.6585, 0.9], p=0.001034

## Four response categories and planned contrasts

The response descriptive table retains all four categories. Planned contrasts are 1 vs 2, 1 vs 3 and 1 vs 4; they are descriptive secondary models and were not used to redefine the primary outcome.
- 1 vs 2 PIR within-person deviation: OR=1.068, 95% CI [0.934, 1.221], p=0.3371, n=1075
- 1 vs 3: estimate unavailable because the sparse contrast did not produce finite GEE estimates (likely separation), n=925
- 1 vs 4 PIR within-person deviation: OR=1.209, 95% CI [0.9854, 1.483], p=0.06898, n=944

## Vigilance

Vigilance 1–4 was modeled as an ordered numeric outcome with subject-clustered GEE Gaussian as a pragmatic trend model.             outcome                             term    n  subjects  clusters  estimate    ci_low   ci_high      p_value                                        model
probe_vigilance_raw                            const 1174        69        47  3.567191  3.405428  3.728954 0.000000e+00 GEE Gaussian exchangeable, subject-clustered
probe_vigilance_raw     pir_within_subject_deviation 1174        69        47 -0.000827 -0.025190  0.023535 9.469231e-01 GEE Gaussian exchangeable, subject-clustered
probe_vigilance_raw                pir_fused_pir_mad 1174        69        47  0.000864 -0.039339  0.041068 9.663874e-01 GEE Gaussian exchangeable, subject-clustered
probe_vigilance_raw pir_fused_pir_robust_slope_per_s 1174        69        47  0.023961  0.019107  0.028815 3.875298e-22 GEE Gaussian exchangeable, subject-clustered
probe_vigilance_raw                            block 1174        69        47 -0.158187 -0.233068 -0.083306 3.465011e-05 GEE Gaussian exchangeable, subject-clustered
probe_vigilance_raw                 time_on_task_sec 1174        69        47 -0.111072 -0.161762 -0.060381 1.749599e-05 GEE Gaussian exchangeable, subject-clustered

This analysis does not establish causality. The focused-vs-not-focused result should be interpreted alongside the vigilance coefficient and its confidence interval, not as a blink/PERCLOS result.

## Sensitivity

The forest plot reports the same core PIR within-person-deviation effect for 10 s, 20 s and 30 s. 30 s remains the only primary window; no window was selected by significance.

## Human-readable conclusion

1. 专注和非专注是否有系统变化：见主模型中 PIR within-person deviation 的 OR/CI；OR>1 表示相对更大的 within-person PIR 与 fully task-focused 概率上升，OR<1 表示下降。
2. 变化主要体现在大小、波动还是趋势：比较 PIR within-person deviation、MAD 和 robust slope 三个项，报告不把未显著项解释为不存在效应。
3. 10/20/30 s 是否稳定：以 forest plot 的方向和置信区间共同判断，不按最显著窗口选主结果。
4. 是否主要由困倦造成：结合 vigilance GEE 结果。如果 PIR 与 vigilance 关联强于与 focused outcome 的关系，应把困倦作为重要替代解释；本版不将其强行解释为注意状态。
5. 是否进入多模态模型：本版 feature/QC 和正式统计已具备进入下一步候选模型的资格，但进入前应锁定本报告版本、保留 subject-aware 评估，并在融合模型中把 vigilance 作为协变量或敏感性分析。

## Limitations

未找到可靠的重复参与者映射，当前以 subject 作为 cluster/random-intercept grouping。模型是 GEE 近似而非 binomial mixed-effects。PIR feature 是统计量，不是 blink/PERCLOS；RITnet/ROI/PIR failure 仍是 QC/missingness。

## Provenance

{
  "source_git_commit": "cfa7d75e82b045ffd14ac3f4b8c0b92166611850",
  "input": "D:\\Project\\厚粲杯\\11_数据\\derived\\nir_v1_scientific_fix_v1\\nir_v1_probe_features.csv",
  "input_sha256": "0eaee4b9515296939c6268f507474443571b003a6491a50f56084551092b556a",
  "input_rows": 4260,
  "primary_rows": 1174,
  "subjects": 69,
  "clusters": 47,
  "crosswalk": "D:\\Project\\厚粲杯\\11_数据\\derived\\beijing_zhuhai_canonical_harmonization_v1\\beijing_zhuhai_person_session_crosswalk.csv",
  "command": "scripts/nir_v1_formal_stats.py --input D:\\Project\\厚粲杯\\11_数据\\derived\\nir_v1_scientific_fix_v1\\nir_v1_probe_features.csv --crosswalk D:\\Project\\厚粲杯\\11_数据\\derived\\beijing_zhuhai_canonical_harmonization_v1\\beijing_zhuhai_person_session_crosswalk.csv --output-root docs\\020-nir\\results\\nir_v1_formal_stats_v1",
  "windows": [
    10,
    20,
    30
  ],
  "centering": "recomputed by subject within each analytic window; training-fold-only required for future ML",
  "figure_files": [
    "figure_1_response_within_pir.png",
    "figure_2_predicted_focused_probability.png",
    "figure_3_vigilance_pir.png",
    "figure_4_effect_forest.png"
  ]
}

Participant mapping audit: a versioned Beijing formal session crosswalk was found and used. The 69 sessions map to 47 repeat_participant_id clusters; identity was not inferred from features.
