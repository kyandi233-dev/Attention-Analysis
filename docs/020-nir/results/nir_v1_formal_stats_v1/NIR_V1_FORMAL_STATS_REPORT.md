# NIR v1 formal statistics v1

Status: `NIR_V1_FORMAL_STATS_V1`. This is the first formal NIR scientific result. It does not change the frozen cohort, NIR runtime, recovery results, or formal statistical conclusions outside this report.

## Analysis definition

Primary analytic set: `nir_quality_tier == primary`, 30 s window, 1174 probes, 69 sessions, and 47 mapped participant clusters. Primary pupil measure: fused full-class PIR. Subject means and within-person deviations were recomputed inside this analytic set before modeling.

Repeated probes were handled with participant-clustered exchangeable GEE logistic regression because stable binomial mixed-effects fitting was not available in the current environment. Continuous predictors were standardized for odds-ratio presentation. Outcome: fully task-focused = response 1; not fully task-focused = responses 2/3/4.

## Primary model

| term | OR | 95% CI | p |
|---|---:|---:|---:|
| PIR within-person deviation | 1.111 | 0.968--1.275 | 0.133 |
| PIR MAD | 0.888 | 0.745--1.058 | 0.184 |
| PIR robust slope | 1.066 | 0.955--1.191 | 0.256 |
| block | 0.710 | 0.555--0.909 | 0.0065 |
| time in block | 0.770 | 0.659--0.900 | 0.0010 |

Response counts were 1/2/3/4 = 885/190/40/59. Response 2/3/4 are not relabeled as one mind-wandering category.

## Planned secondary contrasts

- 1 vs 2: PIR within-person deviation OR 1.068, 95% CI 0.934--1.221, p=0.337, n=1075.
- 1 vs 3: sparse contrast produced no finite GEE estimate, likely separation; n=925. This is reported as unstable, not replaced by an invented estimate.
- 1 vs 4: OR 1.209, 95% CI 0.985--1.483, p=0.069, n=944.

## Vigilance

Vigilance 1--4 was analyzed as an ordered numeric trend using participant-clustered GEE Gaussian. PIR within-person deviation beta=-0.00083, 95% CI -0.0252--0.0235, p=0.947. PIR MAD beta=0.00086, p=0.966. PIR robust slope beta=0.02396, 95% CI 0.0191--0.0288, p=3.88e-22. The strong slope-vigilance association means slope should be considered as a possible task-progress/arousal covariate; it is not evidence that PIR is blink or PERCLOS.

## Window sensitivity

| window | OR for PIR within-person deviation | 95% CI | p |
|---|---:|---:|---:|
| 10 s | 1.143 | 0.999--1.308 | 0.051 |
| 20 s | 1.114 | 0.974--1.275 | 0.116 |
| 30 s | 1.111 | 0.968--1.275 | 0.133 |

30 s remains the only primary window. No window was selected by significance.

## Human-readable conclusion

1. The fully-focused versus not-fully-focused comparison shows a small positive direction for PIR level, but the 30 s primary result is not statistically decisive.
2. PIR MAD and PIR slope do not show a decisive primary attention-state effect. The slope signal is strongly related to vigilance, so it should not be interpreted as attention-specific without adjustment.
3. The PIR-level direction is consistent across 10/20/30 s, but confidence intervals include the null.
4. PIR level and MAD are not strongly related to vigilance in this model; slope is. This supports treating vigilance/task progress as an alternative explanation or covariate rather than claiming a pure attention effect.
5. The NIR feature layer is eligible as a candidate input to the next multimodal model, with subject-aware evaluation and vigilance sensitivity analysis. This report is not a final attention classifier.

## Provenance and limitations

The versioned Beijing formal session crosswalk was used: 69 sessions map to 47 `repeat_participant_id` clusters. No identity was inferred from features. Figure 2 uses covariates on the same standardized scale as the fitted model and holds non-PIR covariates at their analytic-set means.

Machine-learning centering parameters must be recomputed inside each training fold. PIR invalidity, ROI clipping, RITnet failure and segmentation failure remain QC/missingness, not blink/PERCLOS labels. Detailed CSV summaries, figures, input SHA-256 and source commit are in this directory and `provenance.json`.
