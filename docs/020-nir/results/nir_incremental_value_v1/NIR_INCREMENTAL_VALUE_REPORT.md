# NIR incremental value v1

Status: `NIR_INCREMENTAL_VALUE_V1`.

## Design

The frozen NIR primary/30 s/PIR set was joined to existing probe-level behavior and mmWave matrices on the same `(subject, probe)` keys. The join contains 1174 probes, 69 sessions and 47 real participant clusters. `StratifiedGroupKFold` was used, so a repeat participant cannot occur in both train and test. Median imputation and standardization were fitted inside each training fold only.

NIR predictors were limited to PIR within-person level, PIR MAD and PIR slope. No deep learning or large feature expansion was used. Outcome: fully focused = response 1 versus not fully focused = response 2/3/4.

## Out-of-fold performance

| model | n | participant clusters | ROC-AUC | balanced accuracy | F1 | sensitivity | specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Behavior | 1174 | 47 | 0.5659 | 0.5707 | 0.7665 | 0.7435 | 0.3979 |
| Behavior + NIR | 1174 | 47 | 0.5989 | 0.5850 | 0.7471 | 0.6994 | 0.4706 |
| Behavior + mmWave | 1174 | 47 | 0.5714 | 0.5505 | 0.6649 | 0.5751 | 0.5260 |
| Behavior + mmWave + NIR | 1174 | 47 | 0.5803 | 0.5689 | 0.6775 | 0.5876 | 0.5502 |

## NIR incremental value over behavior

1000 participant-cluster bootstrap resamples gave:

- Delta ROC-AUC = +0.0330, 95% CI [-0.0051, 0.0750].
- Delta balanced accuracy = +0.0143, 95% CI [-0.0123, 0.0417].
- Delta F1 = -0.0193, 95% CI [-0.0495, 0.0054].
- Delta sensitivity = -0.0441, 95% CI [-0.0908, -0.0032].
- Delta specificity = +0.0727, 95% CI [0.0264, 0.1198].

## Human-readable answer

NIR adds a modest positive out-of-fold ranking signal over behavior: AUC increases by about 0.033. However, the participant-bootstrap AUC and balanced-accuracy intervals cross zero. The gain is therefore not yet a stable, independently validated classification improvement. NIR is worth retaining as a physiologically interpretable candidate modality, especially for multimodal comparison and covariate adjustment, but it should not be presented as a proven standalone attention classifier.

The NIR addition improves specificity but lowers sensitivity and F1 at the fixed 0.5 threshold. This is why AUC and threshold metrics are reported together.

## RGB/mmWave handoff

The existing formal mmWave probe-level matrix was included. No formal 71-session RGB probe-level feature matrix was found. Existing RGB files under `D:/Project/厚粲杯/11_数据/derived/current_j_rgb_motion_gate_v1` are gate/pilot outputs and were excluded from formal incremental modeling.

The direct competition figure is `figure_incremental_without_vs_with_nir.png`. Full performance and bootstrap tables are `model_performance.csv` and `incremental_deltas_bootstrap.csv`; input hashes and commands are in `provenance.json`.
