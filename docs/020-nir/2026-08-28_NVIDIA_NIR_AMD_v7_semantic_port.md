# NVIDIA NIR semantic port from AMD v7 lean contract

## Baselines and boundary

- AMD source: `origin/amd-DirectML` = `b8a30e803ef21ab3983a4416d4359e905345c2e0`
- NVIDIA baseline: `origin/nvidia-cuda` = `14d33a76a53806c5d5366cd46e06b7f8660a121b`
- NVIDIA working base also retains local identity/profiling history; NVIDIA origin was merged locally before this port.
- No AMD branch was cherry-picked or merged. No DirectML runtime, provider, device semantics or AMD lifecycle code was imported.
- Frozen `ritnet-b16-fp32-uncertainty.onnx` and `.data` were not changed; YOLO was not rerun.

## Ported semantic changes

1. Lean eye schema v6: retain four hard/soft classes, pupil geometry, source-valid facts and three ocular uncertainty means; remove persisted iris/ocular geometry, PIR/OAR, percentile and boundary distributions from cohort output.
2. Pupil-only final geometry and QC composite reasons; four-class segmentation remains intact.
3. Core version upgraded to `fullclass-final-core-v7-pupil-only-lean-schema`; analysis-domain version upgraded to `source-backed-output-mask-v2-pupil-geometry-only`.
4. Compact cohort uncertainty now derives ocular max-probability, top-1/top-2 margin and entropy means directly from `class_probability`, while the full five-output path remains available for sparse QC/qualification.
5. Temporal deltas are limited to hard pupil/ocular fractions, pupil center and the three ocular uncertainty means.

NVIDIA-specific behavior preserved: `CUDAExecutionProvider` first, FP32, fixed batch 16, 640×400 input, CUDA lifecycle guard, producer/inference/CPU-summary overlap, compact labels+class_probability cohort request, full-output qualification path, two-layer scientific/provenance identity, and recovery safeguards.

## Verification

- Full runtime test suite: **61 passed**.
- Real sub-059 first 1,024-eye compact benchmark: CUDA provider, 62.51 eyes/s; fixed 14d33a7 labels-only baseline was 65.04 eyes/s. This is a 3.9% regression, not an improvement; no full subject was started.
- Real 16-eye compact-vs-full parity: labels exact; four soft fractions and three ocular means equal within `1e-6`; 16/16 eyes passed.
- Batch/output contract tests confirm only `labels`, `class_probability` are requested for cohort; five outputs remain available for full QC.
- No formal completion was modified or overwritten.

## Downstream divergence audit

The removed fields are not present in v6 `EYE_METRIC_FIELDS` and are not consumed by the v7 temporal contract or compact cohort QC. Full uncertainty percentile/boundary calculations remain in `summarize_uncertainty(inputs_validated=False)` for explicit validator/sparse QC. Any external consumer expecting v5 iris/PIR/OAR fields must be migrated explicitly; this port does not silently provide those removed columns.

## Status

**PARTIAL.** Semantic code and regression tests pass, CUDA compact parity passes, and NVIDIA backend constraints remain intact. The compact fixed-fragment benchmark is ~3.9% slower than the prior narrow baseline, so no isolated full-subject acceptance was launched and no formal cohort was started. Working-tree ONNX files remain untracked local model artifacts.

