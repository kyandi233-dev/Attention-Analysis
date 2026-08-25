# RITnet Full-Class Post-hoc Extension

## Purpose

The frozen AMD formal pipeline already runs the full four-class RITnet network, but the production `eyes.csv` keeps only pupil-derived geometry. This extension restores the missing structural information needed for scale normalization and QC **without re-running YOLO and without modifying any existing formal artifact**.

The scientific method remains frozen:

```text
source video + saved frame_idx + saved ROI coordinates
        ↓
re-crop the exact native eye ROI
        ↓
same preprocessing: 640×400 + gamma + CLAHE
        ↓
same frozen RITnet FP32 DirectML model, fixed batch=16
        ↓
labels: 0 background / 1 sclera / 2 iris / 3 pupil
        ↓
320×160 full-class metrics + pupil/iris normalization + structural QC
```

No FP16 path and no reduced-resolution RITnet path are used by this extension.

## Fast v1.1 execution policy

The production extension is optimized without changing the RITnet input, precision, weights or batch size:

1. **labels-only ONNX output**: production mode requests only `labels_u8`; the large `pupil_prob` output is not transferred again.
2. **reuse source pupil geometry/confidence**: the already-frozen pupil ellipse, area, center, diameter and confidence from the source `eyes.csv` are reused instead of recomputing them for every eye.
3. **explicit validation mode**: `--validate-pupil` requests `pupil_prob` and recomputes pupil geometry so parity against the original formal result can be verified on a test subject.
4. **CPU/GPU prefetch pipeline**: video decode, ROI crop and RITnet preprocessing for batch N+1 run in a dedicated producer thread while DirectML executes batch N.
5. **RITnet remains fixed b16**: no batch-method change is introduced.
6. **parallel full-class postprocessing**: independent label maps are processed in a small CPU worker pool (`--postprocess-workers`, default 4).

The output summary records wall time plus decode/crop/preprocess/GPU/postprocess/CSV timing so the real bottleneck can be measured on the AMD workstation.

## What is retained

The existing ONNX exposes:

```text
labels_u8  [16,400,640]  # hard four-class labels
pupil_prob [16,400,640]  # class-3 softmax probability
```

Production fast mode needs only `labels_u8` because the source formal `eyes.csv` already contains the frozen pupil geometry and `pupil_confidence`. Validation mode may additionally request `pupil_prob`.

Per eye row the extension writes:

- hard-class pixel counts and fractions for background, sclera, iris and pupil;
- `iris_outer = iris OR pupil`, used to estimate the outer iris reference geometry;
- visible ocular area `sclera OR iris OR pupil`;
- source pupil geometry (reused in production mode) and newly fitted iris-outer geometry;
- scale-normalized metrics:
  - `pupil_to_iris_diameter_ratio`;
  - `pupil_to_iris_ellipse_area_ratio`;
  - `pupil_to_iris_contour_area_ratio`;
- pupil-to-iris center offset and normalized center offset;
- iris fill ratio and ROI-edge-touch flags for QC;
- visible-eye connected-component metrics;
- candidate ocular aperture geometry (bbox, robust vertical aperture, aperture ratios);
- `normalization_valid`, requiring structurally valid pupil/iris geometry, no ROI-edge contact, pupil center inside the iris outer contour, and iris scale larger than pupil scale.

`iris_outer` is fit to class 2 OR class 3 because the class-2 iris tissue contains a pupil hole; the outer iris boundary is the scale reference. Raw class-2 iris pixels are still saved separately.

Sclera and background are retained as hard-class area/fraction and QC information. No iris/sclera/background probabilities are fabricated because the current ONNX does not expose them.

## Time mapping and behavior alignment

The extension copies all original timing columns from source `eyes.csv` unchanged, including:

```text
phase
phase_segment
frame_idx
video_time_ms
unix_ms
phase_time_ms
```

The formal NIR pipeline already maps NIR frames to FocusWave phase windows using the NIR timestamp CSV and behavior-side absolute Unix timestamps. Therefore **RITnet does not need to be rerun to align with SART/behavior trials**.

Trial-level alignment should remain a separate downstream step:

```text
NIR/full-class row unix_ms
        ↕
behavior trial absolute_onset_time / absolute timestamps
```

Use absolute timestamps rather than frame counts. If behavior trial rows contain `absolute_onset_time`, assign NIR samples to trial intervals from the trial onset to the next onset (or to the experimentally defined trial end). This can generate a separate subject-numbered alignment/aggregation table later; it should not rewrite the full-class RITnet CSV.

The achievable alignment resolution is bounded by the NIR frame timestamps/frame interval and acquisition timestamp jitter. It is timestamp-level synchronization, not an assumption that two files have identical row counts.

## Subject-numbered output contract

Every per-subject artifact carries the subject ID. For `sub-031`:

```text
sub-031_ritnet_fullclass.csv
sub-031_ritnet_fullclass_summary.json
sub-031_ritnet_fullclass_manifest.json
sub-031_ritnet_fullclass_completion.json
```

The CSV itself also retains `subject = sub-031`. Subject identity therefore exists at folder, filename and row level.

The cross-subject administrative batch summary remains:

```text
ritnet_fullclass_batch_summary.json
```

## Local validation workflow

From `runtime\nir-formal` after pulling the latest `amd-DirectML`:

```powershell
python -m pytest tests -q
```

Preview source-run selection:

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --device 0 `
  --dry-run
```

First run `sub-031` in **validation mode**:

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "sub-031" `
  --device 0 `
  --postprocess-workers 4 `
  --validate-pupil
```

Review the subject-numbered summary, especially:

```text
elapsed_sec
roi_per_sec
pupil_parity_mismatch_count
pupil_parity_ok_fraction
normalization_valid_fraction
timing_cpu_work_ms
timing_gpu_ms
```

After parity is accepted, run production fast mode without `--validate-pupil`:

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --device 0 `
  --postprocess-workers 4
```

Because validation mode and production mode are distinct completion identities, a validation output is not silently reused as the production fast completion.

## Interpretation boundary

The recommended primary scale-normalized pupil measure is the geometrically normalized pupil/iris ratio from `normalization_valid` frames. The original raw pupil metrics remain useful for within-subject baseline normalization and audit.

Ocular aperture, sclera visibility and visible-eye fractions are candidate openness/QC signals only. The extension does not classify blink events or PERCLOS; those require separate validation and are expected to be cross-checked against the RGB/MediaPipe pipeline.
