# RITnet Full-Class Post-hoc Extension

## Purpose

The frozen AMD formal pipeline already runs the full four-class RITnet network, but the production `eyes.csv` keeps only pupil-derived geometry. This extension restores the information needed for scale normalization and structural QC **without re-running YOLO and without modifying any existing formal artifact**.

The extension reuses, for every source eye row:

```text
video + frame_idx + roi_x1/y1/x2/y2
        ↓
re-crop the exact native eye ROI from the original AVI
        ↓
same RITnet preprocessing (640×400, gamma, CLAHE)
        ↓
same frozen RITnet b16 FP32 DirectML model
        ↓
labels: 0 background / 1 sclera / 2 iris / 3 pupil
        ↓
320×160 full-class metrics + pupil/iris normalization + QC
```

The original `eyes.csv`, `frames.csv`, completion marker and production run are never overwritten.

## What is retained

The current ONNX exposes:

```text
labels_u8  [16,400,640]  # hard four-class labels
pupil_prob [16,400,640]  # class-3 softmax probability
```

Therefore the extension retains everything needed from the available output contract, while **not inventing** iris/sclera/background probability values that the current ONNX does not expose.

Per eye row the extension writes:

- hard-class pixel counts and fractions for background, sclera, iris and pupil;
- `iris_outer = iris OR pupil`, used to estimate the outer iris reference geometry;
- visible ocular area `sclera OR iris OR pupil`;
- pupil and iris-outer contour/ellipse geometry: center, axes, ordered short/long axes, angle, contour area, fitted ellipse area, equivalent diameter and geometric-mean diameter;
- pupil confidence from the existing class-3 probability output;
- scale-normalized metrics:
  - `pupil_to_iris_diameter_ratio`;
  - `pupil_to_iris_ellipse_area_ratio`;
  - `pupil_to_iris_contour_area_ratio`;
- pupil-to-iris center offset and normalized center offset;
- iris fill ratio and ROI-edge-touch flags for QC;
- visible-eye connected-component metrics;
- candidate ocular aperture geometry (bbox, robust vertical aperture, aperture ratios). These are **candidate openness/QC signals only**, not validated blink or PERCLOS labels;
- `normalization_valid`, a conservative structural gate requiring valid pupil/iris ellipses, no ROI-edge contact, and iris scale larger than pupil scale;
- pupil parity checks against the original formal `eyes.csv`, so the post-hoc re-run can verify that the saved ROI coordinates reproduce the original pupil result.

### Why `iris_outer` is not simply `iris_pixels`

RITnet class 2 contains the visible iris tissue but excludes the pupil hole. For a scale reference we need the **outer iris boundary**, so geometry is fit to `class 2 OR class 3`. The raw visible iris class area is still saved separately as `iris_pixels` / `iris_fraction`.

### Sclera and background

Sclera and background are retained as hard-class pixel counts/fractions and as part of visible-eye QC. They are not forced into ellipse geometry because their shape is strongly determined by eyelid exposure and the ROI boundary.

## Subject-numbered output contract

Every per-subject file name contains the subject ID. For `sub-031`, the source formal run directory receives:

```text
sub-031_ritnet_fullclass.csv
sub-031_ritnet_fullclass_summary.json
sub-031_ritnet_fullclass_manifest.json
sub-031_ritnet_fullclass_completion.json
```

The CSV also keeps the original `subject` column. The extension does not rely on folder names alone for identity.

The batch runner additionally writes one root-level administrative file:

```text
ritnet_fullclass_batch_summary.json
```

## Run locally

From the repository root:

```powershell
conda activate nir-amd
cd "D:\AAAWORK\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"
git pull --ff-only
cd runtime\nir-formal
```

Run tests first:

```powershell
python -m pytest tests -q
```

Preview which completed formal run will be selected for each subject:

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --device 0 `
  --dry-run
```

The batch runner reads only formal source runs whose `completion.json` is `status=complete`; it respects `config.yaml` exclusions such as `sub-9504`. If more than one complete run exists for a subject, a current YOLO-b8 run is preferred; otherwise the newest complete formal run is selected, and alternatives are printed in the dry-run output.

Recommended first real validation: one subject only.

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "sub-031" `
  --device 0
```

Review:

```text
sub-031_ritnet_fullclass_summary.json
```

Especially check:

```text
pupil_parity_mismatch_count
pupil_parity_ok_fraction
normalization_valid_fraction
roi_per_sec
```

If pupil parity is acceptable, run all completed subjects:

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --device 0
```

The extension has its own subject-numbered completion marker. A later rerun skips a subject whose full-class completion matches both the source `eyes.csv` SHA256 and the current RITnet model SHA256. Use `--force` only when intentionally recomputing the extension.

## Runtime expectation

This is a RITnet-only pass. It still has to decode the original video and run RITnet for every saved eye ROI, so it is not instantaneous. It does **not** run YOLO, phase detection, bbox inference or tracking. Based on the existing AMD benchmarks, plan roughly in the same order as the RITnet share of the original pipeline; measure `roi_per_sec` on the first complete subject before scheduling the remaining batch.

## Interpretation boundary

The recommended primary normalized pupil measure is the geometrically normalized pupil/iris ratio from structurally valid frames. Raw `pupil_equiv_diameter` remains useful as a source measure and for within-subject baseline normalization.

The extension does not classify blink events. Ocular aperture, sclera visibility and visible-eye fractions are retained so they can later support QC and cross-validation against the RGB/MediaPipe blink pipeline. Blink/PERCLOS thresholds must be validated separately.
