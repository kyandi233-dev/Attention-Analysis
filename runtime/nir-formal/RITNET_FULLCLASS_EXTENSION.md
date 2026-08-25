# RITnet Full-Class Post-hoc Extension

## Purpose

The frozen AMD formal pipeline already ran the full four-class RITnet network, but the production `eyes.csv` retained only pupil-derived geometry. This extension restores the missing structural information needed for pupil scale normalization and structural QC **without re-running YOLO and without modifying any existing formal artifact**.

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

No FP16 path and no reduced-resolution RITnet path are used.

## Primary pupil metric

The primary scale-normalized pupil measure is:

```text
fullclass_pupil_to_iris_diameter_ratio
```

It is **not** raw pupil pixels divided by raw iris pixels. Pupil and outer-iris contours are represented by fitted ellipses, and the ratio uses their geometric-mean diameters:

```text
PIR_D = sqrt(pupil_axis_a * pupil_axis_b)
        ----------------------------------
        sqrt(iris_axis_a * iris_axis_b)
```

For the iris reference, `iris_outer = class 2 OR class 3`, because class 2 is visible iris tissue with the pupil hole excluded. The outer iris boundary is therefore reconstructed before ellipse fitting.

The extension also retains ellipse-area and contour-area ratios as secondary/robustness measures, but `fullclass_pupil_to_iris_diameter_ratio` is the recommended primary variable for later pupil analyses after `fullclass_normalization_valid == True` gating.

## Fast production policy

The production extension is optimized without changing the RITnet input, precision, weights or batch size:

1. production requests only `labels_u8`; the large `pupil_prob` output is not transferred again;
2. frozen source pupil geometry/confidence from `eyes.csv` is reused instead of recomputing it for every eye;
3. `--validate-pupil` is a validation-only path that re-requests pupil probability and recomputes pupil geometry for parity checking;
4. CPU video decode / ROI crop / preprocessing for the next batch overlaps DirectML inference for the current batch;
5. RITnet remains fixed at 640×400, FP32, batch=16;
6. independent full-class postprocessing runs in a small CPU worker pool (default 4).

The summary records decode, ROI crop, preprocessing, GPU, postprocessing, CSV and QC-image write timing separately.

## Retained variables

Per eye row the extension retains:

- background / sclera / iris / pupil hard-class pixel counts and fractions;
- `iris_outer = iris OR pupil`;
- visible ocular area `sclera OR iris OR pupil`;
- source pupil geometry and newly fitted iris-outer geometry;
- pupil/iris diameter, ellipse-area and contour-area ratios;
- pupil-to-iris center offset and normalized center offset;
- iris fill ratio and edge-touch flags;
- visible-eye connected-component metrics;
- candidate ocular aperture geometry;
- `normalization_valid`, requiring structurally valid pupil/iris geometry and conservative spatial checks;
- original timing and identity columns copied unchanged from source `eyes.csv`.

Sclera/background probabilities are not fabricated because the current ONNX does not expose them. Ocular-aperture variables are candidate openness/QC variables only, not validated blink or PERCLOS labels.

## Fixed sparse QC image sampling

The numerical CSV remains complete for every eye row. Segmentation images are saved only as sparse audit material for anomaly review and paper figures.

The policy is frozen in code:

```text
periodic stride: 3000 frames  (~100 s at 30 FPS)
plus: first / middle / last available frame of every phase segment
plus bounded anomaly examples:
    roi_clipped
    ritnet_missing
    normalization_invalid
    ocular_fragmented
maximum per anomaly reason: 2 eye samples / phase / subject
```

Each selected eye produces two PNGs:

```text
*_labels.png
    pure four-class color map
    background black / sclera blue / iris green / pupil red

*_overlay.png
    original grayscale eye ROI with the non-background segmentation
    overlaid semi-transparently
```

All per-subject output names include the subject ID and the extension version, so historical validation outputs are not overwritten. Example:

```text
sub-031_ritnet_fullclass_v1-2-fast-qc.csv
sub-031_ritnet_fullclass_v1-2-fast-qc_summary.json
sub-031_ritnet_fullclass_v1-2-fast-qc_manifest.json
sub-031_ritnet_fullclass_v1-2-fast-qc_completion.json
sub-031_ritnet_fullclass_v1-2-fast-qc_qc_index.csv
sub-031_ritnet_fullclass_v1-2-fast-qc_qc/
    sub-031_block1_s01_f00012345_frame_left_labels.png
    sub-031_block1_s01_f00012345_frame_left_overlay.png
```

The QC index records phase, frame, Unix time, eye, sampling reason, source QC flags and the two image paths. No historical output is deleted.

## Time mapping and behavior alignment

The extension copies these source timing fields unchanged:

```text
phase
phase_segment
frame_idx
video_time_ms
unix_ms
phase_time_ms
```

The formal NIR pipeline already maps NIR frames to FocusWave phase windows using the NIR timestamp CSV and behavior-side absolute Unix timestamps. RITnet therefore does not need to be rerun for SART alignment.

Trial-level alignment remains a separate downstream operation:

```text
full-class row unix_ms
        ↕
behavior trial absolute timestamp
```

Use absolute timestamps rather than frame counts. A later subject-numbered alignment table can assign NIR samples to SART trials/windows without rewriting the full-class CSV.

## Local acceptance and production run

After pulling the latest `amd-DirectML`:

```powershell
cd "D:\AAAWORK\07-竞赛\厚璨杯\021-analysisplan\Attention-Analysis-amd-DirectML"
git pull --ff-only
cd runtime\nir-formal

python -m pytest tests -q
```

Preview all source-run selections:

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --device 0 `
  --postprocess-workers 4 `
  --dry-run
```

Before the multi-hour full batch, run one production-mode subject to validate DirectML execution, fast-path timing and QC PNG output on the AMD workstation:

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "sub-031" `
  --device 0 `
  --postprocess-workers 4
```

Check that the subject summary reports `status=complete` through its completion marker, all expected rows were processed, QC images/index were written, and the timing fields are present.

Then run all completed subjects:

```powershell
python run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --device 0 `
  --postprocess-workers 4
```

The batch runner skips only subject extensions whose completion identity matches the source `eyes.csv`, RITnet model, extension version and frozen QC policy.
