# AMD DirectML v0.2.0

Release target/tag name: `amd-v0.2.0`

## Production combination

- YOLO26n: 640×640, FP32, ONNX Runtime DirectML, fixed batch=8, every frame
- RITnet: 640×400, FP32, ONNX Runtime DirectML, fixed batch=16
- tracking: none
- pupil analysis geometry: 320×160
- FocusWave formal release: v3.1.3

## Why these batch sizes

YOLO batch sweep on the same 1800-frame segment of sub-031 showed b8 as the best tested end-to-end YOLO throughput. Larger batches continued to reduce raw inference time per frame but increased preprocessing/memory overhead enough to reduce total throughput.

RITnet batch sweep on 1680 identical real eye ROIs showed throughput increasing from b8 through b16, with b16 the fastest tested value. Across b8/b10/b12/b14/b16, pupil found status, center, equivalent diameter, ellipse axes, mask area, and pupil confidence were identical in the benchmark outputs.

## Full-pipeline benchmark

Validated major compute chain:

```text
video decode
→ YOLO b8
→ bbox restore / original-frame ROI crop
→ RITnet b16
→ pupil postprocess
```

On the tested AMD machine and the same 1800-frame sub-031 segment:

- processed frames: 1800
- elapsed: ~59.02 s
- processing throughput: ~30.50 FPS
- previous full formal sub-031 throughput: ~20.21 FPS
- observed throughput improvement: ~50.9%

These numbers describe the tested hardware/data segment and are not a cross-device performance guarantee.

## Runtime changes

- `directml_runtime.py` now supports arbitrary positive fixed YOLO batch shapes `[B,3,640,640]`.
- `run_formal_batched.py` is the production single-subject AMD formal runner.
- `run_formal_batch.py` now routes formal batch analysis through the batched runner.
- `config.yaml` package version is `0.2.0` and formal YOLO batch is frozen at 8.
- run directories now distinguish the two model batches explicitly, e.g.:

```text
sub-031_formal_v3.1.3_yolo-b8_ritnet-b16_fp32
```

- completion identity now includes YOLO batch/model SHA256 and RITnet batch/model SHA256, so old `..._yolo_b16_fp32` outputs cannot be silently reused as v0.2.0 completions.
- existing diagnostic/reference code and historical files are retained; no historical artifact is deleted.

## Required model assets

Production formal analysis requires:

```text
models/nir-eye-yolo26n-best-b8.onnx
models/ritnet-b16-fp32.onnx
models/ritnet-b16-fp32.onnx.data
```

The original `models/nir-eye-yolo26n-best.onnx` remains the b1 reference/diagnostic model.

Benchmark variants intended for repository retention after local model commit:

```text
models/nir-eye-yolo26n-best-b4.onnx
models/nir-eye-yolo26n-best-b8.onnx
models/nir-eye-yolo26n-best-b16.onnx
```

Temporary b1/b10/b12/b14 YOLO exports and b8/b10/b12/b14 RITnet exports are benchmark intermediates, not production dependencies.

## Important asset-state note

The b8 YOLO ONNX was generated and benchmarked on the AMD workstation. If it is not yet present in the GitHub model directory, the v0.2.0 runner intentionally raises `FileNotFoundError` rather than falling back to the original b1 model. Commit the validated local b8 model before treating a fresh clone as fully self-contained.
