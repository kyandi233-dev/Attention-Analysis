# AMD DirectML YOLO batch benchmark

> Candidate-only benchmark. The stable `amd-DirectML` formal runtime remains batch-1 YOLO + batch-16 RITnet until this benchmark passes both speed and parity gates.

## Goal

Compare YOLO26n DirectML inference using the same 640×640 FP32 detector at fixed batch sizes 1, 4 and 8. This does **not** reduce frames or image resolution. RITnet is not run in this benchmark; the purpose is to isolate YOLO throughput before changing the formal loop.

## 1. Prepare the candidate branch

```powershell
git fetch origin --prune
git switch amd-DirectML-yolo-batch
git pull --ff-only
cd runtime\nir-formal
```

The stable batch-1 ONNX remains:

```text
models/nir-eye-yolo26n-best.onnx
```

The PyTorch source weight is retained on the NVIDIA branch. Bring a working-tree copy into this candidate branch without modifying the stable AMD branch:

```powershell
git restore --source=origin/nvidia-cuda -- models/nir-eye-yolo26n-best.pt
```

Do not commit generated candidate model files until parity and speed are reviewed.

## 2. Export fixed batch 4 and 8 ONNX variants

```powershell
python export_yolo_batch_variants.py `
  --pt models/nir-eye-yolo26n-best.pt `
  --batches 4,8
```

Expected files:

```text
models/nir-eye-yolo26n-best-b4.onnx
models/nir-eye-yolo26n-best-b8.onnx
```

The helper exports from temporary `.pt` copies so it does not overwrite the stable batch-1 ONNX.

## 3. Benchmark the same 1800 frames

For `sub-031` block1, the previously frozen phase window starts at frame `10186`. Run:

```powershell
python benchmark_yolo_batch.py `
  --video "E:\正式实验\sub-031_\nir\sub-031_nir.avi" `
  --start-frame 10186 `
  --frames 1800 `
  --models `
    models/nir-eye-yolo26n-best.onnx `
    models/nir-eye-yolo26n-best-b4.onnx `
    models/nir-eye-yolo26n-best-b8.onnx `
  --output "D:\_AttentionData\Beijing-NIR\yolo-batch-benchmark-sub031.json"
```

Pass the batch-1 model first; it is used as the parity reference.

## 4. Review gates

Speed gate: compare `fps`, `mean_inference_ms_per_frame` and `mean_total_ms_per_frame`.

Parity gate: batch 4/8 should retain the same selected-eye count on essentially all frames and show only negligible coordinate/confidence differences versus batch 1. Review these fields:

```text
frame_selected_count_agreement
coord_mae_px
coord_max_abs_px
confidence_mae
confidence_max_abs
```

A speed gain without acceptable parity is not sufficient to change the formal runtime.

## 5. Resume stable formal analysis

After the benchmark, return to the stable branch:

```powershell
cd ..\..
git switch amd-DirectML
cd runtime\nir-formal
python run_formal_batch.py --output "D:\_AttentionData\Beijing-NIR\amd-directml"
```

`skip_completed: true` will keep completed subjects and rerun only any subject that was interrupted before a valid `completion.json: complete` was written.
