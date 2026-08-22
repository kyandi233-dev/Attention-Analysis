# Run from runtime/nir-yolo-tracking-ritnet-v1 in the validated eye-ai environment.

python .\run_pipeline.py check-env
python .\run_pipeline.py discover --formal-only

# Formal default: sub-031+, FocusWave v3.1.3 phases,
# per-frame YOLO, RITnet batch=16, FP32, overlay every 3000 frames.
python .\run_pipeline.py formal `
  --video "F:\正式实验\sub-033_\nir\sub-033_nir.avi"

# Same data, CUDA mixed-precision comparison.
python .\run_pipeline.py formal `
  --video "F:\正式实验\sub-033_\nir\sub-033_nir.avi" `
  --ritnet-precision fp16

# Batch-size comparison while keeping FP32 fixed.
python .\run_pipeline.py formal `
  --video "F:\正式实验\sub-033_\nir\sub-033_nir.avi" `
  --ritnet-batch-size 32 --ritnet-precision fp32

# Analyze only selected phases if needed.
python .\run_pipeline.py formal `
  --video "F:\正式实验\sub-033_\nir\sub-033_nir.avi" `
  --phases baseline,instructions,block1

# Legacy diagnostic reproduction remains available.
python .\run_pipeline.py run `
  --video "F:\正式实验\sub-033_\nir\sub-033_nir.avi" `
  --duration-sec 60 --tracker none

python .\run_pipeline.py run `
  --video "F:\正式实验\sub-033_\nir\sub-033_nir.avi" `
  --duration-sec 60 --tracker kcf --redetect-interval 10
