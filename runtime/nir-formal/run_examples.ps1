# Run from runtime/nir-formal in D:\CondaEnvs\nir-amd.

python .\run_pipeline.py check-env
python .\run_pipeline.py discover --formal-only

# Formal default: sub-031+, FocusWave v3.1.3 phases,
# per-frame YOLO, RITnet batch=16, FP32, overlay every 3000 frames.
python .\run_pipeline.py formal `
  --video "F:\正式实验\sub-033_\nir\sub-033_nir.avi"

# Lightweight 20-second end-to-end smoke test inside the real block1 window.
python .\run_pipeline.py formal `
  --video "E:\正式实验\sub-031_\nir\sub-031_nir.avi" `
  --phases block1 --max-frames 600 `
  --output "D:\AttentionModels\pipeline-smoke-output"

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
