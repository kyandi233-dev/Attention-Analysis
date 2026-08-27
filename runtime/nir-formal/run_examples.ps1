# Run from runtime/nir-formal in D:\CondaEnvs\nir-amd.

python .\run_pipeline.py check-env
python .\run_pipeline.py discover --formal-only

# -----------------------------------------------------------------------------
# Historical/base formal producer examples.
# Current base formal: YOLO b8 + RITnet b16 / FP32 / DirectML.
# Do not rerun this producer merely to obtain current full-class evidence when a
# complete formal run with eyes.csv already exists.
# -----------------------------------------------------------------------------

python .\run_formal_batch.py --dry-run

# Single base-formal subject if a genuine producer rerun is required.
python .\run_formal_batched.py `
  --video "F:\正式实验\sub-033_\nir\sub-033_nir.avi" `
  --device 0

# Legacy diagnostics remain available for troubleshooting/reproduction only.
python .\run_pipeline.py run `
  --video "F:\正式实验\sub-033_\nir\sub-033_nir.avi" `
  --duration-sec 60 --tracker none

# -----------------------------------------------------------------------------
# Current ONE canonical RITnet full-class workflow.
# It reuses saved frame_idx + YOLO ROI from a complete formal run, reruns only
# frozen RITnet, saves all 400x640 hard labels and strict provenance.
# The canonical wrappers enforce a clean Git worktree, source-video SHA256 and
# model identity. Do not add old --postprocess-workers/--validate-pupil flags.
# -----------------------------------------------------------------------------

# First inspect only sub-031.
python .\run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "sub-031" `
  --device 0 `
  --chunk-rows 128 `
  --dry-run

# Then run only sub-031 for full evidence-chain acceptance.
python .\run_ritnet_fullclass_batch.py `
  --output "D:\_AttentionData\Beijing-NIR\amd-directml" `
  --subjects "sub-031" `
  --device 0 `
  --chunk-rows 128

# Expand to the AMD cohort only after sub-031 DirectML, completion/hash, resume,
# storage/throughput and QC checks pass and chunk_rows is frozen.
