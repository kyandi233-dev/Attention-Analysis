# NVIDIA NIR full-class production-path wall time

## Run and isolation

- Subject: `sub-170` (68-subject pending-set median-near: 78,367 eyes, 39,315 frames)
- Source: validated formal `v3.1.3` source; same AVI and source identities as production
- Runtime: batch 16, FP32, canonical 640×400 ROI, current five-output uncertainty ONNX, CUDAExecutionProvider
- Path: sequential video/source eye resolution → ROI/preprocess → RITnet → full-class metrics/uncertainty → temporal/source-valid logic → compressed tables → QC → manifest/summary → completion → final validator
- YOLO was not rerun. Formal source and existing completions were not modified. Output root: `D:/Project/厚粲杯/11_数据/nir_fullclass_production_profile_20260828`.

## Measured wall time

Process start was 2026-08-28 05:03:58 and valid completion was published at 05:52:07: **48 min 09 s (2,889 s)**.

| Item | Value |
|---|---:|
| Total frames | 39,315 |
| Total eyes | 78,367 |
| Effective eyes/s | 27.126 |
| Effective frames/s | 13.609 |
| Output bytes | 180,453,066 |
| Eye metric rows | 78,367 |
| Frame coverage rows | 39,315 |
| QC images / pixel evidence eyes | 200 / 16 |
| Completion | `complete` |
| Final validator | valid |

Resource observations during the run: CUDA provider remained active; sampled GPU utilization ranged approximately 5–98%; sampled VRAM remained approximately 11.4–11.7 GiB of 12.23 GiB and did not show monotonic growth; process RAM fell from about 2.0 GB during inference to about 0.6 GB during finalization. No OOM, CUDA/provider error, retry, or failed validation occurred. A persistent telemetry log was not enabled, so an exact run-wide GPU average/P95 and exact CPU/RAM peaks cannot be reconstructed honestly from this run; the ranges above are observations, not averages.

## ONNX output audit

The current graph returns five CPU NumPy arrays per batch of 16, totaling 118,784,000 bytes (113.28 MiB) before any downstream copies:

| Output | Shape | dtype | Bytes/batch | Final role |
|---|---|---|---:|---|
| `labels` | 16×400×640 | uint8 | 4,096,000 | Primary scientific segmentation and hard metrics |
| `class_probability` | 16×4×400×640 | float32 | 65,536,000 | QC/uncertainty soft-class fractions |
| `max_probability` | 16×400×640 | float32 | 16,384,000 | QC/uncertainty |
| `top1_top2_margin` | 16×400×640 | float32 | 16,384,000 | QC/uncertainty |
| `entropy` | 16×400×640 | float32 | 16,384,000 | QC/uncertainty |

All five outputs are materialized by `session.run` on CPU. The four floating-point maps are transient and reduced to per-eye QC facts; they are not persisted as pixel maps. The 40.6 s session stage observed in the prior micro-profile therefore includes execution plus transfer/materialization of roughly 113 MiB per batch, but this run did not independently separate GPU compute from output-transfer time. Avoiding copies is a candidate only; no change was made.

## Queue estimate from actual source counts

The 71 validated sources contain 5,547,232 eyes. Three existing strict skips (`sub-056`, `sub-057`, `sub-058`) account for 231,099 eyes, leaving **5,316,133 eyes across 68 subjects**. Applying the measured 27.126 eyes/s and adding subject switching, finalization, QC and normal variation:

| Estimate | Remaining 68 |
|---|---:|
| Optimistic (raw throughput) | 54.4 h |
| Typical (+10%) | 59.9 h |
| Conservative (+25%) | 68.0 h |

These are production-path estimates, not a parallel-run promise. Subject parallelism remains disabled.

## Decision

**Status: PARTIAL.** The complete subject reached valid completion and long-run VRAM stability was observed, but exact run-wide GPU average/P95 and CPU/RAM peaks were not persistently logged. No scientific output change or runtime failure was observed. Do not start the 71-subject queue until telemetry capture is considered sufficient; no formal config or scientific code was changed.

