# NVIDIA NIR full-class stage-level profile (sub-059, 1024 eyes)

## Scope and evidence

This is an isolated profile on the first 1,024 rows / 529 frames of the same `sub-059` AVI used by the old-vs-current benchmark, RTX 5070, CUDAExecutionProvider, FP32 and batch 16. No formal completion, model, config or result was modified. The legacy path is the canonical `324413c` `runtime/nir-formal/ritnet_onnx_runtime.py` path; the current path is the full five-output runtime.

Important scope distinction: the earlier 15.743 s current number measured labels-only inference plus hard metrics. The profile below measures the complete current five-output path, including probability, max-probability, margin and entropy validation/reduction. Therefore 60.477 s is not a regression of the 15.743 s narrow benchmark; it is a wider output contract.

## Stage profile

The video-access timings use the existing per-frame seek/read workflow. They are included identically for both implementations; the isolated seek pattern is not a safe linear predictor for a complete subject on J:. Runtime-only wall is shown separately.

| Stage | Legacy 324413c (s) | Current (s) | Added (s) | Current pipeline share |
|---|---:|---:|---:|---:|
| Video decode / frame access | 26.701 | 26.701 | 0.000 | 30.5% |
| Source row + ROI geometry resolution | 0.052 | 0.052 | 0.000 | 0.06% |
| Crop + canonical ROI + resize/preprocess | 1.195* | 0.910 | -0.285 | 1.0% |
| H2D / ONNX `session.run` | 9.035 | 40.599 | +31.564 | 46.4% |
| Probability / argmax / uncertainty | N/A† | 16.036 | +16.036 | 18.3% |
| Per-eye metrics / geometry / components | 0.341 | 3.069 | +2.728 | 3.5% |
| Temporal/source-valid mask | 0.080 | 0.080 | 0.000 | 0.09% |
| Python orchestration / residual | 0.001 | 0.001 | 0.000 | <0.01% |
| **Runtime-only wall** | **10.433** | **60.477** | **+50.044** | — |
| **Pipeline wall incl. shared video access** | **37.404** | **87.448** | **+50.044** | **100%** |

\* Includes the shared crop time (0.138 s) plus each implementation's preprocessing. Legacy preprocessing is its own gamma/CLAHE + 640×400 tensor path; current preprocessing is canonical 640×400. They are not byte-identical implementations. † Legacy has no uncertainty output; probability handling is contained in its postprocess, so an old uncertainty stage is not applicable.

Current profile telemetry: CUDA provider confirmed; GPU utilization average 67.5%, P95 99%; VRAM peak 12.24 GB of 12.82 GB (0.58 GB / 4.5% headroom); process CPU peak 595%; RAM peak 2.42 GB. No OOM, provider error, retry or numerical-integrity failure occurred. The measured VRAM margin is below the desired 10–15% production safety margin.

## Findings

- The current complete contract is dominated by ONNX execution/output transfer (46.4%) and full-class probability/uncertainty processing (18.3%). The earlier labels-only 15.74 s benchmark intentionally omitted most of this work.
- Both implementations contain per-eye Python loops. Current hard metrics call native-resolution connected-components/contour/ellipse routines; current uncertainty is also reduced per eye. The code creates contiguous arrays for model input and output slices (`np.stack`/`np.ascontiguousarray`). No duplicate-mask computation was proven from this profile alone.
- The current benchmark is synchronous: prepare → `session.run` → CPU reductions → next batch. No producer/consumer overlap is implemented. The shared video seek/read stage is a large isolated cost, but its scaling is access-pattern dependent.
- No safe optimization was implemented. A change would require a fresh baseline/optimized run and exact output comparison; the observed profile does not by itself demonstrate a >10% low-risk win.

## Optimization candidates (no science changes)

1. **Pinned-buffer / copy-allocation audit (C)** — potentially 5–10%; low implementation risk, low scientific risk if ordering/dtypes stay identical. Not implemented because expected gain is below the requested >10% threshold.
2. **Bounded preprocess producer + inference consumer (A/D)** — potentially >10% only if CPU preparation overlaps GPU execution; medium engineering and operational risk (RAM/VRAM pressure, ordering and failure recovery), low scientific risk with deterministic batching. Requires a separate isolated benchmark.
3. **NumPy/OpenCV vectorization of metric preparation (B)** — potentially useful but uncertain; medium/high implementation risk and medium scientific risk because connected-component/contour/ellipse semantics must remain byte-equivalent. Not attempted.

## Throughput implications

Complete current five-output runtime throughput is 16.93 eyes/s on this isolated 1024-eye run. At that rate, sub-059's 77,725 eyes are approximately 76.4 minutes of runtime-only work; the 68 not-yet-completed source directories contain 5,316,133 eyes (2,746,199 frames), approximately 87.3 hours runtime-only. These are conservative profile-based estimates, not complete-subject wall measurements; shared video seek/finalization/QC/I/O overhead is not safely scalable from this 529-frame isolated access pattern. The earlier 21.9-minute/25-hour planning figures remain valid only for the narrower labels-only benchmark and must not be reused for the complete five-output contract.

## Decision

**Status: PARTIAL.** Stage attribution is complete and no scientific output difference, CUDA failure or OOM was observed. A complete-subject wall-time run and any optimization implementation remain outstanding. Do not start the 71-subject queue, change formal `config.yaml`, or enable subject parallelism based on this profile.
