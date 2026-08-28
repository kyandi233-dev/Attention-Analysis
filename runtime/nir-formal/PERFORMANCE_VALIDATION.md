# NVIDIA RITnet v8 performance validation

This branch benchmarks execution transport only. Production science remains frozen:

- fixed B16 FP32 RITnet, 640x400;
- CUDAExecutionProvider primary, CPU fallback disabled;
- TF32 disabled;
- production cohort outputs remain `labels + class_probability`;
- production pupil geometry remains primary-iris-topology -> OpenCV fitEllipse;
- no YOLO rerun and no production schema/QC/completion changes.

## Benchmark

Activate the existing NVIDIA environment and enter `runtime/nir-formal`.

```powershell
python .\benchmark_ritnet_cuda_transport.py `
  --device 0 `
  --batches 128 `
  --warmup 8 `
  --pool-batches 2 `
  --output-json "D:\ritnet_cuda_transport_benchmark.json"
```

The default timed workload is 128 x B16 = 2048 eyes per mode.

Compared modes:

1. `baseline`: current ordinary `InferenceSession.run` path.
2. `iobinding`: fixed CUDA OrtValue input/output buffers with I/O Binding.
3. `cudagraph`: the same fixed buffers plus ORT CUDA Graph capture/replay.

Timing deliberately includes host-to-device input update and GPU-to-host retrieval of both production cohort outputs, because the current downstream Topology geometry and soft/uncertainty summaries consume CPU arrays.

## Acceptance gate

A candidate mode is not eligible for production unless:

- hard `labels` are bit-for-bit identical to the baseline;
- `class_probability` max absolute difference is <= `1e-6` by default;
- the mode completes without CUDA/CPU fallback;
- measured steady-state throughput materially improves over baseline on the same GPU/environment.

This benchmark is only the first gate. A winning transport path must still pass a real-subject prefix parity run before production integration.
