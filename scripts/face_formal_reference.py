"""Native Py-Feat 2.1.1 CPU reference for the NVIDIA Face parity gate.

ported-from: 51d17c9a6b7db7a1114380910bb111db38293512
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import pandas as pd

from face_formal_dryrun_cuda import _attach_manifest, _find_frame_manifest, _safe_version


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run native Py-Feat Detectorv2 CPU reference on formal RGB Face sample"
    )
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None, help="Diagnostic prefix only; omit for full sample")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_manifest = _find_frame_manifest(sample_dir)
    frames = pd.read_csv(frame_manifest)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        frames = frames.head(args.max_frames).copy()
    if frames.empty:
        raise ValueError("Empty Face frame manifest")

    image_paths = [str(Path(p).resolve()) for p in frames["image_path"].astype(str)]
    missing = [p for p in image_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    version = _safe_version("py-feat")
    if version != "2.1.1":
        raise RuntimeError(f"Expected py-feat 2.1.1, found {version!r}")

    import torch
    from feat import Detectorv2

    init_start = time.perf_counter()
    detector = Detectorv2(device="cpu", identity_model=None)
    init_sec = time.perf_counter() - init_start
    infer_start = time.perf_counter()
    fex = detector.detect(
        image_paths,
        data_type="image",
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=False,
        face_detection_threshold=0.5,
        progress_bar=True,
    )
    infer_sec = time.perf_counter() - infer_start
    raw = _attach_manifest(pd.DataFrame(fex).copy(), frames)

    raw_path = output_dir / "pyfeat_cpu_raw.parquet"
    columns_path = output_dir / "pyfeat_cpu_columns.json"
    manifest_path = output_dir / "pyfeat_cpu_reference_manifest.json"
    raw.to_parquet(raw_path, index=False, engine="pyarrow", compression="zstd")
    columns_path.write_text(json.dumps(list(raw.columns), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "schema_version": "rgb-face-formal-reference-pyfeat-cpu-v0.1",
        "candidate": "pyfeat211_detectorv2_native_cpu_reference",
        "scientific_core": "Py-Feat 2.1.1 Detectorv2; identity_model=None",
        "frame_manifest": str(frame_manifest),
        "expected_input_frames": int(len(frames)),
        "output_rows": int(len(raw)),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "device": "cpu",
        "timing_sec": {"model_initialization": init_sec, "detector_detect": infer_sec},
        "input_frames_per_sec_detect": float(len(frames) / infer_sec) if infer_sec > 0 else None,
        "runtime": {
            "python": platform.python_version(),
            "py_feat": _safe_version("py-feat"),
            "torch": _safe_version("torch"),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "raw_output": str(raw_path),
        "columns_output": str(columns_path),
        "identity_model": None,
        "retention": "all native non-identity Detectorv2 columns retained; frame manifest provenance merged",
        "ported_from": "51d17c9a6b7db7a1114380910bb111db38293512",
    }
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
