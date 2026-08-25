from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Py-Feat Detectorv2 on shared RGB Face benchmark images")
    parser.add_argument("--benchmark-dir", required=True, help=".../_test/face-benchmark/sub-XXX")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    root = Path(args.benchmark_dir)
    manifests = sorted(root.glob("*_face-benchmark_frames.csv"))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected exactly one frame manifest in {root}, found {len(manifests)}")
    manifest_path = manifests[0]
    sample = pd.read_csv(manifest_path)
    if sample.empty or "image_path" not in sample:
        raise ValueError(f"Invalid benchmark frame manifest: {manifest_path}")
    image_paths = [str(Path(p)) for p in sample["image_path"].tolist()]
    missing = [p for p in image_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Benchmark images missing; first missing: {missing[0]}")

    from feat import Detectorv2

    init_start = time.perf_counter()
    detector = Detectorv2(device=args.device)
    init_sec = time.perf_counter() - init_start

    infer_start = time.perf_counter()
    fex = detector.detect(
        image_paths,
        data_type="image",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        progress_bar=True,
    )
    infer_sec = time.perf_counter() - infer_start
    raw = pd.DataFrame(fex).copy()

    output_parquet = root / "pyfeat_raw.parquet"
    output_columns = root / "pyfeat_columns.json"
    output_manifest = root / "pyfeat_benchmark_manifest.json"
    raw.to_parquet(output_parquet, index=False, engine="pyarrow", compression="zstd")
    output_columns.write_text(
        json.dumps(list(raw.columns), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    facebox_columns = [str(v) for v in getattr(fex, "facebox_columns", [])]
    detected_rows = None
    if facebox_columns and set(facebox_columns).issubset(raw.columns):
        faceboxes = raw[facebox_columns].apply(pd.to_numeric, errors="coerce")
        detected_rows = int(faceboxes.notna().any(axis=1).sum())

    try:
        pyfeat_version = importlib.metadata.version("py-feat")
    except Exception:
        pyfeat_version = None
    try:
        torch_version = importlib.metadata.version("torch")
    except Exception:
        torch_version = None

    summary = {
        "schema_version": "rgb-face-benchmark-pyfeat-v0.1",
        "candidate": "pyfeat_detectorv2",
        "benchmark_frame_manifest": str(manifest_path),
        "expected_input_images": int(len(sample)),
        "result_rows": int(len(raw)),
        "detected_face_rows_if_available": detected_rows,
        "output_columns": int(len(raw.columns)),
        "facebox_columns": facebox_columns,
        "device": args.device,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "model_initialization_sec": init_sec,
        "inference_sec": infer_sec,
        "input_images_per_sec": len(sample) / infer_sec if infer_sec > 0 else None,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "py_feat": pyfeat_version,
            "torch": torch_version,
        },
        "raw_output": str(output_parquet),
        "columns_output": str(output_columns),
        "retention": "all native Detectorv2 columns retained; no AU/expression/gaze/head-pose prefiltering",
    }
    output_manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
