from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "rgb-face-formal-dryrun-pyfeat-cuda-v0.1"
REQUIRED_PYFEAT_VERSION = "2.1.1"
FACEBOX_COLUMNS = ["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight", "FaceScore"]


def _find_frame_manifest(root: Path) -> Path:
    candidates = sorted(root.glob("*_face-dryrun_frames.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one *_face-dryrun_frames.csv in {root}, found {len(candidates)}")
    return candidates[0]


def _norm_path(value: str | Path) -> str:
    try:
        return str(Path(value).resolve()).replace("/", "\\").lower()
    except Exception:
        return str(value).replace("/", "\\").lower()


def _safe_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except Exception:
        return None


def _attach_manifest(native: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    if "input" not in native.columns:
        raise ValueError("Py-Feat native output does not contain the expected 'input' column")
    if "image_path" not in frames.columns or "benchmark_index" not in frames.columns:
        raise ValueError("Dry-run frame manifest must contain image_path and benchmark_index")

    manifest = frames.copy()
    manifest["_path_key"] = manifest["image_path"].astype(str).map(_norm_path)
    if manifest["_path_key"].duplicated().any():
        dup = manifest.loc[manifest["_path_key"].duplicated(), "image_path"].iloc[0]
        raise ValueError(f"Duplicate image path in frame manifest: {dup}")

    out = native.copy()
    out["_path_key"] = out["input"].astype(str).map(_norm_path)
    missing = sorted(set(out["_path_key"]) - set(manifest["_path_key"]))
    if missing:
        # Py-Feat may preserve a relative path while the manifest is absolute. Fall back to
        # filename only when filenames are unique on both sides.
        manifest["_name_key"] = manifest["image_path"].astype(str).map(lambda p: Path(p).name.lower())
        out["_name_key"] = out["input"].astype(str).map(lambda p: Path(p).name.lower())
        if manifest["_name_key"].duplicated().any():
            raise ValueError("Could not map Py-Feat input paths to dry-run manifest and filenames are not unique")
        lookup = manifest.drop(columns=["_path_key"]).copy()
        merge_key = "_name_key"
    else:
        lookup = manifest.drop(columns=["_name_key"], errors="ignore").copy()
        merge_key = "_path_key"

    keep = [c for c in lookup.columns if c == merge_key or c not in out.columns]
    out = out.merge(lookup[keep], on=merge_key, how="left", validate="many_to_one")
    if out["benchmark_index"].isna().any():
        raise ValueError("At least one Py-Feat output row could not be mapped to benchmark_index")
    out["benchmark_index"] = pd.to_numeric(out["benchmark_index"], errors="raise").astype(int)

    # Detectorv2 returns detections in per-input score/rank order. Preserve all rows, including
    # multi-face rows and no-face placeholders, and add the canonical row key used downstream.
    out["face_rank"] = out.groupby("benchmark_index", sort=False).cumcount().astype(int)
    numeric_boxes = out[["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"]].apply(
        pd.to_numeric, errors="coerce"
    )
    out["detected"] = (
        numeric_boxes.notna().all(axis=1)
        & numeric_boxes["FaceRectWidth"].gt(0)
        & numeric_boxes["FaceRectHeight"].gt(0)
    )

    x = pd.to_numeric(out["FaceRectX"], errors="coerce")
    y = pd.to_numeric(out["FaceRectY"], errors="coerce")
    w = pd.to_numeric(out["FaceRectWidth"], errors="coerce")
    h = pd.to_numeric(out["FaceRectHeight"], errors="coerce")
    out["rf_bbox_x1"] = x
    out["rf_bbox_y1"] = y
    out["rf_bbox_x2"] = x + w
    out["rf_bbox_y2"] = y + h

    out.drop(columns=["_path_key", "_name_key"], errors="ignore", inplace=True)
    return out


def _count_detected(raw: pd.DataFrame) -> int:
    if "detected" in raw.columns:
        return int(raw["detected"].fillna(False).astype(bool).sum())
    boxes = raw[["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"]].apply(
        pd.to_numeric, errors="coerce"
    )
    return int((boxes.notna().all(axis=1) & boxes["FaceRectWidth"].gt(0) & boxes["FaceRectHeight"].gt(0)).sum())


def _runtime_info(torch: Any) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    devices: list[dict[str, Any]] = []
    for idx in range(device_count):
        props = torch.cuda.get_device_properties(idx)
        devices.append(
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "total_memory_bytes": int(props.total_memory),
                "compute_capability": [int(props.major), int(props.minor)],
            }
        )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "py_feat": _safe_version("py-feat"),
        "torch": _safe_version("torch"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        "cuda_available": cuda_available,
        "devices": devices,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Py-Feat 2.1.1 Detectorv2 native PyTorch/CUDA on formal RGB Face dry-run images"
    )
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-pyfeat-version-mismatch", action="store_true")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_manifest = _find_frame_manifest(sample_dir)
    frames = pd.read_csv(frame_manifest)
    if frames.empty:
        raise ValueError(f"Empty dry-run frame manifest: {frame_manifest}")

    image_paths = [str(Path(p).resolve()) for p in frames["image_path"].astype(str)]
    missing = [p for p in image_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Dry-run image missing; first missing: {missing[0]}")

    pyfeat_version = _safe_version("py-feat")
    if pyfeat_version != REQUIRED_PYFEAT_VERSION and not args.allow_pyfeat_version_mismatch:
        raise RuntimeError(
            f"This project freezes Py-Feat {REQUIRED_PYFEAT_VERSION}; found {pyfeat_version!r}. "
            "Install the frozen version or use --allow-pyfeat-version-mismatch only for diagnostics."
        )

    import torch
    from feat import Detectorv2

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False; refusing to run NVIDIA formal dry-run on CPU")
    if not str(args.device).lower().startswith("cuda"):
        raise ValueError("NVIDIA formal dry-run requires --device cuda or cuda:<index>")

    runtime = _runtime_info(torch)
    torch.cuda.synchronize()
    init_start = time.perf_counter()
    # Identity is intentionally excluded from the scientific core, matching the accepted AMD
    # definition. Do not silently re-enable ArcFace/FaceNet in the formal pipeline.
    detector = Detectorv2(device=args.device, identity_model=None)
    torch.cuda.synchronize()
    init_sec = time.perf_counter() - init_start

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    infer_start = time.perf_counter()
    fex = detector.detect(
        image_paths,
        data_type="image",
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=True,
        face_detection_threshold=0.5,
        progress_bar=True,
    )
    torch.cuda.synchronize()
    infer_sec = time.perf_counter() - infer_start
    peak_memory = int(torch.cuda.max_memory_allocated())

    native = pd.DataFrame(fex).copy()
    identity_cols = [c for c in native.columns if str(c).startswith("Identity")]
    if identity_cols:
        # Defensive only. With identity_model=None these columns should normally be absent/empty.
        native = native.drop(columns=identity_cols)
    raw = _attach_manifest(native, frames)

    raw_path = output_dir / "pyfeat_cuda_raw.parquet"
    columns_path = output_dir / "pyfeat_cuda_columns.json"
    manifest_path = output_dir / "pyfeat_cuda_dryrun_manifest.json"

    write_start = time.perf_counter()
    raw.to_parquet(raw_path, index=False, engine="pyarrow", compression="zstd")
    parquet_write_sec = time.perf_counter() - write_start
    columns_path.write_text(json.dumps(list(raw.columns), ensure_ascii=False, indent=2), encoding="utf-8")

    detected_rows = _count_detected(raw)
    frame_counts = raw.groupby("benchmark_index").size()
    multi_face_frames = int((frame_counts > 1).sum())
    no_face_frames = int(
        raw.groupby("benchmark_index")["detected"].apply(lambda s: not s.fillna(False).astype(bool).any()).sum()
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "candidate": "pyfeat211_detectorv2_native_pytorch_cuda",
        "scientific_core": "Py-Feat 2.1.1 Detectorv2; identity_model=None",
        "frame_manifest": str(frame_manifest),
        "expected_input_frames": int(len(frames)),
        "output_rows": int(len(raw)),
        "detected_face_rows": detected_rows,
        "multi_face_frames": multi_face_frames,
        "no_face_frames": no_face_frames,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "device": str(args.device),
        "timing_sec": {
            "model_initialization": init_sec,
            "detector_detect": infer_sec,
            "parquet_write": parquet_write_sec,
            "detect_plus_write": infer_sec + parquet_write_sec,
        },
        "input_frames_per_sec_detect": float(len(frames) / infer_sec) if infer_sec > 0 else None,
        "input_frames_per_sec_detect_plus_write": float(len(frames) / (infer_sec + parquet_write_sec)) if infer_sec + parquet_write_sec > 0 else None,
        "cuda_peak_memory_allocated_bytes": peak_memory,
        "runtime": runtime,
        "identity_columns_removed_defensively": identity_cols,
        "raw_output": str(raw_path),
        "columns_output": str(columns_path),
        "retention": [
            "all native non-identity Detectorv2 scientific columns retained",
            "multi-face rows retained",
            "no-face placeholders retained when produced by Detectorv2",
            "dry-run frame/timestamp/phase/context provenance merged by input path",
            "canonical benchmark_index + face_rank + detected + rf_bbox compatibility columns added",
        ],
        "notes": [
            "This v0.1 CUDA Gate intentionally uses the already sampled JPEG dry-run images so backend parity is evaluated separately from direct-AVI I/O optimization.",
            "After CUDA scientific parity passes, the NVIDIA full-video runner should decode the original AVI directly while preserving the same timestamp grid and native Detectorv2 semantics.",
            "Identity is outside the accepted project scientific core and is explicitly disabled.",
        ],
    }
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
