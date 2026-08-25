from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import pandas as pd


def _with_benchmark_index(df: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    out = pd.DataFrame(df).reset_index(drop=True).copy()
    if len(out) != len(indices):
        raise RuntimeError(f"LibreFace component row mismatch: expected {len(indices)}, got {len(out)}")
    out.insert(0, "benchmark_index", indices)
    return out


def _load_cached_alignment(path: Path, expected_indices: set[int]) -> tuple[pd.DataFrame, list[int], list[str]] | None:
    if not path.exists():
        return None
    try:
        alignment = pd.read_parquet(path)
    except Exception:
        return None
    required = {"benchmark_index", "alignment_success", "aligned_image_path"}
    if not required.issubset(alignment.columns):
        return None
    observed = set(pd.to_numeric(alignment["benchmark_index"], errors="coerce").dropna().astype(int).tolist())
    if observed != expected_indices:
        return None
    ok = alignment[alignment["alignment_success"].fillna(False).astype(bool)].copy()
    if ok.empty:
        return None
    aligned_paths = [str(Path(p)) for p in ok["aligned_image_path"].tolist()]
    if any(not Path(p).exists() for p in aligned_paths):
        return None
    success_indices = pd.to_numeric(ok["benchmark_index"], errors="raise").astype(int).tolist()
    return alignment, success_indices, aligned_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LibreFace 2.0 on shared RGB Face benchmark images")
    parser.add_argument("--benchmark-dir", required=True, help=".../_test/face-benchmark/sub-XXX")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    root = Path(args.benchmark_dir)
    manifests = sorted(root.glob("*_face-benchmark_frames.csv"))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected exactly one frame manifest in {root}, found {len(manifests)}")
    manifest_path = manifests[0]
    sample = pd.read_csv(manifest_path)
    if sample.empty or "image_path" not in sample or "benchmark_index" not in sample:
        raise ValueError(f"Invalid benchmark frame manifest: {manifest_path}")

    from libreface import (
        estimate_gaze_video,
        get_aligned_image,
        get_au_intensities_and_detect_aus_video,
        get_facial_expression_video,
    )

    temp_dir = root / "libreface_tmp"
    weights_dir = root / "libreface_weights"
    temp_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    alignment_path = root / "libreface_alignment.parquet"
    expected_indices = set(pd.to_numeric(sample["benchmark_index"], errors="raise").astype(int).tolist())
    cached = _load_cached_alignment(alignment_path, expected_indices)
    alignment_reused = cached is not None

    if cached is not None:
        alignment, success_indices, aligned_paths = cached
        alignment_sec = 0.0
        print(f"[libreface] reusing cached alignment: {len(success_indices)}/{len(sample)} faces")
    else:
        aligned_paths: list[str] = []
        success_indices: list[int] = []
        alignment_rows: list[dict[str, object]] = []

        align_start = time.perf_counter()
        for row in sample.itertuples(index=False):
            benchmark_index = int(row.benchmark_index)
            image_path = str(row.image_path)
            try:
                aligned_path, headpose, landmark_data = get_aligned_image(
                    image_path, temp_dir=str(temp_dir), verbose=False
                )
                if not aligned_path:
                    raise RuntimeError("get_aligned_image returned no aligned path")
                aligned_paths.append(str(aligned_path))
                success_indices.append(benchmark_index)
                alignment_rows.append({
                    "benchmark_index": benchmark_index,
                    "alignment_success": True,
                    "alignment_error": None,
                    "aligned_image_path": str(aligned_path),
                    "headpose_json": json.dumps(headpose, ensure_ascii=False, default=str),
                    "landmarks_json": json.dumps(landmark_data, ensure_ascii=False, default=str),
                })
            except Exception as exc:
                alignment_rows.append({
                    "benchmark_index": benchmark_index,
                    "alignment_success": False,
                    "alignment_error": f"{type(exc).__name__}: {exc}",
                    "aligned_image_path": None,
                    "headpose_json": None,
                    "landmarks_json": None,
                })
        alignment_sec = time.perf_counter() - align_start
        alignment = pd.DataFrame(alignment_rows)
        alignment.to_parquet(alignment_path, index=False, engine="pyarrow", compression="zstd")
        if not aligned_paths:
            raise RuntimeError("LibreFace aligned zero benchmark faces")

    au_start = time.perf_counter()
    detected_aus, au_intensities = get_au_intensities_and_detect_aus_video(
        aligned_paths,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weights_download_dir=str(weights_dir),
    )
    au_sec = time.perf_counter() - au_start
    au_detection = _with_benchmark_index(pd.DataFrame(detected_aus), success_indices)
    au_intensity = _with_benchmark_index(pd.DataFrame(au_intensities), success_indices)
    au_detection.to_parquet(root / "libreface_au_detection.parquet", index=False, engine="pyarrow", compression="zstd")
    au_intensity.to_parquet(root / "libreface_au_intensity.parquet", index=False, engine="pyarrow", compression="zstd")

    expr_start = time.perf_counter()
    expression = get_facial_expression_video(
        aligned_paths,
        device=args.device,
        batch_size=args.batch_size,
        weights_download_dir=str(weights_dir),
    )
    expression_sec = time.perf_counter() - expr_start
    expression_df = _with_benchmark_index(pd.DataFrame(expression), success_indices)
    expression_df.to_parquet(root / "libreface_expression.parquet", index=False, engine="pyarrow", compression="zstd")

    gaze_start = time.perf_counter()
    gaze = estimate_gaze_video(
        aligned_paths,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weights_download_dir=str(weights_dir),
    )
    gaze_sec = time.perf_counter() - gaze_start
    gaze_df = _with_benchmark_index(pd.DataFrame(gaze), success_indices)
    gaze_df.to_parquet(root / "libreface_gaze.parquet", index=False, engine="pyarrow", compression="zstd")

    merged = sample.merge(alignment, on="benchmark_index", how="left")
    for prefix, component in [
        ("au_detection", au_detection),
        ("au_intensity", au_intensity),
        ("expression", expression_df),
        ("gaze", gaze_df),
    ]:
        renamed = component.rename(
            columns={c: f"{prefix}__{c}" for c in component.columns if c != "benchmark_index"}
        )
        merged = merged.merge(renamed, on="benchmark_index", how="left")
    merged.to_parquet(root / "libreface_raw.parquet", index=False, engine="pyarrow", compression="zstd")

    try:
        libreface_version = importlib.metadata.version("libreface")
    except Exception:
        libreface_version = None
    try:
        torch_version = importlib.metadata.version("torch")
    except Exception:
        torch_version = None

    summary = {
        "schema_version": "rgb-face-benchmark-libreface-v0.1",
        "candidate": "libreface2",
        "benchmark_frame_manifest": str(manifest_path),
        "expected_input_images": int(len(sample)),
        "aligned_faces": int(len(success_indices)),
        "alignment_valid_fraction": len(success_indices) / len(sample) if len(sample) else None,
        "alignment_reused": alignment_reused,
        "device": args.device,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "timing_sec": {
            "alignment": alignment_sec,
            "au_joint_detection_intensity": au_sec,
            "expression": expression_sec,
            "gaze": gaze_sec,
            "total_measured": alignment_sec + au_sec + expression_sec + gaze_sec,
        },
        "input_images_per_sec_total": len(sample) / (alignment_sec + au_sec + expression_sec + gaze_sec),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "libreface": libreface_version,
            "torch": torch_version,
        },
        "outputs": {
            "alignment": str(alignment_path),
            "au_detection": str(root / "libreface_au_detection.parquet"),
            "au_intensity": str(root / "libreface_au_intensity.parquet"),
            "expression": str(root / "libreface_expression.parquet"),
            "gaze": str(root / "libreface_gaze.parquet"),
            "merged_raw": str(root / "libreface_raw.parquet"),
        },
        "retention": "candidate components saved separately before merged convenience table; no common-AU prefiltering",
    }
    output_manifest = root / "libreface_benchmark_manifest.json"
    output_manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
