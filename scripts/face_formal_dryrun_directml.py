from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from face_real_directml_pyfeat import run_pyfeat


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Py-Feat DirectML backend on formal Face dry-run frames")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--retinaface-batch", type=int, default=8)
    parser.add_argument("--multitask-batch", type=int, default=16)
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir).resolve()
    manifests = sorted(sample_dir.glob("*_face-dryrun_frames.csv"))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected exactly one *_face-dryrun_frames.csv in {sample_dir}, found {len(manifests)}")
    source_manifest = manifests[0]

    # The validated real-300 runner discovers benchmark/continuous manifests.
    # Preserve the dry-run source manifest and write an explicit compatibility
    # alias instead of changing or deleting validation-era inputs.
    subject = source_manifest.name.split("_face-dryrun_frames.csv")[0]
    compat_manifest = sample_dir / f"{subject}_face-benchmark_frames.csv"
    table = pd.read_csv(source_manifest)
    table.to_csv(compat_manifest, index=False, encoding="utf-8-sig")

    run_pyfeat(
        Namespace(
            benchmark_dir=str(sample_dir),
            model_dir=str(Path(args.model_dir).resolve()),
            output_dir=str(Path(args.output_dir).resolve()),
            retinaface_batch=int(args.retinaface_batch),
            multitask_batch=int(args.multitask_batch),
        )
    )

    meta = {
        "stage": "face-formal-dryrun-directml",
        "sample_dir": str(sample_dir),
        "source_manifest": str(source_manifest),
        "compat_manifest": str(compat_manifest),
        "output_dir": str(Path(args.output_dir).resolve()),
        "retinaface_batch": int(args.retinaface_batch),
        "multitask_batch": int(args.multitask_batch),
        "note": "Compatibility alias is retained as test provenance; formal full-cohort runner will consume AVI/timestamps directly.",
    }
    meta_path = Path(args.output_dir).resolve() / "face_formal_dryrun_directml_wrapper.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
