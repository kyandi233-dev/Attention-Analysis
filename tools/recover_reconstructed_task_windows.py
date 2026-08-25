"""Generate timestamp-based recovery windows for NIR and mmWave.

Example::

    python tools/recover_reconstructed_task_windows.py \
      --subject sub-099 \
      --timeline D:/.../sub099_reconstructed_timeline.csv \
      --nir-timestamps J:/Data/sub-099_/nir/sub-099_nir_timestamps.csv \
      --mmwave-timestamps J:/Data/sub-099_/mmwave/sub-099_mmwave_timestamps.csv \
      --output D:/.../sub099_recovery_windows

The output is a mapping/manifest only. It does not modify raw files or run
inference. NIR recovery inference consumes the same timeline with
``run_pipeline.py formal --recovery-timeline``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1] / "runtime" / "nir-formal"
sys.path.insert(0, str(RUNTIME))

from recovery_windows import (  # noqa: E402
    map_timestamp_rows,
    read_reconstructed_blocks,
    read_timestamp_rows,
    write_recovery_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--nir-timestamps", type=Path, required=True)
    parser.add_argument("--mmwave-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    blocks = read_reconstructed_blocks(args.timeline)
    nir = map_timestamp_rows(
        "nir",
        read_timestamp_rows(args.nir_timestamps),
        blocks,
        source=str(args.nir_timestamps.resolve()),
    )
    mmwave = map_timestamp_rows(
        "mmwave",
        read_timestamp_rows(args.mmwave_timestamps),
        blocks,
        source=str(args.mmwave_timestamps.resolve()),
    )
    windows = [*nir, *mmwave]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "recovery_windows.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(windows[0].to_dict()))
        writer.writeheader()
        writer.writerows(window.to_dict() for window in windows)
    manifest = write_recovery_manifest(
        args.output / "recovery_manifest.json",
        subject=args.subject,
        timeline=args.timeline,
        windows=windows,
        limitations=(
            "Reconstructed block boundaries support task-window recovery only. "
            "They do not reconstruct baseline, practice, modality-start, or stop events. "
            "NIR and mmWave are mapped through their own absolute timestamp files; "
            "RGB frame numbers are never used as surrogate indices."
        ),
    )
    print(json.dumps({"manifest": str(manifest.resolve()), "windows": len(windows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
