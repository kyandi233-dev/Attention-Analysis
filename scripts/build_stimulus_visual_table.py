from __future__ import annotations

import argparse
import json
from pathlib import Path

from attention_pipeline.config import load_config
from attention_pipeline.nir_behavior.stimulus_visual import (
    build_stimulus_visual_table,
    resolve_materials_dir,
    write_stimulus_visual_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the 27 formal SART stimulus screens from FocusWave assets "
            "and compute versioned digital luminance/contrast covariates."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/stimulus_visual.yaml",
        help="Stimulus visual config path.",
    )
    parser.add_argument(
        "--materials-dir",
        default=None,
        help=(
            "Local FocusWave/01-MainProgram/素材 directory. If omitted, use "
            "FOCUSWAVE_MATERIALS_DIR, config, or sibling-repository discovery."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    materials = resolve_materials_dir(config, args.materials_dir)
    table, manifest = build_stimulus_visual_table(config, materials)
    csv_path, manifest_path = write_stimulus_visual_outputs(config, table, manifest)

    summary = {
        "status": "complete",
        "conditions": int(len(table)),
        "materials_dir": str(materials),
        "output_csv": str(csv_path),
        "manifest_json": str(manifest_path),
        "stimuli": int(table["stimulus_name"].nunique()),
        "sizes_pct": sorted(int(v) for v in table["stimulus_size_pct"].unique()),
        "note": (
            "relative-luminance values are digital linear-sRGB metrics, not "
            "calibrated physical display luminance in cd/m^2"
        ),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
