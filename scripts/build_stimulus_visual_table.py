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
from attention_pipeline.nir_behavior.stimulus_visual_report import (
    write_stimulus_visual_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the 27 formal SART stimulus screens from FocusWave assets, "
            "compute versioned digital luminance/contrast covariates, and write "
            "report-ready full-screen PNGs."
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

    image_outputs = write_stimulus_visual_images(config, materials, table)
    manifest = dict(manifest)
    manifest["report_images"] = image_outputs

    csv_path, manifest_path = write_stimulus_visual_outputs(config, table, manifest)

    summary = {
        "status": "complete",
        "conditions": int(len(table)),
        "materials_dir": str(materials),
        "output_csv": str(csv_path),
        "manifest_json": str(manifest_path),
        "rendered_images_dir": image_outputs["rendered_images_dir"],
        "full_resolution_condition_pngs": image_outputs[
            "full_resolution_condition_count"
        ],
        "overview_full_png": image_outputs["overview_full"]["path"],
        "overview_central_png": image_outputs["overview_central"]["path"],
        "stimuli": int(table["stimulus_name"].nunique()),
        "sizes_pct": sorted(int(v) for v in table["stimulus_size_pct"].unique()),
        "note": (
            "relative-luminance values are digital linear-sRGB metrics, not "
            "calibrated physical display luminance in cd/m^2; individual condition "
            "PNGs preserve the complete reconstructed task window"
        ),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
