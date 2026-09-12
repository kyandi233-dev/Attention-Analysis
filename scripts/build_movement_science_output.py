"""Build P4 Movement scientific outputs from an existing completed RGB 5.5 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from attention_pipeline.rgb_formal.movement_science_figures import build_movement_science_figures
from attention_pipeline.rgb_formal.movement_science_output import build_movement_science_output


DESCRIPTIVE_COLUMNS = (
    "participant_group_id",
    "session_id",
    "block_id",
    "body_motion_energy_median",
    "exposure_change_abs_median",
    "pose_lateral_right_per_sec_median",
    "pose_vertical_up_per_sec_median",
    "pose_radial_proximity_direction_score_median",
)


def _materialize_descriptive_source(rgb55_root: Path, science_root: Path) -> Path:
    source = rgb55_root / "tables/rgb_probe_pre30s_strict_features.csv"
    frame = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)
    keep = [column for column in DESCRIPTIVE_COLUMNS if column in frame.columns]
    required = {"participant_group_id", "session_id", "block_id", "body_motion_energy_median"}
    missing = sorted(required - set(keep))
    if missing:
        raise ValueError(f"Movement descriptive source missing required columns: {missing}")
    destination = science_root / "Movement/tables/movement_probe_descriptive_source.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, keep].to_csv(destination, index=False, encoding="utf-8-sig")
    return destination


def _remove_legacy_png_only_figures(science_root: Path, manifest: dict[str, object]) -> list[str]:
    removed: list[str] = []
    movement_root = science_root / "Movement"
    for relative in list(manifest.get("generated_figures", [])):
        path = movement_root / str(relative)
        if path.exists():
            path.unlink()
            removed.append(str(relative))
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rgb55-root",
        required=True,
        help="Existing RGB 5.5 output root containing tables/ and models/. Does not rerun video analysis.",
    )
    parser.add_argument(
        "--science-root",
        required=True,
        help="Shared FormalScience root; Movement/ is created underneath it.",
    )
    parser.add_argument(
        "--g1-probe-candidates",
        default=None,
        help=(
            "Optional local authoritative NIR G1 probe_measurement_candidates.csv. "
            "When supplied, P4 also writes the Ocular x Movement artifact-sensitivity audit."
        ),
    )
    parser.add_argument(
        "--non-authoritative",
        action="store_true",
        help="Mark subset/smoke output as non-authoritative.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Fail instead of replacing an existing FormalScience/Movement directory.",
    )
    args = parser.parse_args()

    rgb55_root = Path(args.rgb55_root).expanduser().resolve()
    science_root = Path(args.science_root).expanduser().resolve()
    manifest = build_movement_science_output(
        rgb55_root,
        science_root,
        g1_probe_candidates_path=(Path(args.g1_probe_candidates) if args.g1_probe_candidates else None),
        authoritative=not args.non_authoritative,
        replace=not args.keep_existing,
    )

    # The builder's historical PNG-only figures pre-date the Formal 1.16.12
    # scientific-output contract. Keep their numeric tables, but replace the
    # figures with the canonical no-title PNG+SVG package below.
    removed_legacy = _remove_legacy_png_only_figures(science_root, manifest)
    descriptive_source = _materialize_descriptive_source(rgb55_root, science_root)
    generated = build_movement_science_figures(science_root / "Movement")

    manifest["legacy_png_only_figures_removed"] = removed_legacy
    manifest["movement_probe_descriptive_source"] = str(descriptive_source)
    manifest["generated_figures"] = generated
    manifest["generated_model_figures"] = [
        path for path in generated if "relationships" in path or "coefficients" in path
    ]
    manifest["figure_manifest"] = "manifests/figure_manifest.csv"
    manifest["figure_audit"] = "manifests/figure_audit.csv"
    manifest["figure_contract"] = {
        "internal_title_present": False,
        "formats": ["png_300dpi", "svg"],
        "status": "candidate",
        "p4_refit_performed": False,
        "feature_selection_performed": False,
    }

    manifest_path = science_root / "Movement/manifests/movement_science_output_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
