from __future__ import annotations

import argparse
from pathlib import Path

from attention_pipeline.nir_formal_analysis.pupil_blink_measurement_runner import (
    run_pupil_blink_measurement_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run FocusWave 1.16.3 pupil/blink measurement audit only; "
            "does not run Q1/Q2 inference or supervised learning."
        )
    )
    parser.add_argument("--nir-config", default="configs/nir_analysis_ready.yaml")
    parser.add_argument(
        "--audit-config", default="configs/nir_pupil_blink_measurement_audit_v1.yaml"
    )
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--rgb-blink-events", required=True)
    parser.add_argument("--probe-table", required=True)
    parser.add_argument(
        "--rgb-blink-frames-root",
        default=None,
        help=(
            "Optional RGB analysis-ready root containing "
            "<session>/<session>_blink_candidate_frames.parquet for full time-axis sync audit."
        ),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()

    manifest = run_pupil_blink_measurement_audit(
        nir_config_path=Path(args.nir_config),
        audit_config_path=Path(args.audit_config),
        rgb_blink_events_path=Path(args.rgb_blink_events),
        probe_table_path=Path(args.probe_table),
        output_root=Path(args.output_root),
        rgb_blink_frames_root=(
            Path(args.rgb_blink_frames_root) if args.rgb_blink_frames_root else None
        ),
        paths_config=Path(args.paths_config) if args.paths_config else None,
        subjects=args.subjects,
    )
    print(manifest)


if __name__ == "__main__":
    main()
