from __future__ import annotations

import argparse
from pathlib import Path

from attention_pipeline.config import load_config
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
    parser.add_argument(
        "--rgb-blink-events",
        default=None,
        help=(
            "Optional explicit combined RGB blink-event CSV/Parquet. Default: "
            "<rgb_analysis_tables_root>/tables/rgb_blink_candidate_events.csv from the path registry."
        ),
    )
    parser.add_argument(
        "--probe-table",
        default=None,
        help=(
            "Optional explicit Behavior probe table. Default: "
            "<behavior_output_root>/formal_v3/probe_primary_30s.csv from the path registry."
        ),
    )
    parser.add_argument(
        "--rgb-blink-frames-root",
        default=None,
        help=(
            "Optional explicit RGB analysis-ready root containing "
            "<session>/<session>_blink_candidate_frames.parquet. Default: rgb_analysis_ready_root "
            "from the path registry."
        ),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()

    paths_config = Path(args.paths_config) if args.paths_config else None
    nir_config = load_config(args.nir_config, paths_config=paths_config)

    rgb_blink_events = (
        Path(args.rgb_blink_events)
        if args.rgb_blink_events
        else nir_config.registry_path("rgb_analysis_tables_root")
        / "tables"
        / "rgb_blink_candidate_events.csv"
    )
    probe_table = (
        Path(args.probe_table)
        if args.probe_table
        else nir_config.registry_path("behavior_output_root")
        / "formal_v3"
        / "probe_primary_30s.csv"
    )
    rgb_blink_frames_root = (
        Path(args.rgb_blink_frames_root)
        if args.rgb_blink_frames_root
        else nir_config.registry_path("rgb_analysis_ready_root")
    )

    manifest = run_pupil_blink_measurement_audit(
        nir_config_path=Path(args.nir_config),
        audit_config_path=Path(args.audit_config),
        rgb_blink_events_path=rgb_blink_events,
        probe_table_path=probe_table,
        output_root=Path(args.output_root),
        rgb_blink_frames_root=rgb_blink_frames_root,
        paths_config=paths_config,
        subjects=args.subjects,
    )
    print(manifest)


if __name__ == "__main__":
    main()
