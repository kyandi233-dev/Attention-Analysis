"""Run the formal pupil-only NIR validation/report layer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.formal_analysis.provenance import resolve_git_checkout
from attention_pipeline.nir_pipeline_validation.pupil_validation import run_validation


def _path(config, key: str) -> Path:
    return config.path_value(key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nir_pipeline_validation.yaml")
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--tables-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--visual-table", default=None)
    args = parser.parse_args()

    config = load_config(args.config, paths_config=args.paths_config)
    tables_root = Path(args.tables_root).expanduser().resolve() if args.tables_root else _path(config, "analysis_tables_root")
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else _path(config, "output_root")

    visual_path = None
    if args.visual_table:
        visual_path = Path(args.visual_table).expanduser().resolve()
    else:
        configured = config.section("paths").get("stimulus_visual_table")
        if configured not in (None, ""):
            visual_path = config.path_value("stimulus_visual_table")
    visual = pd.read_csv(visual_path, low_memory=False) if visual_path is not None and visual_path.is_file() else None

    repo_root = Path(__file__).resolve().parents[1]
    require_clean = bool(config.section("provenance").get("require_clean_code_checkout", True))
    code_provenance = resolve_git_checkout(repo_root, role="code", require_clean=require_clean)

    cohort = config.section("cohort")
    topology = {
        "n_sessions": int(cohort["expected_session_count"]),
        "n_analysis_groups": int(cohort["expected_analysis_group_count"]),
        "n_double_session_repeat_groups": int(cohort["expected_double_session_repeat_groups"]),
    }
    figure_cfg = config.section("figures")
    manifest = run_validation(
        tables_root=tables_root,
        output_root=output_root,
        visual_properties=visual,
        formats=tuple(str(x) for x in figure_cfg.get("formats", ["png"])),
        dpi=int(figure_cfg.get("dpi", 220)),
        expected_topology=topology,
        runtime_provenance={"code": code_provenance},
        config_digest=config.digest,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
