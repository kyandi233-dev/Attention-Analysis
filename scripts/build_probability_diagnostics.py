"""Build descriptive probability diagnostics from archived out-of-fold predictions.

Usage
-----
::

    $env:PYTHONPATH = "<worktree>\\src"   # REQUIRED (see the note below)
    python scripts/build_probability_diagnostics.py `
      --run-root <SupervisedRunsV1> `
      --output-root <diagnostics root>

Reads every ``<run>/probe_predictions.csv`` under ``--run-root`` selected by ``--run-glob``
(default ``*__included_missing_aware``, or the explicit ``--predictions`` paths) and writes:

* ``probability_diagnostics.csv`` - one row per (model, analysis_set_id, membership_type)
  with participant-macro AUROC, Brier score, calibration-in-the-large and the calibration
  intercept/slope, each with a 95% participant-cluster bootstrap interval; and
* ``calibration_bins.csv`` - per-model probability-bin reliability table.

This is a **read-only post-processing step**. It never retrains or refits a prediction
model, never re-runs a LOSO fold, and never influences candidate or ``C`` selection. The
frozen primary metric (participant-macro log loss) and the session-level AUROC are
produced by the frozen Task-A runner and are not recomputed here.

Why ``PYTHONPATH`` is mandatory
-------------------------------
The analysis virtual environment contains an editable install of ``attention-analysis``
pointing at a different worktree, so a plain script run would silently import that tree.
Pytest is unaffected because ``pyproject.toml`` prepends ``src``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from attention_pipeline.supervised_learning.evaluation import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
)
from attention_pipeline.supervised_learning.probability_diagnostics import (
    DEFAULT_CALIBRATION_BINS,
    build_probability_diagnostics,
    diagnostic_summary,
)

# Only the frozen membership type is aggregated by default. ``smoke_go_omission__aware``
# writes the same analysis_set_id and model_id as the formal standalone go_omission run, so
# a bare ``*`` glob would merge an exploratory smoke prediction set into a formal row.
DEFAULT_RUN_GLOB = "*__included_missing_aware"


def _read_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Directory whose immediate subdirectories hold probe_predictions.csv.",
    )
    parser.add_argument(
        "--run-glob",
        default=DEFAULT_RUN_GLOB,
        help=(
            "Glob applied inside --run-root to select formal run directories. The default "
            "keeps only the frozen membership type and therefore excludes exploratory smoke "
            "runs such as smoke_go_omission__aware, which would otherwise collide with the "
            "formal go_omission run (same analysis_set_id and model_id)."
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        action="append",
        default=None,
        help="Explicit probe_predictions.csv path (repeatable).",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--n-bins", type=int, default=DEFAULT_CALIBRATION_BINS)
    parser.add_argument("--replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--confidence-level", type=float, default=DEFAULT_CONFIDENCE_LEVEL)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    paths: list[Path] = []
    if args.predictions:
        paths.extend(Path(p) for p in args.predictions)
    if args.run_root is not None:
        root = Path(args.run_root)
        if not root.is_dir():
            raise FileNotFoundError(f"run root not found: {root}")
        paths.extend(sorted(root.glob(f"{args.run_glob}/probe_predictions.csv")))
    if not paths:
        raise FileNotFoundError(
            "no predictions found; pass --run-root or one or more --predictions"
        )

    frames = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"predictions not found: {path}")
        frames.append(_read_predictions(path))
    combined = pd.concat(frames, ignore_index=True)

    diagnostics, calibration_bins = build_probability_diagnostics(
        combined,
        n_bins=args.n_bins,
        replicates=args.replicates,
        seed=args.seed,
        confidence_level=args.confidence_level,
    )

    output = Path(args.output_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing diagnostics output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    diagnostics_path = output / "probability_diagnostics.csv"
    bins_path = output / "calibration_bins.csv"
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    calibration_bins.to_csv(bins_path, index=False, encoding="utf-8-sig")

    summary = {
        "predictions_files": [str(p) for p in paths],
        "n_prediction_files": len(paths),
        "run_glob": args.run_glob if args.run_root is not None else None,
        "n_prediction_rows": int(len(combined)),
        "outputs": {"probability_diagnostics": str(diagnostics_path), "calibration_bins": str(bins_path)},
        **diagnostic_summary(diagnostics),
    }
    summary_path = output / "probability_diagnostics_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["outputs"]["manifest"] = str(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
