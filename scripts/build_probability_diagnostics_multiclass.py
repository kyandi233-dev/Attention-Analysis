"""从归档四分类折外预测构建多类布里尔分数与校准诊断。

用法
----
::

    $env:PYTHONPATH = "<worktree>\\src"   # 必须（见下方说明）
    python scripts/build_probability_diagnostics_multiclass.py `
      --run-root <SupervisedRuns4ClassV1> `
      --output-root <SupervisedRuns4ClassV1\\multiclass_probability_diagnostics> `
      --frozen-summary <SupervisedRuns4ClassV1\\four_class_summary.csv>

读取 ``--run-root`` 下由 ``--run-glob`` 选中的每个 ``<run>/probe_predictions.csv``
（默认 ``*__included_missing_aware__4class*``，也可用 ``--predictions`` 显式指定），写出：

* ``multiclass_probability_diagnostics.csv`` - 每行一个（运行、分析集合、模型、隶属类型、
  路线），含多类布里尔分数、布里尔技能分数、与逐折类别先验常数的配对差、逐类布里尔
  分数、折外合并宏平均 AUROC、顶端标签校准总体偏差、期望校准误差、校准截距/斜率，
  每项都带 95% 参与者簇自助区间；
* ``multiclass_calibration_bins.csv`` - 每模型每置信度分箱的可靠性表；
* ``multiclass_probability_diagnostics_manifest.json`` - 与冻结基线对账的审计记录。

本步骤是**只读后处理**：从不重训或重拟合预测模型、从不重跑任何 LOSO 折、从不影响
候选或 ``C`` 选择。冻结主指标（参与者宏平均多类对数损失）由冻结的 Task-A 运行器产出，
本脚本只重新计算一次用于与冻结汇总对账。

为什么 ``PYTHONPATH`` 是强制的
------------------------------
分析虚拟环境内 ``attention-analysis`` 的 editable 安装指向**另一个** worktree，裸跑脚本
会静默 import 错误的代码树。pytest 不受影响，因为 ``pyproject.toml`` 已把 ``src`` 前置。
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
from attention_pipeline.supervised_learning.probability_diagnostics_multiclass import (
    DEFAULT_CALIBRATION_BINS,
    MulticlassProbabilityDiagnosticsError,
    build_multiclass_probability_diagnostics,
    multiclass_diagnostic_summary,
)

#: 只聚合四分类正式运行目录。探索性冒烟目录 ``smoke4class__*`` 不匹配该模式，
#: 它可能复用正式目录的 analysis_set_id / model_id（``1.16.22`` §4.1 的陷阱）。
DEFAULT_RUN_GLOB = "*__included_missing_aware__4class*"

PRIOR_COLUMN = "prior_baseline_participant_macro_multiclass_log_loss"


def _read_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _read_frozen_prior_log_loss(path: Path) -> dict[tuple[str, str], float]:
    """读取冻结汇总结论里的逐折先验基线数值，供重建对账使用。"""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in ("analysis_set_id", "model_id", PRIOR_COLUMN) if column not in frame.columns]
    if missing:
        raise MulticlassProbabilityDiagnosticsError(
            f"frozen summary missing required columns: {missing}"
        )
    result: dict[tuple[str, str], float] = {}
    for _, row in frame.iterrows():
        key = (str(row["analysis_set_id"]), str(row["model_id"]))
        value = float(row[PRIOR_COLUMN])
        if key in result and abs(result[key] - value) > 1e-12:
            raise MulticlassProbabilityDiagnosticsError(
                f"frozen summary has conflicting prior baselines for {key}"
            )
        result[key] = value
    return result


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
            "Glob applied inside --run-root to select four-class run directories. The default "
            "keeps only the frozen membership type and excludes exploratory smoke runs."
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        action="append",
        default=None,
        help="Explicit probe_predictions.csv path (repeatable).",
    )
    parser.add_argument(
        "--frozen-summary",
        type=Path,
        default=None,
        help=(
            "four_class_summary.csv from the frozen four-class batch. When supplied, the "
            "reconstructed per-fold class priors must reproduce its prior baseline exactly "
            "(fail-closed) before any diagnostic row is written."
        ),
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--n-bins", type=int, default=DEFAULT_CALIBRATION_BINS)
    parser.add_argument("--replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--confidence-level", type=float, default=DEFAULT_CONFIDENCE_LEVEL)
    parser.add_argument("--prior-log-loss-tolerance", type=float, default=1e-9)
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

    frozen = None
    if args.frozen_summary is not None:
        frozen_path = Path(args.frozen_summary)
        if not frozen_path.is_file():
            raise FileNotFoundError(f"frozen summary not found: {frozen_path}")
        frozen = _read_frozen_prior_log_loss(frozen_path)

    diagnostics, calibration_bins, audit = build_multiclass_probability_diagnostics(
        combined,
        n_bins=args.n_bins,
        replicates=args.replicates,
        seed=args.seed,
        confidence_level=args.confidence_level,
        frozen_prior_log_loss=frozen,
        prior_log_loss_tolerance=args.prior_log_loss_tolerance,
    )

    output = Path(args.output_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing diagnostics output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    diagnostics_path = output / "multiclass_probability_diagnostics.csv"
    bins_path = output / "multiclass_calibration_bins.csv"
    audit_path = output / "multiclass_probability_diagnostics_audit.json"
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    calibration_bins.to_csv(bins_path, index=False, encoding="utf-8-sig")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "predictions_files": [str(p) for p in paths],
        "n_prediction_files": len(paths),
        "run_glob": args.run_glob if args.run_root is not None else None,
        "frozen_summary": str(args.frozen_summary) if args.frozen_summary is not None else None,
        "n_prediction_rows": int(len(combined)),
        "outputs": {
            "multiclass_probability_diagnostics": str(diagnostics_path),
            "multiclass_calibration_bins": str(bins_path),
            "audit": str(audit_path),
        },
        "prior_reconstruction": {
            "n_rows_audited": audit["n_prior_reconstruction_rows"],
            "n_checked_against_frozen": audit["n_prior_rows_checked_against_frozen"],
            "n_mismatching_frozen": audit["n_prior_rows_mismatching_frozen"],
        },
        **multiclass_diagnostic_summary(diagnostics),
    }
    summary_path = output / "multiclass_probability_diagnostics_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["outputs"]["manifest"] = str(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
