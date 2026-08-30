"""Run the governed multimodal fusion formal analysis (eight-combination LOSO ladder).

File: multimodal_fusion_analysis.py
Version: multimodal-fusion-v1.0.0
Purpose:
    命令行入口：对齐审计 -> 八组合 LOSO 全量 -> 性能/增量/边际贡献输出。
    输入表只读；输出写 --output-root 下新 run 目录；不修改任何单模态产物，
    不执行 formal_multimodal_v2.yaml fusion 状态解锁。

Usage:
    python scripts/multimodal_fusion_analysis.py \\
        --config configs/multimodal_fusion.yaml \\
        --paths-config configs/paths.local.yaml \\
        --jobs 8
    # smoke: 只跑前 3 个参与者折、M0/M1/M7、仅 Q1、仅逻辑回归
    python scripts/multimodal_fusion_analysis.py --fold-limit 3 \\
        --combinations M0,M1,M7 --outcomes q1 --models logistic

Dependencies:
    attention_pipeline.multimodal_formal.runner
"""
from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multimodal_fusion.yaml")
    parser.add_argument("--paths-config", default=None,
                        help="Machine-local path registry; otherwise ATTENTION_ANALYSIS_PATHS_CONFIG is used.")
    parser.add_argument("--run-id", default=None, help="Output run id; defaults to UTC timestamp.")
    parser.add_argument("--jobs", type=int, default=8, help="Fold-level parallel workers (loky).")
    parser.add_argument("--fold-limit", type=int, default=None,
                        help="Run only the first N participant folds (smoke).")
    parser.add_argument("--combinations", default=None,
                        help="Comma-separated combination subset, e.g. M0,M1,M7.")
    parser.add_argument("--outcomes", default=None, help="Comma-separated outcome subset, e.g. q1.")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model subset, e.g. logistic.")
    args = parser.parse_args()

    from attention_pipeline.multimodal_formal.runner import run_multimodal_fusion

    combinations = [x.strip() for x in args.combinations.split(",") if x.strip()] if args.combinations else None
    outcomes = [x.strip() for x in args.outcomes.split(",") if x.strip()] if args.outcomes else None
    models = [x.strip() for x in args.models.split(",") if x.strip()] if args.models else None
    manifest = run_multimodal_fusion(
        args.config,
        paths_config=args.paths_config,
        run_id=args.run_id,
        jobs=args.jobs,
        fold_limit=args.fold_limit,
        combinations=combinations,
        outcomes=outcomes,
        models=models,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
