"""Run the FocusWave formal Q1 four-class supervised analysis on one comparison-specific probe table.

文件：supervised_learning_analysis_4class.py
版本：1.0.0
功能：Q1 四分类（名义四类）正式分析的命令行入口，对应预注册
      `分析设计/1.16.24-Q1四分类正式分析与两条公平对照路线_20260913.md`。
      与二分类脚本 `scripts/supervised_learning_analysis.py` 保持同构：读取 probe 表、
      在**任何拟合之前**校验 Task-B 样本的结局范围、调用 run_multiclass_from_config，
      然后把 manifest JSON 打到标准输出。
用法（路线 A：复用二分类线已冻结的比较特异特征方案）：
    & $py scripts/supervised_learning_analysis_4class.py `
        --config configs/supervised_learning_4class_v1.yaml `
        --input-table <AS.full.csv> `
        --output-root <run-root> `
        --route A `
        --run-id <run-id>
用法（路线 B：外层训练折内前向逐步特征选择）：
    同上去掉 --config 时也可，仅把 --route 换成 B。
依赖：attention_pipeline.config、attention_pipeline.supervised_learning.entrypoint_multiclass

输出：
    <output-root>/<run-id>/ 下的运行目录（逐 probe 预测、逐折审计、失败表、manifest
    以及四分类专属评估块 multiclass_evaluation.json）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.supervised_learning.entrypoint_multiclass import (
    DEFAULT_MULTICLASS_CONFIG,
    run_multiclass_from_config,
)
from attention_pipeline.supervised_learning.outcome_scope import (
    validate_task_a_required_outcomes,
)


def _read_probe_table(path: Path) -> pd.DataFrame:
    """读取 probe 级输入表；CSV 用 utf-8-sig 以兼容带 BOM 的导出文件。"""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported four-class input format {suffix!r}; use CSV or Parquet")


def build_parser() -> argparse.ArgumentParser:
    """构造与二分类脚本同构的命令行解析器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_MULTICLASS_CONFIG)
    parser.add_argument("--paths-config", default=None)
    parser.add_argument(
        "--input-table",
        default=None,
        help="Optional runtime override for the comparison-specific admitted probe table.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional runtime override for the external output root.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--route",
        required=True,
        choices=("A", "B"),
        help="A = frozen binary-line feature schemes; B = forward selection inside each outer training fold.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数、做运行前结局闸门，然后执行一次四分类运行。"""
    args = build_parser().parse_args(argv)

    if args.input_table is not None:
        input_path = Path(args.input_table).expanduser().resolve()
    else:
        config = load_config(args.config, paths_config=args.paths_config)
        input_path = config.path_value("input_table")
    if not input_path.is_file():
        raise FileNotFoundError(f"four-class supervised input probe table not found: {input_path}")

    # 与二分类脚本同一条运行前闸门：比较特的样本成员资格可以基于结局有效性定义，
    # 因此必须先确认它只用了当前 Q1 目标的冻结来源列，再做任何模型选择或分组拟合。
    validate_task_a_required_outcomes(_read_probe_table(input_path))

    manifest = run_multiclass_from_config(
        args.config,
        paths_config=args.paths_config,
        input_table=input_path,
        output_root=args.output_root,
        route=args.route,
        run_id=args.run_id,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
