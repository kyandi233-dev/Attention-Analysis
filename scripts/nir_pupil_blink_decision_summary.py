from __future__ import annotations

import argparse
from pathlib import Path

from attention_pipeline.nir_formal_analysis.pupil_blink_decision_summary import (
    build_g1_decision_summaries,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build signal-aware, measurement-state-aware summaries from an existing "
            "NIR pupil/blink G1 measurement audit. This does not rerun NIR/RGB inference "
            "and does not freeze scientific parameters."
        )
    )
    parser.add_argument("--audit-root", required=True, help="Directory containing G1 audit CSV outputs")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Summary output directory (default: <audit-root>/decision_summary)",
    )
    args = parser.parse_args()

    audit_root = Path(args.audit_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else audit_root / "decision_summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = build_g1_decision_summaries(audit_root)
    for name, table in summaries.items():
        table.to_csv(output_dir / name, index=False, encoding="utf-8-sig")
        print(f"{name}: rows={len(table)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
