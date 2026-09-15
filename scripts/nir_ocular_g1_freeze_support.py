from __future__ import annotations

import argparse
from pathlib import Path

from attention_pipeline.nir_formal_analysis.ocular_g1_freeze_support import (
    build_ocular_g1_freeze_support,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build supplemental Ocular G1 parameter-freeze evidence from an existing "
            "measurement-audit output. This reads only existing audit CSVs and does not "
            "rerun frame-level NIR/RGB processing or Q1/Q2 analysis."
        )
    )
    parser.add_argument("--audit-root", required=True, help="Existing G1 measurement-audit output directory")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <audit-root>/freeze_support)",
    )
    args = parser.parse_args()

    audit_root = Path(args.audit_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else audit_root / "freeze_support"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = build_ocular_g1_freeze_support(audit_root)
    for name, table in tables.items():
        path = output_dir / name
        table.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{name}: rows={len(table)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
