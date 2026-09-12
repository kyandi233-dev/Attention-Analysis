"""Produce Q1-blind single-modality feature review and explicit freeze status."""
from __future__ import annotations

import argparse
import json

from attention_pipeline.formal_analysis.feature_qualification import (
    SCIENTIFIC_MODALITIES, write_single_modality_review,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="One-row-per-probe CSV or Parquet; labels are ignored")
    parser.add_argument("--catalog", default="configs/single_modality_features_v1.yaml")
    parser.add_argument("--modality", required=True, choices=SCIENTIFIC_MODALITIES)
    parser.add_argument("--decisions", help="Evidence-backed JSON bound to source/catalog SHA256; omit for audit only")
    parser.add_argument("--output-root", required=True, help="New directory; existing outputs are never overwritten")
    args = parser.parse_args()
    result = write_single_modality_review(args.input, args.catalog, args.modality, args.output_root, decisions_path=args.decisions)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
