"""Verify that four-class route A reuses the frozen binary comparison sets verbatim.

方法权威：`分析设计/1.16.24-Q1四分类正式分析与两条公平对照路线_20260913.md` §5.1

路线 A 的定义是「复用二分类线已冻结的比较特异特征方案，只把标签换成四分类」。这只有在
**同一批观测**上才成立，因此本脚本逐集合比对两侧 `run_manifest.json` 的：

- `provenance.input_sha256`（必须逐字节相同）
- `n_input_rows`（必须相同）

并报告是否存在只在一侧出现的集合。任何不一致即 FAIL。

用法：
    python verify_four_class_reuses_binary_sets.py
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

BASE = Path(r"D:\Project\厚粲杯\11_数据\_FormalAnalysis")
BINARY_ROOT = BASE / "SupervisedRunsV1"
FOUR_CLASS_ROOT = BASE / "SupervisedRuns4ClassV1"
BINARY_SUFFIX = "__included_missing_aware"
FOUR_CLASS_SUFFIX = "__included_missing_aware__4classA"


def _provenance(root: Path, suffix: str) -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    for run_dir in sorted(glob.glob(str(root / f"*{suffix}"))):
        manifest_path = os.path.join(run_dir, "run_manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        provenance = manifest.get("provenance", {})
        key = os.path.basename(run_dir)[: -len(suffix)]
        found[key] = {
            "input_sha256": provenance.get("input_sha256"),
            "n_input_rows": manifest.get("n_input_rows"),
            "config_digest": provenance.get("config_digest"),
            "code_sha": provenance.get("code_sha"),
            "task": manifest.get("task"),
        }
    return found


def main() -> int:
    binary = _provenance(BINARY_ROOT, BINARY_SUFFIX)
    four = _provenance(FOUR_CLASS_ROOT, FOUR_CLASS_SUFFIX)
    common = sorted(set(binary) & set(four))

    rows: list[dict[str, object]] = []
    mismatches: list[str] = []
    for key in common:
        same_sha = binary[key]["input_sha256"] == four[key]["input_sha256"]
        same_rows = binary[key]["n_input_rows"] == four[key]["n_input_rows"]
        rows.append(
            {
                "analysis_set_id": key,
                "binary_input_sha256": binary[key]["input_sha256"],
                "four_class_input_sha256": four[key]["input_sha256"],
                "same_sha256": same_sha,
                "n_input_rows": binary[key]["n_input_rows"],
                "same_row_count": same_rows,
            }
        )
        if not (same_sha and same_rows):
            mismatches.append(key)

    report = {
        "status": "PASS" if not mismatches and common else "FAIL",
        "binary_analysis_sets": len(binary),
        "four_class_route_a_sets": len(four),
        "common_sets": len(common),
        "four_class_only_sets": sorted(set(four) - set(binary)),
        "binary_only_sets_not_yet_run": sorted(set(binary) - set(four)),
        "mismatches": mismatches,
        "sets": rows,
    }
    out_path = FOUR_CLASS_ROOT / "four_class_reuses_binary_sets.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sets"}, ensure_ascii=False, indent=2))
    for row in rows:
        print(
            f"  {row['analysis_set_id']:52s} sha相同={row['same_sha256']} "
            f"行数={row['n_input_rows']} 行数相同={row['same_row_count']}"
        )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
