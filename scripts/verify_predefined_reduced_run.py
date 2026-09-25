"""Verify a completed predefined-reduced nested LOSO run against the frozen full archive."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = [
    "participant_group_id",
    "session_id",
    "block_id",
    "probe_event_id",
]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(run_dir: Path, frozen_run_dir: Path, *, tolerance: float = 1e-12) -> dict[str, object]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    audits = json.loads((run_dir / "fold_audits.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(run_dir / "probe_predictions.csv")
    failures = pd.read_csv(run_dir / "failures.csv")
    old_predictions = pd.read_csv(frozen_run_dir / "probe_predictions.csv")

    benchmark_id = str(manifest["predefined_reduced_summary"]["benchmark_model_id"])
    selected_id = str(manifest["predefined_reduced_summary"]["nested_selected_model_id"])
    old_full = old_predictions.loc[old_predictions["model_id"].eq("full")].copy()
    new_full = predictions.loc[predictions["model_id"].eq(benchmark_id)].copy()
    new_selected = predictions.loc[predictions["model_id"].eq(selected_id)].copy()
    merged = old_full.merge(
        new_full,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
        validate="one_to_one",
    )
    key_match = bool(merged["_merge"].eq("both").all())
    probability_difference = np.abs(
        pd.to_numeric(merged["p_q1_equals_1_old"], errors="coerce")
        - pd.to_numeric(merged["p_q1_equals_1_new"], errors="coerce")
    )
    max_probability_difference = float(probability_difference.max())

    leakage_failures: list[str] = []
    selected_candidates: set[str] = set()
    for audit in audits:
        held_out = str(audit["outer_fold_group"])
        outer_train = {str(value) for value in audit["outer_train_group_ids"]}
        outer_test = {str(value) for value in audit["outer_test_group_ids"]}
        if held_out in outer_train or outer_test != {held_out}:
            leakage_failures.append(f"outer:{audit['model_id']}:{held_out}")
        selection = audit.get("selection") or {}
        feature_scheme = selection.get("feature_scheme") or {}
        if str(audit["model_id"]) == selected_id:
            selected_candidates.add(str(feature_scheme.get("feature_set_id", "")))
        for inner in selection.get("inner_fold_audits", []):
            train_groups = {str(value) for value in inner["train_group_ids"]}
            valid_groups = {str(value) for value in inner["validation_group_ids"]}
            if train_groups & valid_groups or held_out in train_groups or held_out in valid_groups:
                leakage_failures.append(
                    f"inner:{audit['model_id']}:{held_out}:{inner.get('inner_fold')}"
                )

    expected_candidates = {"full_11", "behavior_core", "behavior_core_plus_ocular"}
    checks = {
        "manifest_complete": manifest.get("status") == "complete",
        "summary_pass": manifest["predefined_reduced_summary"].get("status") == "PASS",
        "row_count": len(predictions) == 3550,
        "model_row_counts": len(new_full) == 1775 and len(new_selected) == 1775,
        "participant_count": int(predictions["participant_group_id"].nunique()) == 60,
        "failure_table_empty": failures.empty,
        "benchmark_keys_match_frozen": key_match,
        "benchmark_probabilities_match_frozen": max_probability_difference <= tolerance,
        "outer_and_inner_participant_disjoint": not leakage_failures,
        "selected_candidates_declared_only": selected_candidates.issubset(expected_candidates),
        "all_outer_folds_audited": len(audits) == 120,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "status": status,
        "checks": checks,
        "tolerance": tolerance,
        "benchmark_max_absolute_probability_difference": max_probability_difference,
        "benchmark_matched_probe_rows": int(merged["_merge"].eq("both").sum()),
        "new_prediction_rows": int(len(predictions)),
        "new_participant_groups": int(predictions["participant_group_id"].nunique()),
        "fold_audit_rows": int(len(audits)),
        "failure_rows": int(len(failures)),
        "observed_selected_candidates": sorted(selected_candidates),
        "leakage_failures": leakage_failures,
        "frozen_run_dir": str(frozen_run_dir),
        "run_dir": str(run_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--frozen-run-dir", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    report = verify(
        run_dir,
        Path(args.frozen_run_dir).expanduser().resolve(),
        tolerance=args.tolerance,
    )
    report_path = run_dir / "predefined_reduced_verification.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["predefined_reduced_verification"] = report_path.name
    manifest["predefined_reduced_verification"] = report
    manifest["predefined_reduced_output_sha256"][report_path.name] = _sha256(report_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

