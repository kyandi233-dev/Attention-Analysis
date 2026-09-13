"""Build the per-window Behavior analysis sets for the 1.16.25 window-length sensitivity.

Method authority: 分析设计/1.16.25-首轮主窗口长度敏感性预注册_20260913.md

Contract (must not be changed without a new preregistration):
- The ONLY permitted changed variable is the probe-preceding window length (10 s / 20 s / 30 s).
- The 30 s set is NOT rebuilt; the frozen AS.behavior_reference is reused verbatim.
- Membership rule for a window set: all five frozen Behavior features are finite in that window.
- Every metadata column (comparison_models, required_features, required_feature_records,
  feature_identity_mode, required_outcomes, membership_type, keys, labels) is copied verbatim
  from the frozen 30 s table; only analysis_set_id and the five feature columns differ.
- block_id must be lower-cased before joining the producer sensitivity table (B1/B2 vs b1/b2).

Verification anchors asserted by this script:
1. every 10 s / 20 s row is present in the frozen 30 s table (the window sets are strict subsets);
2. the 30 s slice of the sensitivity table reproduces the frozen feature values exactly.

Usage:
    python build_window_sensitivity_sets.py [--write]
Without --write the script only reports; with --write it emits the CSVs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --- frozen paths (read-only inputs) -------------------------------------------------
BASE = Path(r"D:\Project\厚粲杯\11_数据\_FormalAnalysis")
FROZEN_SET = BASE / "SupervisedRunsV1" / "inputs" / "AS.behavior_reference.csv"
SENSITIVITY = BASE / "Behavior" / "formal_v3" / "probe_window_sensitivity.csv"
FROZEN_PRIMARY = BASE / "Behavior" / "formal_v3" / "probe_primary_30s.csv"
OUT_ROOT = BASE / "SupervisedRunsWindowSensitivityV1" / "inputs"

# --- frozen constants ----------------------------------------------------------------
BEHAVIOR_FEATURES = (
    "go_correct_rt_median_ms",
    "go_correct_rt_cv",
    "go_correct_rt_theilsen_slope_ms_per_s",
    "raw_go_omission_rate",
    "commission_rate",
)
JOIN_KEYS = ("session_id", "block_id", "probe_event_id")
MEMBER_RULE = "all_five_behavior_features_finite"
WINDOWS_TO_BUILD = (10, 20)
ANCHOR_WINDOW = 30
MISSINGNESS_AWARE = "included_missing_aware"


def _normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Lower-case block_id and stringify the join keys so the two sources can align."""
    out = frame.copy()
    out["block_id"] = out["block_id"].astype(str).str.lower()
    for key in JOIN_KEYS:
        out[key] = out[key].astype(str)
    return out


def _finite_mask(frame: pd.DataFrame) -> pd.Series:
    numeric = frame[list(BEHAVIOR_FEATURES)].apply(pd.to_numeric, errors="coerce")
    return numeric.notna().all(axis=1)


def build(write: bool) -> dict[str, object]:
    frozen = _normalise_keys(pd.read_csv(FROZEN_SET, low_memory=False))
    sensitivity = _normalise_keys(pd.read_csv(SENSITIVITY, low_memory=False))

    report: dict[str, object] = {
        "method_authority": "分析设计/1.16.25-首轮主窗口长度敏感性预注册_20260913.md",
        "membership_rule": MEMBER_RULE,
        "frozen_set": str(FROZEN_SET),
        "frozen_set_rows": int(len(frozen)),
        "sensitivity_source": str(SENSITIVITY),
        "join_keys": list(JOIN_KEYS),
        "key_normalisation": "block_id lower-cased; session_id/block_id/probe_event_id stringified",
        "windows": {},
        "anchors": {},
    }

    frozen_keys = set(map(tuple, frozen[list(JOIN_KEYS)].to_numpy()))

    # ---- anchor 1: the 30 s slice reproduces the frozen feature values exactly ----------
    anchor = sensitivity[sensitivity["window_seconds_nominal"] == ANCHOR_WINDOW]
    anchor_finite = anchor[_finite_mask(anchor)]
    anchor_merged = frozen[list(JOIN_KEYS) + list(BEHAVIOR_FEATURES)].merge(
        anchor_finite[list(JOIN_KEYS) + list(BEHAVIOR_FEATURES)],
        on=list(JOIN_KEYS),
        suffixes=("_frozen", "_anchor"),
        how="inner",
    )
    anchor_max_abs_diff = {
        feature: float(
            (
                pd.to_numeric(anchor_merged[f"{feature}_frozen"], errors="coerce")
                - pd.to_numeric(anchor_merged[f"{feature}_anchor"], errors="coerce")
            )
            .abs()
            .max()
        )
        for feature in BEHAVIOR_FEATURES
    }
    report["anchors"]["anchor_30s_feature_max_abs_diff"] = anchor_max_abs_diff
    report["anchors"]["anchor_30s_rows_matched"] = int(len(anchor_merged))
    report["anchors"]["anchor_30s_membership_equals_frozen_keyset"] = bool(
        set(map(tuple, anchor_finite[list(JOIN_KEYS)].to_numpy())) == frozen_keys
    )

    # ---- primary table cross-check ------------------------------------------------------
    primary = _normalise_keys(pd.read_csv(FROZEN_PRIMARY, low_memory=False))
    primary_merged = primary[list(JOIN_KEYS) + list(BEHAVIOR_FEATURES)].merge(
        anchor_finite[list(JOIN_KEYS) + list(BEHAVIOR_FEATURES)],
        on=list(JOIN_KEYS),
        suffixes=("_primary", "_anchor"),
        how="inner",
    )
    report["anchors"]["anchor_30s_vs_frozen_primary_rows"] = int(len(primary_merged))
    report["anchors"]["anchor_30s_vs_frozen_primary_max_abs_diff"] = {
        feature: float(
            (
                pd.to_numeric(primary_merged[f"{feature}_primary"], errors="coerce")
                - pd.to_numeric(primary_merged[f"{feature}_anchor"], errors="coerce")
            )
            .abs()
            .max()
        )
        for feature in BEHAVIOR_FEATURES
    }

    # ---- build each window set ----------------------------------------------------------
    metadata_columns = [
        column for column in frozen.columns if column not in set(BEHAVIOR_FEATURES)
    ]
    if write:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for window in WINDOWS_TO_BUILD:
        slice_ = sensitivity[sensitivity["window_seconds_nominal"] == window]
        slice_keys = set(map(tuple, slice_[list(JOIN_KEYS)].to_numpy()))
        subset_ok = slice_keys <= frozen_keys

        # Apply the preregistered membership rule to the window slice FIRST. A raw window
        # row that is not estimable at this window must not be forced to borrow metadata
        # from the frozen 30 s table, and a raw row absent from the frozen table is only a
        # problem if it would have passed this window's own membership rule.
        slice_member = slice_[_finite_mask(slice_)].copy()
        member = frozen[metadata_columns].merge(
            slice_member[list(JOIN_KEYS) + list(BEHAVIOR_FEATURES)],
            on=list(JOIN_KEYS),
            how="inner",
            validate="one_to_one",
        )
        if len(member) != len(slice_member):
            raise AssertionError(
                f"window={window}: metadata join kept {len(member)} of {len(slice_member)} "
                "membership-passing rows; those rows cannot be given frozen analysis-set metadata"
            )
        extra_vs_frozen = set(map(tuple, slice_member[list(JOIN_KEYS)].to_numpy())) - frozen_keys
        if extra_vs_frozen:
            raise AssertionError(
                f"window={window}: {len(extra_vs_frozen)} membership-passing rows are absent "
                "from the frozen 30 s table; the window set would not be a subset"
            )
        set_id = f"AS.behavior_reference__window{window}s"
        member["analysis_set_id"] = set_id
        member["membership_type"] = MISSINGNESS_AWARE
        q1 = member["q1_nominal_4class"].value_counts().sort_index()

        entry: dict[str, object] = {
            "analysis_set_id": set_id,
            "window_seconds": window,
            "rows_before_membership": int(len(slice_)),
            "rows_raw_absent_from_frozen_30s": int(len(slice_keys - frozen_keys)),
            "rows_after_membership": int(len(member)),
            "participant_groups": int(member["participant_group_id"].nunique()),
            "sessions": int(member["session_id"].nunique()),
            "subset_of_frozen_30s": bool(
                set(map(tuple, slice_member[list(JOIN_KEYS)].to_numpy())) <= frozen_keys
            ),
            "q1_nominal_4class_counts": {str(int(k)): int(v) for k, v in q1.items()},
            "commission_rate_finite": int(
                pd.to_numeric(slice_["commission_rate"], errors="coerce").notna().sum()
            ),
        }
        if write:
            out_path = OUT_ROOT / f"{set_id}.csv"
            member.to_csv(out_path, index=False, encoding="utf-8-sig")
            entry["written"] = str(out_path)
        report["windows"][set_id] = entry

    if not report["anchors"]["anchor_30s_membership_equals_frozen_keyset"]:
        raise AssertionError("30 s reconstruction does not reproduce the frozen analysis-set key set")
    for feature, diff in report["anchors"]["anchor_30s_feature_max_abs_diff"].items():
        if diff > 1e-9:
            raise AssertionError(f"30 s anchor mismatch on {feature}: max|diff|={diff}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Emit the per-window analysis-set CSVs.")
    args = parser.parse_args()
    report = build(write=bool(args.write))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
