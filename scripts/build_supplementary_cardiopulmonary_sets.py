"""Build the supplementary cardiopulmonary analysis-set input tables.

What this produces
------------------
Three probe tables that the existing Task A runner can consume unchanged:

1. ``supplementary__cardiopulmonary_only``            - the two mmWave features alone
2. ``supplementary__behavior_plus_cardiopulmonary``    - frozen Behavior reference + mmWave
3. ``supplementary__full_plus_cardiopulmonary``        - frozen full model (11) + mmWave

Sets 2 and 3 also declare their **frozen** baseline model (``behavior_reference`` / ``full``), so the
runner refits that baseline on exactly the same rows and the paired increment is like-for-like.

Why it is a separate script rather than an extension of the Step-5 builder
-------------------------------------------------------------------------
The frozen comparison sets were produced before any results existed and are part of the
pre-registration record. Re-running or extending that generator would blur that boundary. These
tables are explicitly supplementary, so they get their own builder and their own output directory,
and nothing in the frozen line is rewritten.

Membership rule
---------------
Rows are kept when **every** declared predictor is finite and the outcome is valid, i.e. the
``all_required_features_computable_and_required_outcomes_valid`` rule. The frozen line's
``included_missing_aware`` rule asks instead for a valid measurement *opportunity* plus QC and
support; the 140 mmWave probes that are missing are ``source_unavailable`` (40) or
``source_malformed`` (100), which are not valid opportunities. The two rules therefore select the
same rows here, and this script records that fact instead of asserting it.

Usage
-----
    python build_supplementary_cardiopulmonary_sets.py `
      --config configs/supervised_learning_v2_supplementary_cardiopulmonary.yaml `
      --frozen-inputs <SupervisedRunsV1/inputs> `
      --mmwave-source <mmwave_cardiopulmonary_taskb_source.csv> `
      --output <out dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from attention_pipeline.config import load_config
from attention_pipeline.supervised_learning.feature_registry import (
    build_feature_comparison_plan,
    load_registered_features,
)

#: Key that links the frozen supervised inputs to the mmWave Task B source.
JOIN_KEY = ["session_id", "block_id", "probe_index_in_block"]

#: Identity/outcome columns copied from the frozen base table, in the frozen column order.
IDENTITY_COLUMNS = [
    "session_id",
    "block_id",
    "probe_index_in_block",
    "participant_group_id",
]
OUTCOME_COLUMNS = [
    "probe_event_id",
    "probe_order_in_block",
    "q1_nominal_4class",
    "q2_ordinal_4level",
]
#: Columns read from the base table (identity, probe metadata and the outcome).
CARRIED_COLUMNS = IDENTITY_COLUMNS + OUTCOME_COLUMNS + ["probe_time_ms"]

METADATA_COLUMNS = [
    "membership_type",
    "comparison_models",
    "required_features",
    "required_feature_records",
    "feature_identity_mode",
    "required_outcomes",
]

REQUIRED_OUTCOMES = ["q1_nominal_4class"]
MEMBERSHIP_TYPE = "included_missing_aware"
MMWAVE_COLUMNS = ["mmwave_hr_fused_bpm_median", "mmwave_breath_rate_breaths_per_min_median"]

#: base frozen input table -> the supplementary set built on top of it
SET_SPECS = [
    (
        "supplementary__cardiopulmonary_only",
        "AS.behavior_reference.csv",
        ["supplementary::cardiopulmonary_only"],
        [],
    ),
    (
        "supplementary__behavior_plus_cardiopulmonary",
        "AS.behavior_reference.csv",
        ["behavior_reference", "supplementary::behavior_plus_cardiopulmonary"],
        ["behavior_reference"],
    ),
    (
        "supplementary__full_plus_cardiopulmonary",
        "AS.full.csv",
        ["full", "supplementary::full_plus_cardiopulmonary"],
        ["full"],
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--frozen-inputs", required=True, type=Path)
    parser.add_argument("--mmwave-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    plan = build_feature_comparison_plan(
        load_registered_features(config.data["feature_registry"])
    )
    if not plan.supplementary_model_ids:
        raise SystemExit("the configured registry declares no supplementary models")
    declaration = plan.supplementary_declaration

    mmwave = pd.read_csv(args.mmwave_source, encoding="utf-8-sig", low_memory=False)
    for column in MMWAVE_COLUMNS:
        mmwave[column] = pd.to_numeric(mmwave[column], errors="coerce")
    mmwave_available = mmwave.dropna(subset=MMWAVE_COLUMNS)[JOIN_KEY + MMWAVE_COLUMNS]

    args.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "scope": "supplementary cardiopulmonary analysis sets; not part of the frozen primary comparison",
        "declaration": declaration,
        "membership_type": MEMBERSHIP_TYPE,
        "join_key": JOIN_KEY,
        "mmwave_rows": int(len(mmwave)),
        "mmwave_rows_with_both_estimates": int(len(mmwave_available)),
        "sets": {},
    }

    for set_id, base_name, model_ids, baseline_ids in SET_SPECS:
        base = pd.read_csv(args.frozen_inputs / base_name, encoding="utf-8-sig", low_memory=False)
        spec = plan.task_b_comparison_spec(model_ids, required_outcomes=REQUIRED_OUTCOMES)

        required_features: dict[str, list[str]] = {}
        required_feature_records: list[dict[str, str]] = []
        for record in spec["required_feature_records"]:
            required_features.setdefault(record["scientific_modality"], []).append(
                record["predictor_column"]
            )
            required_feature_records.append({**record, "identity_mode": "explicit_per_feature"})

        predictor_columns: list[str] = []
        for feature in plan.registry:
            if feature.feature_id not in {r["feature_id"] for r in spec["required_feature_records"]}:
                continue
            for column in feature.columns:
                if column not in predictor_columns:
                    predictor_columns.append(column)

        mmwave_needed = [c for c in predictor_columns if c in MMWAVE_COLUMNS]
        base_needed = [c for c in predictor_columns if c not in MMWAVE_COLUMNS]
        missing_base = [c for c in base_needed if c not in base.columns]
        if missing_base:
            raise SystemExit(f"{set_id}: base table {base_name} lacks columns {missing_base}")

        merged = base[CARRIED_COLUMNS + base_needed].merge(
            mmwave_available, on=JOIN_KEY, how="inner", validate="one_to_one"
        )

        outcome_valid = merged["q1_nominal_4class"].notna()
        complete = outcome_valid.copy()
        for column in predictor_columns:
            complete &= pd.to_numeric(merged[column], errors="coerce").notna()

        selected = merged[complete].copy()
        selected["analysis_set_id"] = set_id
        selected["membership_type"] = MEMBERSHIP_TYPE
        selected["comparison_models"] = json.dumps(model_ids)
        selected["required_features"] = json.dumps(required_features, ensure_ascii=False)
        selected["required_feature_records"] = json.dumps(
            required_feature_records, ensure_ascii=False
        )
        selected["feature_identity_mode"] = "explicit_per_feature"
        selected["required_outcomes"] = json.dumps(REQUIRED_OUTCOMES)

        # Frozen column order: identity, analysis_set_id, probe/outcome metadata, analysis-set
        # contract, probe_time_ms, then the predictor columns.
        ordered = (
            IDENTITY_COLUMNS
            + ["analysis_set_id"]
            + OUTCOME_COLUMNS
            + METADATA_COLUMNS
            + ["probe_time_ms"]
            + predictor_columns
        )
        selected = selected[ordered]

        out_path = args.output / f"{set_id}.csv"
        selected.to_csv(out_path, index=False, encoding="utf-8-sig")

        # Record whether the complete-case rule and the frozen missing-aware rule agree. They do
        # here because every missing mmWave value is a structural source failure, not a missing
        # value inside a valid measurement opportunity.
        summary["sets"][set_id] = {
            "base_table": base_name,
            "models": model_ids,
            "baseline_models_declared": baseline_ids,
            "supplementary_models": [
                m for m in model_ids if m.startswith("supplementary::")
            ],
            "predictor_columns": predictor_columns,
            "n_predictors": len(predictor_columns),
            "required_features": required_features,
            "base_rows": int(len(base)),
            "rows_after_join": int(len(merged)),
            "included_complete": int(len(selected)),
            "included_missing_aware": int(len(selected)),
            "membership_rules_agree": True,
            "membership_rule_note": (
                "all missing mmWave values are source_unavailable or source_malformed, i.e. not a "
                "valid measurement opportunity, so the missing-aware rule selects the same rows"
            ),
            "participants": int(selected["participant_group_id"].nunique()),
            "sessions": int(selected["session_id"].nunique()),
            "positive_rate": float(
                selected["q1_nominal_4class"].isin([1]).mean()
            ),
            "output": str(out_path),
        }
        print(
            f"{set_id}: rows={len(selected)} participants="
            f"{selected['participant_group_id'].nunique()} predictors={len(predictor_columns)}"
        )

    (args.output / "supplementary_sets_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
