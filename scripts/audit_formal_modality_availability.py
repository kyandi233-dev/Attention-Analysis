"""Audit session-level merge readiness without creating a second identity master table.

The canonical cohort remains authoritative. This script attaches Behavior,
NIR and RGB availability as a sidecar, validates participant-key parity, and
writes a compact machine-readable readiness report.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from attention_pipeline.behavior_formal.extract import discover_subjects
from attention_pipeline.config import load_config
from attention_pipeline.formal_analysis.behavior_adapter import prepare_behavior_runtime_config
from attention_pipeline.formal_analysis.cohort import included_cohort, summarize_cohort
from attention_pipeline.nir_analysis_ready.pupil_only import load_source_manifest


def _atomic_text(path: Path, text: str, *, overwrite: bool) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _canonical_sessions(values: pd.Series) -> pd.Series:
    sessions = values.astype("string").str.strip().str.rstrip("_")
    invalid = ~sessions.str.match(r"^sub-\d+$", na=False)
    if invalid.any():
        raise ValueError(f"invalid session_id values: {sessions.loc[invalid].tolist()}")
    return sessions.map(lambda value: f"sub-{int(str(value).split('-')[1]):03d}")


def audit(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    behavior_config = load_config(args.behavior_config, paths_config=args.paths_config)
    behavior_runtime, cohort = prepare_behavior_runtime_config(behavior_config)
    governed = included_cohort(cohort, require_groups=True)[
        ["session_id", "repeat_participant_id"]
    ].copy()
    governed = governed.rename(columns={"repeat_participant_id": "participant_group_id"})
    if governed["session_id"].duplicated().any():
        raise ValueError("canonical cohort contains duplicate session_id")

    behavior_sessions = set(discover_subjects(behavior_runtime))
    unknown_behavior = behavior_sessions - set(governed["session_id"])
    if unknown_behavior:
        raise ValueError(f"Behavior discovery contains sessions outside governed cohort: {sorted(unknown_behavior)}")

    nir_config = load_config(args.nir_config, paths_config=args.paths_config)
    nir_payload, nir_records = load_source_manifest(nir_config)
    nir = pd.DataFrame(nir_records)
    if nir.empty:
        raise ValueError("NIR source manifest has no available sessions")
    nir["session_id"] = _canonical_sessions(nir["session_id"])
    if nir["session_id"].duplicated().any():
        raise ValueError("NIR source manifest contains duplicate session_id")
    if not set(nir["session_id"]).issubset(set(governed["session_id"])):
        raise ValueError("NIR source manifest contains sessions outside governed cohort")
    nir_group = nir.set_index("session_id")["analysis_group_token"].astype("string")
    expected_group = governed.set_index("session_id")["participant_group_id"].astype("string")
    mismatch = nir_group[nir_group.ne(expected_group.reindex(nir_group.index))]
    if len(mismatch):
        raise ValueError(f"NIR participant-group mismatch: {sorted(mismatch.index.tolist())}")

    unavailable_nir = {
        str(row["session_id"]): (str(row["status"]), str(row["reason"]))
        for row in nir_payload.get("unavailable_sessions", [])
    }
    if set(unavailable_nir) | set(nir["session_id"]) != set(governed["session_id"]):
        raise ValueError("NIR available/unavailable partition does not cover the governed cohort exactly")

    rgb = pd.read_csv(args.rgb_source_map, encoding="utf-8-sig")
    required_rgb = {"session_id", "rgb_status"}
    if not required_rgb.issubset(rgb.columns):
        raise ValueError(f"RGB source map lacks columns: {sorted(required_rgb - set(rgb.columns))}")
    rgb["session_id"] = _canonical_sessions(rgb["session_id"])
    if rgb["session_id"].duplicated().any():
        raise ValueError("RGB source map contains duplicate session_id")
    rgb_status = rgb.set_index("session_id")["rgb_status"].astype("string")

    availability = governed.copy()
    availability["behavior_status"] = availability["session_id"].map(
        lambda session: "available" if session in behavior_sessions else "source_missing"
    )
    available_nir = set(nir["session_id"])
    availability["nir_status"] = availability["session_id"].map(
        lambda session: "available" if session in available_nir else unavailable_nir[session][0]
    )
    availability["nir_reason"] = availability["session_id"].map(
        lambda session: "" if session in available_nir else unavailable_nir[session][1]
    )
    availability["rgb_status"] = availability["session_id"].map(rgb_status).fillna("source_missing")
    availability["behavior_available"] = availability["behavior_status"].eq("available")
    availability["nir_available"] = availability["nir_status"].eq("available")
    availability["rgb_available"] = availability["rgb_status"].eq("available")
    availability["behavior_nir_rgb_common_available"] = availability[
        ["behavior_available", "nir_available", "rgb_available"]
    ].all(axis=1)

    cohort_summary = summarize_cohort(cohort)
    counts = {
        "governed_sessions": int(len(availability)),
        "participant_groups": int(availability["participant_group_id"].nunique()),
        "repeated_participant_groups": cohort_summary.repeated_groups,
        "behavior_available_sessions": int(availability["behavior_available"].sum()),
        "nir_available_sessions": int(availability["nir_available"].sum()),
        "rgb_available_sessions": int(availability["rgb_available"].sum()),
        "behavior_nir_rgb_common_available_sessions": int(
            availability["behavior_nir_rgb_common_available"].sum()
        ),
    }
    expected = {
        "governed_sessions": args.expected_sessions,
        "participant_groups": args.expected_groups,
        "behavior_available_sessions": args.expected_behavior_sessions,
        "nir_available_sessions": args.expected_nir_sessions,
        "rgb_available_sessions": args.expected_rgb_sessions,
        "behavior_nir_rgb_common_available_sessions": args.expected_common_sessions,
    }
    differences = {key: (counts[key], value) for key, value in expected.items() if counts[key] != value}
    if differences:
        raise ValueError(f"availability counts differ from expected (actual, expected): {differences}")

    report: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_SESSION_LEVEL_MERGE_READY",
        "scope": "Behavior/NIR/RGB session-level identity and availability contract",
        "canonical_key": ["participant_group_id", "session_id"],
        "cohort_rule": "canonical cohort is authoritative; availability is an attached sidecar",
        "missingness_rule": "explicit missing; never zero-fill and never redefine identity",
        "paired_analysis_rule": "filter an explicit common-available subset only for that analysis",
        "counts": counts,
        "unavailable_sessions": {
            "behavior": availability.loc[~availability["behavior_available"], "session_id"].tolist(),
            "nir": availability.loc[~availability["nir_available"], "session_id"].tolist(),
            "rgb": availability.loc[~availability["rgb_available"], "session_id"].tolist(),
        },
        "limits": [
            "This audit validates session-level membership, identity and availability only.",
            "Trial/probe/window merges must still pass their own unique-key, timestamp and leakage gates.",
            "Fusion remains deferred until the single-modality producer tables are frozen.",
        ],
    }
    return availability, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, required=True)
    parser.add_argument("--behavior-config", type=Path, default=Path("configs/behavior_formal_v2.yaml"))
    parser.add_argument("--nir-config", type=Path, default=Path("configs/nir_analysis_ready.yaml"))
    parser.add_argument("--rgb-source-map", type=Path, required=True)
    parser.add_argument("--availability-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--expected-sessions", type=int, required=True)
    parser.add_argument("--expected-groups", type=int, required=True)
    parser.add_argument("--expected-behavior-sessions", type=int, required=True)
    parser.add_argument("--expected-nir-sessions", type=int, required=True)
    parser.add_argument("--expected-rgb-sessions", type=int, required=True)
    parser.add_argument("--expected-common-sessions", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    availability, report = audit(args)
    _atomic_text(args.availability_output, availability.to_csv(index=False), overwrite=args.overwrite)
    _atomic_text(
        args.report_output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        overwrite=args.overwrite,
    )
    print(json.dumps(report["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
