"""Build an M1-lineage mmWave Task B source table for the supplementary comparison.

Why a new source table instead of reusing the endpoint guard's
----------------------------------------------------------------
The three supplementary cardiopulmonary analysis sets were first built on
``mmwave_cardiopulmonary_taskb_source.csv`` from the ``e352dce`` endpoint guard. That file's
``producer_commit`` is ``16729b2`` — the snapshot-v1 corrected replay — and **that commit does not
exist in the producer repository any more** (verified: ``git cat-file -t 16729b2`` fails against a
fresh clone of ``greenboo26/focuswave-multimodal-attention-analysis``). A source whose commit cannot
be retrieved can never satisfy the provenance item of ``分析设计/1.15.9`` §3.

The M1 producer-contract line (``01da845e``, carried on
``codex/mmwave-producer-contract-provenance-m1-20260913``) *does* satisfy it: this script's
companion check re-hashes the four files the run manifest recorded and they match, and the branch
repairs all three contract items. So the comparison is rebuilt on M1.

Two identity bridges are required and both are verified here rather than assumed
-------------------------------------------------------------------------------------
1. ``block_id``: M1 writes ``block-1`` / ``block-2`` while the downstream probe universe writes
   ``b1`` / ``b2``. M1 carries ``legacy_block_num``, which is the bridge.
2. ``repeat_participant_id`` -> ``participant_group_id``: M1 is producer-side and has 62
   participants; the downstream governed identity has 61. The bridge is taken from an artefact that
   carries both, and it is checked to be producer-1:1 and session-consistent.

Usage
-----
    python build_m1_cardiopulmonary_taskb_source.py `
      --m1-dir <mmwave_producer_contract_m1_01da845e_20260913> `
      --identity-bridge <mmwave_cardiopulmonary_taskb_source.csv> `
      --base-table <SupervisedRunsV1/inputs/AS.behavior_reference.csv> `
      --output <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

JOIN_KEY = ["session_id", "block_id", "probe_index_in_block"]
MMWAVE_COLUMNS = ["mmwave_hr_fused_bpm_median", "mmwave_breath_rate_breaths_per_min_median"]

OUTPUT_COLUMNS = [
    "session_id",
    "block_id",
    "probe_index_in_block",
    "participant_group_id",
    "repeat_participant_id",
    "probe_id",
    "mmwave_hr_fused_bpm_median",
    "mmwave_breath_rate_breaths_per_min_median",
    "mmwave_state",
    "mmwave_missing_reason",
    "window_name",
    "window_effective_start_unix_ms",
    "window_end_unix_ms",
    "probe_onset_unix_ms",
    "mmwave_hr_usable_window_fraction",
    "mmwave_source_commit",
    "mmwave_source_run_id",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", required=True, type=Path)
    parser.add_argument("--identity-bridge", required=True, type=Path)
    parser.add_argument("--base-table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frames = []
    for name in ("mmwave_probe_merge_ready.csv", "mmwave_probe_merge_ready_E.csv"):
        frame = pd.read_csv(args.m1_dir / name, encoding="utf-8-sig", low_memory=False)
        frame["__batch"] = name
        frames.append(frame)
    m1 = pd.concat(frames, ignore_index=True)

    report: dict[str, object] = {
        "scope": "build an M1-lineage mmWave Task B source table; no model is trained",
        "m1_rows": int(len(m1)),
        "m1_sessions": int(m1["session_id"].nunique()),
        "m1_producer_participants": int(m1["repeat_participant_id"].nunique()),
        "m1_hr_finite": int(m1[MMWAVE_COLUMNS[0]].notna().sum()),
        "m1_br_finite": int(m1[MMWAVE_COLUMNS[1]].notna().sum()),
    }

    # ---- bridge 1: block identifier ---------------------------------------------------------
    if m1["block_id"].str.contains("-").all():
        if "legacy_block_num" not in m1.columns:
            raise SystemExit("M1 uses hyphenated block ids but carries no legacy_block_num bridge")
        block_pairs = (
            m1[["block_id", "legacy_block_num"]].drop_duplicates().sort_values("legacy_block_num")
        )
        if len(block_pairs) != 2:
            raise SystemExit(f"expected exactly two M1 blocks, got {len(block_pairs)}")
        m1["block_id"] = "b" + block_pairs.set_index("block_id").loc[
            m1["block_id"], "legacy_block_num"
        ].to_numpy().astype(str)
        report["block_bridge"] = block_pairs.to_dict("records")
    else:
        report["block_bridge"] = "not required (M1 already uses the downstream convention)"

    # ---- bridge 2: participant identity -----------------------------------------------------
    bridge_frame = pd.read_csv(args.identity_bridge, encoding="utf-8-sig", low_memory=False)
    bridge = bridge_frame[["repeat_participant_id", "participant_group_id"]].drop_duplicates()
    per_producer = bridge.groupby("repeat_participant_id")["participant_group_id"].nunique()
    if int((per_producer > 1).sum()):
        raise SystemExit("the identity bridge maps a producer id to more than one governed group")
    mapping = bridge.set_index("repeat_participant_id")["participant_group_id"]
    missing = sorted(set(m1["repeat_participant_id"]) - set(mapping.index))
    if missing:
        raise SystemExit(f"M1 producer participants absent from the identity bridge: {missing[:5]}")
    m1["participant_group_id"] = m1["repeat_participant_id"].map(mapping)

    # Session-level consistency: one session must not carry two producer ids, and the bridge must
    # agree with the guard's own session->producer assignment.
    guard_sessions = (
        bridge_frame[["session_id", "repeat_participant_id"]].drop_duplicates()
    ).set_index("session_id")["repeat_participant_id"]
    m1_sessions = m1[["session_id", "repeat_participant_id"]].drop_duplicates()
    m1_sessions = m1_sessions.groupby("session_id")["repeat_participant_id"].nunique()
    report["sessions_with_multiple_producer_ids"] = int((m1_sessions > 1).sum())
    shared = sorted(set(m1["session_id"]) & set(guard_sessions.index))
    disagreements = [
        session
        for session in shared
        if m1.loc[m1["session_id"] == session, "repeat_participant_id"].iloc[0]
        != guard_sessions.loc[session]
    ]
    report["session_identity_disagreements"] = disagreements
    report["participant_group_n"] = int(m1["participant_group_id"].nunique())

    # ---- coverage against the downstream outcome-valid universe ------------------------------
    base = pd.read_csv(args.base_table, encoding="utf-8-sig", low_memory=False)
    base_keys = base[JOIN_KEY].drop_duplicates()
    merged = m1.merge(base_keys, on=JOIN_KEY, how="inner", validate="one_to_one")
    report["base_rows"] = int(len(base))
    report["m1_rows_joined_to_base"] = int(len(merged))
    report["m1_hr_finite_after_join"] = int(merged[MMWAVE_COLUMNS[0]].notna().sum())
    report["m1_br_finite_after_join"] = int(merged[MMWAVE_COLUMNS[1]].notna().sum())
    report["participant_groups_after_join"] = int(
        merged.loc[merged[MMWAVE_COLUMNS[0]].notna(), "participant_group_id"].nunique()
    )

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "m1_cardiopulmonary_taskb_source.csv"
    merged[[c for c in OUTPUT_COLUMNS if c in merged.columns]].to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )
    report["output"] = str(out_path)
    (args.output / "m1_source_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
