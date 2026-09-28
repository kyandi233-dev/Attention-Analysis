"""Build the M1 mmWave Task-B source without filtering on behavior availability.

M1 (producer commit 01da845e) replaces the historical snapshot-v1 source lineage.
This adapter only reconciles probe/block/participant identities. It never changes
HR/BR values, windows, QC thresholds, labels, or the estimator.

FW-1313-INPUT-01: --probe-universe must contain the full governed probe key set,
not AS.behavior_reference.csv. Both directions of key-set equality with M1 are
checked. Optional --base-table is retained ONLY to audit the old restriction; it
can no longer select output rows. Model-specific eligibility belongs to Task B.

Example (all paths supplied by the caller)::

    python scripts/build_m1_cardiopulmonary_taskb_source.py \
      --m1-dir <M1 producer directory> \
      --identity-bridge <table with both participant identities> \
      --probe-universe <full governed probe identity table> \
      --base-table <optional historical behavior input> \
      --output <new external output directory>

The output directory must not exist. Original archives are never overwritten.
Formal record: FocusWave-Formal-Analysis/运行记录与证据/
09-28-2-心肺源表筛选修复与真实数据验证.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

JOIN_KEY = ["session_id", "block_id", "probe_index_in_block"]
MMWAVE_COLUMNS = ["mmwave_hr_fused_bpm_median", "mmwave_breath_rate_breaths_per_min_median"]
OUTPUT_COLUMNS = [
    "session_id", "block_id", "probe_index_in_block", "participant_group_id",
    "repeat_participant_id", "probe_id", *MMWAVE_COLUMNS, "mmwave_state",
    "mmwave_missing_reason", "window_name", "window_effective_start_unix_ms",
    "window_end_unix_ms", "probe_onset_unix_ms", "mmwave_hr_usable_window_fraction",
    "mmwave_source_commit", "mmwave_source_run_id",
]


def _require(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    for column in columns:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name}: null/blank identity in {column}")


def _keys(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Normalize key aliases only; reject duplicates rather than dropping them."""
    out = frame.copy()
    if "probe_index_in_block" not in out and "probe_order_in_block" in out:
        out["probe_index_in_block"] = out["probe_order_in_block"]
    _require(out, JOIN_KEY, name)
    if "probe_order_in_block" in out:
        left = pd.to_numeric(out["probe_index_in_block"], errors="raise")
        right = pd.to_numeric(out["probe_order_in_block"], errors="raise")
        if not left.eq(right).all():
            raise ValueError(f"{name}: probe-index aliases disagree")
    out["session_id"] = out["session_id"].astype(str).str.strip()
    block = out["block_id"].astype(str).str.strip().str.lower()
    block = block.str.replace(r"^block-?", "b", regex=True)
    block = block.where(block.str.startswith("b"), "b" + block)
    if not block.str.fullmatch(r"b[1-9][0-9]*").all():
        raise ValueError(f"{name}: invalid block identifier")
    if "legacy_block_num" in out:
        legacy = pd.to_numeric(out["legacy_block_num"], errors="raise")
        if not legacy.eq(pd.to_numeric(block.str[1:])).all():
            raise ValueError(f"{name}: block and legacy_block_num disagree")
    out["block_id"] = block
    index = pd.to_numeric(out["probe_index_in_block"], errors="raise")
    if not (np.isfinite(index).all() and index.eq(np.floor(index)).all() and index.gt(0).all()):
        raise ValueError(f"{name}: probe indices must be positive finite integers")
    out["probe_index_in_block"] = index.astype(int)
    if out.duplicated(JOIN_KEY).any():
        raise ValueError(f"{name}: duplicate probe keys")
    return out


def _key_set(frame: pd.DataFrame) -> set[tuple]:
    return set(frame[JOIN_KEY].itertuples(index=False, name=None))


def _finite(frame: pd.DataFrame, column: str) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="raise")
    return pd.Series(np.isfinite(numeric), index=frame.index)


def build_source(
    m1: pd.DataFrame,
    bridge_frame: pd.DataFrame,
    probe_universe: pd.DataFrame,
    legacy_base: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Preserve the producer universe and values; validate the governed identities."""
    m1 = _keys(m1, "M1")
    universe = _keys(probe_universe, "probe universe")
    if m1.empty or universe.empty:
        raise ValueError("M1 and probe universe must be non-empty")
    _require(m1, ["repeat_participant_id"], "M1")
    _require(bridge_frame, ["session_id", "repeat_participant_id", "participant_group_id"], "bridge")
    _require(universe, ["participant_group_id"], "probe universe")
    missing = sorted(set(MMWAVE_COLUMNS) - set(m1.columns))
    if missing:
        raise ValueError(f"M1: missing measurement columns {missing}")
    m1 = m1.copy()
    bridge_frame = bridge_frame.copy()
    universe = universe.copy()
    for frame in (m1, bridge_frame, universe):
        for column in ("session_id", "repeat_participant_id", "participant_group_id"):
            if column in frame:
                frame[column] = frame[column].astype(str).str.strip()
    if universe.groupby("session_id")["participant_group_id"].nunique().gt(1).any():
        raise ValueError("probe universe: session maps to multiple participant groups")
    bridge = bridge_frame[["repeat_participant_id", "participant_group_id"]].drop_duplicates()
    if bridge.groupby("repeat_participant_id")["participant_group_id"].nunique().gt(1).any():
        raise ValueError("bridge maps a producer id to multiple governed groups")
    mapping = bridge.set_index("repeat_participant_id")["participant_group_id"]
    if set(m1["repeat_participant_id"]) - set(mapping.index):
        raise ValueError("M1 has producer participants absent from the identity bridge")
    mapped = m1["repeat_participant_id"].map(mapping)
    if "participant_group_id" in m1 and not m1["participant_group_id"].eq(mapped).all():
        raise ValueError("M1 participant_group_id conflicts with the explicit identity bridge")
    m1["participant_group_id"] = mapped

    # These checks previously appeared only in the report; they now stop the write.
    for name, frame in (("M1", m1), ("bridge", bridge_frame)):
        if frame.groupby("session_id")["repeat_participant_id"].nunique().gt(1).any():
            raise ValueError(f"{name}: session maps to multiple producer participants")
    guard = bridge_frame[["session_id", "repeat_participant_id"]].drop_duplicates().set_index("session_id")
    sessions = m1[["session_id", "repeat_participant_id"]].drop_duplicates().set_index("session_id")
    if set(sessions.index) - set(guard.index):
        raise ValueError("M1 session absent from identity bridge")
    if not sessions["repeat_participant_id"].eq(guard.loc[sessions.index, "repeat_participant_id"]).all():
        raise ValueError("M1 session identity disagrees with bridge")
    m1_keys, universe_keys = _key_set(m1), _key_set(universe)
    if m1_keys != universe_keys:
        raise ValueError(
            "full probe universe mismatch: "
            f"missing_from_universe={len(m1_keys - universe_keys)}, "
            f"extra_in_universe={len(universe_keys - m1_keys)}; "
            "do not use an already-filtered behavior input as --probe-universe"
        )
    check = m1[JOIN_KEY + ["participant_group_id"]].merge(
        universe[JOIN_KEY + ["participant_group_id"]], on=JOIN_KEY,
        suffixes=("_m1", "_universe"), validate="one_to_one",
    )
    if not check["participant_group_id_m1"].eq(check["participant_group_id_universe"]).all():
        raise ValueError("governed participant identity disagrees with probe universe")

    report = {
        "scope": "M1 source identity assembly; no model fitting or physiological validation",
        "source_schema": "m1-taskb-universe-preserving-v2",
        "m1_rows": len(m1), "m1_sessions": int(m1["session_id"].nunique()),
        "m1_producer_participants": int(m1["repeat_participant_id"].nunique()),
        "participant_group_n": int(m1["participant_group_id"].nunique()),
        "probe_universe_rows": len(universe), "output_rows": len(m1),
        "m1_hr_finite": int(_finite(m1, MMWAVE_COLUMNS[0]).sum()),
        "m1_br_finite": int(_finite(m1, MMWAVE_COLUMNS[1]).sum()),
        "sessions_with_multiple_producer_ids": 0, "session_identity_disagreements": [],
        "probe_key_set_equal": True, "governed_identity_disagreements": 0,
        "behavior_availability_used_for_filtering": False,
        "measurement_values_transformed": False,
        "legacy_base_used_for_filtering": False,
    }
    if legacy_base is not None:
        legacy = _keys(legacy_base, "legacy base (audit only)")
        if _key_set(legacy) - universe_keys:
            raise ValueError("legacy base contains probes outside the governed universe")
        excluded = m1.loc[~pd.MultiIndex.from_frame(m1[JOIN_KEY]).isin(pd.MultiIndex.from_frame(legacy[JOIN_KEY]))]
        report["legacy_base_audit"] = {
            "base_rows": len(legacy), "would_exclude_rows": len(excluded),
            "would_exclude_finite_hr": int(_finite(excluded, MMWAVE_COLUMNS[0]).sum()),
            "would_exclude_finite_br": int(_finite(excluded, MMWAVE_COLUMNS[1]).sum()),
        }
    return m1.loc[:, [c for c in OUTPUT_COLUMNS if c in m1]].reset_index(drop=True), report


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", required=True, type=Path)
    parser.add_argument("--identity-bridge", required=True, type=Path)
    parser.add_argument("--probe-universe", required=True, type=Path)
    parser.add_argument("--base-table", type=Path, help="Optional old behavior input; audit only, never filters output")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output}")
    inputs = {
        "m1_j": args.m1_dir / "mmwave_probe_merge_ready.csv",
        "m1_e": args.m1_dir / "mmwave_probe_merge_ready_E.csv",
        "identity_bridge": args.identity_bridge, "probe_universe": args.probe_universe,
    }
    if args.base_table is not None:
        inputs["legacy_base_audit_only"] = args.base_table
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    before = {name: _sha(path) for name, path in inputs.items()}
    m1 = pd.concat([_read(inputs["m1_j"]), _read(inputs["m1_e"])], ignore_index=True)
    output, report = build_source(
        m1, _read(args.identity_bridge), _read(args.probe_universe),
        _read(args.base_table) if args.base_table is not None else None,
    )
    after = {name: _sha(path) for name, path in inputs.items()}
    if before != after:
        raise RuntimeError("source changed during read; refusing to publish output")
    report["inputs"] = {name: {"path": str(path.resolve()), "sha256": before[name]} for name, path in inputs.items()}
    report["script_sha256"] = _sha(Path(__file__))
    report["runtime"] = {"pandas": pd.__version__, "numpy": np.__version__}
    try:
        repo = Path(__file__).resolve().parents[1]
        report["source_commit"] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        report["source_worktree_status"] = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        report["source_commit"] = None
        report["source_worktree_status"] = None
    args.output.mkdir(parents=True, exist_ok=False)
    out_path = args.output / "m1_cardiopulmonary_taskb_source.csv"
    output.to_csv(out_path, index=False, encoding="utf-8-sig")
    report["output"] = str(out_path.resolve())
    report["output_sha256"] = _sha(out_path)
    report["status"] = "SOURCE_ASSEMBLED_NOT_MODEL_REVALIDATED"
    (args.output / "m1_source_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
