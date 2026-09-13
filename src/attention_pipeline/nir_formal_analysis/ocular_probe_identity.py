from __future__ import annotations

from pathlib import Path

import pandas as pd


CANONICAL_PROBE_KEY = (
    "participant_group_id",
    "session_id",
    "block_id",
    "probe_index_in_block",
)
_EVENT_KEY = ("session_id", "participant_group_id", "probe_event_id")


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return pd.read_csv(p, encoding="utf-8-sig", low_memory=False)


def _normalize_block(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.lower()
    text = text.str.replace("block", "", regex=False).str.replace("-", "", regex=False)
    text = text.where(text.str.startswith("b"), "b" + text)
    return text


def _normalize_g1_identity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "block_id" not in out.columns and "block_num" in out.columns:
        out["block_id"] = out["block_num"]
    if "probe_index_in_block" not in out.columns:
        for alias in ("probe_order_in_block", "probe_index_within_block"):
            if alias in out.columns:
                out["probe_index_in_block"] = out[alias]
                break

    missing = [name for name in CANONICAL_PROBE_KEY if name not in out.columns]
    if missing:
        raise ValueError(f"G1 probe table missing canonical identity fields/aliases: {missing}")

    out["participant_group_id"] = out["participant_group_id"].astype("string").str.strip()
    out["session_id"] = out["session_id"].astype("string").str.strip()
    out["block_id"] = _normalize_block(out["block_id"])
    out["probe_index_in_block"] = pd.to_numeric(
        out["probe_index_in_block"], errors="coerce"
    ).astype("Int64")

    missing_key = out[list(CANONICAL_PROBE_KEY)].isna().any(axis=1)
    if missing_key.any():
        raise ValueError(
            f"G1 probe table has {int(missing_key.sum())} rows with incomplete canonical identity"
        )
    return out


def _validate_probe_event_id(frame: pd.DataFrame) -> pd.DataFrame:
    if "probe_event_id" not in frame.columns:
        raise ValueError("G1 probe table missing probe_event_id provenance required for identity bridge")

    event = frame["probe_event_id"].astype("string").str.strip()
    parsed = event.str.extract(
        r"^(?P<event_session>[^|]+)\|(?P<event_block>[Bb]\d+)\|probe\|(?P<event_probe>\d+)$"
    )
    unparsed = event.notna() & parsed.isna().any(axis=1)
    if unparsed.any():
        raise ValueError(f"G1 probe_event_id has {int(unparsed.sum())} unparsable rows")

    event_block = _normalize_block(parsed["event_block"])
    event_probe = pd.to_numeric(parsed["event_probe"], errors="coerce").astype("Int64")
    session_mismatch = parsed["event_session"].astype("string").ne(frame["session_id"])
    block_mismatch = event_block.ne(frame["block_id"])
    probe_mismatch = event_probe.ne(frame["probe_index_in_block"])
    mismatch = event.notna() & (session_mismatch | block_mismatch | probe_mismatch)
    if mismatch.any():
        raise ValueError(
            f"G1 probe_event_id disagrees with canonical probe identity in {int(mismatch.sum())} rows"
        )

    mapping = frame[list(CANONICAL_PROBE_KEY) + ["probe_event_id"]].drop_duplicates()
    if mapping.duplicated(list(CANONICAL_PROBE_KEY)).any():
        raise ValueError("canonical probe key maps to multiple probe_event_id values")
    if mapping.duplicated(list(_EVENT_KEY)).any():
        raise ValueError("probe_event_id maps to multiple canonical probe keys")
    return mapping


def ensure_ocular_canonical_probe_identity(
    g1_probe_candidates_path: str | Path,
    ocular_probe_wide_path: str | Path,
) -> dict[str, int]:
    """Attach and validate canonical probe identity on the frozen Ocular wide table.

    The function does not alter scientific predictors. It bridges the legacy
    ``probe_event_id`` identity retained by the candidate materializer to the
    canonical ``participant/session/block/probe`` key required by P5, while
    failing closed if the two identity representations disagree.
    """
    g1 = _normalize_g1_identity(_read_csv(g1_probe_candidates_path))
    mapping = _validate_probe_event_id(g1)

    wide_path = Path(ocular_probe_wide_path).expanduser().resolve()
    wide = _read_csv(wide_path)
    for name in _EVENT_KEY:
        if name not in wide.columns:
            raise ValueError(f"Ocular wide table missing identity bridge field: {name}")
    wide["session_id"] = wide["session_id"].astype("string").str.strip()
    wide["participant_group_id"] = wide["participant_group_id"].astype("string").str.strip()
    wide["probe_event_id"] = wide["probe_event_id"].astype("string").str.strip()

    canonical_present = all(name in wide.columns for name in CANONICAL_PROBE_KEY)
    if canonical_present:
        current = _normalize_g1_identity(wide)
        check = current[list(CANONICAL_PROBE_KEY) + ["probe_event_id"]].merge(
            mapping,
            on=list(_EVENT_KEY),
            how="left",
            suffixes=("", "__g1"),
            validate="one_to_one",
        )
        missing_map = check["block_id__g1"].isna() | check["probe_index_in_block__g1"].isna()
        mismatch = (
            check["block_id"].ne(check["block_id__g1"])
            | check["probe_index_in_block"].ne(check["probe_index_in_block__g1"])
        )
        if missing_map.any() or mismatch.any():
            raise ValueError("Ocular wide canonical identity disagrees with G1 identity mapping")
        wide = current
    else:
        drop_existing = [
            name for name in ("block_id", "probe_index_in_block") if name in wide.columns
        ]
        if drop_existing:
            wide = wide.drop(columns=drop_existing)
        wide = wide.merge(
            mapping,
            on=list(_EVENT_KEY),
            how="left",
            validate="one_to_one",
        )

    missing_key = wide[list(CANONICAL_PROBE_KEY)].isna().any(axis=1)
    if missing_key.any():
        raise ValueError(
            f"Ocular wide table has {int(missing_key.sum())} rows without canonical probe identity"
        )
    if wide.duplicated(list(CANONICAL_PROBE_KEY)).any():
        raise ValueError("Ocular wide table has duplicate canonical probe keys")
    if wide.duplicated(list(_EVENT_KEY)).any():
        raise ValueError("Ocular wide table has duplicate probe_event_id identity keys")

    session_groups = wide.groupby("session_id", dropna=False)["participant_group_id"].nunique(
        dropna=True
    )
    if (session_groups > 1).any():
        raise ValueError("Ocular wide table maps a session to multiple participant groups")

    identity_front = list(CANONICAL_PROBE_KEY) + ["probe_event_id"]
    remaining = [name for name in wide.columns if name not in identity_front]
    wide = wide[identity_front + remaining]
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
    return {
        "row_n": int(len(wide)),
        "session_n": int(wide["session_id"].nunique()),
        "participant_group_n": int(wide["participant_group_id"].nunique()),
    }
