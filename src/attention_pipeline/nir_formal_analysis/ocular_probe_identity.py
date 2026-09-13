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


def _normalize_identity(frame: pd.DataFrame, *, source_label: str) -> pd.DataFrame:
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
        raise ValueError(
            f"{source_label} probe table missing canonical identity fields/aliases: {missing}"
        )

    out["participant_group_id"] = out["participant_group_id"].astype("string").str.strip()
    out["session_id"] = out["session_id"].astype("string").str.strip()
    out["block_id"] = _normalize_block(out["block_id"])
    out["probe_index_in_block"] = pd.to_numeric(
        out["probe_index_in_block"], errors="coerce"
    ).astype("Int64")

    missing_key = out[list(CANONICAL_PROBE_KEY)].isna().any(axis=1)
    if missing_key.any():
        raise ValueError(
            f"{source_label} probe table has {int(missing_key.sum())} rows "
            "with incomplete canonical identity"
        )
    return out


def _validate_probe_event_id(
    frame: pd.DataFrame,
    *,
    source_label: str,
) -> pd.DataFrame:
    if "probe_event_id" not in frame.columns:
        raise ValueError(
            f"{source_label} probe table missing probe_event_id provenance required for identity bridge"
        )

    event = frame["probe_event_id"].astype("string").str.strip()
    parsed = event.str.extract(
        r"^(?P<event_session>[^|]+)\|(?P<event_block>[Bb]\d+)\|probe\|(?P<event_probe>\d+)$"
    )
    unparsed = event.notna() & parsed.isna().any(axis=1)
    if unparsed.any():
        raise ValueError(
            f"{source_label} probe_event_id has {int(unparsed.sum())} unparsable rows"
        )

    event_block = _normalize_block(parsed["event_block"])
    event_probe = pd.to_numeric(parsed["event_probe"], errors="coerce").astype("Int64")
    session_mismatch = parsed["event_session"].astype("string").ne(frame["session_id"])
    block_mismatch = event_block.ne(frame["block_id"])
    probe_mismatch = event_probe.ne(frame["probe_index_in_block"])
    mismatch = event.notna() & (session_mismatch | block_mismatch | probe_mismatch)
    if mismatch.any():
        raise ValueError(
            f"{source_label} probe_event_id disagrees with canonical probe identity in {int(mismatch.sum())} rows"
        )

    mapping = frame[list(CANONICAL_PROBE_KEY) + ["probe_event_id"]].drop_duplicates()
    if mapping.duplicated(list(CANONICAL_PROBE_KEY)).any():
        raise ValueError(
            f"{source_label} canonical probe key maps to multiple probe_event_id values"
        )
    if mapping.duplicated(list(_EVENT_KEY)).any():
        raise ValueError(
            f"{source_label} probe_event_id maps to multiple canonical probe keys"
        )
    return mapping


def _identity_mapping(path: str | Path, *, source_label: str) -> pd.DataFrame:
    normalized = _normalize_identity(_read_csv(path), source_label=source_label)
    return _validate_probe_event_id(normalized, source_label=source_label)


def _merge_identity_mappings(
    g1_mapping: pd.DataFrame,
    supplemental_mapping: pd.DataFrame | None,
) -> pd.DataFrame:
    if supplemental_mapping is None:
        return g1_mapping

    overlap = g1_mapping.merge(
        supplemental_mapping,
        on=list(_EVENT_KEY),
        how="inner",
        suffixes=("__g1", "__supplemental"),
        validate="one_to_one",
    )
    if not overlap.empty:
        mismatch = (
            overlap["block_id__g1"].ne(overlap["block_id__supplemental"])
            | overlap["probe_index_in_block__g1"].ne(
                overlap["probe_index_in_block__supplemental"]
            )
        )
        if mismatch.any():
            raise ValueError(
                "G1 and supplemental probe identity sources disagree on "
                f"{int(mismatch.sum())} overlapping probes"
            )

    mapping = pd.concat([g1_mapping, supplemental_mapping], ignore_index=True)
    mapping = mapping.drop_duplicates(list(CANONICAL_PROBE_KEY) + ["probe_event_id"])
    if mapping.duplicated(list(CANONICAL_PROBE_KEY)).any():
        raise ValueError(
            "combined identity sources map one canonical probe key to multiple probe_event_id values"
        )
    if mapping.duplicated(list(_EVENT_KEY)).any():
        raise ValueError(
            "combined identity sources map one probe_event_id to multiple canonical probe keys"
        )
    return mapping


def ensure_ocular_canonical_probe_identity(
    g1_probe_candidates_path: str | Path,
    ocular_probe_wide_path: str | Path,
    *,
    supplemental_probe_identity_path: str | Path | None = None,
) -> dict[str, int]:
    """Attach and validate canonical probe identity on the frozen Ocular wide table.

    G1 is authoritative for probes with NIR measurement candidates. A supplemental
    probe-level table may provide identity only for governed probes absent from G1
    (for example, valid RGB-only Ocular probes). The supplemental source never fills
    NIR predictors. Any overlapping identity disagreement fails closed.
    """
    g1_mapping = _identity_mapping(g1_probe_candidates_path, source_label="G1")
    supplemental_mapping = (
        _identity_mapping(supplemental_probe_identity_path, source_label="supplemental")
        if supplemental_probe_identity_path is not None
        else None
    )
    mapping = _merge_identity_mappings(g1_mapping, supplemental_mapping)

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
        current = _normalize_identity(wide, source_label="Ocular wide")
        check = current[list(CANONICAL_PROBE_KEY) + ["probe_event_id"]].merge(
            mapping,
            on=list(_EVENT_KEY),
            how="left",
            suffixes=("", "__source"),
            validate="one_to_one",
        )
        missing_map = check["block_id__source"].isna() | check[
            "probe_index_in_block__source"
        ].isna()
        mismatch = check["block_id"].ne(check["block_id__source"]) | check[
            "probe_index_in_block"
        ].ne(check["probe_index_in_block__source"])
        if missing_map.any() or mismatch.any():
            raise ValueError(
                "Ocular wide canonical identity disagrees with validated identity sources"
            )
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
