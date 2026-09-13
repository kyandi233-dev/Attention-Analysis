from pathlib import Path

import pandas as pd
import pytest

from attention_pipeline.nir_formal_analysis.ocular_probe_identity import (
    ensure_ocular_canonical_probe_identity,
)


def _g1() -> pd.DataFrame:
    rows = []
    for block in (1, 2):
        for probe in (1, 2):
            for signal in ("geometry", "rseg"):
                rows.append(
                    {
                        "session_id": "sub-001",
                        "participant_group_id": "p01",
                        "block_num": block,
                        "probe_order_in_block": probe,
                        "probe_event_id": f"sub-001|B{block}|probe|{probe}",
                        "signal": signal,
                    }
                )
    return pd.DataFrame(rows)


def _wide(include_rgb_only: bool = False) -> pd.DataFrame:
    rows = [
        {
            "session_id": "sub-001",
            "participant_group_id": "p01",
            "probe_event_id": f"sub-001|B{block}|probe|{probe}",
            "ocular__nir": block * 10 + probe,
        }
        for block in (1, 2)
        for probe in (1, 2)
    ]
    if include_rgb_only:
        rows.extend(
            {
                "session_id": "sub-002",
                "participant_group_id": "p02",
                "probe_event_id": f"sub-002|B{block}|probe|{probe}",
                "ocular__nir": pd.NA,
            }
            for block in (1, 2)
            for probe in (1, 2)
        )
    return pd.DataFrame(rows)


def _supplemental() -> pd.DataFrame:
    rows = []
    for session_id, participant_group_id in (("sub-001", "p01"), ("sub-002", "p02")):
        for block in (1, 2):
            for probe in (1, 2):
                rows.append(
                    {
                        "session_id": session_id,
                        "participant_group_id": participant_group_id,
                        "block_id": f"B{block}",
                        "probe_order_in_block": probe,
                        "probe_event_id": f"{session_id}|B{block}|probe|{probe}",
                    }
                )
    return pd.DataFrame(rows)


def test_identity_bridge_materializes_canonical_key_and_keeps_provenance(
    tmp_path: Path,
) -> None:
    g1 = tmp_path / "g1.csv"
    wide = tmp_path / "wide.csv"
    _g1().to_csv(g1, index=False)
    _wide().to_csv(wide, index=False)

    result = ensure_ocular_canonical_probe_identity(g1, wide)
    out = pd.read_csv(wide)

    assert out.columns[:5].tolist() == [
        "participant_group_id",
        "session_id",
        "block_id",
        "probe_index_in_block",
        "probe_event_id",
    ]
    assert out["block_id"].tolist() == ["b1", "b1", "b2", "b2"]
    assert out["probe_index_in_block"].tolist() == [1, 2, 1, 2]
    assert out["probe_event_id"].is_unique
    assert not out.duplicated(
        ["participant_group_id", "session_id", "block_id", "probe_index_in_block"]
    ).any()
    assert result == {"row_n": 4, "session_n": 1, "participant_group_n": 1}


def test_identity_bridge_uses_supplemental_identity_for_governed_probe_missing_from_g1(
    tmp_path: Path,
) -> None:
    g1 = tmp_path / "g1.csv"
    wide = tmp_path / "wide.csv"
    supplemental = tmp_path / "rgb.csv"
    _g1().to_csv(g1, index=False)
    _wide(include_rgb_only=True).to_csv(wide, index=False)
    _supplemental().to_csv(supplemental, index=False)

    result = ensure_ocular_canonical_probe_identity(
        g1,
        wide,
        supplemental_probe_identity_path=supplemental,
    )
    out = pd.read_csv(wide)

    assert len(out) == 8
    assert result == {"row_n": 8, "session_n": 2, "participant_group_n": 2}
    assert not out[
        ["participant_group_id", "session_id", "block_id", "probe_index_in_block"]
    ].isna().any().any()
    rgb_only = out[out["session_id"].eq("sub-002")]
    assert len(rgb_only) == 4
    assert rgb_only["ocular__nir"].isna().all()


def test_identity_bridge_fails_closed_when_probe_event_disagrees(
    tmp_path: Path,
) -> None:
    frame = _g1()
    frame.loc[frame.index[0], "probe_event_id"] = "sub-001|B2|probe|1"
    g1 = tmp_path / "g1.csv"
    wide = tmp_path / "wide.csv"
    frame.to_csv(g1, index=False)
    _wide().to_csv(wide, index=False)

    with pytest.raises(ValueError, match="probe_event_id disagrees"):
        ensure_ocular_canonical_probe_identity(g1, wide)


def test_identity_bridge_fails_closed_when_sources_disagree(
    tmp_path: Path,
) -> None:
    g1 = tmp_path / "g1.csv"
    wide = tmp_path / "wide.csv"
    supplemental = tmp_path / "rgb.csv"
    _g1().to_csv(g1, index=False)
    _wide(include_rgb_only=True).to_csv(wide, index=False)
    supplemental_frame = _supplemental()
    mask = supplemental_frame["probe_event_id"].eq("sub-001|B1|probe|1")
    supplemental_frame.loc[mask, "probe_order_in_block"] = 2
    supplemental_frame.to_csv(supplemental, index=False)

    with pytest.raises(ValueError, match="disagree"):
        ensure_ocular_canonical_probe_identity(
            g1,
            wide,
            supplemental_probe_identity_path=supplemental,
        )


def test_identity_bridge_is_idempotent_and_validates_existing_canonical_key(
    tmp_path: Path,
) -> None:
    g1 = tmp_path / "g1.csv"
    wide = tmp_path / "wide.csv"
    supplemental = tmp_path / "rgb.csv"
    _g1().to_csv(g1, index=False)
    _wide(include_rgb_only=True).to_csv(wide, index=False)
    _supplemental().to_csv(supplemental, index=False)

    ensure_ocular_canonical_probe_identity(
        g1,
        wide,
        supplemental_probe_identity_path=supplemental,
    )
    ensure_ocular_canonical_probe_identity(
        g1,
        wide,
        supplemental_probe_identity_path=supplemental,
    )
    out = pd.read_csv(wide)
    assert len(out) == 8
    assert out["probe_event_id"].is_unique
