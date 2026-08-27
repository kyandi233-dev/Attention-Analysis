from __future__ import annotations

import csv

import run_ritnet_fullclass_native_extension as implementation
from run_ritnet_fullclass_extension import _install_subject_identity_guard


def test_canonical_entry_backfills_subject_without_modifying_source(tmp_path):
    source = tmp_path / "eyes.csv"
    fields = ["frame_idx", "eye", "roi_x1", "roi_y1", "roi_x2", "roi_y2"]
    with source.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "frame_idx": 100,
                "eye": "frame_left",
                "roi_x1": 10,
                "roi_y1": 20,
                "roi_x2": 110,
                "roi_y2": 80,
            }
        )

    before = source.read_bytes()
    original = implementation._source_rows
    try:
        _install_subject_identity_guard()
        output_fields, rows = implementation._source_rows(source, "sub-031")
    finally:
        implementation._source_rows = original

    assert output_fields[0] == "subject"
    assert rows[0]["subject"] == "sub-031"
    assert source.read_bytes() == before
