import json
from pathlib import Path

from attention_pipeline.nir_formal_analysis.ocular_freeze_evidence import archive_ocular_freeze_evidence


def test_freeze_evidence_archive_is_byte_exact(tmp_path: Path) -> None:
    ocular = tmp_path / "Ocular"
    (ocular / "tables").mkdir(parents=True)
    (ocular / "manifests").mkdir(parents=True)
    manifest_path = ocular / "manifests/ocular_science_output_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")

    supplied = {}
    rows = {
        "temporal_support": "a,b\n1,0.12345678901234567\n",
        "cross_signal_representation": "a,b\n2,0.98765432109876543\n",
        "sync_semantics": "a,b\n3,20.000000000000004\n",
        "source_mode_limit": "a,b\n4,0.38000000000000006\n",
    }
    for key, text in rows.items():
        path = tmp_path / f"{key}.csv"
        path.write_bytes(text.encode("utf-8"))
        supplied[key] = path

    archive_ocular_freeze_evidence(ocular, **supplied)

    expected_names = {
        "temporal_support": "g1_temporal_support_freeze_grid.csv",
        "cross_signal_representation": "g1_cross_signal_representation_summary.csv",
        "sync_semantics": "g1_sync_semantics_split.csv",
        "source_mode_limit": "g1_source_mode_limit_summary.csv",
    }
    for key, name in expected_names.items():
        assert (ocular / "tables" / name).read_bytes() == supplied[key].read_bytes()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["freeze_evidence"]["temporal_support"]["formal_minimum_span_frozen"] is True
    assert manifest["freeze_evidence"]["temporal_support"]["formal_minimum_span_sec"] == 20.0
    assert all(item["status"] == "included_byte_exact" for item in manifest["freeze_evidence"].values())
