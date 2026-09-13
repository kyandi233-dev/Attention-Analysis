from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


EVIDENCE_FILES = {
    "temporal_support": "g1_temporal_support_freeze_grid.csv",
    "cross_signal_representation": "g1_cross_signal_representation_summary.csv",
    "sync_semantics": "g1_sync_semantics_split.csv",
    "source_mode_limit": "g1_source_mode_limit_summary.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_ocular_freeze_evidence(
    ocular_root: str | Path,
    *,
    temporal_support: str | Path,
    cross_signal_representation: str | Path,
    sync_semantics: str | Path,
    source_mode_limit: str | Path,
) -> None:
    root = Path(ocular_root).expanduser().resolve()
    tables = root / "tables"
    manifest_path = root / "manifests/ocular_science_output_manifest.json"
    supplied = {
        "temporal_support": temporal_support,
        "cross_signal_representation": cross_signal_representation,
        "sync_semantics": sync_semantics,
        "source_mode_limit": source_mode_limit,
    }
    evidence: dict[str, dict[str, object]] = {}
    for key, source in supplied.items():
        source_path = Path(source).expanduser().resolve()
        frame = pd.read_csv(source_path, encoding="utf-8-sig", low_memory=False)
        output = EVIDENCE_FILES[key]
        output_path = tables / output
        shutil.copy2(source_path, output_path)
        source_hash = _sha256(source_path)
        output_hash = _sha256(output_path)
        if output_hash != source_hash:
            raise RuntimeError(f"freeze evidence copy hash mismatch: {key}")
        evidence[key] = {
            "status": "included_byte_exact",
            "row_n": int(len(frame)),
            "output": f"tables/{output}",
            "sha256": source_hash,
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence["temporal_support"].update(
        {
            "formal_minimum_span_frozen": True,
            "formal_minimum_span_sec": 20.0,
            "requires_early_and_late_support": True,
        }
    )
    manifest["freeze_evidence"] = evidence
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
