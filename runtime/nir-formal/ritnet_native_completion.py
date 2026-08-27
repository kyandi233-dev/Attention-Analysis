"""Strict completion verification for the canonical RITnet full-class evidence run."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ritnet_label_store import RitnetLabelStore, canonical_digest, sha256_file

FULLCLASS_SCHEMA_VERSION = 2
FULLCLASS_VERSION = "ritnet-fullclass-v2-native640"
# Compatibility aliases for the implementation module. They are the same
# canonical version, not a second supported production path.
NATIVE_EXTENSION_SCHEMA_VERSION = FULLCLASS_SCHEMA_VERSION
NATIVE_EXTENSION_VERSION = FULLCLASS_VERSION


@dataclass(frozen=True)
class CompletionVerification:
    valid: bool
    errors: tuple[str, ...]
    label_store_report: dict[str, Any] | None


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _resolve(base: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _read_key_rows(path: Path, expected_subject: str) -> list[tuple[int, int, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"subject", "native_label_row_ordinal", "frame_idx", "eye"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing identity/key columns: {sorted(missing)}")
        rows: list[tuple[int, int, str]] = []
        for row_number, row in enumerate(reader, start=2):
            subject = str(row.get("subject") or "").strip()
            if subject != expected_subject:
                raise ValueError(
                    f"CSV subject mismatch at row {row_number}: {subject!r} != {expected_subject!r}"
                )
            rows.append(
                (
                    int(row["native_label_row_ordinal"]),
                    int(float(row["frame_idx"])),
                    str(row["eye"]),
                )
            )
        return rows


def _read_index_keys(path: Path) -> list[tuple[int, int, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"row_ordinal", "frame_idx", "eye"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"label_index missing key columns: {sorted(missing)}")
        return [
            (int(row["row_ordinal"]), int(row["frame_idx"]), str(row["eye"]))
            for row in reader
        ]


def verify_native_completion(
    completion_path: Path,
    *,
    expected_identity: dict[str, Any] | None = None,
) -> CompletionVerification:
    """Verify completion after any label-store recovery has had a chance to run.

    Opening the label store can legitimately rebuild torn CSV metadata from an
    already committed NPZ chunk. Therefore artifact SHA256 checks happen only
    after that recovery step. A real recovery intentionally invalidates the old
    completion marker and forces the caller to finalize a new one.
    """
    completion_path = Path(completion_path)
    base = completion_path.parent
    errors: list[str] = []
    store_report_dict: dict[str, Any] | None = None
    try:
        marker = load_json(completion_path)
    except Exception as exc:
        return CompletionVerification(False, (f"completion unreadable: {exc}",), None)

    if marker.get("schema_version") != FULLCLASS_SCHEMA_VERSION:
        errors.append("completion schema_version mismatch")
    if marker.get("extension_version") != FULLCLASS_VERSION:
        errors.append("completion extension_version mismatch")
    if marker.get("status") != "complete":
        errors.append("completion status is not complete")
    if not marker.get("artifact_hashes_verified_at_utc"):
        errors.append("artifact_hashes_verified_at_utc is missing")

    marker_subject = str(marker.get("subject") or "").strip()
    if not marker_subject:
        errors.append("completion subject is missing")

    identity = marker.get("resume_identity")
    if not isinstance(identity, dict):
        errors.append("completion resume_identity is missing")
        identity = {}
    digest = canonical_digest(identity)
    if marker.get("resume_identity_digest") != digest:
        errors.append("completion resume_identity_digest mismatch")
    if expected_identity is not None and identity != expected_identity:
        errors.append("completion resume identity differs from current run identity")
    if identity.get("subject") != marker_subject:
        errors.append("completion subject does not match resume_identity subject")

    try:
        expected_rows = int(marker.get("expected_rows", -1))
        processed_rows = int(marker.get("processed_rows", -2))
        stored_rows = int(marker.get("stored_label_rows", -3))
    except (TypeError, ValueError):
        expected_rows = processed_rows = stored_rows = -1
        errors.append("completion row counts are not valid integers")
    if expected_rows < 0 or processed_rows != expected_rows or stored_rows != expected_rows:
        errors.append(
            f"row-count mismatch expected={expected_rows} processed={processed_rows} stored={stored_rows}"
        )

    artifact_specs = {
        "output_csv": "output_csv_sha256",
        "label_index": "label_index_sha256",
        "chunk_manifest": "chunk_manifest_sha256",
        "store_manifest": "store_manifest_sha256",
        "summary": "summary_sha256",
        "manifest": "manifest_sha256",
        "qc_index": "qc_index_sha256",
    }
    resolved: dict[str, Path] = {}
    for path_key, hash_key in artifact_specs.items():
        value = marker.get(path_key)
        expected_hash = marker.get(hash_key)
        if not value or not expected_hash:
            errors.append(f"missing completion artifact field {path_key}/{hash_key}")
            continue
        path = _resolve(base, value)
        resolved[path_key] = path
        if not path.is_file():
            errors.append(f"missing artifact: {path}")

    # Recovery/verification comes before the top-level artifact hash comparison.
    label_root_value = marker.get("label_store_root")
    if label_root_value:
        label_root = _resolve(base, label_root_value)
        try:
            preopen_manifest = load_json(label_root / "store_manifest.json")
            store = RitnetLabelStore(
                label_root,
                identity=preopen_manifest["identity"],
                eye_mapping=preopen_manifest["eye_mapping"],
                chunk_rows=int(preopen_manifest["chunk_rows"]),
                compression=str(preopen_manifest["compression"]),
            )
            if marker.get("label_store_identity_digest") != store.identity_digest:
                errors.append("label_store_identity_digest mismatch")
            if store.identity.get("resume_identity_digest") != digest:
                errors.append("label store is not linked to this completion resume identity")
            if store.identity.get("subject") != marker_subject:
                errors.append("label store subject does not match completion subject")
            if store.identity.get("extension_version") != FULLCLASS_VERSION:
                errors.append("label store fullclass version mismatch")

            report = store.verify(expected_rows=expected_rows)
            store_report_dict = report.as_dict()
            if not report.valid:
                errors.extend(f"label-store: {message}" for message in report.errors)

            final_store_manifest = load_json(store.store_manifest_path)
            if final_store_manifest.get("status") != "complete":
                errors.append("label store status is not complete")
            if int(final_store_manifest.get("expected_rows", -1)) != expected_rows:
                errors.append("label store expected_rows mismatch")
            if final_store_manifest.get("label_index_sha256") != sha256_file(store.index_path):
                errors.append("label store embedded label_index_sha256 mismatch")
            if final_store_manifest.get("chunk_manifest_sha256") != sha256_file(store.chunk_manifest_path):
                errors.append("label store embedded chunk_manifest_sha256 mismatch")
        except Exception as exc:
            errors.append(f"label-store verification failed: {exc}")
    else:
        errors.append("label_store_root missing")

    # Hash every final artifact only after the store open/recovery step above.
    for path_key, hash_key in artifact_specs.items():
        path = resolved.get(path_key)
        expected_hash = marker.get(hash_key)
        if path is None or not path.is_file() or not expected_hash:
            continue
        if sha256_file(path) != expected_hash:
            errors.append(f"artifact hash mismatch: {path_key}")

    if "output_csv" in resolved and "label_index" in resolved and marker_subject:
        try:
            csv_keys = _read_key_rows(resolved["output_csv"], marker_subject)
            index_keys = _read_index_keys(resolved["label_index"])
            if len(csv_keys) != len(set(csv_keys)):
                errors.append("output CSV keys are not unique")
            if csv_keys != index_keys:
                errors.append("CSV key sequence does not exactly match label_index")
            if len(csv_keys) != expected_rows:
                errors.append(f"output CSV has {len(csv_keys)} rows, expected {expected_rows}")
        except Exception as exc:
            errors.append(f"CSV/index identity verification failed: {exc}")

    required_true_flags = (
        "label_store_verified",
        "label_value_domain_verified",
        "label_shape_verified",
        "label_index_unique_verified",
        "label_csv_key_match_verified",
    )
    for key in required_true_flags:
        if marker.get(key) is not True:
            errors.append(f"completion flag {key} is not true")

    return CompletionVerification(not errors, tuple(errors), store_report_dict)


# Canonical name for new callers. Keep the old function name only as a code-level
# alias so internal implementation imports resolve to the same single contract.
verify_fullclass_completion = verify_native_completion
