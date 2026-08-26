"""Strict completion validation for ritnet-fullclass-v2-native640 artifacts."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ritnet_label_store import RitnetLabelStore, canonical_digest, sha256_file

NATIVE_EXTENSION_SCHEMA_VERSION = 2
NATIVE_EXTENSION_VERSION = "ritnet-fullclass-v2-native640"


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


def _read_key_rows(path: Path) -> list[tuple[int, int, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"native_label_row_ordinal", "frame_idx", "eye"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV missing key columns: {sorted(required - set(reader.fieldnames or []))}")
        return [
            (int(row["native_label_row_ordinal"]), int(float(row["frame_idx"])), str(row["eye"]))
            for row in reader
        ]


def _read_index_keys(path: Path) -> list[tuple[int, int, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            (int(row["row_ordinal"]), int(row["frame_idx"]), str(row["eye"]))
            for row in reader
        ]


def verify_native_completion(
    completion_path: Path,
    *,
    expected_identity: dict[str, Any] | None = None,
) -> CompletionVerification:
    completion_path = Path(completion_path)
    base = completion_path.parent
    errors: list[str] = []
    store_report_dict: dict[str, Any] | None = None
    try:
        marker = load_json(completion_path)
    except Exception as exc:
        return CompletionVerification(False, (f"completion unreadable: {exc}",), None)

    if marker.get("schema_version") != NATIVE_EXTENSION_SCHEMA_VERSION:
        errors.append("completion schema_version mismatch")
    if marker.get("extension_version") != NATIVE_EXTENSION_VERSION:
        errors.append("completion extension_version mismatch")
    if marker.get("status") != "complete":
        errors.append("completion status is not complete")
    if not marker.get("artifact_hashes_verified_at_utc"):
        errors.append("artifact_hashes_verified_at_utc is missing")

    identity = marker.get("resume_identity")
    if not isinstance(identity, dict):
        errors.append("completion resume_identity is missing")
        identity = {}
    digest = canonical_digest(identity)
    if marker.get("resume_identity_digest") != digest:
        errors.append("completion resume_identity_digest mismatch")
    if expected_identity is not None and identity != expected_identity:
        errors.append("completion resume identity differs from current run identity")

    expected_rows = int(marker.get("expected_rows", -1))
    processed_rows = int(marker.get("processed_rows", -2))
    stored_rows = int(marker.get("stored_label_rows", -3))
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
            continue
        if sha256_file(path) != expected_hash:
            errors.append(f"artifact hash mismatch: {path_key}")

    label_root_value = marker.get("label_store_root")
    if label_root_value:
        label_root = _resolve(base, label_root_value)
        try:
            store_manifest = load_json(label_root / "store_manifest.json")
            store = RitnetLabelStore(
                label_root,
                identity=store_manifest["identity"],
                eye_mapping=store_manifest["eye_mapping"],
                chunk_rows=int(store_manifest["chunk_rows"]),
                compression=str(store_manifest["compression"]),
            )
            if marker.get("label_store_identity_digest") != store.identity_digest:
                errors.append("label_store_identity_digest mismatch")
            report = store.verify(expected_rows=expected_rows)
            store_report_dict = report.as_dict()
            if not report.valid:
                errors.extend(f"label-store: {message}" for message in report.errors)
        except Exception as exc:
            errors.append(f"label-store verification failed: {exc}")
    else:
        errors.append("label_store_root missing")

    if "output_csv" in resolved and "label_index" in resolved:
        try:
            csv_keys = _read_key_rows(resolved["output_csv"])
            index_keys = _read_index_keys(resolved["label_index"])
            if len(csv_keys) != len(set(csv_keys)):
                errors.append("output CSV keys are not unique")
            if csv_keys != index_keys:
                errors.append("CSV key sequence does not exactly match label_index")
            if len(csv_keys) != expected_rows:
                errors.append(f"output CSV has {len(csv_keys)} rows, expected {expected_rows}")
        except Exception as exc:
            errors.append(f"CSV/index key verification failed: {exc}")

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
