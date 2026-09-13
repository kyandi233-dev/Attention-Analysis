from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_HANDOFF_COLUMNS = (
    "scientific_feature_id",
    "candidate_representation_id",
    "predictor_column",
    "scientific_modality",
    "source_namespace",
    "required_devices",
    "preprocessing_dependencies",
    "measurement_qc_status",
    "estimability_status",
    "temporal_anchor",
    "temporal_scope",
    "time_legality_status",
    "time_legality_evidence",
    "report_role",
    "registry_ready",
    "researcher_freeze_required",
)

KEY_COLUMNS = ("participant_group_id", "session_id", "block_id", "probe_index_in_block")
MODALITIES = ("behavior", "ocular", "movement")
ALLOWED_SOURCE_NAMESPACES = {
    "behavior": {"behavior"},
    "ocular": {"nir", "rgb"},
    "movement": {"rgb"},
}
ALLOWED_DEVICES = {
    "behavior": set(),
    "ocular": {"nir", "rgb"},
    "movement": {"rgb"},
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return value


def _bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _parse_json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON list, got: {value!r}")
    return [str(item) for item in parsed]


def _normalize_block(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.lower()
    text = text.str.replace("block", "", regex=False).str.replace("-", "", regex=False)
    text = text.replace({"1": "b1", "2": "b2"})
    return text.where(text.str.startswith("b"), "b" + text)


def _normalize_probe_table(frame: pd.DataFrame, modality: str) -> pd.DataFrame:
    out = frame.copy()
    if "participant_group_id" not in out.columns:
        for alias in ("analysis_group_token", "repeat_participant_id"):
            if alias in out.columns:
                out["participant_group_id"] = out[alias]
                break
    if "block_id" not in out.columns and "block_num" in out.columns:
        out["block_id"] = out["block_num"]
    if "probe_index_in_block" not in out.columns:
        for alias in ("probe_order_in_block", "probe_index_within_block"):
            if alias in out.columns:
                out["probe_index_in_block"] = out[alias]
                break

    missing = [name for name in KEY_COLUMNS if name not in out.columns]
    if missing:
        raise ValueError(f"{modality} probe table missing canonical identity fields/aliases: {missing}")

    out["participant_group_id"] = out["participant_group_id"].astype("string").str.strip()
    out["session_id"] = out["session_id"].astype("string").str.strip()
    out["block_id"] = _normalize_block(out["block_id"])
    out["probe_index_in_block"] = pd.to_numeric(out["probe_index_in_block"], errors="coerce").astype("Int64")

    missing_key = out[list(KEY_COLUMNS)].isna().any(axis=1)
    if missing_key.any():
        raise ValueError(f"{modality} probe table has {int(missing_key.sum())} rows with incomplete canonical identity")
    duplicate_n = int(out.duplicated(list(KEY_COLUMNS)).sum())
    if duplicate_n:
        raise ValueError(f"{modality} probe table has {duplicate_n} duplicate canonical probe keys")
    session_groups = out.groupby("session_id", dropna=False)["participant_group_id"].nunique(dropna=True)
    conflict_n = int((session_groups > 1).sum())
    if conflict_n:
        raise ValueError(f"{modality} probe table maps {conflict_n} sessions to multiple participant groups")
    return out


def _load_science_inputs(science_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, dict[str, Any]], dict[str, str]]:
    handoff_paths = {
        "behavior": science_root / "Behavior/tables/behavior_feature_handoff.csv",
        "ocular": science_root / "Ocular/tables/ocular_feature_handoff.csv",
        "movement": science_root / "Movement/tables/movement_feature_handoff.csv",
    }
    manifest_paths = {
        "behavior": science_root / "Behavior/manifests/science_output_manifest.json",
        "ocular": science_root / "Ocular/manifests/ocular_science_output_manifest.json",
        "movement": science_root / "Movement/manifests/movement_science_output_manifest.json",
    }
    handoffs = {name: _read_csv(path) for name, path in handoff_paths.items()}
    manifests = {name: _read_json(path) for name, path in manifest_paths.items()}

    behavior_root = Path(str(manifests["behavior"].get("producer_formal_root", ""))).expanduser()
    behavior_probe = behavior_root / "probe_primary_30s.csv"
    ocular_probe = science_root / "Ocular/tables/ocular_probe_features_wide.csv"
    movement_probe = science_root / "Movement/tables/movement_probe_descriptive_source.csv"
    probe_paths = {
        "behavior": str(behavior_probe.resolve()),
        "ocular": str(ocular_probe.resolve()),
        "movement": str(movement_probe.resolve()),
    }
    probes = {
        "behavior": _normalize_probe_table(_read_csv(behavior_probe), "behavior"),
        "ocular": _normalize_probe_table(_read_csv(ocular_probe), "ocular"),
        "movement": _normalize_probe_table(_read_csv(movement_probe), "movement"),
    }
    return handoffs, probes, manifests, probe_paths


def _key_set(frame: pd.DataFrame) -> set[tuple[str, str, str, int]]:
    return {
        (str(row.participant_group_id), str(row.session_id), str(row.block_id), int(row.probe_index_in_block))
        for row in frame[list(KEY_COLUMNS)].itertuples(index=False)
    }


def _audit_handoffs(
    handoffs: dict[str, pd.DataFrame],
    probes: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], dict[str, list[str]]]:
    summary_rows: list[dict[str, object]] = []
    predictor_rows: list[dict[str, object]] = []
    errors: list[str] = []
    warnings: list[str] = []
    ready_predictors: dict[str, list[str]] = {}

    for modality in MODALITIES:
        handoff = handoffs[modality].copy()
        missing_schema = sorted(set(REQUIRED_HANDOFF_COLUMNS) - set(handoff.columns))
        if missing_schema:
            errors.append(f"{modality}:handoff_missing_columns:{','.join(missing_schema)}")
            ready_predictors[modality] = []
            summary_rows.append(
                {
                    "scientific_modality": modality,
                    "handoff_row_n": int(len(handoff)),
                    "schema_ok": False,
                    "registry_ready_n": 0,
                    "researcher_freeze_pending_n": 0,
                    "registry_ready_predictor_missing_n": None,
                    "registry_ready_time_illegal_n": None,
                    "required_devices_invalid_n": None,
                }
            )
            continue

        actual_modalities = set(handoff["scientific_modality"].dropna().astype(str))
        if actual_modalities != {modality}:
            errors.append(f"{modality}:scientific_modality_mismatch:{sorted(actual_modalities)}")

        ready_mask = handoff["registry_ready"].map(_bool_value)
        freeze_mask = handoff["researcher_freeze_required"].map(_bool_value)
        if int(ready_mask.sum()) == 0:
            errors.append(f"{modality}:no_registry_ready_candidate")
        if int(freeze_mask.sum()):
            warnings.append(f"{modality}:researcher_freeze_pending:{int(freeze_mask.sum())}")

        predictor_missing_n = 0
        time_illegal_n = 0
        device_invalid_n = 0
        modality_ready: list[str] = []
        for row_i, row in handoff.iterrows():
            predictor = str(row.get("predictor_column", "")).strip()
            is_ready = _bool_value(row.get("registry_ready"))
            predictor_exists = bool(predictor) and predictor in probes[modality].columns
            time_status = str(row.get("time_legality_status", "")).strip()
            time_evidence = str(row.get("time_legality_evidence", "")).strip()
            namespace = str(row.get("source_namespace", "")).strip()
            try:
                devices = _parse_json_list(row.get("required_devices"))
                devices_parse_ok = True
            except (json.JSONDecodeError, ValueError, TypeError):
                devices = []
                devices_parse_ok = False

            namespace_ok = namespace in ALLOWED_SOURCE_NAMESPACES[modality]
            device_set = set(devices)
            if modality == "behavior":
                devices_ok = devices_parse_ok and not device_set
            elif modality == "movement":
                devices_ok = devices_parse_ok and device_set == {"rgb"}
            else:
                devices_ok = devices_parse_ok and bool(device_set) and device_set.issubset(ALLOWED_DEVICES[modality])

            time_ok = time_status == "verified_pre_probe_only" and bool(time_evidence) and time_evidence.lower() != "nan"
            if is_ready:
                modality_ready.append(predictor)
                if not predictor_exists:
                    predictor_missing_n += 1
                    errors.append(f"{modality}:registry_ready_predictor_missing:{predictor}")
                if not time_ok:
                    time_illegal_n += 1
                    errors.append(f"{modality}:registry_ready_time_legality_failed:{predictor}")
                if not (namespace_ok and devices_ok):
                    device_invalid_n += 1
                    errors.append(f"{modality}:registry_ready_source_or_device_invalid:{predictor}")

            predictor_rows.append(
                {
                    "scientific_modality": modality,
                    "handoff_row": int(row_i),
                    "predictor_column": predictor,
                    "registry_ready": is_ready,
                    "researcher_freeze_required": _bool_value(row.get("researcher_freeze_required")),
                    "predictor_exists": predictor_exists,
                    "time_legality_ok": time_ok,
                    "source_namespace": namespace,
                    "source_namespace_ok": namespace_ok,
                    "required_devices": json.dumps(devices, ensure_ascii=False),
                    "required_devices_ok": devices_ok,
                    "report_role": str(row.get("report_role", "")),
                    "estimability_status": str(row.get("estimability_status", "")),
                }
            )
        ready_predictors[modality] = modality_ready
        summary_rows.append(
            {
                "scientific_modality": modality,
                "handoff_row_n": int(len(handoff)),
                "schema_ok": True,
                "registry_ready_n": int(ready_mask.sum()),
                "researcher_freeze_pending_n": int(freeze_mask.sum()),
                "registry_ready_predictor_missing_n": predictor_missing_n,
                "registry_ready_time_illegal_n": time_illegal_n,
                "required_devices_invalid_n": device_invalid_n,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(predictor_rows), errors, warnings, ready_predictors


def _overlap_table(probes: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, set[tuple[str, str, str, int]]]:
    keys = {modality: _key_set(frame) for modality, frame in probes.items()}
    common = keys["behavior"] & keys["ocular"] & keys["movement"]
    combinations = [
        ("behavior", keys["behavior"]),
        ("ocular", keys["ocular"]),
        ("movement", keys["movement"]),
        ("behavior_x_ocular", keys["behavior"] & keys["ocular"]),
        ("behavior_x_movement", keys["behavior"] & keys["movement"]),
        ("ocular_x_movement", keys["ocular"] & keys["movement"]),
        ("behavior_x_ocular_x_movement", common),
    ]
    rows = []
    for name, subset in combinations:
        participants = {item[0] for item in subset}
        sessions = {item[1] for item in subset}
        rows.append(
            {
                "key_set": name,
                "probe_n": len(subset),
                "session_n": len(sessions),
                "participant_group_n": len(participants),
            }
        )
    return pd.DataFrame(rows), common


def _common_missingness(
    probes: dict[str, pd.DataFrame],
    ready_predictors: dict[str, list[str]],
    common: set[tuple[str, str, str, int]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for modality in MODALITIES:
        frame = probes[modality].copy()
        frame["__p5_key"] = list(
            zip(
                frame["participant_group_id"].astype(str),
                frame["session_id"].astype(str),
                frame["block_id"].astype(str),
                frame["probe_index_in_block"].astype(int),
            )
        )
        subset = frame[frame["__p5_key"].isin(common)]
        for predictor in ready_predictors.get(modality, []):
            if predictor not in subset.columns:
                finite_n = 0
            else:
                values = pd.to_numeric(subset[predictor], errors="coerce")
                finite_n = int(np.isfinite(values).sum())
            denominator = int(len(common))
            rows.append(
                {
                    "scientific_modality": modality,
                    "predictor_column": predictor,
                    "common_key_n": denominator,
                    "finite_n": finite_n,
                    "missing_n": denominator - finite_n,
                    "finite_fraction": (finite_n / denominator) if denominator else np.nan,
                    "missing_policy": "audit_only_no_zero_fill_no_imputation",
                }
            )
    return pd.DataFrame(rows)


def run_p5_interface_smoke(
    science_root: str | Path,
    *,
    output_root: str | Path | None = None,
    replace: bool = True,
) -> dict[str, Any]:
    """Validate the Behavior+Ocular+Movement handoff boundary without fitting.

    The smoke test is read-only with respect to all three science-modality inputs.
    It never materializes the unified registry and never performs preprocessing,
    feature selection, imputation, LOSO, supervised fitting, or performance scoring.
    """
    science = Path(science_root).expanduser().resolve()
    out = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else science / "P5_InterfaceSmoke"
    )
    if out.exists() and replace:
        shutil.rmtree(out)
    if out.exists() and not replace:
        raise FileExistsError(out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    warnings: list[str] = []
    handoffs, probes, manifests, probe_paths = _load_science_inputs(science)
    handoff_audit, predictor_audit, handoff_errors, handoff_warnings, ready_predictors = _audit_handoffs(handoffs, probes)
    errors.extend(handoff_errors)
    warnings.extend(handoff_warnings)

    overlap, common = _overlap_table(probes)
    if not common:
        errors.append("three_modality_common_probe_key_is_empty")
    missingness = _common_missingness(probes, ready_predictors, common)

    for modality, manifest in manifests.items():
        for flag in ("final_feature_registry_mutated", "formal_registry_mutated", "supervised_model_run", "multimodal_model_run"):
            if flag in manifest and _bool_value(manifest[flag]):
                errors.append(f"{modality}:source_manifest_stopline_violation:{flag}")

    handoff_audit.to_csv(out / "tables/p5_handoff_audit.csv", index=False, encoding="utf-8-sig")
    predictor_audit.to_csv(out / "tables/p5_predictor_audit.csv", index=False, encoding="utf-8-sig")
    overlap.to_csv(out / "tables/p5_key_overlap.csv", index=False, encoding="utf-8-sig")
    missingness.to_csv(out / "tables/p5_common_key_missingness.csv", index=False, encoding="utf-8-sig")

    common_row = overlap[overlap["key_set"].eq("behavior_x_ocular_x_movement")].iloc[0]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "scope": "P5 Behavior + Ocular + Movement interface smoke only",
        "science_root": str(science),
        "output_root": str(out),
        "probe_sources": probe_paths,
        "registry_ready_predictors": ready_predictors,
        "common_probe_n": int(common_row["probe_n"]),
        "common_session_n": int(common_row["session_n"]),
        "common_participant_group_n": int(common_row["participant_group_n"]),
        "errors": errors,
        "warnings": warnings,
        "pending_researcher_freeze_is_warning_not_auto_selection": True,
        "missing_policy": "no_zero_fill_no_imputation",
        "final_feature_registry_mutated": False,
        "supervised_model_run": False,
        "multimodal_model_run": False,
        "performance_metric_computed": False,
        "stop_after_smoke": True,
    }
    (out / "manifests/p5_interface_smoke_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
