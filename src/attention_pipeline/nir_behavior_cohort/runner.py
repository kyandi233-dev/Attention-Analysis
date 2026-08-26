from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..behavior_formal.extract import formal_block_files
from ..config import Config
from ..nir_behavior.discovery import behavior_config, sha256
from .io import (
    alignment_config,
    cohort_output_root,
    load_behavior_qc_frame,
    load_nir_qc_frame,
    selected_cohort_subjects,
)
from .qc import add_review_scores, summarize_behavior_block, summarize_eye_block


def _git_head(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _header_signature(columns: list[str]) -> str:
    canonical = "\x1f".join(columns)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _subject_summary(qc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject, grp in qc.groupby("subject", sort=True):
        rows.append(
            {
                "subject": subject,
                "eye_block_units": int(len(grp)),
                "missing_eye_block_count": int(grp["missing_eye_block"].fillna(True).sum()),
                "has_all_four_eye_blocks": bool(len(grp) == 4 and not grp["missing_eye_block"].fillna(True).any()),
                "zero_usable_pir_unit_count": int(pd.to_numeric(grp["pir_n"], errors="coerce").fillna(0).eq(0).sum()),
                "pir_usable_fraction_min": float(pd.to_numeric(grp["pir_usable_fraction"], errors="coerce").min()),
                "pir_usable_fraction_median": float(pd.to_numeric(grp["pir_usable_fraction"], errors="coerce").median()),
                "pir_usable_fraction_max": float(pd.to_numeric(grp["pir_usable_fraction"], errors="coerce").max()),
                "max_temporal_gap_sec_max": float(pd.to_numeric(grp["max_temporal_gap_sec"], errors="coerce").max()),
                "roi_clipped_fraction_max": float(pd.to_numeric(grp["roi_clipped_fraction"], errors="coerce").max()),
                "ritnet_found_fraction_min": float(pd.to_numeric(grp["ritnet_found_fraction"], errors="coerce").min()),
                "ocular_fragmented_candidate_fraction_max": float(
                    pd.to_numeric(grp["ocular_fragmented_candidate_fraction"], errors="coerce").max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _anomaly_review_table(qc: pd.DataFrame) -> pd.DataFrame:
    score_columns = [column for column in qc.columns if column.endswith("_robust_z")]
    base = [
        "subject",
        "block_num",
        "eye",
        "missing_eye_block",
        "n_nir_rows",
        "duplicate_unix_ms_count",
        "pir_usable_fraction",
        "max_temporal_gap_sec",
        "roi_clipped_fraction",
        "ritnet_found_fraction",
        "ocular_fragmented_candidate_fraction",
    ]
    result = qc[[column for column in (*base, *score_columns) if column in qc.columns]].copy()
    result["structural_review_candidate"] = (
        result["missing_eye_block"].fillna(False)
        | pd.to_numeric(result["n_nir_rows"], errors="coerce").fillna(0).eq(0)
        | pd.to_numeric(result["duplicate_unix_ms_count"], errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(result["pir_usable_fraction"], errors="coerce").fillna(0).eq(0)
    )
    # Distribution scores are intentionally continuous. No |z| threshold is converted
    # into a boolean exclusion/review rule in the 44-person exploratory phase.
    return result


def run_phase1_cohort_qc(
    config: Config,
    *,
    subjects: list[str] | None = None,
) -> dict[str, Any]:
    aconfig = alignment_config(config)
    selected = selected_cohort_subjects(config, subjects)
    if not selected:
        raise RuntimeError("No completed production full-class subjects discovered")

    qc_cfg = config.section("qc")
    provenance_cfg = config.section("provenance")
    hash_fullclass = bool(provenance_cfg.get("hash_fullclass_csv", False))
    output_root = cohort_output_root(config)
    inventory_dir = output_root / "00_inventory"
    qc_dir = output_root / "01_qc"
    provenance_dir = output_root / "provenance"
    for path in (inventory_dir, qc_dir, provenance_dir):
        path.mkdir(parents=True, exist_ok=True)

    discovery_rows: list[dict[str, Any]] = []
    eye_block_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    bconfig = behavior_config(aconfig)

    for index, subject in enumerate(selected, start=1):
        print(f"[PHASE1 {index}/{len(selected)}] {subject}")
        try:
            nir, source = load_nir_qc_frame(config, subject)
            behavior = load_behavior_qc_frame(config, subject)
            behavior_files = formal_block_files(bconfig, subject)

            source_stat = source.csv_path.stat()
            header_columns = pd.read_csv(source.csv_path, nrows=0, encoding="utf-8-sig").columns.tolist()
            block_counts = {
                f"b{block}_{eye}_rows": int(
                    ((nir["block_num"] == block) & (nir["eye"] == eye)).sum()
                )
                for block in (1, 2)
                for eye in ("left", "right")
            }
            discovery_rows.append(
                {
                    "subject": subject,
                    "fullclass_csv": str(source.csv_path),
                    "fullclass_csv_size_bytes": int(source_stat.st_size),
                    "fullclass_csv_mtime_ns": int(source_stat.st_mtime_ns),
                    "fullclass_csv_sha256": sha256(source.csv_path) if hash_fullclass else None,
                    "fullclass_column_count": int(len(header_columns)),
                    "fullclass_header_signature": _header_signature(header_columns),
                    "completion_marker": str(source.completion_path),
                    "completion_marker_sha256": sha256(source.completion_path),
                    "completion_status": source.completion.get("status"),
                    "extension_version": source.completion.get("extension_version"),
                    "alternative_run_count": int(len(source.alternatives)),
                    "alternative_runs": ";".join(str(path) for path in source.alternatives),
                    "behavior_b1": str(behavior_files[0]),
                    "behavior_b2": str(behavior_files[1]),
                    "behavior_rows": int(len(behavior)),
                    "behavior_trials_b1": int((behavior["block_num"] == 1).sum()),
                    "behavior_trials_b2": int((behavior["block_num"] == 2).sum()),
                    "probe_count": int(pd.to_numeric(behavior["is_probe"], errors="coerce").eq(1).sum()),
                    **block_counts,
                    "all_four_eye_blocks": all(value > 0 for value in block_counts.values()),
                    "preflight_error": "",
                }
            )

            for block_num in (1, 2):
                for eye in ("left", "right"):
                    grp = nir[(nir["block_num"] == block_num) & (nir["eye"] == eye)]
                    eye_block_rows.append(
                        summarize_eye_block(
                            grp,
                            subject=subject,
                            block_num=block_num,
                            eye=eye,
                            slope_bin_sec=float(qc_cfg.get("slope_bin_sec", 5.0)),
                            fragmentation_component_count_gt=int(
                                qc_cfg.get("fragmentation_component_count_gt", 1)
                            ),
                            fragmentation_largest_fraction_lt=float(
                                qc_cfg.get("fragmentation_largest_fraction_lt", 0.90)
                            ),
                        )
                    )

            for _block_num, grp in behavior.groupby("block_num", sort=True):
                behavior_rows.append(summarize_behavior_block(grp.copy()))
        except Exception as exc:
            errors.append({"subject": subject, "error": f"{type(exc).__name__}: {exc}"})
            discovery_rows.append(
                {
                    "subject": subject,
                    "preflight_error": f"{type(exc).__name__}: {exc}",
                    "all_four_eye_blocks": False,
                }
            )

    discovery = pd.DataFrame(discovery_rows)
    eye_block_qc = add_review_scores(pd.DataFrame(eye_block_rows)) if eye_block_rows else pd.DataFrame()
    behavior_qc = pd.DataFrame(behavior_rows)
    subject_qc = _subject_summary(eye_block_qc) if not eye_block_qc.empty else pd.DataFrame()
    anomaly = _anomaly_review_table(eye_block_qc) if not eye_block_qc.empty else pd.DataFrame()

    discovery.to_csv(inventory_dir / "cohort_discovery.csv", index=False, encoding="utf-8-sig")
    eye_block_qc.to_csv(qc_dir / "subject_eye_block_qc.csv", index=False, encoding="utf-8-sig")
    subject_qc.to_csv(qc_dir / "subject_qc.csv", index=False, encoding="utf-8-sig")
    behavior_qc.to_csv(qc_dir / "behavior_cohort_qc.csv", index=False, encoding="utf-8-sig")
    anomaly.to_csv(qc_dir / "cohort_anomaly_flags.csv", index=False, encoding="utf-8-sig")

    summary = {
        "pipeline": config.section("pipeline"),
        "config_digest": config.digest,
        "subjects_requested": len(selected),
        "subjects_preflight_ok": int(discovery.get("preflight_error", pd.Series(dtype=str)).fillna("").eq("").sum()),
        "subjects_preflight_failed": int(len(errors)),
        "eye_block_qc_rows": int(len(eye_block_qc)),
        "behavior_qc_rows": int(len(behavior_qc)),
        "fullclass_column_count_distribution": (
            discovery["fullclass_column_count"].dropna().astype(int).value_counts().sort_index().to_dict()
            if "fullclass_column_count" in discovery.columns
            else {}
        ),
        "fullclass_header_signature_count": int(discovery.get("fullclass_header_signature", pd.Series(dtype=str)).dropna().nunique()),
        "errors": errors,
        "interpretation": "Phase 1 descriptive QC only; no exclusion or model-selection rule is applied.",
    }
    _json_dump(inventory_dir / "cohort_preflight_summary.json", summary)

    repo_root = config.path.parent.parent
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": config.section("pipeline"),
        "cohort_config": str(config.path),
        "cohort_config_digest": config.digest,
        "alignment_config": str(aconfig.path),
        "alignment_config_digest": aconfig.digest,
        "git_head": _git_head(repo_root),
        "subjects": selected,
        "hash_fullclass_csv": hash_fullclass,
        "outputs": {
            "cohort_discovery": str(inventory_dir / "cohort_discovery.csv"),
            "preflight_summary": str(inventory_dir / "cohort_preflight_summary.json"),
            "subject_eye_block_qc": str(qc_dir / "subject_eye_block_qc.csv"),
            "subject_qc": str(qc_dir / "subject_qc.csv"),
            "behavior_cohort_qc": str(qc_dir / "behavior_cohort_qc.csv"),
            "cohort_anomaly_flags": str(qc_dir / "cohort_anomaly_flags.csv"),
        },
    }
    _json_dump(provenance_dir / "cohort_manifest.json", manifest)
    (provenance_dir / "analysis_config_snapshot.yaml").write_text(
        config.path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return {**summary, "output_root": str(output_root)}
