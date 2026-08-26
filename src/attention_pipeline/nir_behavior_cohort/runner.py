from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..behavior_formal.extract import formal_block_files
from ..config import Config
from ..nir_behavior.contract import OPTIONAL_NIR_QC_COLUMNS, subject_output_paths
from ..nir_behavior.discovery import (
    alignment_output_root,
    behavior_config,
    sha256,
)
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
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _header_signature(columns: list[str]) -> str:
    canonical = "\x1f".join(columns)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256(path),
        "column_count": int(len(header)),
        "header_signature": _header_signature(header),
    }


def _existing_alignment_status(aconfig: Config, subject: str) -> dict[str, Any]:
    paths = subject_output_paths(alignment_output_root(aconfig), subject)
    completion_path = paths["completion"]
    result: dict[str, Any] = {
        "alignment_completion": str(completion_path),
        "alignment_completion_exists": completion_path.exists(),
        "alignment_status": "",
        "alignment_required_artifact_count": 0,
        "alignment_required_artifacts_present": 0,
        "alignment_all_required_artifacts_present": False,
        "alignment_status_error": "",
    }
    if not completion_path.exists():
        return result

    try:
        payload = json.loads(completion_path.read_text(encoding="utf-8"))
        required = [Path(item) for item in payload.get("required_artifacts", [])]
        present = sum(path.exists() for path in required)
        result.update(
            {
                "alignment_status": str(payload.get("status", "")),
                "alignment_required_artifact_count": int(len(required)),
                "alignment_required_artifacts_present": int(present),
                "alignment_all_required_artifacts_present": bool(
                    required and present == len(required)
                ),
            }
        )
    except Exception as exc:
        result["alignment_status_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _subject_summary(qc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject, grp in qc.groupby("subject", sort=True):
        rows.append(
            {
                "subject": subject,
                "eye_block_units": int(len(grp)),
                "missing_eye_block_count": int(
                    grp["missing_eye_block"].fillna(True).sum()
                ),
                "has_all_four_eye_blocks": bool(
                    len(grp) == 4
                    and not grp["missing_eye_block"].fillna(True).any()
                ),
                "zero_usable_pir_unit_count": int(
                    pd.to_numeric(grp["pir_n"], errors="coerce")
                    .fillna(0)
                    .eq(0)
                    .sum()
                ),
                "pir_usable_fraction_min": float(
                    pd.to_numeric(
                        grp["pir_usable_fraction"], errors="coerce"
                    ).min()
                ),
                "pir_usable_fraction_median": float(
                    pd.to_numeric(
                        grp["pir_usable_fraction"], errors="coerce"
                    ).median()
                ),
                "pir_usable_fraction_max": float(
                    pd.to_numeric(
                        grp["pir_usable_fraction"], errors="coerce"
                    ).max()
                ),
                "internal_coverage_fraction_estimate_min": float(
                    pd.to_numeric(
                        grp["internal_coverage_fraction_estimate"],
                        errors="coerce",
                    ).min()
                ),
                "sampling_rate_hz_estimate_median": float(
                    pd.to_numeric(
                        grp["sampling_rate_hz_estimate"], errors="coerce"
                    ).median()
                ),
                "max_temporal_gap_sec_max": float(
                    pd.to_numeric(
                        grp["max_temporal_gap_sec"], errors="coerce"
                    ).max()
                ),
                "roi_clipped_fraction_max": float(
                    pd.to_numeric(
                        grp["roi_clipped_fraction"], errors="coerce"
                    ).max()
                ),
                "ritnet_found_fraction_min": float(
                    pd.to_numeric(
                        grp["ritnet_found_fraction"], errors="coerce"
                    ).min()
                ),
                "oar_available_fraction_min": float(
                    pd.to_numeric(
                        grp["oar_available_fraction"], errors="coerce"
                    ).min()
                ),
                "oar_p90_available_fraction_min": float(
                    pd.to_numeric(
                        grp["oar_p90_available_fraction"], errors="coerce"
                    ).min()
                ),
                "ocular_fraction_available_min": float(
                    pd.to_numeric(
                        grp.get(
                            "ocular_fraction_available_fraction",
                            pd.Series(dtype=float),
                        ),
                        errors="coerce",
                    ).min()
                ),
                "iris_outer_fraction_available_min": float(
                    pd.to_numeric(
                        grp.get(
                            "iris_outer_fraction_available_fraction",
                            pd.Series(dtype=float),
                        ),
                        errors="coerce",
                    ).min()
                ),
                "ocular_fragmented_candidate_fraction_max": float(
                    pd.to_numeric(
                        grp["ocular_fragmented_candidate_fraction"],
                        errors="coerce",
                    ).max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _anomaly_review_table(qc: pd.DataFrame) -> pd.DataFrame:
    score_columns = [
        column for column in qc.columns if column.endswith("_robust_z")
    ]
    base = [
        "subject",
        "block_num",
        "eye",
        "missing_eye_block",
        "n_nir_rows",
        "unique_unix_ms_count",
        "rows_per_sec_observed",
        "sampling_rate_hz_estimate",
        "internal_coverage_fraction_estimate",
        "duplicate_unix_ms_count",
        "pir_numeric_finite_fraction",
        "pir_normalization_valid_fraction",
        "pir_usable_fraction",
        "max_temporal_gap_sec",
        "roi_clipped_fraction",
        "ritnet_found_fraction",
        "oar_available_fraction",
        "oar_p90_available_fraction",
        "oar_p90_minus_median_median",
        "ocular_fraction_available_fraction",
        "ocular_fraction_median",
        "iris_outer_fraction_available_fraction",
        "iris_outer_fraction_median",
        "pupil_fraction_available_fraction",
        "pupil_fraction_median",
        "ocular_fragmented_candidate_fraction",
    ]
    result = qc[
        [column for column in (*base, *score_columns) if column in qc.columns]
    ].copy()
    result["structural_review_candidate"] = (
        result["missing_eye_block"].fillna(False)
        | pd.to_numeric(
            result["n_nir_rows"], errors="coerce"
        ).fillna(0).eq(0)
        | pd.to_numeric(
            result["duplicate_unix_ms_count"], errors="coerce"
        ).fillna(0).gt(0)
        | pd.to_numeric(
            result["pir_usable_fraction"], errors="coerce"
        ).fillna(0).eq(0)
    )
    # Distribution scores are intentionally continuous. No |z| threshold is
    # converted into a boolean exclusion/review rule in the exploratory phase.
    return result


def _write_phase1_review_bundle(
    *,
    output_root: Path,
    inventory_dir: Path,
    qc_dir: Path,
    provenance_dir: Path,
) -> Path:
    review_dir = output_root / "phase1_review_bundle"
    review_dir.mkdir(parents=True, exist_ok=True)

    sources = (
        inventory_dir / "cohort_preflight_summary.json",
        inventory_dir / "cohort_discovery.csv",
        qc_dir / "subject_eye_block_qc.csv",
        qc_dir / "subject_qc.csv",
        qc_dir / "behavior_cohort_qc.csv",
        qc_dir / "cohort_anomaly_flags.csv",
        provenance_dir / "cohort_manifest.json",
    )
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(
                f"Phase 1 review bundle source missing: {source}"
            )
        shutil.copy2(source, review_dir / source.name)

    (review_dir / "README.md").write_text(
        "# Phase 1 review bundle\n\n"
        "This directory is a small review-only bundle for the NIR + Behavior "
        "cohort Phase 1 preflight/QC.\n\n"
        "It contains cohort inventory, subject×eye×block QC, subject-level "
        "summaries, Behavior QC, continuous anomaly-review scores, and the "
        "provenance manifest. It intentionally does **not** copy frame-level "
        "full-class NIR CSV files.\n\n"
        "Interpretation boundary: all robust-z and candidate fields are "
        "descriptive review aids only. No subject/eye exclusion, QC cutoff, "
        "eye fusion, PIR standardization choice, blink label, or PERCLOS rule "
        "is applied in Phase 1.\n",
        encoding="utf-8",
    )
    return review_dir


def run_phase1_cohort_qc(
    config: Config,
    *,
    subjects: list[str] | None = None,
    output_root_override: str | Path | None = None,
) -> dict[str, Any]:
    aconfig = alignment_config(config)
    selected = selected_cohort_subjects(config, subjects)
    if not selected:
        raise RuntimeError(
            "No completed production full-class subjects discovered"
        )

    qc_cfg = config.section("qc")
    provenance_cfg = config.section("provenance")
    hash_fullclass = bool(
        provenance_cfg.get("hash_fullclass_csv", False)
    )
    output_root = (
        Path(output_root_override).expanduser().resolve()
        if output_root_override is not None
        else cohort_output_root(config)
    )
    inventory_dir = output_root / "00_inventory"
    qc_dir = output_root / "01_qc"
    provenance_dir = output_root / "provenance"
    review_dir = output_root / "phase1_review_bundle"
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
            header_columns = pd.read_csv(
                source.csv_path,
                nrows=0,
                encoding="utf-8-sig",
            ).columns.tolist()
            optional_present = [
                column
                for column in OPTIONAL_NIR_QC_COLUMNS
                if column in header_columns
            ]
            optional_missing = [
                column
                for column in OPTIONAL_NIR_QC_COLUMNS
                if column not in header_columns
            ]
            b1_identity = _file_identity(behavior_files[0])
            b2_identity = _file_identity(behavior_files[1])
            alignment_status = _existing_alignment_status(
                aconfig,
                subject,
            )
            block_counts = {
                f"b{block}_{eye}_rows": int(
                    (
                        (nir["block_num"] == block)
                        & (nir["eye"] == eye)
                    ).sum()
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
                    "fullclass_csv_sha256": (
                        sha256(source.csv_path)
                        if hash_fullclass
                        else None
                    ),
                    "fullclass_column_count": int(len(header_columns)),
                    "fullclass_header_signature": _header_signature(
                        header_columns
                    ),
                    "optional_qc_columns_present": ";".join(
                        optional_present
                    ),
                    "optional_qc_columns_missing": ";".join(
                        optional_missing
                    ),
                    "completion_marker": str(source.completion_path),
                    "completion_marker_sha256": sha256(
                        source.completion_path
                    ),
                    "completion_status": source.completion.get("status"),
                    "extension_version": source.completion.get(
                        "extension_version"
                    ),
                    "alternative_run_count": int(
                        len(source.alternatives)
                    ),
                    "alternative_runs": ";".join(
                        str(path) for path in source.alternatives
                    ),
                    "behavior_b1": b1_identity["path"],
                    "behavior_b1_size_bytes": b1_identity["size_bytes"],
                    "behavior_b1_sha256": b1_identity["sha256"],
                    "behavior_b1_column_count": b1_identity[
                        "column_count"
                    ],
                    "behavior_b1_header_signature": b1_identity[
                        "header_signature"
                    ],
                    "behavior_b2": b2_identity["path"],
                    "behavior_b2_size_bytes": b2_identity["size_bytes"],
                    "behavior_b2_sha256": b2_identity["sha256"],
                    "behavior_b2_column_count": b2_identity[
                        "column_count"
                    ],
                    "behavior_b2_header_signature": b2_identity[
                        "header_signature"
                    ],
                    "behavior_rows": int(len(behavior)),
                    "behavior_trials_b1": int(
                        (behavior["block_num"] == 1).sum()
                    ),
                    "behavior_trials_b2": int(
                        (behavior["block_num"] == 2).sum()
                    ),
                    "probe_count": int(
                        pd.to_numeric(
                            behavior["is_probe"], errors="coerce"
                        )
                        .eq(1)
                        .sum()
                    ),
                    **alignment_status,
                    **block_counts,
                    "all_four_eye_blocks": all(
                        value > 0 for value in block_counts.values()
                    ),
                    "preflight_error": "",
                }
            )

            for block_num in (1, 2):
                for eye in ("left", "right"):
                    grp = nir[
                        (nir["block_num"] == block_num)
                        & (nir["eye"] == eye)
                    ]
                    eye_block_rows.append(
                        summarize_eye_block(
                            grp,
                            subject=subject,
                            block_num=block_num,
                            eye=eye,
                            slope_bin_sec=float(
                                qc_cfg.get("slope_bin_sec", 5.0)
                            ),
                            fragmentation_component_count_gt=int(
                                qc_cfg.get(
                                    "fragmentation_component_count_gt",
                                    1,
                                )
                            ),
                            fragmentation_largest_fraction_lt=float(
                                qc_cfg.get(
                                    "fragmentation_largest_fraction_lt",
                                    0.90,
                                )
                            ),
                        )
                    )

            for _block_num, grp in behavior.groupby(
                "block_num",
                sort=True,
            ):
                behavior_rows.append(
                    summarize_behavior_block(grp.copy())
                )
        except Exception as exc:
            errors.append(
                {
                    "subject": subject,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            discovery_rows.append(
                {
                    "subject": subject,
                    "preflight_error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "all_four_eye_blocks": False,
                }
            )

    discovery = pd.DataFrame(discovery_rows)
    eye_block_qc = (
        add_review_scores(pd.DataFrame(eye_block_rows))
        if eye_block_rows
        else pd.DataFrame()
    )
    behavior_qc = pd.DataFrame(behavior_rows)
    subject_qc = (
        _subject_summary(eye_block_qc)
        if not eye_block_qc.empty
        else pd.DataFrame()
    )
    anomaly = (
        _anomaly_review_table(eye_block_qc)
        if not eye_block_qc.empty
        else pd.DataFrame()
    )

    discovery_path = inventory_dir / "cohort_discovery.csv"
    summary_path = inventory_dir / "cohort_preflight_summary.json"
    eye_qc_path = qc_dir / "subject_eye_block_qc.csv"
    subject_qc_path = qc_dir / "subject_qc.csv"
    behavior_qc_path = qc_dir / "behavior_cohort_qc.csv"
    anomaly_path = qc_dir / "cohort_anomaly_flags.csv"
    manifest_path = provenance_dir / "cohort_manifest.json"

    discovery.to_csv(
        discovery_path,
        index=False,
        encoding="utf-8-sig",
    )
    eye_block_qc.to_csv(
        eye_qc_path,
        index=False,
        encoding="utf-8-sig",
    )
    subject_qc.to_csv(
        subject_qc_path,
        index=False,
        encoding="utf-8-sig",
    )
    behavior_qc.to_csv(
        behavior_qc_path,
        index=False,
        encoding="utf-8-sig",
    )
    anomaly.to_csv(
        anomaly_path,
        index=False,
        encoding="utf-8-sig",
    )

    alignment_completed = int(
        (
            discovery.get(
                "alignment_status",
                pd.Series(dtype=str),
            )
            .fillna("")
            .eq("complete")
        ).sum()
    )
    summary = {
        "pipeline": config.section("pipeline"),
        "config_digest": config.digest,
        "subjects_requested": len(selected),
        "subjects_preflight_ok": int(
            discovery.get(
                "preflight_error",
                pd.Series(dtype=str),
            )
            .fillna("")
            .eq("")
            .sum()
        ),
        "subjects_preflight_failed": int(len(errors)),
        "eye_block_qc_rows": int(len(eye_block_qc)),
        "behavior_qc_rows": int(len(behavior_qc)),
        "existing_schema2_alignment_complete_subjects": (
            alignment_completed
        ),
        "fullclass_column_count_distribution": (
            discovery["fullclass_column_count"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
            if "fullclass_column_count" in discovery.columns
            else {}
        ),
        "fullclass_header_signature_count": int(
            discovery.get(
                "fullclass_header_signature",
                pd.Series(dtype=str),
            )
            .dropna()
            .nunique()
        ),
        "review_bundle": str(review_dir),
        "errors": errors,
        "interpretation": (
            "Phase 1 descriptive QC only; no exclusion, eye-fusion, "
            "standardization, blink/PERCLOS, or model-selection rule is "
            "applied."
        ),
    }
    _json_dump(summary_path, summary)

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
        "output_root_override": (
            str(output_root_override)
            if output_root_override is not None
            else None
        ),
        "analysis_policy": config.section("analysis_policy"),
        "outputs": {
            "cohort_discovery": str(discovery_path),
            "preflight_summary": str(summary_path),
            "subject_eye_block_qc": str(eye_qc_path),
            "subject_qc": str(subject_qc_path),
            "behavior_cohort_qc": str(behavior_qc_path),
            "cohort_anomaly_flags": str(anomaly_path),
            "phase1_review_bundle": str(review_dir),
        },
    }
    _json_dump(manifest_path, manifest)
    (
        provenance_dir / "analysis_config_snapshot.yaml"
    ).write_text(
        config.path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    review_bundle = _write_phase1_review_bundle(
        output_root=output_root,
        inventory_dir=inventory_dir,
        qc_dir=qc_dir,
        provenance_dir=provenance_dir,
    )
    return {
        **summary,
        "output_root": str(output_root),
        "review_bundle": str(review_bundle),
    }
