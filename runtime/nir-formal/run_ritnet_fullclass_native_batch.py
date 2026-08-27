"""Internal batch selection/dispatch for the canonical final full-class workflow.

Historical formal sources are accepted only after strict completion validation.
For each subject this runner selects one unambiguous source and dispatches the
canonical ``run_ritnet_fullclass_extension.py`` final <=1 GiB pipeline. Legacy
label-chunk/compression controls are intentionally absent.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from formal_completion import validate_completion
from ritnet_fullclass_contract import FULLCLASS_VERSION, normalize_subject
from ritnet_label_store import sha256_file

EXTENSION = PACKAGE_ROOT / "run_ritnet_fullclass_extension.py"


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _formal_identity(marker: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": marker.get("run_id"),
        "subject": marker.get("subject"),
        "video": marker.get("video"),
        "focuswave_release": marker.get("focuswave_release"),
        "phases": marker.get("phases"),
        "expected_frames": marker.get("expected_frames"),
        "yolo_batch_size": marker.get("yolo_batch_size"),
        "yolo_model_sha256": marker.get("yolo_model_sha256"),
        "ritnet_enabled": marker.get("ritnet_enabled"),
        "ritnet_batch_size": marker.get("ritnet_batch_size"),
        "ritnet_precision": marker.get("ritnet_precision"),
        "ritnet_model_sha256": marker.get("ritnet_model_sha256"),
    }


def discover_source_runs(
    output_root: Path,
    subjects: set[str] | None = None,
    requested_subjects: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Discover strictly complete source runs, narrowing the filesystem scan first.

    A subject filter is applied to directory names before completion validation so
    a single-subject dry-run does not inspect every historical formal directory.
    The marker subject is still authoritative after validation.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    subjects = set(requested_subjects) if requested_subjects is not None else subjects
    if subjects:
        run_dirs = {
            run_dir
            for subject in subjects
            for run_dir in output_root.glob(f"{subject}_formal_*")
        }
    else:
        run_dirs = set(output_root.glob("sub-*_formal_*"))
    for run_dir in sorted(run_dirs):
        if not run_dir.is_dir():
            continue
        validation = validate_completion(run_dir)
        if not validation.valid or not validation.marker:
            continue
        marker = validation.marker
        eyes_path = run_dir / "eyes.csv"
        if not eyes_path.is_file():
            continue
        subject = normalize_subject(marker.get("subject") or run_dir.name.split("_formal_", 1)[0])
        grouped.setdefault(subject, []).append(
            {
                "run_dir": run_dir,
                "marker": marker,
                "validation_reason": validation.reason,
                "eyes_sha256": sha256_file(eyes_path),
                "formal_identity": _formal_identity(marker),
            }
        )
    return grouped


def _same_source_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left["formal_identity"] == right["formal_identity"] and left["eyes_sha256"] == right["eyes_sha256"]


def select_run(
    candidates: list[dict[str, Any]],
    *,
    expected_yolo_batch_size: int | None = None,
) -> tuple[dict[str, Any], list[Path], str]:
    if not candidates:
        raise RuntimeError("select_run requires at least one validated source candidate")
    original_candidates = list(candidates)
    # Compatibility callers may still pass the historical production batch gate.
    # The canonical selector itself does not prefer one validated source based on
    # retroactive YOLO metadata; it only uses that gate when explicitly requested.
    if expected_yolo_batch_size is not None:
        explicit = [
            item for item in candidates
            if item["marker"].get("yolo_batch_size") not in (None, "")
            and int(item["marker"].get("yolo_batch_size")) == int(expected_yolo_batch_size)
        ]
        missing = [item for item in candidates if item["marker"].get("yolo_batch_size") in (None, "")]
        matching = explicit or missing
        if not matching:
            available = sorted({item["marker"].get("yolo_batch_size") for item in candidates}, key=str)
            raise RuntimeError(
                "No validated formal source matches configured production yolo.batch_size="
                f"{expected_yolo_batch_size}; available={available}"
            )
        if len(matching) == 1:
            item = matching[0]
            reason = (
                "legacy_source_yolo_batch_size_not_recorded_accepted"
                if item in missing else "unique_validated_source_matching_configured_yolo_batch_size"
            )
            return item, [other["run_dir"] for other in original_candidates if other is not item], reason
        candidates = matching
    if len(candidates) == 1:
        return candidates[0], [], "unique_validated_historical_formal_source"

    reference = candidates[0]
    if not all(_same_source_identity(reference, item) for item in candidates[1:]):
        details = [
            {
                "run_dir": str(item["run_dir"]),
                "run_id": item["marker"].get("run_id"),
                "eyes_sha256": item["eyes_sha256"],
                "yolo_batch_size": item["marker"].get("yolo_batch_size"),
            }
            for item in candidates
        ]
        raise RuntimeError(
            "Ambiguous validated historical formal sources for one subject; refusing silent selection: "
            + json.dumps(details, ensure_ascii=False, sort_keys=True)
        )
    candidates_sorted = sorted(candidates, key=lambda item: str(item["run_dir"]).lower())
    selected = candidates_sorted[0]
    return selected, [item["run_dir"] for item in candidates_sorted[1:]], "equivalent_duplicate_historical_sources_same_identity_and_eyes_sha256"


def parse_subjects(text: str | None) -> list[str] | None:
    if text is None:
        return None
    return [normalize_subject(value) for value in text.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch final <=1 GiB RITnet full-class producer")
    parser.add_argument("--output", type=Path, required=True, help="Historical formal output root")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--subjects", help="Optional comma-separated subject filter")
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output.resolve()
    config_path = args.config.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    config = load_config(config_path)
    requested = parse_subjects(args.subjects)
    excluded = {
        normalize_subject(value)
        for value in config.get("batch", {}).get("subjects", {}).get("exclude", [])
    }
    discovery_subjects = set(requested) if requested is not None else None
    grouped = discover_source_runs(output_root, requested_subjects=requested)
    if requested is None:
        subjects = sorted(
            (subject for subject in grouped if subject not in excluded),
            key=lambda value: int(value[4:]),
        )
    else:
        missing = [subject for subject in requested if subject not in grouped]
        if missing:
            raise FileNotFoundError("No strictly validated source run found for: " + ", ".join(missing))
        subjects = [subject for subject in requested if subject not in excluded]
    if not subjects:
        raise RuntimeError("No strictly validated formal source runs selected")

    expected_yolo_batch_size = int(config.get("yolo", {}).get("batch_size", 8))
    selections = []
    for subject in subjects:
        selected, alternatives, selection_reason = select_run(grouped[subject])
        marker = selected["marker"]
        source_yolo_batch_size_recorded = not (
            marker.get("yolo_batch_size") is None
            or str(marker.get("yolo_batch_size")).strip() == ""
        )
        selections.append(
            {
                "subject": subject,
                "run_dir": str(selected["run_dir"]),
                "source_run_id": marker.get("run_id"),
                "source_yolo_batch_size": marker.get("yolo_batch_size"),
                "source_yolo_batch_size_recorded": source_yolo_batch_size_recorded,
                "source_yolo_batch_size_note": (
                    None
                    if source_yolo_batch_size_recorded
                    else "not_recorded_in_legacy_completion_marker"
                ),
                "source_eyes_sha256": selected["eyes_sha256"],
                "source_validation": selected["validation_reason"],
                "selection_reason": selection_reason,
                "alternatives": [str(path) for path in alternatives],
            }
        )

    preview = {
        "fullclass_version": FULLCLASS_VERSION,
        "selected_count": len(selections),
        "configured_source_yolo_batch_size": expected_yolo_batch_size,
        "source_completion_contract_enforced": True,
        "source_ambiguity_mtime_selection_allowed": False,
        "source_video_content_sha256_frozen_by_child": True,
        "legacy_label_chunk_storage_enabled": False,
        "selections": selections,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    continue_on_error = bool(config.get("batch", {}).get("continue_on_error", True))
    results = []
    for index, item in enumerate(selections, start=1):
        command = [
            sys.executable,
            str(EXTENSION),
            "--run-dir", item["run_dir"],
            "--config", str(config_path),
            "--device", str(args.device),
            "--source-selection-reason", item["selection_reason"],
        ]
        for alternative in item["alternatives"]:
            command.extend(["--source-alternative-run", alternative])
        print(f"[RUN {index}/{len(selections)}] {item['subject']}: {item['run_dir']}")
        print("  " + subprocess.list2cmdline(command))
        completed = subprocess.run(command, cwd=PACKAGE_ROOT)
        results.append(
            {
                "subject": item["subject"],
                "run_dir": item["run_dir"],
                "source_run_id": item["source_run_id"],
                "source_eyes_sha256": item["source_eyes_sha256"],
                "selection_reason": item["selection_reason"],
                "alternatives": item["alternatives"],
                "returncode": completed.returncode,
                "status": "completed_or_strictly_skipped" if completed.returncode == 0 else "failed",
            }
        )
        if completed.returncode != 0 and not continue_on_error:
            break

    summary_path = output_root / "ritnet_fullclass_batch_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Batch summary -> {summary_path}")
    return 1 if any(result["status"] == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
