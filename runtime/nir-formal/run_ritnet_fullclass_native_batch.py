"""Internal batch implementation for the single canonical RITnet full-class workflow.

Source formal runs are never trusted from ``completion.status`` alone. Every
candidate must pass ``formal_completion.validate_completion`` before it can be
selected. If multiple validated candidates exist for one subject, selection is
strict: prefer the configured production YOLO batch size and only auto-resolve
true duplicates whose formal identity and ``eyes.csv`` bytes are identical.
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

# Always route subjects through the canonical user-facing single-subject gate.
EXTENSION = PACKAGE_ROOT / "run_ritnet_fullclass_extension.py"


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _formal_identity(marker: dict[str, Any]) -> dict[str, Any]:
    """Fields that make a formal source scientifically distinct for reuse."""
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


def discover_source_runs(output_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Return only formal runs that pass the full completion contract."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run_dir in sorted(output_root.glob("sub-*_formal_*")):
        if not run_dir.is_dir():
            continue
        validation = validate_completion(run_dir)
        if not validation.valid or not validation.marker:
            continue
        marker = validation.marker
        eyes_path = run_dir / "eyes.csv"
        if not eyes_path.is_file():
            # Defensive: validate_completion already requires this artifact.
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
    return (
        left["formal_identity"] == right["formal_identity"]
        and left["eyes_sha256"] == right["eyes_sha256"]
    )


def select_run(
    candidates: list[dict[str, Any]],
    *,
    expected_yolo_batch_size: int,
) -> tuple[dict[str, Any], list[Path], str]:
    """Select one validated formal source without mtime-based ambiguity.

    A source with a different YOLO batch size is not silently substituted because
    full-class reuses its YOLO boxes. Multiple matching candidates are only
    auto-resolved when their formal identity and ``eyes.csv`` bytes are identical.
    """
    if not candidates:
        raise RuntimeError("select_run requires at least one validated source candidate")

    matching = [
        item
        for item in candidates
        if int(item["marker"].get("yolo_batch_size", -1)) == int(expected_yolo_batch_size)
    ]
    if not matching:
        available = sorted({item["marker"].get("yolo_batch_size") for item in candidates}, key=str)
        raise RuntimeError(
            "No validated formal source matches configured production yolo.batch_size="
            f"{expected_yolo_batch_size}; available={available}"
        )

    if len(matching) == 1:
        selected = matching[0]
        alternatives = [item["run_dir"] for item in candidates if item is not selected]
        return selected, alternatives, "unique_validated_source_matching_configured_yolo_batch_size"

    reference = matching[0]
    if not all(_same_source_identity(reference, item) for item in matching[1:]):
        details = [
            {
                "run_dir": str(item["run_dir"]),
                "run_id": item["marker"].get("run_id"),
                "eyes_sha256": item["eyes_sha256"],
                "yolo_batch_size": item["marker"].get("yolo_batch_size"),
            }
            for item in matching
        ]
        raise RuntimeError(
            "Ambiguous validated formal sources for one subject; refusing silent selection: "
            + json.dumps(details, ensure_ascii=False, sort_keys=True)
        )

    matching_sorted = sorted(matching, key=lambda item: str(item["run_dir"]).lower())
    selected = matching_sorted[0]
    alternatives = [item["run_dir"] for item in candidates if item is not selected]
    return selected, alternatives, "equivalent_duplicate_sources_same_formal_identity_and_eyes_sha256"


def parse_subjects(text: str | None) -> list[str] | None:
    if text is None:
        return None
    return [normalize_subject(value) for value in text.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch canonical RITnet full-class producer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--subjects", help="Optional comma-separated subject filter")
    parser.add_argument("--device", default="0")
    # Kept temporarily for CLI compatibility while the old all-label store is
    # removed in later checklist steps. It is not a scientific parameter.
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--compression", choices=("npz_compressed", "npz_stored"), default="npz_compressed")
    # Accepted for compatibility with the canonical outer wrapper; child runs
    # hash source videos unconditionally through run_ritnet_fullclass_extension.py.
    parser.add_argument("--hash-video", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")
    output_root = args.output.resolve()
    config_path = args.config.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    config = load_config(config_path)
    grouped = discover_source_runs(output_root)
    requested = parse_subjects(args.subjects)
    excluded = {
        normalize_subject(value)
        for value in config.get("batch", {}).get("subjects", {}).get("exclude", [])
    }
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
        selected, alternatives, selection_reason = select_run(
            grouped[subject],
            expected_yolo_batch_size=expected_yolo_batch_size,
        )
        marker = selected["marker"]
        selections.append(
            {
                "subject": subject,
                "run_dir": str(selected["run_dir"]),
                "source_run_id": marker.get("run_id"),
                "source_yolo_batch_size": marker.get("yolo_batch_size"),
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
        "source_video_content_sha256_enforced_by_child": True,
        "model_mismatch_override_allowed": False,
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
            "--chunk-rows", str(args.chunk_rows),
            "--compression", str(args.compression),
        ]
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
                "returncode": completed.returncode,
                "status": "completed" if completed.returncode == 0 else "failed",
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
