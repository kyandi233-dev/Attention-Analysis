"""Sequential batch launcher for ritnet-fullclass-v2-native640."""
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

from ritnet_fullclass_contract import NATIVE_EXTENSION_VERSION, normalize_subject

EXTENSION = PACKAGE_ROOT / "run_ritnet_fullclass_native_extension.py"


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def load_completion(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "completion.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def discover_source_runs(output_root: Path) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for run_dir in sorted(output_root.glob("sub-*_formal_*")):
        if not run_dir.is_dir():
            continue
        marker = load_completion(run_dir)
        if not marker or marker.get("status") != "complete":
            continue
        if marker.get("max_frames") is not None or marker.get("partial_phase_selection"):
            continue
        if not (run_dir / "eyes.csv").is_file():
            continue
        subject = normalize_subject(marker.get("subject") or run_dir.name.split("_formal_", 1)[0])
        grouped.setdefault(subject, []).append((run_dir, marker))
    return grouped


def select_run(candidates: list[tuple[Path, dict[str, Any]]]) -> tuple[Path, dict[str, Any], list[Path]]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            int(item[1].get("yolo_batch_size", -1) == 8),
            (item[0] / "completion.json").stat().st_mtime_ns,
        ),
        reverse=True,
    )
    selected_dir, selected_marker = ranked[0]
    return selected_dir, selected_marker, [path for path, _ in ranked[1:]]


def parse_subjects(text: str | None) -> list[str] | None:
    if text is None:
        return None
    return [normalize_subject(value) for value in text.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch native640 RITnet evidence producer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--subjects", help="Optional comma-separated subject filter")
    parser.add_argument("--device", default="0")
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--compression", choices=("npz_compressed", "npz_stored"), default="npz_compressed")
    parser.add_argument("--hash-video", action="store_true")
    parser.add_argument("--allow-model-mismatch", action="store_true")
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
            raise FileNotFoundError("No complete source run found for: " + ", ".join(missing))
        subjects = [subject for subject in requested if subject not in excluded]
    if not subjects:
        raise RuntimeError("No complete formal source runs selected")

    selections = []
    for subject in subjects:
        run_dir, marker, alternatives = select_run(grouped[subject])
        selections.append(
            {
                "subject": subject,
                "run_dir": str(run_dir),
                "source_yolo_batch_size": marker.get("yolo_batch_size"),
                "alternatives": [str(path) for path in alternatives],
            }
        )
    preview = {
        "extension": NATIVE_EXTENSION_VERSION,
        "selected_count": len(selections),
        "chunk_rows": args.chunk_rows,
        "compression": args.compression,
        "hash_video": bool(args.hash_video),
        "method": "RITnet FP32 fixed-b16 640x400 + full hard-label evidence store",
        "warning": "chunk_rows=128 remains provisional until sub-031 throughput/compression benchmark",
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
        if args.hash_video:
            command.append("--hash-video")
        if args.allow_model_mismatch:
            command.append("--allow-model-mismatch")
        print(f"[RUN {index}/{len(selections)}] {item['subject']}: {item['run_dir']}")
        print("  " + subprocess.list2cmdline(command))
        completed = subprocess.run(command, cwd=PACKAGE_ROOT)
        results.append(
            {
                "subject": item["subject"],
                "run_dir": item["run_dir"],
                "returncode": completed.returncode,
                "status": "completed" if completed.returncode == 0 else "failed",
            }
        )
        if completed.returncode != 0 and not continue_on_error:
            break

    summary_path = output_root / "ritnet_fullclass_v2_native640_batch_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Batch summary -> {summary_path}")
    return 1 if any(result["status"] == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
