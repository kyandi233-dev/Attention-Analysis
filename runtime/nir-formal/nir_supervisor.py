"""Low-noise supervisor for the formal NVIDIA NIR batch.

The supervisor is deliberately conservative: it never kills a live analysis,
never edits completion markers by hand, and only starts a recovery run when no
formal batch or pipeline process is alive.  It writes one JSON event per state
transition so a chat monitor can report anomalies without repeating normal
progress.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUBJECT_RE = re.compile(r"sub-\d{3}")
TERMINAL = {"complete", "failed", "smoke_complete"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def powershell_processes() -> list[dict[str, Any]]:
    command = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object {$_.CommandLine -match 'run_formal_batch|run_pipeline.py'} | "
        "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if not result.stdout.strip():
        return []
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else [value]


def subject_from_command(command: str) -> str | None:
    match = re.search(r"sub-\d{3}(?=[_\\/])", command)
    return match.group(0) if match else None


def command_key(command: str) -> str:
    """Return script arguments, ignoring the venv/base Python executable."""
    normalized = " ".join(command.replace('"', "").split()).lower()
    marker = ".exe "
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    return normalized


def scan(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for directory in sorted(root.iterdir() if root.exists() else []):
        if not directory.is_dir():
            continue
        # Ignore legacy pre-batch8 output directories; they are audit evidence,
        # not candidates for automatic recovery or completion counting.
        if not directory.name.endswith("_yolo8_b16_fp32"):
            continue
        marker = read_json(directory / "completion.json")
        if marker:
            rows.append(
                {
                    "name": directory.name,
                    "subject": marker.get("subject") or subject_from_command(str(marker.get("video", ""))),
                    "status": marker.get("status"),
                    "processed": marker.get("processed_frames"),
                    "expected": marker.get("expected_frames"),
                    "error": marker.get("error"),
                    "failure_stage": marker.get("failure_stage"),
                    "updated": marker.get("finished_at_utc") or marker.get("started_at_utc"),
                }
            )
    processes = powershell_processes()
    for process in processes:
        process["command"] = str(process.get("CommandLine") or "")
        process["subject"] = subject_from_command(process["command"])
        process["kind"] = "batch" if "run_formal_batch.py" in process["command"] else "pipeline"
    return rows, processes


def emit(event_path: Path, event: dict[str, Any]) -> None:
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": now(), **event}, ensure_ascii=False) + "\n")


def launch(runtime: Path, root: Path, subjects: str | None = None, force: bool = False) -> int:
    python = runtime / ".venv_nir_gpu" / "Scripts" / "python.exe"
    if not python.exists():
        python = Path(r"D:\Project\厚粲杯\08_算法\.venv_nir_gpu\Scripts\python.exe")
    runner = runtime / "run_formal_batch.py"
    log_dir = root / "_runtime_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (log_dir / f"supervisor_{stamp}.out.log").open("w", encoding="utf-8")
    err = (log_dir / f"supervisor_{stamp}.err.log").open("w", encoding="utf-8")
    args = [str(runner)]
    if subjects:
        args += ["--subjects", subjects]
    if force:
        args.append("--force")
    args += [
        "--backend", "pytorch-cuda", "--device", "0",
        "--ritnet-precision", "fp32", "--ritnet-batch-size", "16",
        "--yolo-batch-size", "8",
    ]
    process = subprocess.Popen([str(python), *args], cwd=runtime, stdout=out, stderr=err)
    out.close()
    err.close()
    return process.pid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    root = args.output_root.resolve()
    runtime = args.runtime.resolve()
    state_path = root / "_runtime_logs" / "supervisor_state.json"
    event_path = root / "_runtime_logs" / "supervisor_events.jsonl"
    state = read_json(state_path) or {"last_signature": "", "retries": {}}

    while True:
        rows, processes = scan(root)
        # A Windows venv launcher and its base interpreter expose the same
        # command line as two PIDs.  They are one execution chain, not two
        # analyses, so deduplicate by the full command before judging overlap.
        unique: dict[tuple[str, str], dict[str, Any]] = {}
        for process in processes:
            key = (process["kind"], command_key(process["command"]))
            unique.setdefault(key, process)
        instances = list(unique.values())
        active_batches = [p for p in instances if p["kind"] == "batch"]
        active_pipelines = [p for p in instances if p["kind"] == "pipeline"]
        by_subject: dict[str, list[dict[str, Any]]] = {}
        for process in active_pipelines:
            if process.get("subject"):
                by_subject.setdefault(process["subject"], []).append(process)

        failed = [r for r in rows if r.get("status") == "failed"]
        incomplete_without_process = [
            r for r in rows
            if r.get("status") in {"initializing", "running"}
            and not by_subject.get(r.get("subject", ""))
        ]
        duplicates = [subject for subject, values in by_subject.items() if len(values) > 1]
        complete_count = sum(r.get("status") == "complete" for r in rows)
        signature = json.dumps(
            {
                "complete": complete_count,
                "failed": [(r.get("subject"), r.get("error")) for r in failed],
                "orphaned": [r.get("subject") for r in incomplete_without_process],
                "duplicates": duplicates,
                "batch": bool(active_batches),
                "pipeline": sorted(by_subject),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if signature != state.get("last_signature"):
            emit(event_path, {"type": "state_change", "summary": json.loads(signature)})
            state["last_signature"] = signature

        if duplicates:
            emit(event_path, {"type": "duplicate_process", "subjects": duplicates, "action": "none_kill"})

        recovery_started = False
        if incomplete_without_process and not active_batches and not active_pipelines:
            subject = incomplete_without_process[0].get("subject")
            if subject:
                key = f"orphaned:{subject}"
                retries = int(state.setdefault("retries", {}).get(key, 0))
                if retries < args.max_retries:
                    pid = launch(runtime, root, subject, force=True)
                    state["retries"][key] = retries + 1
                    emit(event_path, {"type": "auto_repair", "subject": subject, "reason": "orphaned_state", "pid": pid})
                    recovery_started = True

        if failed and not active_batches and not active_pipelines:
            recoverable = next((r for r in failed if "CUDA" in str(r.get("error", "")) or r.get("failure_stage") == "unhandled_exit"), None)
            if recoverable:
                subject = recoverable.get("subject")
                key = f"failed:{subject}"
                retries = int(state.setdefault("retries", {}).get(key, 0))
                if subject and retries < args.max_retries:
                    pid = launch(runtime, root, subject, force=True)
                    state["retries"][key] = retries + 1
                    emit(event_path, {"type": "auto_repair", "subject": subject, "reason": "recoverable_failure", "pid": pid})
                    recovery_started = True

        # If the batch runner itself disappeared between subjects, resume the
        # normal queue after targeted recovery decisions.  A live batch or
        # pipeline always wins, so this cannot create a competing instance.
        if (
            not active_batches
            and not active_pipelines
            and not recovery_started
            and complete_count < 72
            and any(r.get("status") != "complete" for r in rows)
        ):
            pid = launch(runtime, root)
            emit(event_path, {"type": "auto_resume", "reason": "batch_process_absent", "pid": pid})

        if complete_count >= 72 and not active_batches and not active_pipelines:
            emit(event_path, {"type": "complete", "complete": complete_count})
            write_json(state_path, state)
            return 0

        write_json(state_path, state)
        time.sleep(max(10, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
