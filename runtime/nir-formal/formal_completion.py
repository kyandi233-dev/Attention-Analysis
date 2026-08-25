"""Atomic completion markers, strict validation, and formal-run lifecycle guards."""
from __future__ import annotations

import atexit
import csv
import json
import os
import socket
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
MARKER_NAME = "completion.json"
LOCK_NAME = ".run.lock"
REQUIRED_ARTIFACTS = (
    "phase_windows.json",
    "frames.csv",
    "eyes.csv",
    "summary.json",
    "run_manifest.json",
)
TERMINAL_STATUSES = {"complete", "recovery_complete", "smoke_complete", "failed"}


@dataclass(frozen=True)
class CompletionValidation:
    valid: bool
    reason: str
    marker: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunLock:
    path: Path
    token: str
    pid: int
    host: str


class RunLockError(RuntimeError):
    """Raised when another live process owns a formal output directory."""


def normalize_path(value: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(value)))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON in the destination directory and atomically replace ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_completion(run_dir: Path, payload: Mapping[str, Any]) -> Path:
    marker = Path(run_dir) / MARKER_NAME
    atomic_write_json(marker, payload)
    return marker


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes

            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                process_query_limited_information, False, int(pid)
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            error = ctypes.windll.kernel32.GetLastError()
            return error not in {87}  # ERROR_INVALID_PARAMETER => PID does not exist.
        except Exception:
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_lock(path: Path) -> dict[str, Any] | None:
    try:
        value = _load_json(path)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def acquire_run_lock(run_dir: Path) -> RunLock:
    """Atomically claim one formal output directory, recovering only provably stale locks."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / LOCK_NAME
    host = socket.gethostname()
    pid = os.getpid()

    for _ in range(3):
        token = uuid.uuid4().hex
        payload = {
            "schema_version": 1,
            "token": token,
            "pid": pid,
            "host": host,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
        }
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            return RunLock(path=path, token=token, pid=pid, host=host)
        except FileExistsError:
            existing = _read_lock(path)
            if existing is None:
                raise RunLockError(f"Formal run lock exists but is unreadable: {path}")
            owner_host = str(existing.get("host") or "")
            try:
                owner_pid = int(existing.get("pid", -1))
            except (TypeError, ValueError):
                owner_pid = -1
            if owner_host and owner_host != host:
                raise RunLockError(
                    f"Formal run already locked by host={owner_host} pid={owner_pid}: {path}"
                )
            if _pid_is_running(owner_pid):
                raise RunLockError(
                    f"Formal run already active with pid={owner_pid} on {host}: {path}"
                )
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    raise RunLockError(f"Could not acquire formal run lock after stale-lock recovery: {path}")


def release_run_lock(lock: RunLock) -> None:
    """Release a lock only when the on-disk token still belongs to this process."""
    existing = _read_lock(lock.path)
    if not existing or existing.get("token") != lock.token:
        return
    try:
        lock.path.unlink()
    except FileNotFoundError:
        pass


def expected_frame_keys(windows: Iterable[Mapping[str, Any]]) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    for window in windows:
        phase = str(window["phase"])
        segment = int(window["segment"])
        start = int(window["start_frame_idx"])
        end = int(window["end_frame_idx"])
        if end < start:
            raise ValueError(f"Invalid phase window {phase} segment {segment}: {start}..{end}")
        for frame_idx in range(start, end + 1):
            key = (phase, segment, frame_idx)
            if key in keys:
                raise ValueError(f"Duplicate expected frame key: {key}")
            keys.add(key)
    return keys


def _identity_matches(marker: Mapping[str, Any], expected: Mapping[str, Any]) -> str | None:
    for key, value in expected.items():
        actual = marker.get(key)
        if key == "video":
            if normalize_path(str(actual or "")) != normalize_path(str(value)):
                return f"identity mismatch for {key}: {actual!r} != {value!r}"
        elif actual != value:
            return f"identity mismatch for {key}: {actual!r} != {value!r}"
    return None


def _read_frame_keys(path: Path) -> tuple[int, set[tuple[str, int, int]], int]:
    count = 0
    failures = 0
    keys: set[tuple[str, int, int]] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"phase", "phase_segment", "frame_idx", "status"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"frames.csv is missing columns: {sorted(required)}")
        for row in reader:
            count += 1
            key = (str(row["phase"]), int(row["phase_segment"]), int(row["frame_idx"]))
            if key in keys:
                raise ValueError(f"duplicate frame key in frames.csv: {key}")
            keys.add(key)
            failures += int(str(row.get("status", "")) == "video_read_failed")
    return count, keys, failures


def validate_completion(
    run_dir: Path,
    expected_identity: Mapping[str, Any] | None = None,
    *,
    accepted_statuses: tuple[str, ...] = ("complete",),
) -> CompletionValidation:
    """Validate a formal run without trusting process exit code or summary existence."""
    run_dir = Path(run_dir)
    marker_path = run_dir / MARKER_NAME
    if not marker_path.is_file():
        return CompletionValidation(False, f"missing {MARKER_NAME}")

    try:
        marker = _load_json(marker_path)
    except Exception as exc:
        return CompletionValidation(False, f"unreadable {MARKER_NAME}: {exc}")
    if not isinstance(marker, dict):
        return CompletionValidation(False, f"{MARKER_NAME} must contain a JSON object")
    if marker.get("schema_version") != SCHEMA_VERSION:
        return CompletionValidation(False, "unsupported completion schema", marker)
    if marker.get("status") not in accepted_statuses:
        return CompletionValidation(
            False,
            f"status {marker.get('status')!r} is not accepted ({', '.join(accepted_statuses)})",
            marker,
        )
    if expected_identity:
        mismatch = _identity_matches(marker, expected_identity)
        if mismatch:
            return CompletionValidation(False, mismatch, marker)

    required_artifacts = tuple(marker.get("required_artifacts") or ())
    if required_artifacts != REQUIRED_ARTIFACTS:
        return CompletionValidation(False, "required_artifacts contract mismatch", marker)
    for name in required_artifacts:
        artifact = run_dir / name
        if not artifact.is_file():
            return CompletionValidation(False, f"missing required artifact: {name}", marker)
        if marker_path.stat().st_mtime_ns < artifact.stat().st_mtime_ns:
            return CompletionValidation(False, f"completion marker predates {name}", marker)

    try:
        phase_windows = _load_json(run_dir / "phase_windows.json")
        summary = _load_json(run_dir / "summary.json")
        manifest = _load_json(run_dir / "run_manifest.json")
        if not isinstance(phase_windows, list) or not isinstance(summary, dict) or not isinstance(manifest, dict):
            raise ValueError("formal JSON artifacts have unexpected top-level types")
        expected_keys = expected_frame_keys(phase_windows)
        frame_count, actual_keys, failure_count = _read_frame_keys(run_dir / "frames.csv")
        with (run_dir / "eyes.csv").open(newline="", encoding="utf-8-sig") as handle:
            list(csv.reader(handle))
    except Exception as exc:
        return CompletionValidation(False, f"artifact validation failed: {exc}", marker)

    missing = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys
    checks = {
        "expected_frames": len(expected_keys),
        "processed_frames": frame_count,
        "missing_expected_frame_count": len(missing),
        "unexpected_frame_count": len(unexpected),
        "video_read_failure_count": failure_count,
    }
    for key, actual in checks.items():
        if int(marker.get(key, -1)) != int(actual):
            return CompletionValidation(
                False,
                f"marker {key}={marker.get(key)!r} but artifact value is {actual}",
                marker,
            )

    if int(summary.get("processed_frames", -1)) != frame_count:
        return CompletionValidation(False, "summary processed_frames mismatch", marker)
    if summary.get("subject") != marker.get("subject"):
        return CompletionValidation(False, "summary subject mismatch", marker)
    if normalize_path(str(summary.get("video", ""))) != normalize_path(str(marker.get("video", ""))):
        return CompletionValidation(False, "summary video mismatch", marker)
    if summary.get("phases") != marker.get("phases"):
        return CompletionValidation(False, "summary phases mismatch", marker)
    if bool(summary.get("truncated_for_smoke_test")) != bool(marker.get("truncated_for_smoke_test")):
        return CompletionValidation(False, "summary truncation flag mismatch", marker)

    effective = manifest.get("effective_parameters") or {}
    package = manifest.get("package") or {}
    if package.get("version") != marker.get("package_version"):
        return CompletionValidation(False, "manifest package version mismatch", marker)
    if effective.get("phases") != marker.get("phases"):
        return CompletionValidation(False, "manifest phases mismatch", marker)
    if effective.get("max_frames") != marker.get("max_frames"):
        return CompletionValidation(False, "manifest max_frames mismatch", marker)

    if marker.get("status") in {"complete", "recovery_complete"}:
        if (
            marker.get("max_frames") is not None
            or marker.get("truncated_for_smoke_test")
            or (marker.get("status") == "complete" and marker.get("partial_phase_selection"))
        ):
            return CompletionValidation(False, "published run cannot be truncated", marker)
        if marker.get("status") == "recovery_complete" and marker.get("recovery_mode") is not True:
            return CompletionValidation(False, "recovery_complete requires recovery_mode", marker)
        if int(marker.get("decoded_frames", -1)) != len(expected_keys):
            return CompletionValidation(False, "decoded_frames mismatch", marker)
        if missing or unexpected or failure_count:
            return CompletionValidation(False, "complete run has missing/unexpected/failed frames", marker)

    return CompletionValidation(True, "valid completion marker", marker)


def _arg_value(name: str) -> str | None:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return None
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else None


def _formal_guard_spec() -> tuple[Path, dict[str, Any]] | None:
    """Derive the deterministic formal run directory before CUDA/model initialization."""
    if "formal" not in sys.argv:
        return None

    package_root = Path(__file__).resolve().parent
    config_text = _arg_value("--config")
    config_path = Path(config_text).resolve() if config_text else package_root / "config.yaml"
    try:
        import yaml

        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except Exception:
        return None

    video_text = _arg_value("--video")
    subject_text = _arg_value("--subject")
    if video_text:
        video = Path(video_text)
        subject = video.stem.removesuffix("_nir")
        video_identity = str(video.resolve())
    elif subject_text:
        subject = str(subject_text).strip().rstrip("_")
        if not subject.startswith("sub-"):
            subject = f"sub-{subject}"
        video_identity = None
    else:
        return None

    backend = _arg_value("--backend") or str(
        config.get("inference", {}).get("backend", "pytorch-cuda")
    )
    precision = _arg_value("--ritnet-precision") or str(
        config["ritnet"].get("precision", "fp32")
    )
    batch_size = int(
        _arg_value("--ritnet-batch-size") or config["ritnet"].get("batch_size", 16)
    )
    yolo_batch_size = int(
        _arg_value("--yolo-batch-size") or config.get("inference", {}).get("yolo_batch_size", 1)
    )
    phases_text = _arg_value("--phases")
    phases = (
        [value.strip().lower() for value in phases_text.split(",") if value.strip()]
        if phases_text
        else [str(value).strip().lower() for value in config["formal"].get("phases", [])]
    )
    configured_phases = [str(value).strip().lower() for value in config["formal"].get("phases", [])]
    is_full_phase_run = phases == configured_phases
    max_frames_text = _arg_value("--max-frames")
    max_frames = int(max_frames_text) if max_frames_text is not None else None
    recovery_timeline_text = _arg_value("--recovery-timeline")
    recovery_mode = bool(recovery_timeline_text)

    release = str(config["formal"].get("focuswave_release", "v3.1.3"))
    output_text = _arg_value("--output")
    if output_text:
        output_root = Path(output_text)
    else:
        output_root = Path(config["output"]["root"])
        if not output_root.is_absolute():
            output_root = package_root / output_root

    suffixes: list[str] = []
    if backend == "ort-cuda":
        suffixes.append("ort-cuda")
    if not is_full_phase_run:
        suffixes.append("partial-" + "-".join(phases))
    if recovery_mode:
        suffixes.append("recovery")
    if max_frames is not None:
        suffixes.append(f"smoke{max_frames}")
    run_suffix = "_" + "_".join(suffixes) if suffixes else ""
    yolo_suffix = f"_yolo{yolo_batch_size}" if yolo_batch_size > 1 else ""
    run_name = f"{subject}_formal_{release}{yolo_suffix}_b{batch_size}_{precision}{run_suffix}"
    run_dir = output_root / run_name

    use_ritnet = bool(config["ritnet"].get("enabled", True)) and "--skip-ritnet" not in sys.argv
    marker = {
        "schema_version": SCHEMA_VERSION,
        "status": "initializing",
        "subject": subject,
        "video": video_identity,
        "package_version": str(config.get("package", {}).get("version", "")),
        "focuswave_release": release,
        "inference_backend": backend,
        "phases": phases,
        "ritnet_enabled": use_ritnet,
        "ritnet_precision": precision if use_ritnet else "disabled",
        "ritnet_batch_size": batch_size if use_ritnet else 0,
        "yolo_batch_size": yolo_batch_size,
        "max_frames": max_frames,
        "partial_phase_selection": not is_full_phase_run,
        "recovery_mode": recovery_mode,
        "timeline_file": (
            str(Path(recovery_timeline_text).resolve())
            if recovery_timeline_text
            else str((Path(video_identity).parent.parent / "beh" / "master_timeline.csv").resolve())
            if video_identity
            else None
        ),
        "processed_frames": 0,
        "decoded_frames": 0,
        "failure_stage": "initialization",
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "required_artifacts": list(REQUIRED_ARTIFACTS),
    }
    return run_dir, marker


_FORMAL_GUARD_LOCK: RunLock | None = None
_FORMAL_GUARD_RUN_DIR: Path | None = None
_FORMAL_GUARD_ERROR: tuple[str, str] | None = None
_FORMAL_GUARD_PREVIOUS_EXCEPTHOOK = sys.excepthook


def _formal_guard_excepthook(exc_type, exc_value, traceback) -> None:
    global _FORMAL_GUARD_ERROR
    _FORMAL_GUARD_ERROR = (getattr(exc_type, "__name__", str(exc_type)), str(exc_value))
    _FORMAL_GUARD_PREVIOUS_EXCEPTHOOK(exc_type, exc_value, traceback)


def _finalize_formal_guard() -> None:
    global _FORMAL_GUARD_LOCK
    if _FORMAL_GUARD_LOCK is None or _FORMAL_GUARD_RUN_DIR is None:
        return
    try:
        marker_path = _FORMAL_GUARD_RUN_DIR / MARKER_NAME
        marker: dict[str, Any] = {}
        if marker_path.is_file():
            try:
                loaded = _load_json(marker_path)
                if isinstance(loaded, dict):
                    marker = loaded
            except Exception:
                marker = {}
        if marker.get("status") not in TERMINAL_STATUSES:
            marker.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "failure_stage": marker.get("failure_stage") or "unhandled_exit",
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                }
            )
            if _FORMAL_GUARD_ERROR:
                marker["exception_type"] = _FORMAL_GUARD_ERROR[0]
                marker["error"] = _FORMAL_GUARD_ERROR[1]
            else:
                marker["error"] = "formal process exited without publishing a terminal completion status"
            write_completion(_FORMAL_GUARD_RUN_DIR, marker)
    finally:
        release_run_lock(_FORMAL_GUARD_LOCK)
        _FORMAL_GUARD_LOCK = None


def _install_formal_guard() -> None:
    global _FORMAL_GUARD_LOCK, _FORMAL_GUARD_RUN_DIR
    spec = _formal_guard_spec()
    if spec is None:
        return
    run_dir, marker = spec
    lock = acquire_run_lock(run_dir)
    _FORMAL_GUARD_LOCK = lock
    _FORMAL_GUARD_RUN_DIR = run_dir
    write_completion(run_dir, marker)
    sys.excepthook = _formal_guard_excepthook
    atexit.register(_finalize_formal_guard)


_install_formal_guard()
