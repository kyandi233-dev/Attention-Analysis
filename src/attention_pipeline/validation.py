from __future__ import annotations

import subprocess

from .config import Config
from .io import block_windows, load_timestamps, subject_paths
from .protocol import validate_protocol


def validate_all(config: Config) -> dict:
    protocol = validate_protocol(config)
    raw_root = config.path_value("raw_root")
    known_missing = config.data["subjects"].get("known_missing", {})
    subject_results = []
    for subject in config.data["subjects"]["include"]:
        paths = subject_paths(raw_root, subject)
        expected_missing = set(known_missing.get(subject, []))
        files = {}
        for key in ("master_timeline", "nir_video", "nir_timestamps", "rgb_video", "rgb_timestamps"):
            exists = paths[key].exists()
            files[key] = {"exists": exists, "declared_unusable": key in expected_missing}
        block_ok = False
        nir_written = 0
        nir_dropped = 0
        if paths["master_timeline"].exists():
            block_ok = len(block_windows(paths["master_timeline"])) == 6
        if paths["nir_timestamps"].exists():
            ts = load_timestamps(paths["nir_timestamps"])
            nir_written = int((~ts["is_dropped"] & ts["unix_ms"].notna()).sum())
            nir_dropped = int(ts["is_dropped"].sum())
        unexpected = [key for key, value in files.items() if not value["exists"] and not value["declared_unusable"]]
        subject_results.append({
            "subject": subject,
            "ok": not unexpected and block_ok,
            "unexpected_missing": unexpected,
            "files": files,
            "formal_blocks": 6 if block_ok else 0,
            "nir_written_frames": nir_written,
            "nir_dropped_rows": nir_dropped,
            "known_issues": config.data["subjects"].get("known_issues", {}).get(subject, {}),
        })
    runtime_results = {}
    checks = {
        "main_python": ["cv2", "numpy", "pandas", "yaml", "scipy", "mediapipe"],
        "pypupilext_python": ["cv2", "numpy", "pandas", "pypupilext"],
    }
    for name, modules in checks.items():
        executable = config.section("runtimes").get(name)
        command = [str(executable), "-c", ";".join(f"import {module}" for module in modules)]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
            runtime_results[name] = {
                "executable": executable,
                "ok": completed.returncode == 0,
                "required_modules": modules,
                "error": completed.stderr.strip()[-500:] if completed.returncode else "",
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            runtime_results[name] = {"executable": executable, "ok": False, "required_modules": modules, "error": str(exc)}
    return {
        "ok": protocol["ok"] and all(item["ok"] for item in subject_results) and all(x["ok"] for x in runtime_results.values()),
        "protocol": protocol,
        "subjects": subject_results,
        "runtimes": runtime_results,
    }
