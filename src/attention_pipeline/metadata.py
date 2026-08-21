from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import Config


def source_id(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(chunk_bytes))
    return f"sha256-head:{digest.hexdigest()[:16]}:size:{stat.st_size}"


def run_metadata(config: Config, sources: list[Path] | None = None) -> dict:
    timezone = config.section("pipeline").get("timezone", "Asia/Shanghai")
    return {
        "pipeline_version": config.section("pipeline")["version"],
        "config_digest": config.digest,
        "generated_at": datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds"),
        "sources": [
            {"path": str(path.resolve()), "source_id": source_id(path)}
            for path in (sources or [])
        ],
    }

