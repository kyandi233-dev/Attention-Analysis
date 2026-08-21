from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    path: Path
    data: dict[str, Any]
    digest: str

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"配置缺少字典节: {name}")
        return value

    def path_value(self, name: str) -> Path:
        raw = self.section("paths").get(name)
        if raw is None:
            raise KeyError(f"配置缺少路径: paths.{name}")
        path = Path(str(raw))
        if not path.is_absolute():
            path = (self.path.parent.parent / path).resolve()
        return path


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("配置根节点必须是字典")
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return Config(config_path, data, digest)


def gate_allows(config: Config, stage: str) -> bool:
    return bool(config.section("stages").get(stage, False))

