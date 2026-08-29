from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .path_registry import PathRegistry, load_path_registry


@dataclass(frozen=True)
class Config:
    path: Path
    data: dict[str, Any]
    digest: str
    path_registry: PathRegistry | None = None

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"配置缺少字典节: {name}")
        return value

    def _require_registry(self) -> PathRegistry:
        if self.path_registry is None:
            raise ValueError(
                "该配置引用机器路径注册表，但当前未加载。请使用 --paths-config "
                "或设置 ATTENTION_ANALYSIS_PATHS_CONFIG。"
            )
        return self.path_registry

    def registry_path(self, name: str) -> Path:
        return self._require_registry().path_value(name)

    def registry_paths(self, name: str) -> list[Path]:
        return self._require_registry().path_values(name)

    def path_value(self, name: str) -> Path:
        raw = self.section("paths").get(name)
        if raw is None:
            raise KeyError(f"配置缺少路径: paths.{name}")

        if isinstance(raw, dict):
            path_key = raw.get("path_key")
            if path_key is None:
                raise ValueError(
                    f"paths.{name} 为字典时必须包含 path_key，不能把机器绝对路径写进科学配置"
                )
            return self.registry_path(str(path_key))

        if isinstance(raw, str) and raw.startswith("@path:"):
            return self.registry_path(raw.split(":", 1)[1])

        path = Path(str(raw))
        if not path.is_absolute():
            path = (self.path.parent.parent / path).resolve()
        return path

    def path_values(self, name: str) -> list[Path]:
        raw = self.section("paths").get(name)
        if raw is None:
            raise KeyError(f"配置缺少路径: paths.{name}")

        if isinstance(raw, dict):
            path_key = raw.get("path_key")
            if path_key is None:
                raise ValueError(f"paths.{name} 为字典时必须包含 path_key")
            return self.registry_paths(str(path_key))

        if isinstance(raw, str) and raw.startswith("@path:"):
            return self.registry_paths(raw.split(":", 1)[1])

        values = raw if isinstance(raw, list) else [raw]
        result: list[Path] = []
        for item in values:
            path = Path(str(item))
            if not path.is_absolute():
                path = (self.path.parent.parent / path).resolve()
            result.append(path)
        return result


def load_config(
    path: str | Path,
    *,
    paths_config: str | Path | None = None,
    use_env_paths: bool = True,
) -> Config:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("配置根节点必须是字典")
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    registry_source: str | Path | None = paths_config
    if registry_source is None and use_env_paths:
        registry_source = os.environ.get("ATTENTION_ANALYSIS_PATHS_CONFIG")
    registry = load_path_registry(registry_source) if registry_source is not None else None
    return Config(config_path, data, digest, registry)


def gate_allows(config: Config, stage: str) -> bool:
    return bool(config.section("stages").get(stage, False))
