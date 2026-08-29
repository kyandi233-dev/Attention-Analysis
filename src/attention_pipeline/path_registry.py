from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*")


def _canonical_digest(data: dict[str, Any]) -> str:
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _resolve_one(raw: Any, base_dir: Path) -> Path:
    if raw is None:
        raise ValueError("路径值不能为 null")
    text = os.path.expanduser(os.path.expandvars(str(raw).strip()))
    unresolved = _ENV_PATTERN.findall(text)
    if unresolved:
        raise ValueError(f"路径仍包含未解析环境变量: {unresolved}")
    path = Path(text)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


@dataclass(frozen=True)
class PathRegistry:
    """Machine-local path registry separated from scientific configuration."""

    path: Path
    data: dict[str, Any]
    digest: str

    def _paths_section(self) -> dict[str, Any]:
        value = self.data.get("paths")
        if not isinstance(value, dict):
            raise KeyError("路径注册表缺少字典节: paths")
        return value

    def raw(self, name: str) -> Any:
        paths = self._paths_section()
        if name not in paths:
            raise KeyError(f"路径注册表缺少逻辑路径: paths.{name}")
        return paths[name]

    def path_value(self, name: str) -> Path:
        raw = self.raw(name)
        if isinstance(raw, list):
            raise TypeError(f"paths.{name} 是路径列表，请使用 path_values()")
        return _resolve_one(raw, self.path.parent)

    def path_values(self, name: str) -> list[Path]:
        raw = self.raw(name)
        values = raw if isinstance(raw, list) else [raw]
        if not values:
            raise ValueError(f"paths.{name} 不能为空列表")
        return [_resolve_one(item, self.path.parent) for item in values]

    def has(self, name: str) -> bool:
        return name in self._paths_section()


def load_path_registry(path: str | Path | None = None) -> PathRegistry:
    """
    Load the machine-local path registry.

    If ``path`` is omitted, ``ATTENTION_ANALYSIS_PATHS_CONFIG`` is used. The
    registry may contain absolute paths, paths relative to the registry file,
    ``~``, and environment variables such as ``${FOCUSWAVE_FORMAL_RAW_ROOT}``.
    """
    source = path or os.environ.get("ATTENTION_ANALYSIS_PATHS_CONFIG")
    if source is None:
        raise ValueError(
            "未提供路径注册表。请使用 --paths-config 或设置 "
            "ATTENTION_ANALYSIS_PATHS_CONFIG。"
        )
    config_path = Path(os.path.expanduser(os.path.expandvars(str(source)))).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("路径注册表根节点必须是字典")
    version = data.get("version")
    if version != 1:
        raise ValueError(f"不支持的路径注册表版本: {version!r}; 当前要求 version: 1")
    if not isinstance(data.get("paths"), dict):
        raise ValueError("路径注册表必须包含 paths 字典")
    return PathRegistry(config_path, data, _canonical_digest(data))
