# -*- coding: utf-8 -*-
"""Launch LabelImg reliably from a rebuildable Python 3.10 environment on Windows.

This preserves the historical workaround used during NIR eye annotation:
- bypass the pip-generated labelImg-script.py wrapper, whose shebang can break on
  non-ASCII installation paths;
- copy Qt platform plugins to an ASCII-only directory before importing PyQt5.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path


def _default_ascii_plugin_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "attention-analysis" / "qtplugins"
    return Path("C:/attention-analysis/qtplugins")


PLUGINS_ASCII = Path(os.environ.get("QT_PLUGINS_ASCII", str(_default_ascii_plugin_dir())))
VENV_PLUGINS = Path(sys.prefix) / "Lib" / "site-packages" / "PyQt5" / "Qt5" / "plugins"


def ensure_ascii_plugins() -> None:
    src_dll = VENV_PLUGINS / "platforms" / "qwindows.dll"
    dst_dll = PLUGINS_ASCII / "platforms" / "qwindows.dll"
    if src_dll.exists() and not dst_dll.exists():
        try:
            shutil.copytree(VENV_PLUGINS, PLUGINS_ASCII, dirs_exist_ok=True)
        except OSError as exc:
            print(f"warning: copy Qt plugins failed: {exc}", file=sys.stderr)
    if dst_dll.exists():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(PLUGINS_ASCII / "platforms")
        os.environ["QT_PLUGIN_PATH"] = str(PLUGINS_ASCII)
    else:
        print("warning: Qt platform plugin not found; GUI may fail to start", file=sys.stderr)


ensure_ascii_plugins()

from labelImg.labelImg import main  # noqa: E402


if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\.pyw?|\.exe)?$", "", sys.argv[0])
    raise SystemExit(main())
