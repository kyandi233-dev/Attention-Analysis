# -*- coding: utf-8 -*-
"""labelImg 干净启动器（UTF-8）。

规避中文路径导致的两个问题：
  1. pip 生成的 labelImg-script.py 用 GBK 写 shebang，Python 3 按 UTF-8 读报
     Non-UTF-8 code → 本文件直接调 labelImg.labelImg.main，绕过该包装脚本。
  2. Qt 从含中文的路径加载 qwindows.dll 平台插件会阻塞/报 “no Qt platform plugin”
     → 启动前把 PyQt5 插件目录拷贝到纯 ASCII 路径，并用 QT_QPA_PLATFORM_PLUGIN_PATH /
     QT_PLUGIN_PATH 指给 Qt（必须在 import PyQt5 之前设置）。

用法：
    venv-labelimg\\Scripts\\python.exe venv-labelimg\\Scripts\\labelimg_launch.py ^
        datasets\\nir-eye-dataset-v1\\images\\batch1 ^
        datasets\\nir-eye-dataset-v1\\images\\batch1\\classes.txt ^
        datasets\\nir-eye-dataset-v1\\labels_yolo\\batch1
"""
import os
import re
import shutil
import sys
from pathlib import Path

PLUGINS_ASCII = Path(os.environ.get("QT_PLUGINS_ASCII", "C:/Users/Kyand/AppData/Local/qtplugins"))
VENV_PLUGINS = Path(__file__).resolve().parent.parent / "Lib/site-packages/PyQt5/Qt5/plugins"


def ensure_ascii_plugins() -> None:
    src_dll = VENV_PLUGINS / "platforms" / "qwindows.dll"
    dst_dll = PLUGINS_ASCII / "platforms" / "qwindows.dll"
    if src_dll.exists() and not dst_dll.exists():
        try:
            shutil.copytree(VENV_PLUGINS, PLUGINS_ASCII, dirs_exist_ok=True)
        except OSError as exc:  # 拷贝失败不阻塞启动，仅告警
            print(f"warning: copy Qt plugins failed: {exc}", file=sys.stderr)
    if dst_dll.exists():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(PLUGINS_ASCII / "platforms")
        os.environ["QT_PLUGIN_PATH"] = str(PLUGINS_ASCII)
    else:
        print("warning: Qt platform plugin not found; GUI may fail to start", file=sys.stderr)


ensure_ascii_plugins()

from labelImg.labelImg import main  # noqa: E402  (需在设置环境变量后导入)

if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\.pyw?|\.exe)?$", "", sys.argv[0])
    sys.exit(main())
