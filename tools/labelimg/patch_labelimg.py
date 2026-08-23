"""Apply the Windows/PyQt5 integer-value compatibility patch to LabelImg 1.8.6.

LabelImg 1.8.6 can pass floats into Qt setValue() calls during scrolling/zooming.
Recent PyQt5 builds require integers. This script applies the four historical
`int(...)` conversions used during this project's NIR eye annotation.

The patch is idempotent: running it again after a successful patch makes no changes.
"""
from __future__ import annotations

import sys
from pathlib import Path


REPLACEMENTS = {
    "bar.setValue(bar.value() + bar.singleStep() * units)":
        "bar.setValue(int(bar.value() + bar.singleStep() * units))",
    "self.zoom_widget.setValue(value)":
        "self.zoom_widget.setValue(int(value))",
    "h_bar.setValue(new_h_bar_value)":
        "h_bar.setValue(int(new_h_bar_value))",
    "v_bar.setValue(new_v_bar_value)":
        "v_bar.setValue(int(new_v_bar_value))",
}


def main() -> int:
    target = Path(sys.prefix) / "Lib" / "site-packages" / "labelImg" / "labelImg.py"
    if not target.exists():
        raise FileNotFoundError(
            f"LabelImg source not found at {target}. Install tools/labelimg/requirements.txt first."
        )

    text = target.read_text(encoding="utf-8")
    changed = 0
    for old, new in REPLACEMENTS.items():
        if new in text:
            continue
        if old not in text:
            raise RuntimeError(f"Expected LabelImg 1.8.6 source pattern not found: {old}")
        text = text.replace(old, new, 1)
        changed += 1

    if changed:
        target.write_text(text, encoding="utf-8")
        print(f"patched {changed} setValue call(s): {target}")
    else:
        print(f"LabelImg patch already applied: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
