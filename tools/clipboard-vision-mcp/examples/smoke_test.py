"""Quick local sanity check — does NOT require Groq.

Run after `pip install -e .`:
  1. Copy any image to your clipboard (e.g. screenshot).
  2. python examples/smoke_test.py
"""

from clipboard_vision_mcp.clipboard import save_clipboard_image, ClipboardError


def main() -> None:
    try:
        path = save_clipboard_image()
    except ClipboardError as e:
        print(f"FAIL: {e}")
        return
    print(f"OK: clipboard image saved to {path}")


if __name__ == "__main__":
    main()
